"""Mamba2 ranking encoder with dependency-free fallback and regularisers."""

from __future__ import annotations

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class _FallbackBlock(nn.Module):
    """Small causal-ish sequence block used when mamba-ssm is not installed."""

    def __init__(self, d_model: int, kernel_size: int = 5, expand: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size - 1, groups=d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * expand), nn.GELU(), nn.Linear(d_model * expand, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)[:, : x.shape[1]]
        gate, value = self.in_proj(y).chunk(2, dim=-1)
        y = self.out_proj(torch.sigmoid(gate) * torch.tanh(value))
        h = self.norm(residual + self.dropout(y))
        return self.ffn_norm(h + self.dropout(self.ffn(h)))


class _TorchMamba2(nn.Module):
    """Dependency-free Mamba2-style selective state-space block.

    This is a compact PyTorch implementation intended for Windows/CPU
    environments where ``mamba-ssm`` (the CUDA fused kernel) cannot be
    installed.  It keeps the key Mamba2 ingredients: an expanded input and
    gate projection, causal depthwise convolution, input-dependent delta
    (selective time step), and a recurrent diagonal state-space update.  The
    state is diagonal and one scalar per channel, which is considerably
    cheaper than materialising ``[batch, time, channel, d_state]`` while
    retaining the selective memory behaviour needed by this MVP.

    The public call signature matches ``mamba_ssm.Mamba2`` so the encoder can
    switch to the fused implementation without changing model code.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, **_: object):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(expand * d_model)
        self.d_conv = max(2, int(d_conv))
        # Mamba's input projection produces a state/input branch and a gate.
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, self.d_conv,
                              padding=self.d_conv - 1,
                              groups=self.d_inner, bias=True)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        # A is constrained to be negative through softplus for stable decay.
        self.a_log = nn.Parameter(torch.zeros(self.d_inner))
        self.d_skip = nn.Parameter(torch.ones(self.d_inner))
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("_TorchMamba2 expects [batch, time, d_model]")
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        # Conv1d with left padding is causal; drop the right-hand padding.
        u = self.conv(u.transpose(1, 2)).transpose(1, 2)[:, :x.shape[1]]
        u = F.silu(u)
        # Delta is selective: every token/channel gets its own positive step.
        delta = F.softplus(self.dt_proj(u)).clamp_(min=1e-4, max=5.0)
        decay = torch.exp(-delta * F.softplus(self.a_log).view(1, 1, -1))
        state = torch.zeros_like(u[:, 0])
        outputs = []
        # Sequence lengths are short (weekly lookbacks are usually <= 64),
        # making this explicit scan both stable and fast enough on CPU.
        for t in range(u.shape[1]):
            state = decay[:, t] * state + (1.0 - decay[:, t]) * u[:, t]
            outputs.append(state)
        y = torch.stack(outputs, dim=1)
        y = self.norm(y * self.d_skip.view(1, 1, -1))
        return self.out_proj(y * F.silu(gate))


def _make_mamba(d_model: int, d_state: int, d_conv: int, expand: int):
    try:
        from mamba_ssm import Mamba2  # type: ignore

        try:
            return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        except TypeError:
            return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=32)
    except Exception:
        # Keep the Mamba2 architecture available without CUDA/fused kernels.
        # The encoder exposes ``backend='torch_mamba2'`` for experiment
        # reports, distinguishing this implementation from the legacy conv
        # fallback retained below for backwards compatibility.
        try:
            return _TorchMamba2(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        except Exception:
            return None


class MambaEncoder(nn.Module):
    """Encode ``[batch, time, features]`` into an L2-normalised vector."""

    def __init__(self, input_dim: int, d_model: int = 64, embedding_dim: int = 128,
                 n_layers: int = 4, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        blocks = []
        self.using_mamba = True
        self.backend = "mamba_ssm"
        for _ in range(n_layers):
            mamba = _make_mamba(d_model, d_state, d_conv, expand)
            if mamba is None:
                self.using_mamba = False
                self.backend = "torch_fallback"
                blocks.append(_FallbackBlock(d_model, d_conv + 1, expand=expand))
            else:
                if isinstance(mamba, _TorchMamba2):
                    self.backend = "torch_mamba2"
                blocks.append(mamba)
        # Mixed fused/fallback blocks are possible if a custom Mamba install
        # only supports a subset of layers.  Keep a per-block type flag so a
        # single failed layer cannot silently change residual semantics for
        # every other block.
        self._mamba_block_flags = [not isinstance(b, _FallbackBlock) for b in blocks]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(d_model, embedding_dim)

    def _forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden states at every time step, before the embedding head."""
        h = self.input_proj(x)
        for block, is_mamba in zip(self.blocks, self._mamba_block_flags):
            if getattr(self, 'activation_checkpointing', False) and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                output = checkpoint(block, h, use_reentrant=False)
            else:
                output = block(h)
            h = h + self.dropout(output) if is_mamba else output
        return self.norm(h)

    def forward(self, x: torch.Tensor, normalize: bool = True,
                return_sequence: bool = False) -> torch.Tensor:
        """Encode a sequence.

        ``return_sequence=True`` is used by multi-positive temporal contrastive
        learning: the j-th future prefix gets its own embedding.  The default
        remains the final-token embedding used for score inference.
        """
        h = self._forward_hidden(x)
        if return_sequence:
            z = self.embedding(h)
            return F.normalize(z, dim=-1) if normalize else z
        z = self.embedding(h[:, -1])
        return F.normalize(z, dim=-1) if normalize else z


class PairRanker(nn.Module):
    """Shared encoder with multiple score heads averaged for ranking."""

    def __init__(self, input_dim: int, n_score_heads: int = 8, **encoder_kwargs):
        super().__init__()
        self.encoder = MambaEncoder(input_dim, **encoder_kwargs)
        dim = encoder_kwargs.get("embedding_dim", 128)
        self.n_score_heads = max(int(n_score_heads), 1)
        self.score = nn.Linear(dim, self.n_score_heads)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x, normalize=True)
        # Each head produces an independent scalar score.  The mean is the
        # score consumed by the ranking loss and by inference/backtesting.
        return z, self.score(z).mean(dim=-1)

    def pair_forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1, s1 = self(x1)
        z2, s2 = self(x2)
        return z1, z2, s1, s2


def ranknet_loss(score1: torch.Tensor, score2: torch.Tensor, y: torch.Tensor,
                 temperature: float = 1.0, return_mask: bool = False):
    """Stable RankNet loss where ``y`` is -1 or +1."""
    diff = (score1 - score2) / max(float(temperature), 1e-6)
    mask = y.abs() > 0
    loss = F.softplus(-y[mask] * diff[mask]).mean() if mask.any() else diff.sum() * 0.0
    return (loss, mask) if return_mask else loss


def extreme_score_loss(score: torch.Tensor, label: torch.Tensor,
                       temperature: float = 1.0, return_mask: bool = False,
                       top_weight: float = 1.0, bottom_weight: float = 1.0,
                       sample_weight: torch.Tensor | None = None):
    """Binary logistic loss for a *single-stock* score.

    ``label`` is -1 for the bottom cross-sectional tail and +1 for the top
    tail; zeros are the middle 60% and are ignored.  This is deliberately a
    per-stock objective (there is no pair subtraction as in RankNet), while
    the score remains comparable within each date because labels are formed
    from that date's cross-section.
    """
    mask = label.abs() > 0
    margin = score / max(float(temperature), 1e-6)
    if mask.any():
        per_item = F.softplus(-label[mask] * margin[mask])
        weights = torch.where(label[mask] > 0,
                              torch.as_tensor(float(top_weight), dtype=margin.dtype, device=margin.device),
                              torch.as_tensor(float(bottom_weight), dtype=margin.dtype, device=margin.device))
        if sample_weight is not None:
            weights = weights * torch.broadcast_to(sample_weight.to(score), score.shape)[mask]
        loss = (per_item * weights).sum() / weights.sum().clamp_min(1e-12)
    else:
        loss = margin.sum() * 0.0
    return (loss, mask) if return_mask else loss


def _soft_rank(x: torch.Tensor, temperature: float = 0.1,
               valid: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable approximation to the ascending rank of each row.

    ``x`` is ``[..., N]``.  The pairwise sigmoid is inexpensive for the
    weekly cross-sections used by the project (typically 100--200 names) and
    gives a useful gradient to the score head.  A larger temperature makes
    the approximation smoother; it is deliberately separate from the
    InfoNCE temperature.
    """
    tau = max(float(temperature), 1e-4)
    # For a high x_i, more x_j satisfy x_i > x_j, hence a larger rank.
    pair = (x.unsqueeze(-1) - x.unsqueeze(-2)) / tau
    probs = torch.sigmoid(pair)
    if valid is not None:
        # The last axis indexes the item being compared against.  Excluding
        # invalid names here (rather than after summation) avoids changing a
        # valid stock's rank merely because a missing name was filled with 0.
        probs = probs * valid.to(probs.dtype).unsqueeze(-2)
    return probs.sum(dim=-1)


def extreme_return_mask(realized_return, mask=None, fraction=0.1):
    """Select equal-size return tails per date, without using predicted scores."""
    if not 0 < fraction <= 0.5:
        raise ValueError('tail fraction must be in (0, 0.5]')
    valid = torch.isfinite(realized_return)
    if mask is not None:
        valid = valid & mask.bool()
    selected = torch.zeros_like(valid)
    for row in range(len(realized_return)):
        indices = valid[row].nonzero().flatten()
        n = len(indices)
        if n < 2:
            continue
        k = min(max(1, int(n * fraction)), n // 2)
        order = indices[torch.argsort(realized_return[row, indices].detach(), stable=True)]
        selected[row, order[:k]] = True
        selected[row, order[-k:]] = True
    return selected


def rank_ic_loss(score: torch.Tensor, realized_return: torch.Tensor,
                 mask: torch.Tensor | None = None, temperature: float = 0.1,
                 return_ic: bool = False, sample_weight: torch.Tensor | None = None):
    """Differentiable cross-sectional Spearman/RankIC objective.

    ``score`` and ``realized_return`` may be ``[N]`` or ``[B, N]``.  Each row
    is one date (or week), so ranks are never mixed across dates.  Invalid or
    non-tradable names can be excluded with ``mask``.  The loss is
    ``1 - mean(RankIC)``; maximizing RankIC therefore minimizes this term.
    Rows with fewer than three valid names or zero variance contribute zero.

    This is intentionally a soft-rank Pearson correlation rather than a
    detached metric: gradients flow through the model scores while returns
    remain fixed labels.  ``return_ic=True`` returns ``(loss, mean_ic)`` for
    logging.
    """
    if score.ndim == 1:
        score = score.unsqueeze(0)
        realized_return = realized_return.unsqueeze(0)
        squeezed = True
    elif score.ndim == 2:
        squeezed = False
    else:
        raise ValueError("rank_ic_loss expects score with shape [N] or [B, N]")
    if realized_return.shape != score.shape:
        raise ValueError("score and realized_return must have the same shape")
    valid = torch.isfinite(score) & torch.isfinite(realized_return)
    if mask is not None:
        valid = valid & mask.to(device=score.device, dtype=torch.bool)
    # Masking with the row mean keeps invalid values out of both ranks and the
    # correlation.  The subsequent per-row validity check handles sparse
    # cross-sections without introducing a NaN into the batch loss.
    safe_score = torch.where(valid, score, torch.zeros_like(score))
    safe_ret = torch.where(valid, realized_return, torch.zeros_like(realized_return))
    sr = _soft_rank(safe_score, temperature, valid)
    # A constant zero rank for invalid names must not affect valid ranks.  We
    # re-centre with valid counts below and multiply invalid entries out.
    sr = sr * valid.to(sr.dtype)
    rr = _soft_rank(safe_ret.detach(), temperature, valid).detach() * valid.to(sr.dtype)
    count = valid.sum(-1)
    # Ranks still use the complete valid cross-section. Only the correlation
    # moments are weighted, so low-weight names remain ranking competitors.
    weights = valid.to(score.dtype)
    if sample_weight is not None:
        weights = weights * torch.broadcast_to(sample_weight.to(score), score.shape)
    denom = weights.sum(-1).clamp_min(1e-12)
    sr = sr - ((sr * weights).sum(-1, keepdim=True) / denom.unsqueeze(-1))
    rr = rr - ((rr * weights).sum(-1, keepdim=True) / denom.unsqueeze(-1))
    cov = (sr * rr * weights).sum(-1)
    var_s = (sr.square() * weights).sum(-1)
    var_r = (rr.square() * weights).sum(-1)
    ic = cov / (var_s.mul(var_r).clamp_min(1e-8).sqrt())
    row_ok = (count >= 3) & (var_s > 1e-8) & (var_r > 1e-8)
    ic_safe = torch.where(row_ok, ic, torch.zeros_like(ic))
    n_ok = row_ok.sum()
    mean_ic = ic_safe.sum() / n_ok.clamp_min(1).to(ic_safe.dtype)
    loss = 1.0 - mean_ic
    if return_ic:
        return loss, mean_ic.detach()
    return loss


def block_rank_ic_loss(score: torch.Tensor, realized_return: torch.Tensor,
                       mask: torch.Tensor | None = None, n_blocks: int = 4,
                       temperature: float = 0.1,
                       return_stats: bool = False):
    """Rank scores only *between* return quantile blocks.

    ``realized_return`` is sorted independently for every cross-section and
    split into ``n_blocks`` (four by default).  Pairwise constraints are
    generated only for stocks in different blocks; stocks in the same block
    are deliberately left unconstrained.  This reduces noisy gradients when
    two names have nearly identical realised returns while still teaching
    the score head the coarse top-to-bottom ordering.  Return values are
    detached labels, so no future value enters the model input.

    The loss is a smooth pairwise logistic objective:

    ``softplus(-sign(block_i-block_j) * (score_i-score_j) / temperature)``

    for each valid cross-block pair ``i < j``.  Rows with fewer than two
    valid names contribute zero.  With ``return_stats=True`` the function
    returns ``(loss, {"pair_count": ..., "pair_accuracy": ...})``; the
    default remains a scalar tensor for drop-in use in existing training
    loops.
    """
    if score.ndim == 1:
        score = score.unsqueeze(0)
        realized_return = realized_return.unsqueeze(0)
        squeezed = True
    elif score.ndim == 2:
        squeezed = False
    else:
        raise ValueError("block_rank_ic_loss expects score with shape [N] or [B, N]")
    if realized_return.shape != score.shape:
        raise ValueError("score and realized_return must have the same shape")
    blocks_n = max(int(n_blocks), 2)
    valid = torch.isfinite(score) & torch.isfinite(realized_return)
    if mask is not None:
        valid = valid & mask.to(device=score.device, dtype=torch.bool)

    # Assign labels from detached realised returns.  A small row loop is
    # intentional: cross-sections are at most a few hundred names and this
    # avoids fragile tie handling in a vectorised quantile implementation.
    block = torch.full(score.shape, -1, dtype=torch.long, device=score.device)
    for b in range(score.shape[0]):
        idx = torch.nonzero(valid[b], as_tuple=False).flatten()
        count = int(idx.numel())
        if count < 2:
            continue
        order = torch.argsort(realized_return[b, idx].detach(), descending=False,
                              stable=True)
        # Position based assignment gives balanced blocks even when returns
        # contain ties or the number of names is not divisible by n_blocks.
        positions = torch.empty(count, dtype=torch.long, device=score.device)
        positions[order] = torch.arange(count, device=score.device)
        block[b, idx] = torch.div(positions * blocks_n, count,
                                  rounding_mode="floor").clamp_max(blocks_n - 1)

    # Use one copy of every unordered pair.  Besides avoiding duplicate work,
    # this keeps the reported pair count interpretable (for 100 names split
    # into four equal blocks it is 3,750 cross-block pairs).
    n = score.shape[-1]
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=score.device), diagonal=1)
    bi = block.unsqueeze(-1)
    bj = block.unsqueeze(-2)
    pair_mask = upper.unsqueeze(0) & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    pair_mask = pair_mask & (bi >= 0) & (bj >= 0) & (bi != bj)
    direction = torch.sign((bi - bj).to(score.dtype))
    diff = score.unsqueeze(-1) - score.unsqueeze(-2)
    tau = max(float(temperature), 1e-6)
    pair_count = pair_mask.sum()
    if pair_count.item() == 0:
        loss = score.sum() * 0.0
        accuracy = score.sum().detach() * 0.0
    else:
        margins = direction * diff / tau
        loss = F.softplus(-margins[pair_mask]).mean()
        accuracy = (margins[pair_mask] > 0).to(score.dtype).mean().detach()
    if return_stats:
        stats = {
            "pair_count": float(pair_count.detach().item()),
            "pair_accuracy": float(accuracy.item()),
            "block_count": float(blocks_n),
        }
        return loss, stats
    return loss


# Explicit aliases make the objective discoverable under both terminology
# used in experiment configs and the research notes.
quartile_block_loss = block_rank_ic_loss
block_rank_loss = block_rank_ic_loss


def extreme_label_stats(label: torch.Tensor) -> dict[str, float]:
    """Return positive/negative tail counts and balance diagnostics."""
    x = label.detach().reshape(-1)
    pos = int((x > 0).sum().item())
    neg = int((x < 0).sum().item())
    mid = int((x == 0).sum().item())
    total = pos + neg
    return {
        "positive_count": float(pos),
        "negative_count": float(neg),
        "middle_count": float(mid),
        "negative_to_positive": float(neg / max(pos, 1)),
        "tail_balance": float(min(pos, neg) / max(total, 1)),
    }


@torch.no_grad()
def contrastive_diagnostics(z_query: torch.Tensor, z_key: torch.Tensor,
                            labels: torch.Tensor | None = None) -> dict[str, float]:
    """Measure positive/negative counts and cosine separation for InfoNCE.

    For date-grouped tensors ``[B, N, D]`` positives are diagonal pairs and
    all off-diagonal same-date pairs are negatives.  The returned means make
    it easy to detect an accidental excess of negatives or a representation
    that treats every future window as the same stock.
    """
    q = F.normalize(z_query, dim=-1)
    # For multi-positive keys [B,N,H,D], aggregate diagnostics over horizons
    # while preserving the same-date stock denominator.
    if z_key.ndim == 4 and z_query.ndim == 3:
        vals = [contrastive_diagnostics(q, z_key[:, :, j, :]) for j in range(z_key.shape[2])]
        if not vals:
            return {"positive_pairs": 0.0, "negative_pairs": 0.0,
                    "negative_to_positive": 0.0, "positive_cosine": 0.0,
                    "negative_cosine": 0.0, "hardest_negative_cosine": 0.0}
        out = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        # Pair counts are additive across independent horizon tasks.
        out["positive_pairs"] = float(sum(v["positive_pairs"] for v in vals))
        out["negative_pairs"] = float(sum(v["negative_pairs"] for v in vals))
        out["negative_to_positive"] = out["negative_pairs"] / max(out["positive_pairs"], 1.0)
        return out
    k = F.normalize(z_key, dim=-1)
    if q.ndim == 3:
        sim = torch.einsum("bnd,bmd->bnm", q, k)
        n = sim.shape[-1]
        eye = torch.eye(n, dtype=torch.bool, device=sim.device).unsqueeze(0).expand(sim.shape[0], -1, -1)
    elif q.ndim == 2:
        sim = q @ k.transpose(0, 1)
        n = sim.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    else:
        raise ValueError("contrastive_diagnostics expects [N,D] or [B,N,D]")
    pos = sim[eye]
    neg = sim[~eye]
    out = {
        "positive_pairs": float(pos.numel()),
        "negative_pairs": float(neg.numel()),
        "negative_to_positive": float(neg.numel() / max(pos.numel(), 1)),
        "positive_cosine": float(pos.mean().item()) if pos.numel() else 0.0,
        "negative_cosine": float(neg.mean().item()) if neg.numel() else 0.0,
        "hardest_negative_cosine": float(neg.max().item()) if neg.numel() else 0.0,
    }
    if labels is not None:
        out.update({f"score_{k}": v for k, v in extreme_label_stats(labels).items()})
    return out


def temporal_infonce(z_query: torch.Tensor, z_key: torch.Tensor,
                     temperature: float = 0.1,
                     stock_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric in-batch contrastive loss for same-stock future windows.

    ``z_query[k]`` and ``z_key[k]`` are the same stock at two separated times;
    other rows in the batch are negatives. The future encoder is the same
    encoder during MVP pretraining, so no future tensor is used at inference.
    """
    # Multi-positive future prefixes: z_key is [B, N, H, D].  Each horizon is
    # treated as a separate positive task and losses are averaged over H, so
    # increasing the number of future views does not silently reweight the
    # whole objective.
    if z_query.ndim == 3 and z_key.ndim == 4:
        b, n, h, _ = z_key.shape
        if n < 2 or h < 1:
            return z_query.sum() * 0.0
        q = F.normalize(z_query, dim=-1)
        k = F.normalize(z_key, dim=-1)
        losses = []
        for j in range(h):
            logits = torch.einsum('bnd,bmd->bnm', q, k[:, :, j, :]) / max(float(temperature), 1e-6)
            labels = torch.arange(n, device=q.device).expand(b, n)
            losses.append(0.5 * (F.cross_entropy(logits.reshape(b*n, n), labels.reshape(-1)) +
                                 F.cross_entropy(logits.transpose(1, 2).reshape(b*n, n), labels.reshape(-1))))
        return torch.stack(losses).mean()

    # For date-grouped batches, dimensions are [batch_dates, stocks, embed].
    # Compute negatives within each date's stock cross-section, so no sample
    # from another date can become a false negative.
    if z_query.ndim == 3:
        b, n, _ = z_query.shape
        if n < 2:
            return z_query.sum() * 0.0
        q = F.normalize(z_query, dim=-1)
        k = F.normalize(z_key, dim=-1)
        logits = torch.einsum('bnd,bmd->bnm', q, k) / max(float(temperature), 1e-6)
        labels = torch.arange(n, device=q.device).expand(b, n)
        return 0.5 * (F.cross_entropy(logits.reshape(b*n, n), labels.reshape(-1)) +
                      F.cross_entropy(logits.transpose(1, 2).reshape(b*n, n), labels.reshape(-1)))
    if z_query.shape[0] < 2:
        return z_query.sum() * 0.0
    q = F.normalize(z_query, dim=-1)
    k = F.normalize(z_key, dim=-1)
    logits = q @ k.transpose(0, 1) / max(float(temperature), 1e-6)
    labels = torch.arange(q.shape[0], device=q.device)
    # A shuffled loader can put two windows of the same stock in one batch.
    # They are not valid negatives: mask those pairs while preserving the
    # diagonal positive.  This keeps the loss focused on temporal identity
    # rather than allowing the model to separate a stock from itself.
    if stock_ids is not None:
        sid = stock_ids.to(logits.device).reshape(-1)
        same = sid[:, None].eq(sid[None, :])
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, torch.finfo(logits.dtype).min)
        logits_t = logits.transpose(0, 1)
    else:
        logits_t = logits.transpose(0, 1)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits_t, labels))


def group_decorrelation_loss(z: torch.Tensor, group_size: int = 8,
                             balance_weight: float = 0.1, variance_weight: float = 0.1):
    """Penalise covariance between different groups of embedding dimensions.

    The covariance is computed across the batch after centring and scaling,
    while the diagonal is excluded.  Small variance and unbalanced groups are
    softly penalised to avoid the trivial constant-vector solution.
    """
    if z.ndim != 2 or z.shape[0] < 2:
        return z.sum() * 0.0
    n, dim = z.shape
    groups = torch.arange(dim, device=z.device) // max(int(group_size), 1)
    x = z - z.mean(0, keepdim=True)
    std = x.std(0, unbiased=False).clamp_min(1e-4)
    xn = x / std
    cov = xn.T @ xn / max(n - 1, 1)
    different = groups[:, None] != groups[None, :]
    off_group = cov[different]
    decor = (off_group.square().mean() if off_group.numel() else cov.sum() * 0.0)
    group_energy = z.reshape(n, -1, max(int(group_size), 1)).square().mean((0, 2))
    balance = (group_energy - group_energy.mean()).square().mean()
    variance = F.relu(0.05 - std).square().mean()
    return decor + balance_weight * balance + variance_weight * variance


def output_decorrelation_loss(head_scores: torch.Tensor,
                              mask: torch.Tensor | None = None) -> torch.Tensor:
    """Optional ``|offdiag(Corr(s_1, ..., s_K))|_F^2`` objective.

    For [B, N, K], compute a separate K-by-K correlation across N stocks
    for each week, then average the off-diagonal squared Frobenius norms.
    A mask excludes missing labels. Constant heads are stabilized by epsilon;
    this penalty alone does not guarantee non-collapsing or profitable heads.
    """
    if head_scores.ndim < 2:
        raise ValueError("head_scores must have shape [..., n_heads]")
    x = head_scores
    if x.shape[-2] < 2 or x.shape[-1] < 2:
        return x.sum() * 0.0
    valid = torch.isfinite(x).all(-1)
    if mask is not None:
        valid = valid & mask.bool()
    count = valid.sum(-1)
    safe = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
    mean = safe.sum(-2, keepdim=True) / count.clamp_min(1)[..., None, None]
    centered = (safe - mean) * valid.unsqueeze(-1)
    cov = centered.transpose(-2, -1) @ centered / (count-1).clamp_min(1)[..., None, None]
    std = cov.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2)).clamp_min(1e-12)
    offdiag = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
    per_week = offdiag.square().sum(dim=(-2,-1))
    usable = count >= 2
    return torch.where(usable, per_week, torch.zeros_like(per_week)).sum() / usable.sum().clamp_min(1)
