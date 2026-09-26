"""L-G-L-G-L industry attention with one market token and temporal-only InfoNCE."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import MambaEncoder


class GatedAttention(nn.Module):
    def __init__(self, dim, heads, dropout=.1, gate=.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2*dim, dim))
        self.dropout = nn.Dropout(dropout)
        self.attn_gate = nn.Parameter(torch.full((dim,), math.atanh(gate)))
        self.ffn_gate = nn.Parameter(torch.full((dim,), math.atanh(gate)))

    def forward(self, x, valid):
        h = self.norm1(x)
        update, _ = self.attn(h, h, h, key_padding_mask=~valid, need_weights=False)
        x = x + torch.tanh(self.attn_gate)*self.dropout(update)
        return x + torch.tanh(self.ffn_gate)*self.dropout(self.ffn(self.norm2(x)))


class TemporalTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['embedding_dim']
        self.encoder = MambaEncoder(cfg['n_features'], d_model=cfg['d_model'],
            embedding_dim=dim, n_layers=cfg['n_layers'], dropout=cfg.get('dropout', .1))
        self.projector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, windows):
        return F.normalize(self.projector(self.encoder(windows, normalize=False)), dim=-1)


class HierarchicalRanker(nn.Module):
    def __init__(self, cfg, industry_ids):
        super().__init__()
        self.temporal = TemporalTower(cfg)
        dim = cfg['embedding_dim']
        ids = torch.as_tensor(industry_ids, dtype=torch.long)
        self.register_buffer('industry_ids', ids)
        self.n_industries = int(ids.max())+1
        counts = torch.bincount(ids)
        if not (counts == counts[0]).all():
            raise ValueError('Industry groups must have equal stock counts')
        self.group_size = int(counts[0])
        self.industry_tokens = nn.Parameter(torch.randn(self.n_industries, dim)*.02)
        self.market_token = nn.Parameter(torch.randn(1, dim)*.02)
        self.market_projection = nn.Sequential(nn.Linear(cfg.get('market_features',6),dim), nn.GELU(), nn.Linear(dim,dim))
        layers = cfg.get('global_layers',2)
        args = dict(dim=dim, heads=cfg.get('attention_heads',4), dropout=cfg.get('dropout',.1), gate=cfg.get('relation_gate_init',.1))
        self.local_layers = nn.ModuleList([GatedAttention(**args) for _ in range(layers+1)])
        self.global_layers = nn.ModuleList([GatedAttention(**args) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim,1)
        self.industry_head = nn.Linear(dim,1) if float(cfg.get('industry_head_weight',0.))>0 else None
        self.chunk_size = cfg.get('encoder_chunk_size',580)
        self.activation_checkpointing = cfg.get('activation_checkpointing',True)

    def encode(self, windows):
        from torch.utils.checkpoint import checkpoint
        flat = windows.reshape(-1, *windows.shape[-2:])
        result=[]
        for chunk in flat.split(self.chunk_size):
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                h = checkpoint(self.temporal.encoder, chunk, normalize=False, use_reentrant=False)
            else:
                h = self.temporal.encoder(chunk, normalize=False)
            result.append(h)
        return torch.cat(result).reshape(*windows.shape[:2], -1)

    def relate(self, temporal, market, available, industry_ids=None):
        ids = self.industry_ids if industry_ids is None else industry_ids
        order = torch.argsort(ids, stable=True)
        inverse = torch.argsort(order)
        batch, stocks, dim = temporal.shape
        mask = available[:, order].reshape(batch,self.n_industries,self.group_size)
        h = temporal[:, order].reshape(batch,self.n_industries,self.group_size,dim)
        h = h.masked_fill(~mask[..., None], 0.)
        group_valid = mask.any(-1)
        pooled = h.sum(2)/mask.sum(2).clamp_min(1)[..., None]
        reps = pooled + self.industry_tokens[None]
        market_token = self.market_token[None].expand(batch,-1,-1) + self.market_projection(market)[:,None]
        market_token = market_token + (pooled*group_valid[...,None]).sum(1,keepdim=True)/group_valid.sum(1).clamp_min(1)[:,None,None]
        for index, local in enumerate(self.local_layers):
            tokens = torch.cat([h,reps[:,:,None]],2).reshape(batch*self.n_industries,self.group_size+1,dim)
            valid = torch.cat([mask,torch.ones_like(mask[:,:,:1])],2).reshape(batch*self.n_industries,-1)
            tokens = local(tokens,valid).reshape(batch,self.n_industries,self.group_size+1,dim)
            h, reps = tokens[:,:,:self.group_size],tokens[:,:,-1]
            if index < len(self.global_layers):
                shared = torch.cat([reps,market_token],1)
                valid = torch.cat([group_valid,torch.ones_like(group_valid[:,:1])],1)
                shared = self.global_layers[index](shared,valid)
                reps,market_token = shared[:,:-1],shared[:,-1:]
        h = h.reshape(batch,stocks,dim)[:,inverse]
        z = F.normalize(self.final_norm(h),dim=-1)
        industry = self.industry_head(reps).squeeze(-1) if self.industry_head is not None else None
        return z,self.score(z).squeeze(-1),industry

    def forward(self, windows, market, available):
        temporal = self.encode(windows)
        z,scores,industry = self.relate(temporal,market,available)
        return temporal,z,scores,industry


def industry_exposure_loss(scores, industry_ids, valid, industries=None):
    """Week-wise share of score variance explained by industry means (eta squared).

    Minimising it makes the cross-industry part of the score flat while leaving
    within-industry ordering free.  Returns values in [0, 1]; a random score of
    the usual width already sits near ``groups/stocks``, so read it together
    with ``industry_exposure_baseline``.
    """
    if industries is None:industries=int(industry_ids.max())+1
    losses=[]
    for row,(score,group,keep) in enumerate(zip(scores,industry_ids,valid)):
        score,score_group=score[keep],group[keep]
        if score_group.numel()<3 or float(score.std())<=1e-6:continue
        z=(score-score.mean())/(score.std(unbiased=False)+1e-6)
        onehot=F.one_hot(score_group.long(),industries).to(z.dtype)
        totals=(onehot.T@z).square()/onehot.sum(0).clamp_min(1)
        losses.append(totals.sum()/z.numel())
    return torch.stack(losses).mean() if losses else scores.sum()*0


def industry_exposure_baseline(counts_per_industry):
    """Eta squared expected from a score unrelated to industry.

    Averaging over the assignments of one week's scores to its own industry
    groups gives ``E[eta2] = (G - (N-G)/(N-1)) / N`` for ``G`` non-empty groups
    holding ``N`` names: exactly 1 when every name is its own group, 0 when the
    week has a single group, and about ``G/N`` for a balanced market.
    """
    kept=[int(count) for count in counts_per_industry if count>0]
    stocks=sum(kept);groups=len(kept)
    if stocks<3 or groups<2:return None
    return (groups-(stocks-groups)/(stocks-1))/stocks


def top_k_weight_vector(scores, capacity, iterations=40):
    """Water-filling projection of each row onto ``{0 <= w <= capacity, sum w = 1}``.

    With ``capacity = 1/top_n`` this is the traded book itself: the names above the
    threshold saturate at ``1/top_n``, the rest get exactly zero, and only the
    marginal names around the cut carry gradient.  The threshold is solved without
    gradient and the result is renormalised, so the row sums to one differentiably.
    """
    lower = scores.min(dim=1, keepdim=True).values - 1.
    upper = scores.max(dim=1, keepdim=True).values
    for _ in range(iterations):
        middle = (lower + upper) / 2
        overflow = (scores - middle).clamp(min=0., max=capacity).sum(dim=1, keepdim=True) > 1.
        lower = torch.where(overflow, middle, lower)
        upper = torch.where(overflow, upper, middle)
    weights = (scores - lower).clamp(min=0., max=capacity)
    return weights / weights.sum(dim=1, keepdim=True)


def weekly_stickiness_loss(scores, past_scores, valid, past_valid, temperature_mult=.5, capacity=None, dead_zone=None):
    """Mean L1 distance between this week's and last week's score-induced weights.

    Turnover, not ranking, is what the costs are spent on here, and mass is
    conserved (the universe is fixed, so the FP source term is zero), which makes
    ``|w_t - w_{t-1}|_1 / 2`` the minimal flux carrying the change, i.e. the traded
    amount itself.  ``capacity`` projects onto the Top-N weight simplex, which is
    the book exactly; otherwise the Gibbs measure ``softmax(s / temperature_mult sd)``
    is used, which is smoother but also charges churn below rank N.  Uncovered
    names carry zero mass rather than diluting the comparison.

    ``dead_zone`` is stated in traded fraction of the book and turns the penalty
    into the hinge ``relu(|dw|_1 - 2 dead_zone)``, i.e. the no-trade region: churn
    the execution band already tolerates is free and only the excess is billed,
    which is the policy the band measures rather than a demand for a frozen book.
    """
    losses=[]
    for score,past,keep,past_keep in zip(scores,past_scores,valid,past_valid):
        if int(keep.sum())<3 or int(past_keep.sum())<3:continue
        if capacity is None:
            temperature=score.detach()[keep].std(unbiased=True).clamp_min(1e-6)*temperature_mult
            current=torch.zeros_like(score); previous=torch.zeros_like(past)
            current[keep]=F.softmax(score[keep]/temperature,dim=-1)
            previous[past_keep]=F.softmax(past[past_keep]/temperature,dim=-1)
        else:
            masked=torch.where(keep,score,score.detach().min()-2.)
            past_masked=torch.where(past_keep,past,past.detach().min()-2.)
            current=top_k_weight_vector(masked[None],capacity)[0]
            previous=top_k_weight_vector(past_masked[None],capacity)[0]
        flow=(current-previous).abs().sum()
        losses.append(flow if not dead_zone else F.relu(flow-2*dead_zone))
    return torch.stack(losses).mean() if losses else scores.sum()*0


def masked_infonce(query, key, valid, temperature=.1, groups=None):
    """Week-local contrastive task. ``groups`` optionally restricts negatives
    to the same industry, turning the task from market-wide identification into
    within-industry identification (the anchor's own future stays positive)."""
    losses=[]
    for row,(q,k,mask) in enumerate(zip(query,key,valid)):
        q,k=q[mask],k[mask]
        if len(q)<2:continue
        logits=F.normalize(q,dim=-1)@F.normalize(k,dim=-1).T/temperature
        labels=torch.arange(len(q),device=q.device)
        if groups is not None:
            same=groups[row][mask]
            same=same[:,None]==same[None,:]
            logits=logits.masked_fill(~same,-1e9)
        losses.append((F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2)
    return torch.stack(losses).mean() if losses else query.sum()*0
