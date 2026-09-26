"""Trace a real cached week through the saved single-head encoder, without training."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.data import FEATURES, dataset_arrays, aggregate_weekly
from quant_mvp.model import PairRanker


def main():
    torch.set_num_threads(2)
    source = Path('outputs/real_weekly_500_full_rankic_10ep')
    cfg = yaml.safe_load((source/'config.yaml').read_text())
    panel = pd.read_parquet(cfg['cache_path'])
    symbol, date = '000002.SZ', pd.Timestamp('2025-03-14')
    X, _, dates, symbols = dataset_arrays(panel, cfg['lookback'], cfg['n_features'], cfg['train_end'])
    ti = list(dates).index(date)
    si = symbols.index(symbol)
    raw = panel.set_index(['date', 'symbol']).reindex(pd.MultiIndex.from_product([dates, symbols]))[FEATURES].to_numpy().reshape(len(dates), len(symbols), -1)
    filled = np.nan_to_num(raw, nan=0., posinf=0., neginf=0.)
    train = filled[:cfg['train_end']].reshape(-1, len(FEATURES))
    mu, sd = train.mean(0), train.std(0)+1e-6
    scaled = np.clip((filled-mu)/sd, -8, 8)
    cs_mu, cs_sd = scaled[ti].mean(0), scaled[ti].std(0)+1e-6
    final = np.clip((scaled[ti, si]-cs_mu)/cs_sd, -6, 6)
    np.testing.assert_allclose(final, X[ti, si], atol=1e-6)
    row = panel[(panel.symbol == symbol) & (panel.date == date)].iloc[0]
    model = PairRanker(cfg['n_features'], n_score_heads=1, d_model=cfg['d_model'],
                       embedding_dim=cfg['embedding_dim'], n_layers=cfg['n_layers'])
    model.load_state_dict(torch.load(source/'best.pt', map_location='cpu', weights_only=True))
    model.eval()
    assert model.encoder.backend == 'torch_mamba2'
    inputs = torch.from_numpy(X[ti-39:ti+1, si][None])
    with torch.no_grad():
        projected = model.encoder.input_proj(inputs)
        block = model.encoder.blocks[0]
        u, gate = block.in_proj(projected).chunk(2, dim=-1)
        conv = block.conv(u.transpose(1, 2)).transpose(1, 2)[:, :40]
        conv_manual = (u[0, -4:, 0]*block.conv.weight[0, 0]).sum() + block.conv.bias[0]
        torch.testing.assert_close(conv_manual, conv[0, -1, 0])
        activated = torch.nn.functional.silu(conv)
        delta = torch.nn.functional.softplus(block.dt_proj(activated)).clamp(min=1e-4, max=5.)
        decay = torch.exp(-delta * torch.nn.functional.softplus(block.a_log)[None, None])
        hidden = model.encoder._forward_hidden(inputs)
        unnormalized = model.encoder.embedding(hidden[:, -1])
        embedding, score = model(inputs)
    saved = pd.read_parquet(source/'predictions.parquet')
    expected = saved[(saved.symbol == symbol) & (saved.date == str(date.date()))].score.iloc[0]
    np.testing.assert_allclose(float(score[0]), expected, atol=1e-6)
    result = dict(symbol=symbol, date=str(date.date()), checkpoint=str(source/'best.pt'),
        backend=model.encoder.backend, window_start=str(pd.Timestamp(dates[ti-39]).date()),
        train_start=str(pd.Timestamp(dates[0]).date()), train_end=str(pd.Timestamp(dates[cfg['train_end']-1]).date()),
        previous_close=float(panel[(panel.symbol == symbol) & (panel.date == dates[ti-1])].close.iloc[0]),
        weekly={k: float(row[k]) for k in ['open', 'high', 'low', 'close', 'volume', 'amount']},
        features=[dict(name=f, raw=float(raw[ti, si, k]), training_mean=float(mu[k]),
            training_std=float(sd[k]), after_training_z=float(scaled[ti, si, k]),
            cross_section_mean=float(cs_mu[k]), cross_section_std=float(cs_sd[k]),
            model_input=float(X[ti, si, k])) for k, f in enumerate(FEATURES)],
        shapes=dict(input=list(inputs.shape), projected=list(projected.shape),
                    expanded_branch=list(u.shape), hidden=list(hidden.shape), embedding=list(embedding.shape)),
        first_block_channel0=dict(last4_expanded_inputs=u[0, -4:, 0].tolist(),
            conv_weights=block.conv.weight[0, 0].tolist(), conv_bias=float(block.conv.bias[0].detach()),
            conv_result=float(conv_manual), after_silu=float(activated[0, -1, 0]),
            delta=float(delta[0, -1, 0]), decay=float(decay[0, -1, 0])),
        projected_last_week_first8=projected[0, -1, :8].tolist(),
        embedding=embedding[0].tolist(), embedding_norm=float(embedding.norm()),
        pre_normalization_norm=float(unnormalized.norm()), score=float(score[0]), saved_score=float(expected))
    daily_path = Path('data/yahoo_500_raw')/(symbol+'.parquet')
    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
        week = daily[(daily.date > date-pd.Timedelta(days=7)) & (daily.date <= date)].copy()
        agg = aggregate_weekly(week).iloc[0]
        for col in result['weekly']:
            np.testing.assert_allclose(agg[col], result['weekly'][col], rtol=1e-7)
        week.date = week.date.dt.strftime('%Y-%m-%d')
        result['daily_bars'] = week[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].to_dict('records')
    out = Path('outputs/single_score_500_style_analysis/embedding_example.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
