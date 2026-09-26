"""Minimal daily cross-sectional ranking prototype.

The package deliberately keeps data acquisition and modelling independent so
the same cached dataset can be reused by notebooks, training jobs and plots.
"""

from .data import DataConfig, DailyDataStore, WindowPairDataset, TemporalScoreDataset, TemporalCrossSectionDataset, add_features, load_data, dataset_arrays
from .model import (MambaEncoder, PairRanker, ranknet_loss, extreme_score_loss,
                    rank_ic_loss, extreme_label_stats, contrastive_diagnostics,
                    temporal_infonce, group_decorrelation_loss, output_decorrelation_loss,
                    block_rank_ic_loss, block_rank_loss, quartile_block_loss)

__all__ = [
    "DataConfig",
    "DailyDataStore",
    "WindowPairDataset",
    "TemporalScoreDataset",
    "TemporalCrossSectionDataset",
    "add_features",
    "load_data", "dataset_arrays",
    "MambaEncoder",
    "PairRanker",
    "ranknet_loss",
    "extreme_score_loss",
    "rank_ic_loss",
    "extreme_label_stats",
    "contrastive_diagnostics",
    "temporal_infonce",
    "group_decorrelation_loss",
    "output_decorrelation_loss",
    "block_rank_ic_loss",
    "block_rank_loss",
    "quartile_block_loss",
]
