"""Development checkout shim; installed package lives under ``src/quant_mvp``."""
from pathlib import Path
_src = Path(__file__).resolve().parents[1] / "src" / "quant_mvp"
if str(_src) not in __path__:
    __path__.append(str(_src))
from .data import DataConfig, DailyDataStore, WindowPairDataset, TemporalScoreDataset, TemporalCrossSectionDataset, add_features, load_data, dataset_arrays
from .model import (MambaEncoder, PairRanker, ranknet_loss, extreme_score_loss,
                    temporal_infonce, group_decorrelation_loss,
                    block_rank_ic_loss, block_rank_loss, quartile_block_loss)

__all__ = ["DataConfig", "DailyDataStore", "WindowPairDataset", "TemporalScoreDataset", "TemporalCrossSectionDataset", "add_features", "load_data", "dataset_arrays", "MambaEncoder", "PairRanker", "ranknet_loss", "extreme_score_loss", "temporal_infonce", "group_decorrelation_loss", "block_rank_ic_loss", "block_rank_loss", "quartile_block_loss"]
