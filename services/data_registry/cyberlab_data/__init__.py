"""Dataset registry and preprocessing adapters (spec Step 13)."""

from cyberlab_data.cicids2017 import CICIDS2017, prepare, write_fixture
from cyberlab_data.common import (
    BENIGN,
    DatasetBlocked,
    PreparedDataset,
    dataset_record_id,
    is_default_split,
    load_split,
    manifest_summary,
    prepared_dir_for,
)
from cyberlab_data.registry import ADAPTERS, DatasetAdapter, get_adapter

__all__ = [
    "ADAPTERS",
    "BENIGN",
    "CICIDS2017",
    "DatasetAdapter",
    "DatasetBlocked",
    "PreparedDataset",
    "dataset_record_id",
    "get_adapter",
    "is_default_split",
    "load_split",
    "manifest_summary",
    "prepared_dir_for",
    "prepare",
    "write_fixture",
]
