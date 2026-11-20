from pathlib import Path

import pandas as pd
import pytest

from cyberlab_data import DatasetBlocked, get_adapter, load_split
from cyberlab_data.unsw_nb15 import prepare, write_fixture
from cyberlab_protocol.intent import split_from_goal
from cyberlab_schemas import SplitSpec


def test_registry_lookup() -> None:
    assert get_adapter("UNSW-NB15").key == "unsw_nb15"
    assert get_adapter("unsw-nb15-v1@abc123").key == "unsw_nb15"
    assert get_adapter("CIC-IDS2017").key == "cicids2017"
    assert get_adapter(None).key == "cicids2017"


def test_unsw_official_split(tmp_path: Path) -> None:
    write_fixture(tmp_path / "raw")
    p = prepare(tmp_path, SplitSpec(strategy="official"))
    m = p.manifest
    assert m.status == "verified", m.blocked_reasons
    assert "BENIGN" in m.class_counts and "Backdoor" in m.class_counts
    assert "Reconnaissance" in m.class_counts  # stray spaces normalised
    assert not any(f in m.features for f in ("id", "label", "attack_cat", "proto"))
    assert "proto=tcp" in m.features  # an all-zero "other" column is dropped as constant
    df = pd.read_parquet(p.clean_path)
    assert set(df.loc[df["split"] == "test", "Part"]) == {"test"}
    assert set(df.loc[df["split"] != "test", "Part"]) == {"train"}
    assert m.split_overlap_rows == 0
    x, y = load_split(p.root, "test", "binary")
    assert set(y) == {"BENIGN", "ATTACK"} and len(x) == len(y)


def test_unsw_stratified_and_no_days(tmp_path: Path) -> None:
    write_fixture(tmp_path / "raw")
    assert prepare(tmp_path).manifest.status == "verified"
    with pytest.raises(DatasetBlocked):
        prepare(tmp_path, SplitSpec(strategy="by_day", test_days=["Friday"]))


def test_unsw_missing_files_explains(tmp_path: Path) -> None:
    with pytest.raises(DatasetBlocked) as exc:
        prepare(tmp_path)
    assert "UNSW_NB15_training-set.csv" in str(exc.value)


def test_official_split_from_goal() -> None:
    assert split_from_goal("Use the official train/test split of UNSW-NB15.") == {
        "strategy": "official"
    }
    assert split_from_goal("Compare models with macro-F1.") is None


def test_netflow_adapter_drops_identifiers(tmp_path: Path) -> None:
    from cyberlab_data import ADAPTERS

    a = ADAPTERS["nf_unsw_nb15_v2"]
    a.write_fixture(tmp_path / "raw")
    m = a.prepare(tmp_path).manifest
    assert m.status == "verified", m.blocked_reasons
    assert not any(f in m.features for f in ("IPV4_SRC_ADDR", "L4_SRC_PORT", "Label", "Attack"))
    assert "BENIGN" in m.class_counts and "DDoS" in m.class_counts


def test_ctu13_by_scenario(tmp_path: Path) -> None:
    import json

    from cyberlab_data import ADAPTERS

    a = ADAPTERS["ctu13"]
    a.write_fixture(tmp_path / "raw")
    p = a.prepare(tmp_path, SplitSpec(strategy="by_group", test_days=["3"]))
    assert p.manifest.status == "verified", p.manifest.blocked_reasons
    assert set(p.manifest.class_counts) == {"BENIGN", "Botnet"}  # background excluded
    df = pd.read_parquet(p.clean_path)
    assert set(df.loc[df["split"] == "test", "Group"]) == {"3"}
    notes = json.loads((p.root / "preprocessing_notes.json").read_text())
    assert notes["background_or_unlabelled_flows_excluded"] > 0


def test_cross_dataset_split(tmp_path: Path) -> None:
    from cyberlab_data import ADAPTERS, DatasetBlocked, prepare_split

    for key in ("nf_unsw_nb15_v2", "nf_ton_iot_v2"):
        ADAPTERS[key].write_fixture(tmp_path / key / "raw")
    split = SplitSpec(strategy="cross_dataset", test_days=["NF-ToN-IoT-v2"])
    p = prepare_split(ADAPTERS["nf_unsw_nb15_v2"], tmp_path, split)
    assert p.manifest.status == "verified", p.manifest.blocked_reasons
    df = pd.read_parquet(p.clean_path)
    assert set(df.loc[df["split"] == "test", "Part"]) == {"test"}
    assert p.manifest.split_overlap_rows == 0
    # CIC-IDS2017 has a different feature format: refused, not silently substituted
    import pytest

    with pytest.raises(DatasetBlocked):
        prepare_split(ADAPTERS["cicids2017"], tmp_path, split)


def test_phishing_text_grouped_split(tmp_path: Path) -> None:
    from cyberlab_data import ADAPTERS

    a = ADAPTERS["phishing_urls"]
    a.write_fixture(tmp_path / "raw")
    p = a.prepare(tmp_path, SplitSpec(strategy="grouped"))
    assert p.manifest.status == "verified", p.manifest.blocked_reasons
    assert p.manifest.features == ["text:URL"]
    df = pd.read_parquet(p.clean_path)
    train_hosts = set(df.loc[df["split"] == "train", "Group"])
    assert not train_hosts & set(df.loc[df["split"] == "test", "Group"])


def test_preprocessing_audit_accounts_for_every_row(tmp_path: Path) -> None:
    import json

    from cyberlab_data import ADAPTERS, prepare, write_fixture

    write_fixture(tmp_path / "cic" / "raw", rows_per_class=60)
    p = prepare(tmp_path / "cic", benign_cap=50)
    audit = json.loads((p.root / "data_preprocessing_audit.json").read_text())
    assert audit["status"] == "PASS" and audit["consistent"], audit["problems"]
    names = [s["stage"] for s in audit["stages"]]
    assert names[1:] == [
        "missing or infinite values",
        "exact duplicates",
        "conflicting labels",
        "class down-sampling",
    ]
    assert audit["total_removed"] == audit["raw_rows"] - p.manifest.n_rows_clean
    assert audit["stages"][-1]["removed_per_class"]  # BENIGN capped
    for key in ("ctu13", "nf_unsw_nb15_v2", "phishing_urls"):
        a = ADAPTERS[key]
        a.write_fixture(tmp_path / key / "raw")
        q = a.prepare(tmp_path / key)
        r = json.loads((q.root / "data_preprocessing_audit.json").read_text())
        assert r["consistent"], (key, r["problems"])
