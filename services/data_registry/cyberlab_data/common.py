"""Shared, dataset-independent preparation: cleaning, de-duplication, leakage-safe splits,
the split gate and the manifest. Each dataset adapter only loads its raw files into a frame with
a canonical `Class` column (the normal-traffic class is always called BENIGN) and an optional
group column (CIC-IDS2017: capture `Day`; UNSW-NB15: official `Part` train/test).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cyberlab_schemas import DatasetManifest, SplitSpec

BENIGN = "BENIGN"


class DatasetBlocked(RuntimeError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class PreparedDataset:
    manifest: DatasetManifest
    root: Path

    @property
    def clean_path(self) -> Path:
        return self.root / "clean.parquet"


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    source_url: str
    license: str
    citation: str
    citation_doi: str
    download_url: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_default_split(split: SplitSpec | None) -> bool:
    return split is None or split.model_dump() == SplitSpec().model_dump()


def split_key(split: SplitSpec) -> str:
    """Short stable id of a split definition (part of the prepared directory name)."""
    return hashlib.sha256(split.model_dump_json().encode()).hexdigest()[:12]


def prepared_dir_for(root: Path, split: SplitSpec | None) -> Path:
    """Each split the protocols ask for gets its own prepared copy (the default keeps the
    original location), so one study's split can never be silently replaced by another's."""
    return root / "prepared" if is_default_split(split) else root / f"prepared-{split_key(split)}"  # type: ignore[arg-type]


def dataset_record_id(dataset_id: str, split: SplitSpec | None) -> str:
    return dataset_id if is_default_split(split) else f"{dataset_id}@{split_key(split)}"  # type: ignore[arg-type]


def _test_mask(df: pd.DataFrame, split: SplitSpec) -> np.ndarray | None:
    """Rows fixed to the test set by the split design, or None for a stratified split."""
    if split.strategy == "by_day":
        if "Day" not in df.columns:
            raise DatasetBlocked(["a by-day split needs capture days (CIC-IDS2017 only)"])
        test_days = {d.capitalize() for d in split.test_days}
        if not test_days:
            raise DatasetBlocked(["by_day split needs test_days"])
        return df["Day"].isin(test_days).to_numpy()
    if split.strategy == "official":
        if "Part" not in df.columns:
            raise DatasetBlocked(["this dataset has no official train/test partition"])
        return (df["Part"] == "test").to_numpy()
    return None


def assign_split(df: pd.DataFrame, split: SplitSpec, rng: np.random.Generator) -> np.ndarray:
    labels = np.empty(len(df), dtype=object)
    fixed = _test_mask(df, split)
    if fixed is not None:
        labels[:] = "train"
        labels[fixed] = "test"
        train_idx = np.flatnonzero(labels == "train")
        n_val = int(len(train_idx) * split.val_fraction)
        labels[rng.choice(train_idx, size=n_val, replace=False)] = "val"
        return labels
    for _cls, idx in df.groupby("Class").indices.items():
        idx = rng.permutation(idx)
        n_test = max(1, round(len(idx) * split.test_fraction))
        n_val = round(len(idx) * split.val_fraction) if len(idx) - n_test > 2 else 0
        labels[idx[:n_test]] = "test"
        labels[idx[n_test : n_test + n_val]] = "val"
        labels[idx[n_test + n_val :]] = "train"
    return labels


def finalize(
    df: pd.DataFrame,
    *,
    root: Path,
    split: SplitSpec,
    dataset_id: str,
    info: DatasetInfo,
    files: dict[str, str],
    group_column: str | None,
    benign_cap: int | None,
    class_cap: int | None,
    extra_notes: dict[str, Any] | None = None,
) -> PreparedDataset:
    """Numeric cleaning, de-duplication, label-conflict removal, caps, split, gate, manifest."""
    n_raw = len(df)
    meta = {"Class", *([group_column] if group_column else [])}
    features = [c for c in df.columns if c not in meta]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    n_bad = int(df[features].isna().any(axis=1).sum())
    df = df.dropna(subset=features)
    df[features] = df[features].astype("float32")
    constant = [c for c in features if df[c].nunique() <= 1]
    features = [c for c in features if c not in constant]

    # Keep the first copy: for an official partition the train file is read first, so a test
    # row that duplicates a training row is the one removed (no leakage, counted below).
    before = len(df)
    df = df.drop_duplicates(subset=[*features, "Class"])
    n_dups = before - len(df)
    feat_hash = pd.util.hash_pandas_object(df[features], index=False)
    conflict_mask = feat_hash.duplicated(keep=False) & (
        df.assign(_h=feat_hash.to_numpy()).groupby("_h")["Class"].transform("nunique").to_numpy()
        > 1
    )
    n_conflicts = int(conflict_mask.sum())
    df = df.loc[~conflict_mask]

    rng = np.random.default_rng(split.seed)
    caps: list[pd.DataFrame] = []
    for cls, g in df.groupby("Class"):
        cap = benign_cap if cls == BENIGN else class_cap
        if cap is not None and len(g) > cap:
            g = g.iloc[rng.choice(len(g), size=cap, replace=False)]
        caps.append(g)
    df = pd.concat(caps).reset_index(drop=True)
    df["split"] = assign_split(df, split, rng)

    reasons: list[str] = []
    feat_hash = pd.util.hash_pandas_object(df[features], index=False)
    train_h = set(feat_hash[df["split"] == "train"])
    overlap = int(feat_hash[df["split"] == "test"].isin(train_h).sum())
    if overlap:
        reasons.append(f"{overlap} test rows also appear in train (leakage)")
    test_counts = df.loc[df["split"] == "test", "Class"].value_counts()
    missing = sorted(set(df["Class"]) - set(test_counts.index))
    if split.strategy in ("by_day", "official"):
        # A fixed external test set (unseen days, official partition) may lack some classes:
        # that is part of the design and recorded in the manifest. Only a one-class test blocks.
        if len(test_counts) < 2:
            reasons.append("the test set contains fewer than two classes")
    elif missing:
        reasons.append(f"classes absent from test split: {missing}")

    out = prepared_dir_for(root, split)
    out.mkdir(parents=True, exist_ok=True)
    keep = [*features, "Class", *([group_column] if group_column else []), "split"]
    df[keep].to_parquet(out / "clean.parquet", index=False)
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        name=info.name,
        source_url=info.source_url,
        license=info.license,
        citation=info.citation,
        files=files,
        n_rows_raw=n_raw,
        n_rows_clean=len(df),
        n_duplicates_removed=n_dups + n_conflicts,
        features=features,
        label_column="Class",
        class_counts={str(k): int(v) for k, v in df["Class"].value_counts().items()},
        split_counts={str(k): int(v) for k, v in df["split"].value_counts().items()},
        split_overlap_rows=overlap,
        status="blocked" if reasons else "verified",
        blocked_reasons=reasons,
    )
    notes = {
        "rows_with_inf_or_nan_dropped": n_bad,
        "exact_duplicates_dropped": n_dups,
        "conflicting_label_rows_dropped": n_conflicts,
        "constant_features_dropped": constant,
        "classes_absent_from_test": missing,
        "benign_cap": benign_cap,
        "class_cap": class_cap,
        "clean_parquet_sha256": sha256_file(out / "clean.parquet"),
        "split": split.model_dump(),
        **(extra_notes or {}),
    }
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (out / "preprocessing_notes.json").write_text(json.dumps(notes, indent=2))
    return PreparedDataset(manifest, out)


def load_split(
    prepared_dir: Path, which: str, label_scheme: str = "multiclass"
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(prepared_dir / "clean.parquet")
    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    part = df[df["split"] == which]
    y = part["Class"]
    if label_scheme == "binary":
        y = (y != BENIGN).map({True: "ATTACK", False: BENIGN})
    return part[manifest["features"]], y.reset_index(drop=True)


def manifest_summary(m: DatasetManifest) -> dict[str, Any]:
    return {
        "dataset_id": m.dataset_id,
        "status": m.status,
        "rows": m.n_rows_clean,
        "classes": m.class_counts,
        "splits": m.split_counts,
        "blocked_reasons": m.blocked_reasons,
    }
