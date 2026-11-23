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
    if split.strategy == "by_group":
        if "Group" not in df.columns:
            raise DatasetBlocked(["this dataset has no capture groups"])
        groups = {str(g) for g in split.test_days}
        if not groups:
            raise DatasetBlocked(["by_group split needs the test groups (test_days)"])
        return df["Group"].astype(str).isin(groups).to_numpy()
    if split.strategy in ("official", "cross_dataset"):
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
    if split.strategy == "grouped":
        # whole groups (e.g. a website's domain) go to one split, so no group is in both
        if "Group" not in df.columns:
            raise DatasetBlocked(["a grouped split needs a Group column"])
        groups = df["Group"].astype(str).to_numpy()
        uniq = rng.permutation(np.unique(groups))
        sizes = pd.Series(groups).value_counts()
        target_test, target_val = split.test_fraction * len(df), split.val_fraction * len(df)
        assign: dict[str, str] = {}
        n_test = n_val = 0
        for g in uniq:
            if n_test < target_test:
                assign[g], n_test = "test", n_test + int(sizes[g])
            elif n_val < target_val:
                assign[g], n_val = "val", n_val + int(sizes[g])
            else:
                assign[g] = "train"
        labels[:] = [assign[g] for g in groups]
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
    audit = PreprocessingAudit(list((extra_notes or {}).get("pre_stages", [])))
    audit.start(df)
    meta = {"Class", *([group_column] if group_column else [])}
    features = [c for c in df.columns if c not in meta]
    # text features (modality 'text') are named 'text:<column>' and kept as strings
    text = [c for c in features if c.startswith("text:")]
    numeric = [c for c in features if c not in text]
    if numeric:
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
        # values beyond float32 (some NetFlow-v2 byte-rate fields) would become infinite when
        # stored as float32 below, so they count as invalid here and the row is removed
        big = np.finfo(np.float32).max
        df[numeric] = df[numeric].mask(df[numeric].abs() > big)
    for c in text:
        df[c] = df[c].astype("string").str.strip().mask(lambda v: v == "")
    n_bad = int(df[features].isna().any(axis=1).sum())
    df = audit.step(
        df,
        df.dropna(subset=features),
        "missing or infinite values",
        "rows with an infinite or missing feature value",
    )
    if numeric:
        df[numeric] = df[numeric].astype("float32")
    for c in text:
        df[c] = df[c].astype(str)
    constant = [c for c in features if df[c].nunique() <= 1]
    features = [c for c in features if c not in constant]

    # Keep the first copy: for an official partition the train file is read first, so a test
    # row that duplicates a training row is the one removed (no leakage, counted below).
    before = len(df)
    df = audit.step(
        df,
        df.drop_duplicates(subset=[*features, "Class"]),
        "exact duplicates",
        "same features and label as an earlier row",
    )
    n_dups = before - len(df)
    feat_hash = pd.util.hash_pandas_object(df[features], index=False)
    conflict_mask = feat_hash.duplicated(keep=False) & (
        df.assign(_h=feat_hash.to_numpy()).groupby("_h")["Class"].transform("nunique").to_numpy()
        > 1
    )
    n_conflicts = int(conflict_mask.sum())
    df = audit.step(
        df, df.loc[~conflict_mask], "conflicting labels", "identical features with different labels"
    )

    rng = np.random.default_rng(split.seed)
    caps: list[pd.DataFrame] = []
    for cls, g in df.groupby("Class"):
        cap = benign_cap if cls == BENIGN else class_cap
        if cap is not None and len(g) > cap:
            g = g.iloc[rng.choice(len(g), size=cap, replace=False)]
        caps.append(g)
    df = audit.step(
        df,
        pd.concat(caps).reset_index(drop=True),
        "class down-sampling",
        f"seeded cap per class (BENIGN {benign_cap}, others {class_cap})",
    )
    df["split"] = assign_split(df, split, rng)

    reasons: list[str] = []
    feat_hash = pd.util.hash_pandas_object(df[features], index=False)
    train_h = set(feat_hash[df["split"] == "train"])
    overlap = int(feat_hash[df["split"] == "test"].isin(train_h).sum())
    if overlap:
        reasons.append(f"{overlap} test rows also appear in train (leakage)")
    test_counts = df.loc[df["split"] == "test", "Class"].value_counts()
    missing = sorted(set(df["Class"]) - set(test_counts.index))
    if group_column == "Group" and split.strategy in ("grouped", "by_group"):
        g_train = set(df.loc[df["split"] == "train", "Group"].astype(str))
        shared = g_train & set(df.loc[df["split"] == "test", "Group"].astype(str))
        if shared:
            reasons.append(f"{len(shared)} groups appear in both train and test (group leakage)")
    if split.strategy in ("by_day", "official", "by_group", "cross_dataset"):
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
        "single_feature_shortcuts": shortcut_scan(df[df["split"] == "train"], features),
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
    report = audit.finish(len(df), {str(k): int(v) for k, v in df["split"].value_counts().items()})
    notes["preprocessing_audit_ok"] = report["consistent"]
    (out / "data_preprocessing_audit.json").write_text(json.dumps(report, indent=2))
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


def shortcut_scan(train: pd.DataFrame, features: list[str], top: int = 5) -> list[dict[str, Any]]:
    """Features whose value alone almost decides the class on the training rows (a leakage
    or shortcut warning, spec 13.2). Purity = share of rows whose value maps to one class."""
    out: list[dict[str, Any]] = []
    n = len(train)
    if n == 0:
        return out
    for f in features:
        if f.startswith("text:"):
            continue
        col = train[f]
        if col.nunique() > 50:  # many distinct values would look "pure" by chance
            col = pd.qcut(col, q=50, duplicates="drop", labels=False)
        counts = pd.crosstab(col, train["Class"])
        purity = float(counts.max(axis=1).sum() / n)
        out.append({"feature": f, "purity": round(purity, 4), "values": int(len(counts))})
    out.sort(key=lambda r: -float(r["purity"]))
    return [r for r in out[:top] if float(r["purity"]) >= 0.99]


class PreprocessingAudit:
    """Stage-by-stage row accounting (professor's plan, Step 1.2): for every operation the input
    rows, removed rows, remaining rows, the reason, and the class distribution before and after.
    It checks input = removed + remaining at every stage and that the stages chain together;
    an inconsistent audit blocks manuscript generation."""

    def __init__(self, pre_stages: list[dict[str, Any]] | None = None) -> None:
        self.stages: list[dict[str, Any]] = [dict(s) for s in (pre_stages or [])]

    @staticmethod
    def _classes(df: pd.DataFrame) -> dict[str, int]:
        return {str(k): int(v) for k, v in df["Class"].value_counts().items()}

    def start(self, df: pd.DataFrame) -> None:
        self.stages.append(
            {
                "stage": "loaded for cleaning",
                "input_rows": len(df),
                "removed_rows": 0,
                "remaining_rows": len(df),
                "reason": "rows read from the raw files after the dataset-specific steps above",
                "classes_after": self._classes(df),
            }
        )

    def step(
        self, before: pd.DataFrame, after: pd.DataFrame, stage: str, reason: str
    ) -> pd.DataFrame:
        cb, ca = self._classes(before), self._classes(after)
        self.stages.append(
            {
                "stage": stage,
                "input_rows": len(before),
                "removed_rows": len(before) - len(after),
                "remaining_rows": len(after),
                "reason": reason,
                "removed_per_class": {c: cb[c] - ca.get(c, 0) for c in cb if cb[c] != ca.get(c, 0)},
                "class_distribution_changed": cb != ca,
                "classes_after": ca,
            }
        )
        return after

    def finish(self, final_rows: int, split_counts: dict[str, int]) -> dict[str, Any]:
        problems = []
        for i, st in enumerate(self.stages):
            if st["input_rows"] != st["removed_rows"] + st["remaining_rows"]:
                problems.append(f"stage '{st['stage']}': input != removed + remaining")
            if i and self.stages[i - 1]["remaining_rows"] != st["input_rows"]:
                problems.append(
                    f"stage '{st['stage']}' starts with {st['input_rows']} rows but the previous "
                    f"stage ended with {self.stages[i - 1]['remaining_rows']}"
                )
        if self.stages and self.stages[-1]["remaining_rows"] != final_rows:
            problems.append("the last stage does not end with the final row count")
        if sum(split_counts.values()) != final_rows:
            problems.append("train + validation + test rows do not add up to the final rows")
        first = self.stages[0]["input_rows"] if self.stages else final_rows
        return {
            "status": "PASS" if not problems else "BLOCK",
            "consistent": not problems,
            "raw_rows": first,
            "final_rows": final_rows,
            "total_removed": first - final_rows,
            "split_rows": split_counts,
            "stages": self.stages,
            "problems": problems,
        }
