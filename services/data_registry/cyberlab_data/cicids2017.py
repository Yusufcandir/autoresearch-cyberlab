"""CIC-IDS2017 adapter: provenance, cleaning, de-duplication and leakage-safe splits.

Input: the official `MachineLearningCSV` files (one CSV per capture day), placed by a human in
`<data_dir>/datasets/cicids2017/raw/`. The platform never runs arbitrary download commands;
`download()` only fetches from the allowlisted official source.

Known data issues handled here (documented in the manifest so they appear in the paper):
- column names have leading spaces; `Fwd Header Length` appears twice (the copy is dropped);
- `Flow Bytes/s` and `Flow Packets/s` contain inf/NaN values (rows dropped and counted);
- many exact duplicate flows (dropped) and some identical feature vectors with different
  labels (conflicts, dropped) so the same row can never sit in both train and test;
- the "Web Attack" labels use a broken dash character in some copies (normalised).
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cyberlab_schemas import DatasetManifest, SplitSpec

DATASET_ID = "cicids2017-mlcsv-v1"


@dataclass(frozen=True)
class _Info:
    name: str = "CIC-IDS2017 (MachineLearningCSV)"
    source_url: str = "https://www.unb.ca/cic/datasets/ids-2017.html"
    download_url: str = (
        "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"
    )
    license: str = (
        "Canadian Institute for Cybersecurity dataset licence: free for research use; "
        "redistribution not permitted; cite Sharafaldin et al. (2018)."
    )
    citation: str = (
        "I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani. Toward Generating a New Intrusion "
        "Detection Dataset and Intrusion Traffic Characterization. ICISSP 2018. "
        "doi:10.5220/0006639801080116"
    )
    citation_doi: str = "10.5220/0006639801080116"


CICIDS2017 = _Info()

LABEL_GROUPS = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Bot": "Bot",
    "Infiltration": "Rare",
    "Heartbleed": "Rare",
}

DAY_PATTERN = re.compile(r"(monday|tuesday|wednesday|thursday|friday)", re.I)


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_label(raw: str) -> str:
    s = str(raw).strip()
    s = re.sub(r"Web Attack\s*\W+\s*", "Web Attack - ", s)
    return s


def download(root: Path, url: str = CICIDS2017.download_url) -> Path:  # pragma: no cover - network
    """Fetch the official zip (about 230 MB) into raw/ and unzip it. Allowlisted URL only."""
    import httpx

    if url != CICIDS2017.download_url:
        raise ValueError("only the allowlisted official CIC-IDS2017 URL may be downloaded")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / "MachineLearningCSV.zip"
    with httpx.stream("GET", url, timeout=httpx.Timeout(60, read=600), follow_redirects=True) as r:
        r.raise_for_status()
        with target.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    with zipfile.ZipFile(target) as z:
        for member in z.namelist():
            if member.endswith(".csv") and ".." not in member:
                (raw / Path(member).name).write_bytes(z.read(member))
    return raw


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")], errors="ignore")
    day = DAY_PATTERN.search(path.name)
    df["Day"] = day.group(1).capitalize() if day else "Unknown"
    return df


def prepare(
    root: Path,
    split: SplitSpec | None = None,
    benign_cap: int | None = 400_000,
    class_cap: int | None = None,
) -> PreparedDataset:
    """Clean, de-duplicate, split and hash CIC-IDS2017. Raises DatasetBlocked on gate failure."""
    split = split or SplitSpec()
    raw = root / "raw"
    csvs = sorted(raw.glob("*.csv")) if raw.exists() else []
    if not csvs:
        raise DatasetBlocked(
            [f"no CSV files in {raw}; download MachineLearningCSV.zip from {CICIDS2017.source_url}"]
        )
    files = {p.name: sha256_file(p) for p in csvs}
    df = pd.concat([_read_csv(p) for p in csvs], ignore_index=True)
    if "Label" not in df.columns:
        raise DatasetBlocked(["no Label column found"])
    n_raw = len(df)

    df["Label"] = df["Label"].map(normalize_label)
    unknown = sorted(set(df["Label"]) - set(LABEL_GROUPS))
    if unknown:
        raise DatasetBlocked([f"unknown labels: {unknown}"])
    df["Class"] = df["Label"].map(LABEL_GROUPS)

    features = [c for c in df.columns if c not in {"Label", "Class", "Day"}]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    n_bad = int(df[features].isna().any(axis=1).sum())
    df = df.dropna(subset=features)
    df[features] = df[features].astype("float32")
    constant = [c for c in features if df[c].nunique() <= 1]
    features = [c for c in features if c not in constant]

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
        cap = benign_cap if cls == "BENIGN" else class_cap
        if cap is not None and len(g) > cap:
            g = g.iloc[rng.choice(len(g), size=cap, replace=False)]
        caps.append(g)
    df = pd.concat(caps).reset_index(drop=True)
    df["split"] = _assign_split(df, split, rng)

    reasons: list[str] = []
    feat_hash = pd.util.hash_pandas_object(df[features], index=False)
    train_h = set(feat_hash[df["split"] == "train"])
    overlap = int(feat_hash[df["split"] == "test"].isin(train_h).sum())
    if overlap:
        reasons.append(f"{overlap} test rows also appear in train (leakage)")
    test_counts = df.loc[df["split"] == "test", "Class"].value_counts()
    missing = sorted(set(df["Class"]) - set(test_counts.index))
    if missing:
        reasons.append(f"classes absent from test split: {missing}")

    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    df[[*features, "Class", "Day", "split"]].to_parquet(out / "clean.parquet", index=False)
    manifest = DatasetManifest(
        dataset_id=DATASET_ID,
        name=CICIDS2017.name,
        source_url=CICIDS2017.source_url,
        license=CICIDS2017.license,
        citation=CICIDS2017.citation,
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
        "benign_cap": benign_cap,
        "class_cap": class_cap,
        "clean_parquet_sha256": sha256_file(out / "clean.parquet"),
        "split": split.model_dump(),
    }
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (out / "preprocessing_notes.json").write_text(json.dumps(notes, indent=2))
    return PreparedDataset(manifest, out)


def _assign_split(df: pd.DataFrame, split: SplitSpec, rng: np.random.Generator) -> np.ndarray:
    labels = np.empty(len(df), dtype=object)
    if split.strategy == "by_day":
        test_days = {d.capitalize() for d in split.test_days}
        if not test_days:
            raise DatasetBlocked(["by_day split needs test_days"])
        labels[:] = "train"
        labels[df["Day"].isin(test_days).to_numpy()] = "test"
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


def load_split(
    prepared_dir: Path, which: str, label_scheme: str = "multiclass"
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(prepared_dir / "clean.parquet")
    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    part = df[df["split"] == which]
    y = part["Class"]
    if label_scheme == "binary":
        y = (y != "BENIGN").map({True: "ATTACK", False: "BENIGN"})
    return part[manifest["features"]], y.reset_index(drop=True)


# --------------------------------------------------------------------------- test fixture

FIXTURE_FEATURES = [
    " Destination Port",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Fwd Packet Length Max",
    "Flow Bytes/s",
    " Flow Packets/s",
    " Fwd Header Length",
    " Fwd Header Length",
    " SYN Flag Count",
    " Init_Win_bytes_forward",
]


def write_fixture(raw_dir: Path, rows_per_class: int = 120, seed: int = 7) -> list[Path]:
    """Write small synthetic CSVs with CIC-IDS2017's real column quirks. Test use only."""
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "Monday-WorkingHours.pcap_ISCX.csv": ["BENIGN"],
        "Wednesday-workingHours.pcap_ISCX.csv": ["BENIGN", "DoS Hulk", "DoS slowloris"],
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": ["BENIGN", "PortScan"],
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": [
            "BENIGN",
            "Web Attack \x96 Brute Force",
        ],
    }
    paths = []
    for i, (name, labels) in enumerate(plan.items()):
        frames = []
        for j, label in enumerate(labels):
            centre = (i + 1) * 10 + j * 25
            x = rng.normal(centre, 4, size=(rows_per_class, len(FIXTURE_FEATURES)))
            frame = pd.DataFrame(np.abs(x).round(2))
            frame[" Label"] = label
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df.iloc[0, 6] = np.inf  # Flow Bytes/s inf
        df.iloc[1, 7] = np.nan
        df = pd.concat([df, df.iloc[2:6]], ignore_index=True)  # exact duplicates
        header = ",".join([*FIXTURE_FEATURES, " Label"])
        path = raw_dir / name
        with path.open("w", encoding="latin-1", newline="") as fh:
            fh.write(header + "\n")
            df.to_csv(fh, header=False, index=False)
        paths.append(path)
    return paths


def manifest_summary(m: DatasetManifest) -> dict[str, Any]:
    return {
        "dataset_id": m.dataset_id,
        "status": m.status,
        "rows": m.n_rows_clean,
        "classes": m.class_counts,
        "splits": m.split_counts,
        "blocked_reasons": m.blocked_reasons,
    }
