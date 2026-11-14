"""UNSW-NB15 adapter (official training/testing-set CSVs).

Input: `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`, placed by a human in
`<data_dir>/datasets/unsw_nb15/raw/` (UNSW distributes them through its own download page; the
platform never fetches from mirrors).

Known data issues handled here (recorded in the preprocessing notes):
- `id` is a row counter that follows file order and therefore class blocks: it is dropped,
  never used as a feature (a known leakage source in published UNSW-NB15 results);
- `label` (0/1) duplicates the attack category and is dropped; the target is `attack_cat`;
- `attack_cat` has stray spaces and both "Backdoor" and "Backdoors" in some copies
  (normalised); the normal-traffic class "Normal" is called BENIGN like every dataset here;
- `proto`, `service` and `state` are categorical: one-hot encoded for the most frequent values
  (all other values share an `other` column). The vocabulary is fixed from the training file
  only, so the test file never shapes the features;
- the official partition shares many identical rows between the training and testing files;
  de-duplication keeps the training copy, so no test row also appears in training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cyberlab_data.common import (
    BENIGN,
    DatasetBlocked,
    DatasetInfo,
    PreparedDataset,
    finalize,
    sha256_file,
)
from cyberlab_schemas import SplitSpec

DATASET_ID = "unsw-nb15-v1"

UNSW_NB15 = DatasetInfo(
    name="UNSW-NB15 (official training and testing sets)",
    source_url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
    license=(
        "UNSW Canberra Cyber dataset terms: free for academic research use; cite Moustafa and "
        "Slay (2015)."
    ),
    citation=(
        "N. Moustafa, J. Slay. UNSW-NB15: a comprehensive data set for network intrusion "
        "detection systems (UNSW-NB15 network data set). Military Communications and "
        "Information Systems Conference (MilCIS) 2015. doi:10.1109/MilCIS.2015.7348942"
    ),
    citation_doi="10.1109/MilCIS.2015.7348942",
)

CATEGORIES = {
    "Normal": BENIGN,
    "Generic": "Generic",
    "Exploits": "Exploits",
    "Fuzzers": "Fuzzers",
    "DoS": "DoS",
    "Reconnaissance": "Reconnaissance",
    "Analysis": "Analysis",
    "Backdoor": "Backdoor",
    "Backdoors": "Backdoor",
    "Shellcode": "Shellcode",
    "Worms": "Worms",
}
CATEGORICAL = ("proto", "service", "state")
TOP_VALUES = 15
DROPPED = ("id", "label", "attack_cat")

MANUAL_STEPS = (
    "UNSW-NB15 is downloaded from UNSW's own page and cannot be fetched automatically. "
    "1) Open https://research.unsw.edu.au/projects/unsw-nb15-dataset and follow the download "
    "link. 2) From the 'Training and Testing Sets' folder take UNSW_NB15_training-set.csv and "
    "UNSW_NB15_testing-set.csv (about 30 MB together). 3) Copy both into {raw} and run "
    "`cyberlab dataset prepare --name unsw-nb15`."
)


def _find(raw: Path, part: str) -> Path | None:
    hits = sorted(p for p in raw.glob("*.csv") if part in p.name.lower())
    return hits[0] if hits else None


def _read(path: Path, part: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = sorted({"attack_cat", *CATEGORICAL} - set(df.columns))
    if missing:
        raise DatasetBlocked([f"{path.name}: columns missing {missing} (not the official file?)"])
    df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df["Part"] = part
    return df


def prepare(
    root: Path,
    split: SplitSpec | None = None,
    benign_cap: int | None = None,
    class_cap: int | None = None,
) -> PreparedDataset:
    split = split or SplitSpec()
    if split.strategy == "by_day":
        raise DatasetBlocked(
            ["UNSW-NB15 has no capture days; use the official or a stratified split"]
        )
    raw = root / "raw"
    train_csv = _find(raw, "training") if raw.exists() else None
    test_csv = _find(raw, "testing") if raw.exists() else None
    if train_csv is None or test_csv is None:
        raise DatasetBlocked([MANUAL_STEPS.format(raw=raw)])
    files = {p.name: sha256_file(p) for p in (train_csv, test_csv)}
    train, test = _read(train_csv, "train"), _read(test_csv, "test")

    unknown = sorted((set(train["attack_cat"]) | set(test["attack_cat"])) - set(CATEGORIES))
    if unknown:
        raise DatasetBlocked([f"unknown attack categories: {unknown}"])
    vocab = {
        c: [str(v) for v in train[c].astype(str).value_counts().index[:TOP_VALUES]]
        for c in CATEGORICAL
    }
    df = pd.concat([train, test], ignore_index=True)  # train first: dedup keeps train copies
    df["Class"] = df["attack_cat"].map(CATEGORIES)
    for c in CATEGORICAL:
        values = df[c].astype(str)
        for v in vocab[c]:
            df[f"{c}={v}"] = (values == v).astype("float32")
        df[f"{c}=other"] = (~values.isin(vocab[c])).astype("float32")
    df = df.drop(columns=[*DROPPED, *CATEGORICAL], errors="ignore")
    return finalize(
        df,
        root=root,
        split=split,
        dataset_id=DATASET_ID,
        info=UNSW_NB15,
        files=files,
        group_column="Part",
        benign_cap=benign_cap,
        class_cap=class_cap,
        extra_notes={
            "dropped_columns": list(DROPPED),
            "one_hot_vocabulary_from_training_file": vocab,
            "official_rows": {"train": len(train), "test": len(test)},
        },
    )


# --------------------------------------------------------------------------- test fixture

_NUMERIC = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "smean", "dmean"]


def write_fixture(raw_dir: Path, rows_per_class: int = 80, seed: int = 11) -> list[Path]:
    """Small synthetic files with UNSW-NB15's column layout and quirks. Test use only."""
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cats = ["Normal", "Generic", "Exploits", "DoS", "Backdoors", " Reconnaissance "]
    paths = []
    for part, n in (("training", rows_per_class), ("testing", rows_per_class // 2)):
        frames = []
        for j, cat in enumerate(cats):
            x = np.abs(rng.normal(10 + 20 * j, 4, size=(n, len(_NUMERIC)))).round(3)
            f = pd.DataFrame(x, columns=_NUMERIC)
            f["proto"] = rng.choice(["tcp", "udp", "arp", "ospf"], size=n)
            f["service"] = rng.choice(["-", "http", "dns"], size=n)
            f["state"] = rng.choice(["FIN", "INT", "CON"], size=n)
            f["attack_cat"] = cat
            f["label"] = int(cat != "Normal")
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
        df.insert(0, "id", np.arange(1, len(df) + 1))
        if part == "testing":
            # the official files share rows: copy a few training-like rows into the test file
            df = pd.concat([df, df.iloc[:3]], ignore_index=True)
        path = raw_dir / f"UNSW_NB15_{part}-set.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths
