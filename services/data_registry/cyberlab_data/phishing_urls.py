"""PhiUSIIL phishing URL adapter (Prasad and Chandra, 2024): the platform's TEXT modality.

The model reads only the URL string (feature `text:URL`), not the engineered page features in
the same file, so this is a genuinely different modality from the flow-feature datasets.

Input: `PhiUSIIL_Phishing_URL_Dataset.csv` from the UCI Machine Learning Repository, placed by
a human in `<data_dir>/datasets/phishing_urls/raw/`.

Handled here (recorded in the preprocessing notes):
- label 1 means a legitimate URL (BENIGN) and 0 a phishing URL (Phishing), as documented by UCI;
- only the URL is kept; the engineered columns are left out (several are computed from the page
  content and would not be available for a URL alone);
- every URL's host is its `Group`: the default `grouped` split keeps all URLs of one website in
  the same split, so a model cannot pass the test by memorising domains it saw in training.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

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

DATASET_ID = "phiusiil-urls-v1"
PHIUSIIL = DatasetInfo(
    name="PhiUSIIL Phishing URL dataset (URL text)",
    source_url="https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
    license="CC BY 4.0 (UCI Machine Learning Repository); cite Prasad and Chandra (2024).",
    citation=(
        "A. Prasad, S. Chandra. PhiUSIIL: A diverse security profile empowered phishing URL "
        "detection framework based on similarity index and incremental learning. Computers & "
        "Security 136, 2024. doi:10.1016/j.cose.2023.103545"
    ),
    citation_doi="10.1016/j.cose.2023.103545",
)
MANUAL_STEPS = (
    "Download PhiUSIIL_Phishing_URL_Dataset.csv from "
    "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset (about 55 MB), copy it "
    "into {raw} and run `cyberlab dataset prepare --name phishing-urls`."
)


def _host(url: str) -> str:
    u = url if "://" in url else f"http://{url}"
    try:
        host = urlsplit(u).hostname or url
    except ValueError:
        host = url
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def prepare(
    root: Path,
    split: SplitSpec | None = None,
    benign_cap: int | None = 400_000,
    class_cap: int | None = None,
) -> PreparedDataset:
    split = split or SplitSpec(strategy="grouped")
    if split.strategy not in ("grouped", "stratified_dedup"):
        raise DatasetBlocked(["the phishing URL dataset supports grouped or stratified splits"])
    raw = root / "raw"
    csvs = sorted(raw.glob("*.csv")) if raw.exists() else []
    if not csvs:
        raise DatasetBlocked([MANUAL_STEPS.format(raw=raw)])
    path = csvs[0]
    df = pd.read_csv(path, low_memory=False)
    cols = {c.strip().lower(): c for c in df.columns}
    if "url" not in cols or "label" not in cols:
        raise DatasetBlocked([f"{path.name}: needs URL and label columns"])
    url = df[cols["url"]].astype(str)
    label = pd.to_numeric(df[cols["label"]], errors="coerce")
    if not set(label.dropna().unique()) <= {0, 1}:
        raise DatasetBlocked([f"{path.name}: label must be 0 (phishing) or 1 (legitimate)"])
    out = pd.DataFrame(
        {
            "text:URL": url,
            "Class": np.where(label == 1, BENIGN, "Phishing"),
            "Group": url.map(_host),
        }
    )
    out = out[label.notna().to_numpy()]
    return finalize(
        out,
        root=root,
        split=split,
        dataset_id=DATASET_ID,
        info=PHIUSIIL,
        files={path.name: sha256_file(path)},
        group_column="Group",
        benign_cap=benign_cap,
        class_cap=class_cap,
        extra_notes={
            "modality": "text",
            "text_feature": "URL",
            "engineered_columns_not_used": len(df.columns) - 2,
            "distinct_hosts": int(out["Group"].nunique()),
        },
    )


def write_fixture(raw_dir: Path, rows_per_class: int = 150, seed: int = 21) -> list[Path]:
    """Synthetic URLs with the dataset's columns. Test use only."""
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    words = ["login", "verify", "account", "secure", "update", "bank", "paypa1", "signin"]
    rows = []
    for i in range(rows_per_class):
        site = f"site{i % 60}.com"
        rows.append({"URL": f"https://www.{site}/page/{i}", "TLD": "com", "label": 1})
        bad = "-".join(rng.choice(words, size=2))
        host = f"{bad}{i % 40}.xyz"
        rows.append(
            {"URL": f"http://{host}/{rng.choice(words)}.php?id={i}", "TLD": "xyz", "label": 0}
        )
    path = raw_dir / "PhiUSIIL_Phishing_URL_Dataset.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return [path]
