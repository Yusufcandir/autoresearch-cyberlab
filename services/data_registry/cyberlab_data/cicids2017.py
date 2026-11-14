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

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cyberlab_data.common import (  # noqa: F401 - re-exported for existing imports
    DatasetBlocked,
    DatasetInfo,
    PreparedDataset,
    dataset_record_id,
    finalize,
    is_default_split,
    load_split,
    manifest_summary,
    prepared_dir_for,
    sha256_file,
    split_key,
)
from cyberlab_schemas import SplitSpec

DATASET_ID = "cicids2017-mlcsv-v1"


CICIDS2017 = DatasetInfo(
    name="CIC-IDS2017 (MachineLearningCSV)",
    source_url="https://www.unb.ca/cic/datasets/ids-2017.html",
    download_url="http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip",
    license=(
        "Canadian Institute for Cybersecurity dataset licence: free for research use; "
        "redistribution not permitted; cite Sharafaldin et al. (2018)."
    ),
    citation=(
        "I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani. Toward Generating a New Intrusion "
        "Detection Dataset and Intrusion Traffic Characterization. ICISSP 2018. "
        "doi:10.5220/0006639801080116"
    ),
    citation_doi="10.5220/0006639801080116",
)

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


def normalize_label(raw: str) -> str:
    s = str(raw).strip()
    s = re.sub(r"Web Attack\s*\W+\s*", "Web Attack - ", s)
    return s


MANUAL_STEPS = (
    "CIC-IDS2017 is distributed by UNB after a short registration form, so it cannot be fetched "
    "automatically. 1) Open https://www.unb.ca/cic/datasets/ids-2017.html and request the dataset. "
    "2) Download MachineLearningCSV.zip (about 230 MB). 3) Copy it to {raw} and run "
    "`cyberlab dataset unzip`, or extract the eight CSV files into {raw} yourself."
)


def download(root: Path, url: str = CICIDS2017.download_url) -> Path:  # pragma: no cover - network
    """Try the allowlisted official URL; if UNB serves its registration page, explain the manual steps."""
    import httpx

    if url != CICIDS2017.download_url:
        raise ValueError("only the allowlisted official CIC-IDS2017 URL may be downloaded")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / "MachineLearningCSV.zip"
    with httpx.stream("GET", url, timeout=httpx.Timeout(60, read=600), follow_redirects=True) as r:
        r.raise_for_status()
        if "zip" not in r.headers.get("content-type", "") and "octet-stream" not in r.headers.get(
            "content-type", ""
        ):
            raise DatasetBlocked([MANUAL_STEPS.format(raw=raw)])
        with target.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    unzip(root)
    return raw


def unzip(root: Path) -> list[Path]:
    """Extract every CSV from the zip(s) in raw/, flattening folders (path-traversal safe)."""
    raw = root / "raw"
    zips = sorted(raw.glob("*.zip"))
    if not zips:
        raise DatasetBlocked([MANUAL_STEPS.format(raw=raw)])
    out: list[Path] = []
    for target in zips:
        with zipfile.ZipFile(target) as z:
            for member in z.namelist():
                name = Path(member).name
                if name.lower().endswith(".csv") and not name.startswith("."):
                    dest = raw / name
                    with z.open(member) as src, dest.open("wb") as fh:
                        while chunk := src.read(1 << 20):
                            fh.write(chunk)
                    out.append(dest)
    return out


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
    if split.strategy == "official":
        raise DatasetBlocked(["CIC-IDS2017 has no official train/test partition"])
    raw = root / "raw"
    csvs = sorted(raw.glob("*.csv")) if raw.exists() else []
    if not csvs:
        raise DatasetBlocked([MANUAL_STEPS.format(raw=raw)])
    files = {p.name: sha256_file(p) for p in csvs}
    df = pd.concat([_read_csv(p) for p in csvs], ignore_index=True)
    if "Label" not in df.columns:
        raise DatasetBlocked(["no Label column found"])
    df["Label"] = df["Label"].map(normalize_label)
    unknown = sorted(set(df["Label"]) - set(LABEL_GROUPS))
    if unknown:
        raise DatasetBlocked([f"unknown labels: {unknown}"])
    df["Class"] = df["Label"].map(LABEL_GROUPS)
    df = df.drop(columns=["Label"])
    return finalize(
        df,
        root=root,
        split=split,
        dataset_id=DATASET_ID,
        info=CICIDS2017,
        files=files,
        group_column="Day",
        benign_cap=benign_cap,
        class_cap=class_cap,
    )


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
