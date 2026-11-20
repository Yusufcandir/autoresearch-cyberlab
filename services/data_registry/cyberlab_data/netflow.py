"""NetFlow-v2 family adapters (Sarhan, Layeghy and Portmann, 2022).

Four public NIDS datasets (UNSW-NB15, CSE-CIC-IDS2018, ToN-IoT, BoT-IoT) converted by the
University of Queensland into ONE shared set of 43 NetFlow features. Because the features are
identical, a model trained on one can be tested on another: the cross-dataset protocol.

Input: the variant's CSV (e.g. `NF-UNSW-NB15-v2.csv`) placed by a human in
`<data_dir>/datasets/<key>/raw/` (downloaded from https://staff.itee.uq.edu.au/marius/NIDS_datasets/).

Handled here (recorded in the preprocessing notes):
- host identifiers are not features: source and destination IP addresses and ports describe
  the lab network, not the attack, and would not transfer to another network;
- `Label` (0/1) repeats the target and is dropped; the class is the `Attack` column, with
  "Benign" called BENIGN like every dataset on this platform;
- the files are large (up to tens of millions of rows): a two-pass, seeded per-class sample
  keeps at most `class_cap` rows of each class (BENIGN capped by `benign_cap`), so preparation
  fits in memory and every class keeps all of its rows when it is small.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

CITATION = (
    "M. Sarhan, S. Layeghy, M. Portmann. Towards a Standard Feature Set for Network Intrusion "
    "Detection System Datasets. Mobile Networks and Applications, 2022. "
    "doi:10.1007/s11036-021-01843-0"
)
LICENSE = (
    "University of Queensland NIDS datasets: free for research use; cite Sarhan, Layeghy and "
    "Portmann (2022) and the original dataset."
)
SOURCE = "https://staff.itee.uq.edu.au/marius/NIDS_datasets/"
IDENTIFIERS = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT")
CHUNK = 500_000


@dataclass(frozen=True)
class Variant:
    key: str
    display: str
    file_stem: str  # e.g. NF-UNSW-NB15-v2
    origin: str


VARIANTS = {
    v.key: v
    for v in (
        Variant("nf_unsw_nb15_v2", "NF-UNSW-NB15-v2", "NF-UNSW-NB15-v2", "UNSW-NB15"),
        Variant(
            "nf_cse_cic_ids2018_v2",
            "NF-CSE-CIC-IDS2018-v2",
            "NF-CSE-CIC-IDS2018-v2",
            "CSE-CIC-IDS2018",
        ),
        Variant("nf_ton_iot_v2", "NF-ToN-IoT-v2", "NF-ToN-IoT-v2", "ToN-IoT"),
        Variant("nf_bot_iot_v2", "NF-BoT-IoT-v2", "NF-BoT-IoT-v2", "BoT-IoT"),
    )
}


def info_for(v: Variant) -> DatasetInfo:
    return DatasetInfo(
        name=f"{v.display} (NetFlow v2 version of {v.origin})",
        source_url=SOURCE,
        license=LICENSE,
        citation=CITATION,
        citation_doi="10.1007/s11036-021-01843-0",
    )


def manual_steps(v: Variant) -> str:
    return (
        f"{v.display} is downloaded from the University of Queensland page and cannot be fetched "
        f"automatically. 1) Open {SOURCE} and download {v.display}. 2) Unzip it and copy "
        f"{v.file_stem}.csv into {{raw}}. 3) Run `cyberlab dataset prepare --name {v.key}`."
    )


def _find_csv(raw: Path, v: Variant) -> Path | None:
    if not raw.exists():
        return None
    want = v.file_stem.lower().replace("-", "").replace("_", "")
    for p in sorted(raw.glob("*.csv")):
        if p.stem.lower().replace("-", "").replace("_", "") == want:
            return p
    csvs = sorted(raw.glob("*.csv"))
    return csvs[0] if len(csvs) == 1 else None


def load_netflow(
    v: Variant, root: Path, benign_cap: int | None, class_cap: int | None, seed: int
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    """Read the variant's CSV with a seeded two-pass per-class sample. Returns the frame
    (features + Class), the file hashes and notes."""
    raw = root / "raw"
    path = _find_csv(raw, v)
    if path is None:
        raise DatasetBlocked([manual_steps(v).format(raw=raw)])
    head = pd.read_csv(path, nrows=5)
    if "Attack" not in head.columns:
        raise DatasetBlocked([f"{path.name}: no Attack column (not a NetFlow-v2 file?)"])
    counts: dict[str, int] = {}
    for ch in pd.read_csv(path, usecols=["Attack"], chunksize=CHUNK):
        for k, n in ch["Attack"].astype(str).str.strip().value_counts().items():
            counts[str(k)] = counts.get(str(k), 0) + int(n)
    rate = {}
    for k, n in counts.items():
        cap = benign_cap if k.lower() == "benign" else class_cap
        rate[k] = 1.0 if cap is None or n <= cap else cap / n
    rng = np.random.default_rng(seed)
    parts = []
    for ch in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        att = ch["Attack"].astype(str).str.strip()
        keep = rng.random(len(ch)) < att.map(rate).to_numpy()
        parts.append(ch.loc[keep])
    df = pd.concat(parts, ignore_index=True)
    df["Class"] = (
        df["Attack"].astype(str).str.strip().map(lambda a: BENIGN if a.lower() == "benign" else a)
    )
    dropped = [c for c in (*IDENTIFIERS, "Label", "Attack") if c in df.columns]
    df = df.drop(columns=dropped)
    in_file = int(sum(counts.values()))
    notes = {
        "pre_stages": [
            {
                "stage": "seeded per-class sampling while reading",
                "input_rows": in_file,
                "removed_rows": in_file - len(df),
                "remaining_rows": len(df),
                "reason": "at most benign_cap / class_cap rows per class, read in chunks",
                "classes_after": {str(k): int(v) for k, v in df["Class"].value_counts().items()},
            }
        ],
        "dropped_columns": dropped,
        "rows_in_file": int(sum(counts.values())),
        "class_counts_in_file": counts,
        "sampling_rate_per_class": {k: round(r, 6) for k, r in rate.items()},
    }
    return df, {path.name: sha256_file(path)}, notes


def make_prepare(v: Variant) -> Any:
    def prepare(
        root: Path,
        split: SplitSpec | None = None,
        benign_cap: int | None = 400_000,
        class_cap: int | None = 400_000,
    ) -> PreparedDataset:
        split = split or SplitSpec()
        if split.strategy in ("by_day", "official", "by_group"):
            raise DatasetBlocked([f"{v.display} has no days, groups or official partition"])
        df, files, notes = load_netflow(v, root, benign_cap, class_cap, split.seed)
        return finalize(
            df,
            root=root,
            split=split,
            dataset_id=f"{v.key.replace('_', '-')}",
            info=info_for(v),
            files=files,
            group_column=None,
            benign_cap=benign_cap,
            class_cap=class_cap,
            extra_notes=notes,
        )

    return prepare


# --------------------------------------------------------------------------- test fixture

FIXTURE_COLUMNS = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
]


def make_fixture(v: Variant) -> Any:
    def write_fixture(raw_dir: Path, rows_per_class: int = 80, seed: int = 5) -> list[Path]:
        """Synthetic NetFlow-v2 shaped rows. The same attack families in every variant (with a
        variant-specific shift), so cross-dataset tests have something to transfer. Test use only."""
        rng = np.random.default_rng(seed + sum(map(ord, v.key)))
        raw_dir.mkdir(parents=True, exist_ok=True)
        shift = (sum(map(ord, v.key)) % 7) / 2
        frames = []
        for j, att in enumerate(["Benign", "DDoS", "Scanning", "Exploits"]):
            x = np.abs(
                rng.normal(10 + 15 * j + shift, 3, size=(rows_per_class, len(FIXTURE_COLUMNS)))
            )
            f = pd.DataFrame(x.round(1), columns=FIXTURE_COLUMNS)
            f["IPV4_SRC_ADDR"] = [f"10.0.0.{i % 250}" for i in range(rows_per_class)]
            f["IPV4_DST_ADDR"] = "192.168.1.1"
            f["Label"] = int(att != "Benign")
            f["Attack"] = att
            frames.append(f)
        path = raw_dir / f"{v.file_stem}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        return [path]

    return write_fixture
