"""CTU-13 adapter (Garcia, Grill, Stiborek and Zunino, 2014).

Thirteen captures of REAL botnet traffic mixed with real normal and background traffic on the
CTU university network. Input: the labelled bidirectional NetFlow files (`*.binetflow`), one per
scenario, placed by a human under `<data_dir>/datasets/ctu13/raw/` (scenario folders `1` to `13`
or the files themselves).

Handled here (recorded in the preprocessing notes):
- labels: flows labelled as botnet traffic become Botnet, normal flows become BENIGN; background
  flows have no verified label and are excluded, as in the original evaluation;
- host identifiers (StartTime, SrcAddr, DstAddr, Sport, Dport) are not features;
- `Proto`, `Dir` and `State` are one-hot encoded for their most frequent values;
- every scenario is a `Group`, so a `by_group` split can test on botnets never seen in training.
"""

from __future__ import annotations

import re
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

DATASET_ID = "ctu13-binetflow-v1"
CTU13 = DatasetInfo(
    name="CTU-13 (labelled bidirectional NetFlows, real botnet traffic)",
    source_url="https://www.stratosphereips.org/datasets-ctu13",
    license="Stratosphere Laboratory, CTU Prague: free for research use; cite Garcia et al. (2014).",
    citation=(
        "S. Garcia, M. Grill, J. Stiborek, A. Zunino. An empirical comparison of botnet detection "
        "methods. Computers & Security 45, 2014. doi:10.1016/j.cose.2014.05.011"
    ),
    citation_doi="10.1016/j.cose.2014.05.011",
)
MANUAL_STEPS = (
    "CTU-13 is downloaded from the Stratosphere Laboratory and cannot be fetched automatically. "
    "1) Open https://www.stratosphereips.org/datasets-ctu13 and download the dataset. 2) Copy each "
    "scenario's labelled `*.binetflow` file into {raw}, in folders named 1 to 13 (or keep the "
    "original folder names). 3) Run `cyberlab dataset prepare --name ctu13`."
)
NUMERIC = ("Dur", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes")
CATEGORICAL = ("Proto", "Dir", "State")
TOP_VALUES = 12


def _label(raw: str) -> str | None:
    s = str(raw)
    if "Botnet" in s:
        return "Botnet"
    if "Normal" in s:
        return BENIGN
    return None  # background or unknown: excluded


def _group(path: Path, raw: Path) -> str:
    rel = path.relative_to(raw)
    if len(rel.parts) > 1:
        return rel.parts[0]
    m = re.search(r"(\d+)", path.stem)
    return m.group(1) if m else path.stem


def prepare(
    root: Path,
    split: SplitSpec | None = None,
    benign_cap: int | None = 400_000,
    class_cap: int | None = 400_000,
) -> PreparedDataset:
    split = split or SplitSpec()
    if split.strategy in ("by_day", "official"):
        raise DatasetBlocked(
            ["CTU-13 has capture scenarios, not days or an official partition; use by_group"]
        )
    raw = root / "raw"
    files = sorted(raw.rglob("*.binetflow")) if raw.exists() else []
    if not files:
        raise DatasetBlocked([MANUAL_STEPS.format(raw=raw)])
    frames = []
    excluded = 0
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if "Label" not in df.columns:
            raise DatasetBlocked([f"{f.name}: no Label column (not a labelled binetflow file?)"])
        df["Class"] = df["Label"].map(_label)
        excluded += int(df["Class"].isna().sum())
        df = df.dropna(subset=["Class"])
        df["Group"] = _group(f, raw)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    pre_stages = [
        {
            "stage": "background flows excluded",
            "input_rows": len(df) + excluded,
            "removed_rows": excluded,
            "remaining_rows": len(df),
            "reason": "background flows carry no verified label",
            "classes_after": {str(k): int(v) for k, v in df["Class"].value_counts().items()},
        }
    ]
    vocab = {
        c: [str(v) for v in df[c].astype(str).value_counts().index[:TOP_VALUES]]
        for c in CATEGORICAL
        if c in df.columns
    }
    out = df[[c for c in NUMERIC if c in df.columns]].copy()
    for c, values in vocab.items():
        col = df[c].astype(str)
        for v in values:
            out[f"{c}={v}"] = (col == v).astype("float32")
        out[f"{c}=other"] = (~col.isin(values)).astype("float32")
    out["Class"] = df["Class"].to_numpy()
    out["Group"] = df["Group"].to_numpy()
    return finalize(
        out,
        root=root,
        split=split,
        dataset_id=DATASET_ID,
        info=CTU13,
        files={str(f.relative_to(raw)): sha256_file(f) for f in files},
        group_column="Group",
        benign_cap=benign_cap,
        class_cap=class_cap,
        extra_notes={
            "pre_stages": pre_stages,
            "background_or_unlabelled_flows_excluded": excluded,
            "one_hot_vocabulary": vocab,
            "groups": sorted(str(g) for g in df["Group"].unique()),
        },
    )


def write_fixture(raw_dir: Path, rows_per_class: int = 60, seed: int = 13) -> list[Path]:
    """Synthetic binetflow files for three scenarios. Test use only."""
    rng = np.random.default_rng(seed)
    paths = []
    labels = [
        "flow=From-Normal-V42-Stribrka",
        "flow=From-Botnet-V42-UDP-DNS",
        "flow=Background-UDP",
    ]
    for scen in ("1", "2", "3"):
        d = raw_dir / scen
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for j, label in enumerate(labels):
            for _ in range(rows_per_class):
                rows.append(
                    {
                        "StartTime": "2011/08/10 09:46:53",
                        "Dur": abs(rng.normal(1 + 5 * j, 0.5)),
                        "Proto": rng.choice(["tcp", "udp"]),
                        "SrcAddr": "147.32.84.165",
                        "Sport": int(rng.integers(1024, 65000)),
                        "Dir": rng.choice(["->", "<->"]),
                        "DstAddr": "8.8.8.8",
                        "Dport": 53,
                        "State": rng.choice(["CON", "FSPA_FSPA", "INT"]),
                        "sTos": 0,
                        "dTos": 0,
                        "TotPkts": int(abs(rng.normal(5 + 20 * j, 2))),
                        "TotBytes": int(abs(rng.normal(500 + 900 * j, 50))),
                        "SrcBytes": int(abs(rng.normal(200 + 400 * j, 30))),
                        "Label": label,
                    }
                )
        p = d / f"capture{scen}.binetflow"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths.append(p)
    return paths
