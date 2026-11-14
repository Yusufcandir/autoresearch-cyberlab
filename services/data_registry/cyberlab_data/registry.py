"""Registered datasets. A study names its dataset once (the new-study form); every agent, the
broker and the manuscript look the adapter up here instead of assuming CIC-IDS2017."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cyberlab_data import cicids2017, unsw_nb15
from cyberlab_data.common import DatasetInfo


@dataclass(frozen=True)
class DatasetAdapter:
    key: str  # folder under DATASET_ROOT and the CLI --name
    display: str  # short name used in prompts and the form
    dataset_id: str
    info: DatasetInfo
    paper: dict[str, Any]  # dataset paper for the reference list (DOI verified like any other)
    prepare: Callable[..., Any]
    write_fixture: Callable[..., Any]
    manual_steps: str
    split_strategies: tuple[str, ...]
    search_query: str
    preprocessing: list[str] = field(default_factory=list)  # true method sentences
    unzip: Callable[..., Any] | None = None
    download: Callable[..., Any] | None = None

    def root(self, dataset_root: Path) -> Path:
        return dataset_root / self.key


CIC = DatasetAdapter(
    key="cicids2017",
    display="CIC-IDS2017",
    dataset_id=cicids2017.DATASET_ID,
    info=cicids2017.CICIDS2017,
    paper={
        "title": "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic "
        "Characterization",
        "year": 2018,
        "authors": ["Iman Sharafaldin", "Arash Habibi Lashkari", "Ali A. Ghorbani"],
        "venue": "ICISSP 2018",
        "key": "Sharafaldin2018dataset",
    },
    prepare=cicids2017.prepare,
    write_fixture=cicids2017.write_fixture,
    manual_steps=cicids2017.MANUAL_STEPS,
    split_strategies=("stratified_dedup", "by_day"),
    search_query="CIC-IDS2017 intrusion detection machine learning",
    preprocessing=[
        "Constant features and the duplicated Fwd Header Length column were dropped.",
        "Fine-grained labels were grouped into classes: BENIGN, DoS, DDoS, PortScan, BruteForce "
        "(FTP/SSH-Patator), WebAttack, Bot, and Rare (Infiltration and Heartbleed).",
    ],
    unzip=cicids2017.unzip,
    download=cicids2017.download,
)

UNSW = DatasetAdapter(
    key="unsw_nb15",
    display="UNSW-NB15",
    dataset_id=unsw_nb15.DATASET_ID,
    info=unsw_nb15.UNSW_NB15,
    paper={
        "title": "UNSW-NB15: a comprehensive data set for network intrusion detection systems "
        "(UNSW-NB15 network data set)",
        "year": 2015,
        "authors": ["Nour Moustafa", "Jill Slay"],
        "venue": "MilCIS 2015",
        "key": "Moustafa2015unswnb15",
    },
    prepare=unsw_nb15.prepare,
    write_fixture=unsw_nb15.write_fixture,
    manual_steps=unsw_nb15.MANUAL_STEPS,
    split_strategies=("stratified_dedup", "official"),
    search_query="UNSW-NB15 intrusion detection machine learning",
    preprocessing=[
        "The row identifier (id) and the binary label column were not used as features: id "
        "follows the file order, which groups rows by class, and label repeats the target.",
        "The categorical fields proto, service and state were one-hot encoded for their most "
        "frequent values in the training file; all other values share one 'other' column.",
        "Constant features were dropped.",
        "The attack categories were used as classes (Normal traffic is called BENIGN): "
        "Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode and "
        "Worms.",
    ],
)

ADAPTERS: dict[str, DatasetAdapter] = {a.key: a for a in (CIC, UNSW)}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get_adapter(name_or_id: str | None) -> DatasetAdapter:
    """By key, display name, dataset id or a split-specific record id ('id@split');
    unknown or empty names fall back to CIC-IDS2017 (the platform's original dataset)."""
    if not name_or_id:
        return CIC
    want = _norm(name_or_id.split("@")[0])
    for a in ADAPTERS.values():
        if want in {_norm(a.key), _norm(a.display), _norm(a.dataset_id)}:
            return a
    for a in ADAPTERS.values():  # 'UNSW-NB15 (official sets)' and similar free text
        if _norm(a.display) in want:
            return a
    return CIC


def choices() -> list[tuple[str, str]]:
    return [(a.display, a.info.name) for a in ADAPTERS.values()]
