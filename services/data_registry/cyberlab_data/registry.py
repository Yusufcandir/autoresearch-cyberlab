"""Registered datasets. A study names its dataset once (the new-study form); every agent, the
broker and the manuscript look the adapter up here instead of assuming CIC-IDS2017."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cyberlab_data import cicids2017, ctu13, netflow, unsw_nb15
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
    # (root, benign_cap, class_cap, seed) -> (frame with features + Class, file hashes, notes);
    # only datasets in a shared feature format can be combined for cross-dataset testing
    load: Callable[..., Any] | None = None
    observed: bool = False  # captured on a real network (spec 7.5: label synthetic vs observed)

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

CTU = DatasetAdapter(
    key="ctu13",
    display="CTU-13",
    dataset_id=ctu13.DATASET_ID,
    info=ctu13.CTU13,
    paper={
        "title": "An empirical comparison of botnet detection methods",
        "year": 2014,
        "authors": ["Sebastian Garcia", "Martin Grill", "Jan Stiborek", "Alejandro Zunino"],
        "venue": "Computers & Security",
        "key": "Garcia2014empirical",
    },
    prepare=ctu13.prepare,
    write_fixture=ctu13.write_fixture,
    manual_steps=ctu13.MANUAL_STEPS,
    split_strategies=("stratified_dedup", "by_group"),
    search_query="CTU-13 botnet detection machine learning",
    preprocessing=[
        "Flows labelled as botnet traffic form the Botnet class and normal flows the BENIGN "
        "class; background flows have no verified label and were excluded.",
        "Host identifiers (start time, addresses and ports) were not used as features.",
        "The categorical fields Proto, Dir and State were one-hot encoded for their most "
        "frequent values; all other values share one 'other' column.",
        "Constant features were dropped.",
    ],
    observed=True,
)


def _netflow(v: netflow.Variant) -> DatasetAdapter:
    def load(root: Path, benign_cap: int | None, class_cap: int | None, seed: int) -> Any:
        return netflow.load_netflow(v, root, benign_cap, class_cap, seed)

    return DatasetAdapter(
        key=v.key,
        display=v.display,
        dataset_id=v.key.replace("_", "-"),
        info=netflow.info_for(v),
        paper={
            "title": "Towards a Standard Feature Set for Network Intrusion Detection System "
            "Datasets",
            "year": 2021,
            "authors": ["Mohanad Sarhan", "Siamak Layeghy", "Marius Portmann"],
            "venue": "Mobile Networks and Applications",
            "key": "Sarhan2022standard",
        },
        prepare=netflow.make_prepare(v),
        write_fixture=netflow.make_fixture(v),
        manual_steps=netflow.manual_steps(v),
        split_strategies=("stratified_dedup", "cross_dataset"),
        search_query=f"{v.origin} NetFlow intrusion detection cross-dataset generalisation",
        preprocessing=[
            f"The NetFlow-v2 version of {v.origin} was used: 43 NetFlow features shared by all "
            "datasets of the family.",
            "Source and destination IP addresses and ports were not used as features; they "
            "describe the capture network, not the attack.",
            "The binary Label column repeats the target and was dropped; the Attack column gives "
            "the classes (Benign traffic is called BENIGN).",
            "Each class was sampled with a fixed seed to at most facts.benign_cap; smaller "
            "classes kept all of their rows.",
            "Constant features were dropped.",
        ],
        load=load,
    )


ADAPTERS: dict[str, DatasetAdapter] = {
    a.key: a for a in (CIC, UNSW, CTU, *(_netflow(v) for v in netflow.VARIANTS.values()))
}


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
    # 'UNSW-NB15 (official sets)' and similar free text; the longest name wins, so
    # 'NF-UNSW-NB15-v2' is not mistaken for 'UNSW-NB15'
    for a in sorted(ADAPTERS.values(), key=lambda a: -len(a.display)):
        if _norm(a.display) in want:
            return a
    return CIC


def prepare_split(
    adapter: DatasetAdapter,
    dataset_root: Path,
    split: Any,
    benign_cap: int | None = 400_000,
) -> Any:
    """Prepare data for a protocol's split. A cross-dataset split trains on this dataset and
    tests on another one with the same features (the NetFlow-v2 family)."""
    if getattr(split, "strategy", None) != "cross_dataset":
        return adapter.prepare(adapter.root(dataset_root), split, benign_cap=benign_cap)
    from cyberlab_data.common import BENIGN, DatasetBlocked, DatasetInfo, finalize

    if not split.test_days:
        raise DatasetBlocked(["a cross-dataset split needs the test dataset (test_days[0])"])
    other = get_adapter(split.test_days[0])
    if adapter.load is None or other.load is None or other.key == adapter.key:
        raise DatasetBlocked(
            [
                f"cross-dataset testing needs two different datasets with a shared feature "
                f"format (NetFlow-v2 family); got {adapter.display} and {other.display}"
            ]
        )
    a, files_a, notes_a = adapter.load(
        adapter.root(dataset_root), benign_cap, benign_cap, split.seed
    )
    b, files_b, notes_b = other.load(other.root(dataset_root), benign_cap, benign_cap, split.seed)
    shared = sorted((set(a.columns) & set(b.columns)) - {"Class"})
    if len(shared) < 10:
        raise DatasetBlocked([f"only {len(shared)} shared features between the datasets"])
    a, b = a[[*shared, "Class"]].copy(), b[[*shared, "Class"]].copy()
    a["Part"], b["Part"] = "train", "test"
    both = pd.concat([a, b], ignore_index=True)  # train first: dedup keeps training copies
    info = DatasetInfo(
        name=f"{adapter.display} (train) and {other.display} (test)",
        source_url=adapter.info.source_url,
        license=adapter.info.license,
        citation=adapter.info.citation,
        citation_doi=adapter.info.citation_doi,
    )
    attack_a = sorted(set(a["Class"]) - {BENIGN})
    attack_b = sorted(set(b["Class"]) - {BENIGN})
    return finalize(
        both,
        root=adapter.root(dataset_root),
        split=split,
        dataset_id=adapter.dataset_id,
        info=info,
        files={
            **{f"train/{k}": v for k, v in files_a.items()},
            **{f"test/{k}": v for k, v in files_b.items()},
        },
        group_column="Part",
        benign_cap=None,
        class_cap=None,
        extra_notes={
            "train_dataset": adapter.display,
            "test_dataset": other.display,
            "shared_features": len(shared),
            "attack_classes_train_only": sorted(set(attack_a) - set(attack_b)),
            "attack_classes_test_only": sorted(set(attack_b) - set(attack_a)),
            "train_notes": notes_a,
            "test_notes": notes_b,
        },
    )


def choices() -> list[tuple[str, str]]:
    return [(a.display, a.info.name) for a in ADAPTERS.values()]
