"""Text adapter (multilingual documents) for the language/tabular capability (AT-16)."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cyberlab_schemas.plugins import (
    SCHEMA_INVALID,
    Capability,
    PluginError,
    ValidationReport,
    group_split,
    require_view,
)

CAPABILITY = Capability(
    name="text",
    version="1",
    supported_modalities=("text",),
    feature_views=("token_counts", "hashed_bow", "group_split"),
    notes="Language-agnostic tokenisation; no external models are downloaded.",
)


class TextDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    language: str = Field(pattern=r"^[a-z]{2}$")
    group: str = Field(min_length=1, description="source document / author / translation family")


class TextDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: Literal["text"] = "text"
    documents: list[TextDocument] = Field(min_length=2)


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _parse(manifest: Mapping[str, Any]) -> TextDataset:
    try:
        return TextDataset.model_validate(manifest)
    except ValidationError as exc:
        raise PluginError(
            SCHEMA_INVALID,
            "text dataset does not match the schema",
            {"errors": exc.errors(include_url=False)},
        ) from exc


class TextAdapter:
    def capabilities(self) -> Capability:
        return CAPABILITY

    def validate_dataset(self, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
        ds = _parse(dataset_manifest)
        problems = []
        ids = [d.id for d in ds.documents]
        if len(set(ids)) != len(ids):
            problems.append("duplicate document ids")
        labels = Counter(d.label for d in ds.documents)
        if len(labels) < 2:
            problems.append("needs at least two labels")
        return ValidationReport(
            ok=not problems,
            modality="text",
            n_items=len(ds.documents),
            groups=sorted({d.group for d in ds.documents}),
            problems=problems,
            stats={
                "labels": dict(labels),
                "languages": dict(Counter(d.language for d in ds.documents)),
            },
        ).as_dict()

    def feature_view(self, name: str, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
        require_view(CAPABILITY, name)
        ds = _parse(dataset_manifest)
        if name == "group_split":
            split = group_split([d.group for d in ds.documents])
            return {"view": name, **split}
        if name == "token_counts":
            return {
                "view": name,
                "rows": {
                    d.id: dict(Counter(t.lower() for t in _TOKEN.findall(d.text)))
                    for d in ds.documents
                },
            }
        dim = 64
        rows = {}
        for d in ds.documents:
            vec = [0] * dim
            for t in _TOKEN.findall(d.text.lower()):
                vec[int(hashlib.md5(t.encode(), usedforsecurity=False).hexdigest(), 16) % dim] += 1
            rows[d.id] = vec
        return {"view": name, "dim": dim, "rows": rows}


def fixture() -> dict[str, Any]:
    """Small synthetic multilingual fixture (test use only)."""
    docs = [
        ("d1", "normal login from office network", "benign", "en", "s1"),
        ("d2", "nombreux échecs de connexion ssh", "attack", "fr", "s2"),
        ("d3", "routine backup completed", "benign", "en", "s3"),
        ("d4", "çok sayıda başarısız giriş denemesi", "attack", "tr", "s4"),
        ("d5", "scheduled update installed", "benign", "en", "s5"),
        ("d6", "port scan detected on many hosts", "attack", "en", "s6"),
    ]
    return {
        "modality": "text",
        "documents": [
            {"id": i, "text": t, "label": lab, "language": lang, "group": g}
            for i, t, lab, lang, g in docs
        ],
    }
