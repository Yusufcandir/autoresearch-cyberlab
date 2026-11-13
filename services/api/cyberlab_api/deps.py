"""Shared FastAPI dependencies (kept separate so the API and dashboard modules don't import each other)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException

from cyberlab_common.config import Settings
from cyberlab_orchestrator.lab import Lab

_LAB: dict[str, Lab] = {}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_lab() -> Lab:
    if "lab" not in _LAB:
        settings = get_settings()
        missing = settings.missing_required()
        if missing:
            raise HTTPException(503, f"not configured: missing {missing}")
        _LAB["lab"] = Lab(settings)
    return _LAB["lab"]
