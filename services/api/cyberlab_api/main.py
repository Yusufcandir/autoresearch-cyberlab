"""No-op reference API: health endpoints only (Step 05).

Real endpoints from spec section 10.1 are added in Step 09.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from cyberlab_common.config import Settings

app = FastAPI(title="AutoResearch-CyberLab API", version="0.1.0")


@lru_cache
def get_settings() -> Settings:
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
def ready(settings: SettingsDep) -> JSONResponse:
    missing = settings.missing_required()
    if missing:
        body: dict[str, Any] = {
            "type": "about:blank",
            "title": "Not ready",
            "status": 503,
            "detail": "Required configuration is missing; dependent features are blocked.",
            "missing": missing,
        }
        return JSONResponse(body, status_code=503, media_type="application/problem+json")
    return JSONResponse({"status": "ready"})
