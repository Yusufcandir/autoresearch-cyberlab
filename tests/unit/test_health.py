from fastapi.testclient import TestClient

from cyberlab_api.main import app, get_settings
from cyberlab_common.config import Settings


def _client(settings: Settings) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_live() -> None:
    assert _client(Settings()).get("/health/live").json() == {"status": "live"}


def test_ready_blocks_when_config_missing() -> None:
    resp = _client(Settings(_env_file=None)).get("/health/ready")
    assert resp.status_code == 503
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert "DATABASE_URL" in resp.json()["missing"]


def test_ready_when_configured() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost/db",
        redis_url="redis://localhost:6379/0",
        audit_log_destination="file:///tmp/audit.log",
    )
    assert _client(settings).get("/health/ready").status_code == 200
