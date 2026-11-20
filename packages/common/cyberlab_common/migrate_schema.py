"""`cyberlab db-upgrade`: apply Alembic migrations. An existing database created before
migrations existed is stamped at the baseline first, so nothing is re-created."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from cyberlab_common.db import Database


def upgrade(db: Database) -> str:
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    with db.engine.begin() as conn:
        cfg.attributes["connection"] = conn
        tables = set(inspect(conn).get_table_names())
        if "alembic_version" not in tables and "projects" in tables:
            command.stamp(cfg, "0001_baseline")
        command.upgrade(cfg, "head")
        row = conn.exec_driver_sql("SELECT version_num FROM alembic_version").fetchone()
    db._add_missing_columns()  # forward-only safety net for nullable columns
    return str(row[0]) if row else "none"
