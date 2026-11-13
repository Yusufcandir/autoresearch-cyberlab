"""Copying the canonical database to another engine (SQLite -> PostgreSQL move)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cyberlab_common.db import Database, User
from cyberlab_common.migrate import copy_database


def test_copy_database_copies_every_row_and_refuses_a_non_empty_target(tmp_path: Path) -> None:
    src_url, dst_url = f"sqlite:///{tmp_path / 'a.db'}", f"sqlite:///{tmp_path / 'b.db'}"
    src = Database(src_url)
    src.create_all()
    with src.session() as s:
        s.add(User(username="alice", password_hash="x", role="pi"))  # noqa: S106
    counts = copy_database(src_url, dst_url)
    assert counts["users"] == 1
    with pytest.raises(RuntimeError, match="not empty"):
        copy_database(src_url, dst_url)
