"""Baseline: the schema as of November 2026 (all tables of cyberlab_common.db).

Revision ID: 0001_baseline
Revises:
"""

from __future__ import annotations

from alembic import op

from cyberlab_common.db import Base

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(op.get_bind())


def downgrade() -> None:
    raise RuntimeError("the baseline cannot be downgraded; restore a backup instead")
