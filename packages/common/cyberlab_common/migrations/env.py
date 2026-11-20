"""Alembic environment: the target metadata is the platform's SQLAlchemy models."""

from __future__ import annotations

from alembic import context

from cyberlab_common.db import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = config.attributes["connection"]
    context.configure(connection=connectable, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
