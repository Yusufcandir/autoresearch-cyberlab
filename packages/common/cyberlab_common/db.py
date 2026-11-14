"""Canonical relational model (spec Steps 08 and 20).

SQLite for development and tests, PostgreSQL in production; the code uses only portable
SQLAlchemy types. Every mutable record carries actor and timestamps, and every change is
also written to `audit_events`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def new_id() -> str:
    return str(uuid.uuid4())


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16))  # admin | pi | researcher | reviewer
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(300))
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    spec: Mapped[dict[str, Any]] = mapped_column(JSON)
    spec_version: Mapped[int] = mapped_column(Integer, default=1)
    stage: Mapped[str] = mapped_column(String(40), default="created")
    status_label: Mapped[str] = mapped_column(String(20), default="Proposed")
    blocked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    paused: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    gpu_minutes_used: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    # Soft delete (NFR-009): hidden from everyone; records and evidence are kept for audit and
    # retention, and an admin can restore the study.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("project_id", "user_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    role: Mapped[str] = mapped_column(String(16))  # pi | researcher | reviewer


class ProtocolVersion(Base):
    __tablename__ = "protocols"
    __table_args__ = (UniqueConstraint("project_id", "version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    version: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    body: Mapped[dict[str, Any]] = mapped_column(JSON)
    rationale: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ApprovalRecord(Base):
    __tablename__ = "approvals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    gate: Mapped[str] = mapped_column(String(40))
    subject_version: Mapped[int] = mapped_column(Integer)
    subject_sha256: Mapped[str] = mapped_column(String(64))
    approver_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    decision: Mapped[str] = mapped_column(String(10))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class DatasetRecord(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class JobRecord(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    experiment_id: Mapped[str] = mapped_column(String(80))
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    state: Mapped[str] = mapped_column(String(16), default="queued")
    external_ref: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RunRecord(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    experiment_id: Mapped[str] = mapped_column(String(80))
    model_name: Mapped[str] = mapped_column(String(80))
    seed: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    receipt: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ArtifactRecord(Base):
    __tablename__ = "artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class CitationRecord(Base):
    __tablename__ = "citations"
    __table_args__ = (UniqueConstraint("project_id", "key"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    key: Mapped[str] = mapped_column(String(120))
    doi: Mapped[str | None] = mapped_column(String(200), nullable=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[Any]] = mapped_column(JSON, default=list)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    cited_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verification_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16))  # Verified | Unverified
    source: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ClaimRecord(Base):
    __tablename__ = "claims"
    # "<project8>:<model>.<metric>" ids are longer than a UUID
    id: Mapped[str] = mapped_column(String(128), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    body: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ManuscriptRecord(Base):
    __tablename__ = "manuscripts"
    __table_args__ = (UniqueConstraint("project_id", "version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    version: Mapped[int] = mapped_column(Integer)
    markdown: Mapped[str] = mapped_column(Text)
    bibtex: Mapped[str] = mapped_column(Text, default="")
    sha256: Mapped[str] = mapped_column(String(64))
    evidence_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    review: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | released
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentNote(Base):
    """Agent outputs (prose, plans). Always labelled Proposed: never evidence."""

    __tablename__ = "agent_notes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    agent: Mapped[str] = mapped_column(String(40))
    kind: Mapped[str] = mapped_column(String(40))
    content: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    actor: Mapped[str] = mapped_column(String(64))
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(60))
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OutboxEvent(Base):
    """Transactional outbox (spec 10.2): events are written in the same transaction as state."""

    __tablename__ = "outbox"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(60))
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn: Any, _rec: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine = make_engine(url)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Tiny forward-only migration: add nullable columns that newer code expects."""
        insp = inspect(self.engine)
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if not insp.has_table(table.name):
                    continue
                existing = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name not in existing and col.nullable:
                        ddl = col.type.compile(dialect=self.engine.dialect)
                        conn.execute(
                            text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}')
                        )

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


def audit(
    s: Session,
    actor: str,
    action: str,
    project_id: str | None = None,
    detail: dict[str, Any] | None = None,
    emit: bool = True,
) -> None:
    """Append an audit event and (optionally) an outbox event in the caller's transaction."""
    payload = json.loads(json.dumps(detail or {}, default=str))
    s.add(AuditEvent(actor=actor, action=action, project_id=project_id, detail=payload))
    if emit:
        s.add(OutboxEvent(event_type=action, project_id=project_id, payload=payload))
