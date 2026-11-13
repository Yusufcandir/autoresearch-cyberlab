"""Identity, roles and project isolation (spec Step 07).

Passwords: PBKDF2-SHA256 (stdlib). Sessions: HMAC-signed bearer tokens with expiry.
Every project read or write goes through `require_project_role`, so a user who is not a
member gets the same 404 as for a project that does not exist (no existence leak, AT-02).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from cyberlab_common.db import Membership, Project, User

ROLES = ("admin", "pi", "researcher", "reviewer")
PROJECT_RANK = {"reviewer": 1, "researcher": 2, "pi": 3}
_ITERATIONS = 240_000


class AuthError(Exception):
    """Authentication failed (401)."""


class NotFoundOrForbidden(Exception):
    """The resource does not exist or the user may not see it (404)."""


class Forbidden(Exception):
    """The user can see the resource but lacks the role for this action (403)."""


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("password must be at least 10 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, iters, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
    return hmac.compare_digest(digest.hex(), digest_hex)


def issue_token(user_id: str, secret: str, ttl_seconds: int = 12 * 3600) -> str:
    expiry = int(time.time()) + ttl_seconds
    body = f"{user_id}.{expiry}"
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    return body + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")


def read_token(token: str, secret: str) -> str:
    try:
        user_id, expiry, sig = token.rsplit(".", 2)
    except ValueError as exc:
        raise AuthError("malformed token") from exc
    body = f"{user_id}.{expiry}"
    good = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    expected = base64.urlsafe_b64encode(good).decode().rstrip("=")
    if not hmac.compare_digest(expected, sig):
        raise AuthError("bad signature")
    if int(expiry) < time.time():
        raise AuthError("token expired")
    return user_id


@dataclass(frozen=True)
class Principal:
    id: str
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def create_user(s: Session, username: str, password: str, role: str) -> User:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if s.scalar(select(User).where(User.username == username)):
        raise ValueError(f"user {username!r} already exists")
    user = User(username=username, password_hash=hash_password(password), role=role)
    s.add(user)
    s.flush()
    return user


def authenticate(s: Session, username: str, password: str) -> Principal:
    user = s.scalar(select(User).where(User.username == username, User.active.is_(True)))
    if user is None or not verify_password(password, user.password_hash):
        raise AuthError("invalid username or password")
    return Principal(user.id, user.username, user.role)


def principal_from_token(s: Session, token: str, secret: str) -> Principal:
    user = s.get(User, read_token(token, secret))
    if user is None or not user.active:
        raise AuthError("unknown user")
    return Principal(user.id, user.username, user.role)


def project_role(s: Session, principal: Principal, project_id: str) -> str | None:
    if principal.is_admin:
        return "pi" if s.get(Project, project_id) else None
    m = s.scalar(
        select(Membership).where(
            Membership.project_id == project_id, Membership.user_id == principal.id
        )
    )
    return m.role if m else None


def require_project_role(
    s: Session, principal: Principal, project_id: str, minimum: str = "reviewer"
) -> Project:
    project = s.get(Project, project_id)
    role = project_role(s, principal, project_id) if project else None
    # A deleted study behaves as if it did not exist (an admin restores it from the console).
    if project is None or role is None or project.deleted_at is not None:
        raise NotFoundOrForbidden(project_id)
    if PROJECT_RANK[role] < PROJECT_RANK[minimum]:
        raise Forbidden(f"requires project role {minimum}")
    return project


def accessible_projects(s: Session, principal: Principal) -> list[Project]:
    live = Project.deleted_at.is_(None)
    if principal.is_admin:
        return list(s.scalars(select(Project).where(live).order_by(Project.created_at.desc())))
    q = (
        select(Project)
        .join(Membership, Membership.project_id == Project.id)
        .where(Membership.user_id == principal.id, live)
        .order_by(Project.created_at.desc())
    )
    return list(s.scalars(q))


def add_member(s: Session, project_id: str, user_id: str, role: str) -> None:
    if role not in PROJECT_RANK:
        raise ValueError("project role must be pi, researcher or reviewer")
    s.add(Membership(project_id=project_id, user_id=user_id, role=role))
