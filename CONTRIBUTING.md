# Contributing

1. Branch from `main` (`feature/<step>-<short-name>`); never push directly to `main`.
2. Run `uv run ruff check . && uv run mypy && uv run pytest` before opening a PR.
3. Fill in the PR template, including which spec step and acceptance test (AT-xx) it serves.
4. Dependencies are pinned in `uv.lock`; change them only in a dedicated PR.
5. Schema changes go through migrations (Alembic, from Step 08).
