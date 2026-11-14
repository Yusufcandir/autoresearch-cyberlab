"""Copy the canonical database to another engine (e.g. SQLite -> PostgreSQL in Docker).

The target must be empty. Tables are copied in foreign-key order inside one transaction and
row counts are verified afterwards, so a failed copy leaves the target unchanged.
"""

from __future__ import annotations

from sqlalchemy import String, func, insert, select

from cyberlab_common.db import Base, Database


def copy_database(source_url: str, target_url: str, batch: int = 1000) -> dict[str, int]:
    src = Database(source_url)
    src.create_all()  # same schema upgrade the app runs at start (adds missing columns)
    dst = Database(target_url)
    dst.create_all()
    tables = Base.metadata.sorted_tables
    with dst.engine.connect() as check:
        used = {t.name: check.scalar(select(func.count()).select_from(t)) for t in tables}
    if any(used.values()):
        raise RuntimeError(f"target database is not empty: {[k for k, v in used.items() if v]}")
    counts: dict[str, int] = {}
    with src.engine.connect() as s:
        data = {t.name: [dict(r._mapping) for r in s.execute(select(t))] for t in tables}
    # SQLite does not enforce VARCHAR lengths, PostgreSQL does: check everything first.
    too_long = [
        f"{t.name}.{c.name}: {max(len(str(r[c.name])) for r in data[t.name] if r[c.name] is not None)}"
        f" > {c.type.length}"
        for t in tables
        for c in t.columns
        if isinstance(c.type, String)
        and c.type.length
        and any(r[c.name] is not None and len(str(r[c.name])) > c.type.length for r in data[t.name])
    ]
    if too_long:
        raise RuntimeError(f"values longer than the target columns allow: {too_long}")
    with dst.engine.begin() as d:
        for t in tables:
            rows = data[t.name]
            for i in range(0, len(rows), batch):
                d.execute(insert(t), rows[i : i + batch])
            counts[t.name] = len(rows)
    with dst.engine.connect() as check:
        for t in tables:
            got = check.scalar(select(func.count()).select_from(t))
            if got != counts[t.name]:
                raise RuntimeError(f"{t.name}: copied {counts[t.name]} rows but found {got}")
    if dst.engine.dialect.name == "postgresql":
        # Rows were inserted with their original integer ids: move each id sequence past them,
        # otherwise the next insert would reuse an existing id.
        with dst.engine.begin() as d:
            for t in tables:
                pk = list(t.primary_key.columns)
                if len(pk) == 1 and pk[0].autoincrement is True and counts[t.name]:
                    # table and column names come from our own schema, never from input
                    sql = (
                        f"SELECT setval(pg_get_serial_sequence('{t.name}', '{pk[0].name}'), "  # noqa: S608
                        f"(SELECT MAX({pk[0].name}) FROM {t.name}))"
                    )
                    d.exec_driver_sql(sql)
    return counts
