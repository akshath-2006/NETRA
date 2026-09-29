"""Engine and session management.

The PRAGMA settings below are the entire reason SQLite copes with several
camera workers writing at once. They are worth understanding rather than
copying:

* ``journal_mode=WAL`` -- readers never block the writer and the writer never
  blocks readers. Without this the API would stall every time a worker wrote.
* ``busy_timeout=5000`` -- when two workers do collide, the loser waits up to
  five seconds and retries instead of raising "database is locked" mid-demo.
* ``synchronous=NORMAL`` -- safe under WAL, and much faster than FULL.

Swapping to Postgres later means changing the URL and dropping the pragmas.
Nothing else in the codebase knows which database it is talking to.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.logging import get_logger
from app.store.models import Base

log = get_logger("db")

_engines: dict[str, Engine] = {}


def _apply_sqlite_pragmas(dbapi_connection, _record) -> None:
    cur = dbapi_connection.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def get_engine(database: Path | str, echo: bool = False) -> Engine:
    """Create (or reuse) the engine for a database path."""
    key = str(database)
    if key in _engines:
        return _engines[key]

    if key.startswith(("postgresql", "mysql", "sqlite:")):
        url = key                                   # already a URL
    else:
        Path(key).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{key}"

    engine = create_engine(url, echo=echo, future=True)
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _apply_sqlite_pragmas)

    _engines[key] = engine
    return engine


def get_sessionmaker(database: Path | str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database), expire_on_commit=False, future=True)


def _add_missing_columns(engine: Engine) -> None:
    """Add columns that later milestones introduced.

    ``create_all`` creates missing *tables* but never alters existing ones, so
    without this an upgrade would mean telling you to delete your database.
    SQLite ALTER TABLE ADD COLUMN is cheap and non-destructive; a real
    deployment would use Alembic.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            try:
                existing = {row[1] for row in
                            conn.execute(text(f"PRAGMA table_info('{table.name}')"))}
            except Exception:
                continue
            if not existing:
                continue
            for column in table.columns:
                if column.name in existing or column.primary_key:
                    continue
                ddl = f"{column.name} {column.type.compile(engine.dialect)}"
                try:
                    conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {ddl}"))
                    log.info("added column %s.%s", table.name, column.name)
                except Exception as exc:      # foreign keys etc. -- not fatal
                    log.debug("could not add %s.%s: %s", table.name, column.name, exc)


def init_db(database: Path | str) -> Engine:
    """Create any missing tables and columns. Safe to call every startup."""
    engine = get_engine(database)
    Base.metadata.create_all(engine)
    if str(engine.url).startswith("sqlite"):
        _add_missing_columns(engine)
    log.info("schema ready at %s (%d tables)", database, len(Base.metadata.tables))
    return engine
