from collections.abc import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _sqlite_pragmas(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA foreign_keys=ON')
    cursor.execute('PRAGMA journal_mode=WAL')
    cursor.execute('PRAGMA busy_timeout=5000')
    cursor.close()


def _sqlite_register_functions(dbapi_connection, _connection_record):
    from .domain.search import normalize_title_text

    dbapi_connection.create_function('normalize_title', 1, normalize_title_text)


def configure_engine(database_url: str | None = None) -> Engine:
    global engine, _session_factory
    url = database_url or get_settings().resolved_database_url
    if engine is not None:
        engine.dispose()
    connect_args = {'check_same_thread': False} if url.startswith('sqlite') else {}
    engine = create_engine(url, connect_args=connect_args)
    if url.startswith('sqlite'):
        event.listen(engine, 'connect', _sqlite_pragmas)
        event.listen(engine, 'connect', _sqlite_register_functions)
    _session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)
    return engine


def get_engine() -> Engine:
    if engine is None:
        configure_engine()
    assert engine is not None
    return engine


def SessionLocal() -> Session:
    """Always return a session from the currently configured engine."""
    global _session_factory
    if _session_factory is None:
        configure_engine()
    assert _session_factory is not None
    return _session_factory()


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


configure_engine()
