from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from platform_app.config import clear_settings_cache
from platform_app.database import Base


def test_alembic_upgrade_creates_snapshot_event_membership(tmp_path, monkeypatch):
    database = tmp_path / 'migrated.db'
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{database.as_posix()}')
    clear_settings_cache()
    try:
        command.upgrade(Config('alembic.ini'), 'head')
        engine = create_engine(f'sqlite:///{database.as_posix()}')
        try:
            schema = inspect(engine)
            assert 'snapshot_label_events' in schema.get_table_names()
            columns = {column['name'] for column in schema.get_columns('snapshot_label_events')}
            assert columns == {'event_id', 'snapshot_id'}
            content_columns = {column['name'] for column in schema.get_columns('content_items')}
            assert 'label_version' in content_columns
            with engine.connect() as connection:
                version = connection.execute(text('SELECT version_num FROM alembic_version')).scalar_one()
            assert version == '0003_content_label_version'
        finally:
            engine.dispose()
    finally:
        clear_settings_cache()


def test_alembic_bootstraps_an_unstamped_create_all_database(tmp_path, monkeypatch):
    database = tmp_path / 'legacy-create-all.db'
    url = f'sqlite:///{database.as_posix()}'
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text('DROP TABLE snapshot_label_events'))
    engine.dispose()

    monkeypatch.setenv('DATABASE_URL', url)
    clear_settings_cache()
    try:
        command.upgrade(Config('alembic.ini'), 'head')
        migrated = create_engine(url)
        try:
            schema = inspect(migrated)
            assert 'snapshot_label_events' in schema.get_table_names()
            with migrated.connect() as connection:
                version = connection.execute(text('SELECT version_num FROM alembic_version')).scalar_one()
            assert version == '0003_content_label_version'
            content_columns = {column['name'] for column in schema.get_columns('content_items')}
            assert 'label_version' in content_columns
        finally:
            migrated.dispose()
    finally:
        clear_settings_cache()
