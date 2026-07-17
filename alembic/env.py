from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, inspect, pool, text

from platform_app.config import get_settings
from platform_app.database import Base
from platform_app import models  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    return get_settings().resolved_database_url


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={'paramstyle': 'named'},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration['sqlalchemy.url'] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        tables = set(inspect(connection).get_table_names())
        # Early releases used create_all and therefore have no Alembic stamp.
        # Their schema matches 0001, so stamp that baseline before applying upgrades.
        if 'content_items' in tables and 'alembic_version' not in tables:
            connection.execute(text(
                'CREATE TABLE alembic_version '
                '(version_num VARCHAR(32) NOT NULL PRIMARY KEY)'
            ))
            connection.execute(text(
                "INSERT INTO alembic_version (version_num) VALUES ('0001_initial')"
            ))
            connection.commit()
        elif connection.in_transaction():
            connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
