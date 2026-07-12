import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from platform_app.config import clear_settings_cache
from platform_app.database import Base, configure_engine, get_engine


@pytest.fixture()
def platform_env(tmp_path, monkeypatch):
    db_path = tmp_path / 'test.db'
    media_root = tmp_path / 'media'
    media_root.mkdir()
    data_dir = tmp_path / 'platform_data'
    data_dir.mkdir()
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path.as_posix()}')
    monkeypatch.setenv('ALLOWED_MEDIA_ROOTS', str(media_root))
    monkeypatch.setenv('PLATFORM_DATA_DIR', str(data_dir))
    monkeypatch.setenv('INGEST_API_KEY', 'test-ingest-key')
    monkeypatch.setenv('CSRF_ENABLED', 'true')
    monkeypatch.setenv('AUTO_CREATE_TABLES', 'true')
    monkeypatch.setenv('TRAINING_LABEL_THRESHOLD', '2')
    clear_settings_cache()
    configure_engine(f'sqlite:///{db_path.as_posix()}')
    Base.metadata.drop_all(bind=get_engine())
    Base.metadata.create_all(bind=get_engine())
    yield {
        'db_path': db_path,
        'media_root': media_root,
        'data_dir': data_dir,
    }
    clear_settings_cache()


@pytest.fixture()
def client(platform_env):
    from platform_app.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_client(client):
    response = client.post('/api/v1/auth/setup', json={
        'username': 'owner',
        'password': 'change-this-to-a-long-password',
    })
    assert response.status_code == 201
    csrf = client.cookies.get('preference_platform_csrf')
    client.headers.update({'X-CSRF-Token': csrf})
    return client


def make_image(path: Path, size=(256, 256), color=(120, 40, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color=color).save(path)
    return path
