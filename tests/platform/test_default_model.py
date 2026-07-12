from pathlib import Path

from platform_app.config import clear_settings_cache, get_settings
from platform_app.database import SessionLocal
from platform_app.default_model import ensure_default_model
from platform_app.models import ModelVersion, utcnow
from sqlalchemy import select


def test_ensure_default_model_when_no_active(platform_env, monkeypatch, tmp_path):
    # Fake a tiny "checkpoint" file; meta load will fail softly.
    checkpoint = tmp_path / 'fake_model.pth'
    checkpoint.write_bytes(b'not-a-real-torch-file')
    monkeypatch.setenv('DEFAULT_MODEL_PATH', str(checkpoint))
    monkeypatch.setenv('DEFAULT_MODEL_VERSION', 'default-test')
    clear_settings_cache()

    with SessionLocal() as session:
        model = ensure_default_model(session)
        assert model is not None
        assert model.version == 'default-test'
        assert model.status == 'active'
        assert Path(model.checkpoint_path) == checkpoint.resolve()

        again = ensure_default_model(session)
        assert again is not None
        assert again.id == model.id


def test_ensure_default_skips_when_active_exists(platform_env, monkeypatch, tmp_path):
    checkpoint = tmp_path / 'fake_model.pth'
    checkpoint.write_bytes(b'x')
    monkeypatch.setenv('DEFAULT_MODEL_PATH', str(checkpoint))
    monkeypatch.setenv('DEFAULT_MODEL_VERSION', 'default-test')
    clear_settings_cache()

    with SessionLocal() as session:
        manual = ModelVersion(
            version='manual-upload',
            status='active',
            checkpoint_path=str(checkpoint),
            decision_threshold=0.6,
            temperature=1.0,
            metrics={'source': 'manual'},
            activated_at=utcnow(),
        )
        session.add(manual)
        session.commit()

        result = ensure_default_model(session)
        assert result is not None
        assert result.version == 'manual-upload'
        defaults = session.scalars(select(ModelVersion).where(ModelVersion.version == 'default-test')).all()
        assert defaults == []
