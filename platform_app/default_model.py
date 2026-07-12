"""Register and activate DEFAULT_MODEL_PATH when no active model exists."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import ModelVersion, utcnow


def _checkpoint_meta(path: Path) -> tuple[float, float, dict, str]:
    threshold = 0.5
    temperature = 1.0
    metrics: dict = {'source': 'default-env'}
    data_hash = ''
    try:
        import torch

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            threshold = float(checkpoint.get('decision_threshold', threshold))
            temperature = float(checkpoint.get('temperature', temperature))
            data_hash = str(checkpoint.get('data_manifest_hash', '') or '')
            report = checkpoint.get('metrics')
            if isinstance(report, dict):
                metrics = {**report, 'source': 'default-env'}
            elif 'val_acc' in checkpoint:
                metrics = {
                    'source': 'default-env',
                    'val_acc': float(checkpoint.get('val_acc') or 0),
                }
    except Exception:
        pass
    return threshold, temperature, metrics, data_hash


def ensure_default_model(session: Session) -> ModelVersion | None:
    """If nothing is active, register/activate the env default checkpoint.

    Manual upload + activate always wins: as long as any row is ``active``,
    this function is a no-op.
    """
    active = session.scalar(select(ModelVersion).where(ModelVersion.status == 'active'))
    if active is not None:
        return active

    settings = get_settings()
    path = settings.resolved_default_model_path
    if path is None or not path.is_file():
        return None

    version = (settings.default_model_version or 'default-env').strip() or 'default-env'
    model = session.scalar(select(ModelVersion).where(ModelVersion.version == version))
    if model is None:
        threshold, temperature, metrics, data_hash = _checkpoint_meta(path)
        model = ModelVersion(
            version=version,
            status='active',
            checkpoint_path=str(path),
            decision_threshold=threshold,
            temperature=temperature,
            metrics=metrics,
            data_manifest_hash=data_hash,
            activated_at=utcnow(),
        )
        session.add(model)
    else:
        model.checkpoint_path = str(path)
        model.status = 'active'
        model.activated_at = utcnow()
        if not model.metrics:
            model.metrics = {'source': 'default-env'}
    session.commit()
    session.refresh(model)
    return model
