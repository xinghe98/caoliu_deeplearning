"""Active model cache for the prediction worker."""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ModelVersion


class ModelManager:
    def __init__(self):
        self._predictor = None
        self._model_id: str | None = None
        self._model_version: str | None = None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def model_version(self) -> str | None:
        return self._model_version

    def load_active(self, session: Session, predictor_factory=None):
        model = session.scalar(select(ModelVersion).where(ModelVersion.status == 'active'))
        if model is None:
            from .default_model import ensure_default_model

            model = ensure_default_model(session)
        if model is None:
            raise RuntimeError('没有已发布的 active 模型，且 DEFAULT_MODEL_PATH 不可用')
        return self._load(model, predictor_factory)

    def load_version(self, session: Session, version: str, predictor_factory=None):
        model = session.scalar(select(ModelVersion).where(ModelVersion.version == version))
        if model is None:
            raise RuntimeError(f'模型版本不存在: {version}')
        return self._load(model, predictor_factory)

    def _load(self, model: ModelVersion, predictor_factory=None):
        model_id = model.id
        if self._predictor is not None and self._model_id == model_id:
            # Keep threshold/temperature in sync with DB without reloading weights.
            if hasattr(self._predictor, 'decision_threshold'):
                self._predictor.decision_threshold = model.decision_threshold
            if hasattr(self._predictor, 'temperature'):
                self._predictor.temperature = model.temperature
            return model, self._predictor

        if not Path(model.checkpoint_path).is_file():
            raise RuntimeError(f'模型文件缺失: {model.checkpoint_path}')
        if predictor_factory is None:
            from predict import Predictor
            predictor_factory = Predictor
        predictor = predictor_factory(model.checkpoint_path)
        if hasattr(predictor, 'decision_threshold'):
            predictor.decision_threshold = model.decision_threshold
        if hasattr(predictor, 'temperature'):
            predictor.temperature = model.temperature
        self._predictor = predictor
        self._model_id = model_id
        self._model_version = model.version
        return model, predictor

    def invalidate(self) -> None:
        self._predictor = None
        self._model_id = None
        self._model_version = None
