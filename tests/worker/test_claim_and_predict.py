from pathlib import Path

from sqlalchemy import select

from platform_app.database import SessionLocal
from platform_app.models import ContentItem, Job, MediaAsset, ModelVersion, Prediction
from platform_app.worker import PredictionWorker
from tests.conftest import make_image


class FakePredictor:
    def __init__(self, _path):
        self.decision_threshold = 0.5
        self.temperature = 1.0

    def predict_from_paths(self, paths, title):
        return {
            'probability': 0.91,
            'prediction': 1,
            'decision_threshold': self.decision_threshold,
        }


def test_worker_predict_from_paths(platform_env):
    image = make_image(platform_env['media_root'] / 'w.jpg')
    with SessionLocal() as session:
        content = ContentItem(
            content_key='btih:2222222222222222222222222222222222222222',
            content_group_id='g1',
            title_clean='worker item',
            magnet_uri='magnet:?xt=urn:btih:2222222222222222222222222222222222222222',
            info_hash='2222222222222222222222222222222222222222',
        )
        session.add(content)
        session.flush()
        session.add(MediaAsset(content_id=content.id, source_path=str(image), ordinal=1, file_size=10))
        ckpt = platform_env['data_dir'] / 'fake.pth'
        ckpt.write_bytes(b'fake')
        model = ModelVersion(version='v-test', status='active', checkpoint_path=str(ckpt), metrics={'pr_auc': 0.5})
        session.add(model)
        job = Job(job_type='predict', payload={'content_id': content.id})
        session.add(job)
        session.commit()
        content_id = content.id

    worker = PredictionWorker(batch_size=8, predictor_factory=FakePredictor)
    assert worker.run_once() is True
    with SessionLocal() as session:
        prediction = session.scalar(select(Prediction).where(Prediction.content_id == content_id))
        assert prediction is not None
        assert prediction.probability == 0.91
        job = session.scalar(select(Job).where(Job.job_type == 'predict'))
        assert job.status == 'succeeded'


def test_only_one_worker_can_claim_a_job(platform_env):
    with SessionLocal() as session:
        job = Job(job_type='predict', payload={'content_id': 'missing'})
        session.add(job)
        session.commit()
    first = PredictionWorker(worker_id='first')
    second = PredictionWorker(worker_id='second')
    with SessionLocal() as session_a, SessionLocal() as session_b:
        claimed_a = first._claim_batch(session_a)
        claimed_b = second._claim_batch(session_b)
    assert len(claimed_a) == 1
    assert claimed_b == []


def test_worker_honors_job_model_version_after_activation_changes(platform_env):
    image = make_image(platform_env['media_root'] / 'versioned.jpg')
    with SessionLocal() as session:
        content = ContentItem(
            content_key='url:versioned-job',
            content_group_id='versioned-job',
            title_clean='versioned worker item',
        )
        session.add(content)
        session.flush()
        session.add(MediaAsset(content_id=content.id, source_path=str(image), ordinal=1, file_size=10))
        old_path = platform_env['data_dir'] / 'old-worker.pth'
        new_path = platform_env['data_dir'] / 'new-worker.pth'
        old_path.write_bytes(b'old')
        new_path.write_bytes(b'new')
        session.add_all([
            ModelVersion(version='worker-old', status='archived', checkpoint_path=str(old_path)),
            ModelVersion(version='worker-new', status='active', checkpoint_path=str(new_path)),
        ])
        session.add(Job(
            job_type='predict',
            payload={'content_id': content.id, 'model_version': 'worker-old'},
        ))
        session.commit()
        content_id = content.id

    worker = PredictionWorker(batch_size=8, predictor_factory=FakePredictor)
    assert worker.run_once() is True
    with SessionLocal() as session:
        prediction = session.scalar(select(Prediction).where(Prediction.content_id == content_id))
        assert prediction.model_version == 'worker-old'
