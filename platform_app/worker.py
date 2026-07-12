"""Database-backed prediction worker for the local preference platform."""

import argparse
import socket
import time
from datetime import timedelta
from pathlib import Path

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import selectinload
from tqdm import tqdm

from .database import SessionLocal, configure_engine
from .model_manager import ModelManager
from .models import ContentItem, Job, Prediction, WorkerHeartbeat, utcnow
from .training import create_snapshot


class PredictionWorker:
    def __init__(self, worker_id: str | None = None, batch_size: int = 16, predictor_factory=None):
        self.worker_id = worker_id or socket.gethostname()
        self.batch_size = batch_size
        self.predictor_factory = predictor_factory
        self.manager = ModelManager()

    def _heartbeat(self, session, model_version: str = '') -> None:
        row = session.scalar(select(WorkerHeartbeat).where(WorkerHeartbeat.worker_id == self.worker_id))
        if row is None:
            session.add(WorkerHeartbeat(worker_id=self.worker_id, model_version=model_version, last_seen_at=utcnow()))
        else:
            row.model_version = model_version
            row.last_seen_at = utcnow()
        session.commit()

    def _claim_batch(self, session, job_type: str = 'predict') -> list[Job]:
        now = utcnow()
        candidate_ids = list(session.scalars(
            select(Job.id)
            .where(
                Job.job_type == job_type,
                or_(Job.status == 'pending', (Job.status == 'running') & (Job.lease_expires_at < now)),
            )
            # Newest first so library (created_at desc) fills scores first.
            .order_by(Job.created_at.desc())
            .limit(self.batch_size)
        ).all())
        jobs = []
        for job_id in candidate_ids:
            claimed = session.execute(
                update(Job)
                .where(
                    Job.id == job_id,
                    Job.job_type == job_type,
                    or_(Job.status == 'pending', and_(Job.status == 'running', Job.lease_expires_at < now)),
                )
                .values(
                    status='running',
                    locked_by=self.worker_id,
                    lease_expires_at=now + timedelta(minutes=5),
                    attempts=Job.attempts + 1,
                )
            )
            if claimed.rowcount:
                jobs.append(session.get(Job, job_id))
        if jobs:
            session.commit()
        return jobs

    def _complete_job(self, session, job: Job, *, status: str, error: str = '', payload: dict | None = None) -> bool:
        values = {
            'status': status,
            'locked_by': None,
            'lease_expires_at': None,
            'last_error': error,
            'finished_at': utcnow() if status == 'succeeded' else None,
        }
        if payload is not None:
            values['payload'] = payload
        result = session.execute(
            update(Job)
            .where(Job.id == job.id, Job.status == 'running', Job.locked_by == self.worker_id)
            .values(**values)
        )
        return bool(result.rowcount)

    def _save_prediction(self, session, content: ContentItem, model, result: dict) -> None:
        existing = session.scalar(select(Prediction).where(
            Prediction.content_id == content.id,
            Prediction.model_version == model.version,
        ))
        if existing is None:
            session.add(Prediction(
                content_id=content.id,
                model_version=model.version,
                probability=result['probability'],
                decision_threshold=result['decision_threshold'],
                prediction=result['prediction'],
            ))
        else:
            existing.probability = result['probability']
            existing.decision_threshold = result['decision_threshold']
            existing.prediction = result['prediction']

    def _predict_queue_size(self, session) -> int:
        return int(session.scalar(
            select(func.count()).select_from(Job).where(
                Job.job_type == 'predict',
                Job.status.in_(('pending', 'running')),
            )
        ) or 0)

    def process_predict_jobs(self, session, jobs: list[Job]) -> None:
        model, predictor = self.manager.load_active(session, predictor_factory=self.predictor_factory)
        self._heartbeat(session, model.version)
        queue_left = self._predict_queue_size(session)
        ok = 0
        fail = 0
        progress = tqdm(
            jobs,
            desc=f'预测 {model.version}',
            unit='条',
            leave=True,
            dynamic_ncols=True,
        )
        for job in progress:
            try:
                content = session.scalar(
                    select(ContentItem)
                    .options(selectinload(ContentItem.media))
                    .where(ContentItem.id == job.payload['content_id'])
                )
                if content is None or not content.media:
                    raise RuntimeError('内容不存在或没有可用图片')
                paths = [Path(media.source_path) for media in sorted(content.media, key=lambda item: item.ordinal)]
                title = content.title_clean or content.title_raw
                if hasattr(predictor, 'predict_from_paths'):
                    result = predictor.predict_from_paths(paths, title)
                else:
                    result = predictor.predict(str(paths[0].parent), title)
                self._save_prediction(session, content, model, result)
                if not self._complete_job(session, job, status='succeeded'):
                    session.rollback()
                    continue
                ok += 1
                queue_left = max(0, queue_left - 1)
                progress.set_postfix(ok=ok, fail=fail, p=f'{result["probability"]:.3f}', left=queue_left)
            except Exception as exc:
                session.rollback()
                attempts = session.scalar(select(Job.attempts).where(Job.id == job.id)) or 0
                self._complete_job(
                    session,
                    job,
                    status='failed' if attempts >= 3 else 'pending',
                    error=str(exc),
                )
                fail += 1
                queue_left = max(0, queue_left - 1)
                progress.set_postfix(ok=ok, fail=fail, err=str(exc)[:24], left=queue_left)
            session.commit()
        progress.close()

    def process_export_jobs(self, session, jobs: list[Job]) -> None:
        progress = tqdm(jobs, desc='导出训练包', unit='个', leave=True, dynamic_ncols=True)
        for job in progress:
            try:
                snapshot = create_snapshot(session)
                if not self._complete_job(
                    session,
                    job,
                    status='succeeded',
                    payload={**(job.payload or {}), 'snapshot_id': snapshot.id},
                ):
                    session.rollback()
                    continue
                progress.set_postfix(status='ok')
            except Exception as exc:
                session.rollback()
                attempts = session.scalar(select(Job.attempts).where(Job.id == job.id)) or 0
                self._complete_job(
                    session,
                    job,
                    status='failed' if attempts >= 3 else 'pending',
                    error=str(exc),
                )
                progress.set_postfix(status='fail', err=str(exc)[:24])
            session.commit()
        progress.close()

    def run_once(self) -> bool:
        with SessionLocal() as session:
            export_jobs = self._claim_batch(session, 'export_training_snapshot')
            if export_jobs:
                self.process_export_jobs(session, export_jobs)
                return True
            jobs = self._claim_batch(session, 'predict')
            if not jobs:
                self._heartbeat(session)
                return False
            self.process_predict_jobs(session, jobs)
            return True


def main():
    parser = argparse.ArgumentParser(description='个人偏好平台预测 worker')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=5.0)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    configure_engine()
    from .database import SessionLocal
    from .default_model import ensure_default_model

    with SessionLocal() as session:
        model = ensure_default_model(session)
        if model is not None:
            print(f'active 模型: {model.version} -> {model.checkpoint_path}')
        else:
            print('警告: 无 active 模型，且 DEFAULT_MODEL_PATH 不可用')
    worker = PredictionWorker(batch_size=args.batch_size)
    while True:
        processed = worker.run_once()
        if args.once:
            return
        if not processed:
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
