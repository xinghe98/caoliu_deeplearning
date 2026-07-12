"""Enqueue predict jobs for contents missing a score under the active model."""

from __future__ import annotations

import argparse

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from platform_app.config import clear_settings_cache, get_settings
from platform_app.database import SessionLocal, configure_engine
from platform_app.default_model import ensure_default_model
from platform_app.models import ContentItem, Job, ModelVersion, Prediction
from platform_app.services import queue_prediction_job


def requeue(*, only_missing: bool = True, limit: int | None = None) -> dict[str, int]:
    clear_settings_cache()
    configure_engine()
    with SessionLocal() as session:
        active = ensure_default_model(session) or session.scalar(
            select(ModelVersion).where(ModelVersion.status == 'active')
        )
        if active is None:
            raise SystemExit('没有 active 模型，请配置 DEFAULT_MODEL_PATH 或先发布模型')

        contents = list(session.scalars(
            select(ContentItem)
            .options(selectinload(ContentItem.media))
            .where(ContentItem.status == 'ready')
            .order_by(ContentItem.created_at.desc())
        ).all())

        pending_jobs = session.scalars(
            select(Job).where(Job.job_type == 'predict', Job.status.in_(('pending', 'running')))
        ).all()
        job_ids = {
            (job.payload or {}).get('content_id')
            for job in pending_jobs
            if (job.payload or {}).get('content_id')
        }
        scored = set()
        if only_missing:
            scored = set(session.scalars(
                select(Prediction.content_id).where(Prediction.model_version == active.version)
            ).all())

        queued = 0
        skipped = 0
        for content in contents:
            if only_missing and content.id in scored:
                skipped += 1
                continue
            if content.id in job_ids:
                skipped += 1
                continue
            if not content.media:
                skipped += 1
                continue
            queue_prediction_job(session, content)
            queued += 1
            if limit is not None and queued >= limit:
                break
        session.commit()
        pending = session.scalar(
            select(func.count()).select_from(Job).where(
                Job.job_type == 'predict', Job.status == 'pending'
            )
        ) or 0
        return {
            'queued': queued,
            'skipped': skipped,
            'pending_jobs': int(pending),
            'model': active.version,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='为缺少预测分的内容补建 predict 任务')
    parser.add_argument('--all', action='store_true', help='忽略已有分数，全部重排队')
    parser.add_argument('--limit', type=int, default=None, help='最多新建多少任务')
    args = parser.parse_args(argv)
    stats = requeue(only_missing=not args.all, limit=args.limit)
    print(
        f"model={stats['model']} queued={stats['queued']} "
        f"skipped={stats['skipped']} pending_jobs={stats['pending_jobs']}"
    )
    print('请确保 worker 在运行: python -m platform_app.worker')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
