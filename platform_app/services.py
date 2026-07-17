import hashlib
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import desc, exists, select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .domain.keys import (
    canonical_key,
    content_group_id,
    extract_info_hash_from_magnet,
    normalize_info_hash,
    validate_magnet_uri,
)
from .domain.media import inspect_image, is_under_roots
from .models import (
    ContentItem, IdempotencyRecord, Job, LabelEvent, MediaAsset, ModelVersion,
    Prediction, SnapshotLabelEvent, ViewEvent, utcnow,
)
from .schemas import ContentRead


def validate_media_path(raw_path: str, *, inspect: bool = True) -> tuple[Path, dict | None]:
    path = Path(raw_path).resolve()
    settings = get_settings()
    if not path.is_file():
        raise HTTPException(status_code=422, detail=f'图片不存在: {path}')
    if not is_under_roots(path, settings.media_roots):
        raise HTTPException(status_code=403, detail='图片路径不在允许的媒体根目录内')
    meta = inspect_image(path) if inspect else None
    return path, meta


def active_model_version(session: Session) -> str | None:
    return session.scalar(select(ModelVersion.version).where(ModelVersion.status == 'active'))


def content_read(content: ContentItem, model_version: str | None) -> ContentRead:
    latest_prediction = next(
        (prediction for prediction in content.predictions if prediction.model_version == model_version),
        None,
    ) if model_version else None
    return ContentRead(
        id=content.id,
        content_key=content.content_key,
        title_clean=content.title_clean or content.title_raw,
        source_url=content.source_url,
        magnet_uri=content.magnet_uri,
        status=content.status,
        current_label=content.current_label,
        created_at=content.created_at,
        media=[media for media in sorted(content.media, key=lambda item: item.ordinal)],
        probability=latest_prediction.probability if latest_prediction else None,
        decision_threshold=latest_prediction.decision_threshold if latest_prediction else None,
        model_version=latest_prediction.model_version if latest_prediction else None,
        dataset_role=content.dataset_role,
    )


def content_query():
    return select(ContentItem).options(selectinload(ContentItem.media), selectinload(ContentItem.predictions))


def queue_prediction_job(
    session: Session,
    content: ContentItem,
    *,
    model_version: str | None = None,
) -> Job:
    payload = {'content_id': content.id}
    if model_version is not None:
        payload['model_version'] = model_version
    job = Job(job_type='predict', payload=payload)
    session.add(job)
    return job


def queue_missing_predictions_for_model(session: Session, model_version: str) -> int:
    scored_ids = set(session.scalars(
        select(Prediction.content_id).where(Prediction.model_version == model_version)
    ).all())
    pending_ids = {
        (job.payload or {}).get('content_id')
        for job in session.scalars(
            select(Job).where(Job.job_type == 'predict', Job.status.in_(('pending', 'running')))
        ).all()
        if (job.payload or {}).get('model_version') == model_version
    }
    contents = session.scalars(
        content_query().where(ContentItem.status == 'ready')
    ).all()
    queued = 0
    for content in contents:
        if content.id in scored_ids or content.id in pending_ids or not content.media:
            continue
        queue_prediction_job(session, content, model_version=model_version)
        queued += 1
    return queued


def maybe_queue_training_snapshot(session: Session) -> Job | None:
    settings = get_settings()
    query = select(LabelEvent).where(
        LabelEvent.source == 'explicit_web',
        ~exists(select(SnapshotLabelEvent.event_id).where(SnapshotLabelEvent.event_id == LabelEvent.id)),
    )
    count = len(session.scalars(query).all())
    if count < settings.training_label_threshold:
        return None
    existing = session.scalar(
        select(Job).where(
            Job.job_type == 'export_training_snapshot',
            Job.status.in_(('pending', 'running')),
        )
    )
    if existing:
        return existing
    job = Job(job_type='export_training_snapshot', payload={})
    session.add(job)
    return job


def parse_cursor(cursor: str | None) -> tuple[datetime | None, str | None]:
    if not cursor:
        return None, None
    if cursor.count('|') != 1:
        raise HTTPException(status_code=422, detail='无效 cursor')
    created_at, content_id = cursor.split('|', 1)
    if not created_at or not content_id:
        raise HTTPException(status_code=422, detail='无效 cursor')
    try:
        parsed = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        uuid.UUID(content_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail='无效 cursor') from exc
    return parsed, content_id


def make_cursor(content: ContentItem) -> str:
    created_at = content.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return f'{created_at.isoformat()}|{content.id}'


def parse_score_cursor(cursor: str | None) -> tuple[float | None, str | None]:
    if not cursor:
        return None, None
    if cursor.count('|') != 1:
        raise HTTPException(status_code=422, detail='无效 cursor')
    score_raw, content_id = cursor.split('|', 1)
    if score_raw == '' or not content_id:
        raise HTTPException(status_code=422, detail='无效 cursor')
    try:
        score = float(score_raw)
        uuid.UUID(content_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail='无效 cursor') from exc
    return score, content_id


def make_score_cursor(content_id: str, probability: float | None) -> str:
    score = -1.0 if probability is None else float(probability)
    return f'{score:.10f}|{content_id}'


def _score(item: ContentItem, model_version: str | None) -> float:
    prediction = next(
        (value for value in item.predictions if value.model_version == model_version),
        None,
    ) if model_version else None
    return prediction.probability if prediction else 0.5


def _skipped_content_ids(session: Session) -> set[str]:
    cutoff = utcnow() - timedelta(days=get_settings().skip_cooldown_days)
    rows = session.scalars(
        select(ViewEvent.content_id).where(
            ViewEvent.event_type == 'skip',
            ViewEvent.created_at >= cutoff,
        )
    ).all()
    return set(rows)


def feed_contents(session: Session, limit: int, mode: str) -> list[ContentItem]:
    model_version = active_model_version(session)
    skipped = _skipped_content_ids(session)
    contents = list(session.scalars(
        content_query().where(ContentItem.status == 'ready', ContentItem.current_label.is_(None))
    ).all())
    contents = [item for item in contents if item.id not in skipped]
    if not contents:
        return []

    # One item per content group.
    seen_groups: set[str] = set()
    deduped: list[ContentItem] = []
    for item in sorted(contents, key=lambda value: value.created_at, reverse=True):
        if item.content_group_id in seen_groups:
            continue
        seen_groups.add(item.content_group_id)
        deduped.append(item)
    contents = deduped

    ranked = sorted(contents, key=lambda item: _score(item, model_version), reverse=True)
    if mode == 'score':
        return ranked[:limit]
    if mode == 'newest':
        return sorted(contents, key=lambda value: value.created_at, reverse=True)[:limit]
    if mode == 'random':
        random.shuffle(contents)
        return contents[:limit]
    if mode == 'uncertain':
        return sorted(contents, key=lambda value: abs(_score(value, model_version) - 0.5))[:limit]

    recommended_count = max(1, round(limit * get_settings().feed_recommendation_ratio))
    exploration_count = max(0, limit - recommended_count)
    recommended = ranked[:recommended_count]
    remaining = [item for item in contents if item.id not in {value.id for value in recommended}]
    uncertain = sorted(remaining, key=lambda value: abs(_score(value, model_version) - 0.5))[: exploration_count // 2]
    random_pool = [item for item in remaining if item.id not in {value.id for value in uncertain}]
    random.shuffle(random_pool)
    return (recommended + uncertain + random_pool[: exploration_count - len(uncertain)])[:limit]


def build_content_key_and_group(payload) -> tuple[str, str, str, str]:
    try:
        magnet = validate_magnet_uri(payload.magnet_uri)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    info_hash = normalize_info_hash(payload.info_hash) or extract_info_hash_from_magnet(magnet)
    try:
        key = payload.content_key or canonical_key(payload.source_url, info_hash, magnet)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    group = payload.content_group_id or content_group_id(
        info_hash=info_hash,
        magnet_uri=magnet,
        title=payload.title_clean or payload.title_raw,
        content_key=key,
    )
    return key, group, magnet, info_hash


def get_idempotent_response(
    session: Session,
    user_id: str,
    key: str,
    request_hash: str,
    *,
    legacy_request_hash: str | None = None,
    resource_id: str | None = None,
) -> IdempotencyRecord | None:
    record = session.scalar(
        select(IdempotencyRecord).where(IdempotencyRecord.user_id == user_id, IdempotencyRecord.key == key)
    )
    if record is None:
        return None
    legacy_match = (
        legacy_request_hash is not None
        and record.request_hash == legacy_request_hash
        and resource_id is not None
        and record.response_body.get('id') == resource_id
    )
    if record.request_hash != request_hash and not legacy_match:
        raise HTTPException(status_code=409, detail='Idempotency-Key 已用于不同请求')
    return record


def save_idempotent_response(
    session: Session,
    user_id: str,
    key: str,
    request_hash: str,
    status_code: int,
    response_body: dict,
) -> IdempotencyRecord:
    record = IdempotencyRecord(
        user_id=user_id,
        key=key,
        request_hash=request_hash,
        status_code=status_code,
        response_body=response_body,
    )
    session.add(record)
    return record
