from contextlib import asynccontextmanager
from pathlib import Path
import secrets

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import exists, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import (
    COOKIE_NAME,
    check_login_rate_limit,
    clear_login_failures,
    clear_session_cookies,
    create_session,
    enforce_csrf,
    password_hash,
    record_login_failure,
    request_body_hash,
    require_user,
)
from .candidates import import_candidate
from .config import get_settings
from .database import configure_engine, get_session
from .domain.labels import LabelConflictError, apply_label, undo_label
from .domain.search import parse_search_tokens
from .models import (
    AuthSession,
    ContentItem,
    Job,
    LabelEvent,
    MediaAsset,
    ModelVersion,
    Prediction,
    SnapshotLabelEvent,
    TrainingSnapshot,
    User,
    ViewEvent,
    WorkerHeartbeat,
    utcnow,
)
from .schemas import (
    CandidateComparison,
    ContentPage,
    ContentRead,
    EventCreate,
    IngestContent,
    IngestResult,
    JobRead,
    JobStatsRead,
    LabelCreate,
    LabelEventRead,
    LabelResultRead,
    LoginRequest,
    ModelRead,
    ModelRegister,
    SessionRead,
    SetupAdmin,
    TrainingSnapshotRead,
    TrainingStatusRead,
    WatchedUpdate,
)
from .services import (
    build_content_key_and_group,
    active_model_version,
    content_query,
    content_read,
    feed_contents,
    get_idempotent_response,
    labeled_at_map,
    make_cursor,
    make_score_cursor,
    make_time_cursor,
    maybe_queue_training_snapshot,
    parse_cursor,
    parse_score_cursor,
    queue_prediction_job,
    queue_missing_predictions_for_model,
    save_idempotent_response,
    set_content_watched,
    validate_media_path,
    watched_content_ids,
    watched_exists_clause,
)
from .training import create_snapshot


def _run_startup_migrations() -> None:
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[1]
    command.upgrade(Config(str(root / 'alembic.ini')), 'head')


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_engine()
    if get_settings().auto_create_tables:
        _run_startup_migrations()
    from .database import SessionLocal
    from .default_model import ensure_default_model

    with SessionLocal() as session:
        ensure_default_model(session)
    yield


app = FastAPI(title='个人偏好平台 API', version='0.2.0', lifespan=lifespan)


def get_content_or_404(session: Session, content_id: str) -> ContentItem:
    content = session.scalar(content_query().where(ContentItem.id == content_id))
    if content is None:
        raise HTTPException(status_code=404, detail='内容不存在')
    return content


@app.post('/api/v1/auth/setup', response_model=SessionRead, status_code=201)
def setup_admin(payload: SetupAdmin, response: Response, session: Session = Depends(get_session)):
    if session.scalar(select(User.id).limit(1)) is not None:
        raise HTTPException(status_code=409, detail='管理员账户已初始化')
    user = User(username=payload.username, password_hash=password_hash.hash(payload.password))
    session.add(user)
    session.flush()
    create_session(session, user, response)
    return SessionRead(username=user.username)


@app.post('/api/v1/auth/login', response_model=SessionRead)
def login(payload: LoginRequest, response: Response, session: Session = Depends(get_session)):
    check_login_rate_limit(payload.username)
    user = session.scalar(select(User).where(User.username == payload.username))
    if user is None or not user.is_active or not password_hash.verify(payload.password, user.password_hash):
        record_login_failure(payload.username)
        raise HTTPException(status_code=401, detail='用户名或密码错误')
    clear_login_failures(payload.username)
    create_session(session, user, response)
    return SessionRead(username=user.username)


@app.post('/api/v1/auth/logout', status_code=204)
def logout(
    response: Response,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
    _: None = Depends(enforce_csrf),
):
    sessions = session.scalars(select(AuthSession).where(AuthSession.user_id == user.id)).all()
    for auth_session in sessions:
        session.delete(auth_session)
    session.commit()
    clear_session_cookies(response)
    return None


@app.get('/api/v1/auth/session', response_model=SessionRead)
def auth_session(user: User = Depends(require_user)):
    return SessionRead(username=user.username)


@app.get('/health/live')
def health_live():
    return {'status': 'ok'}


@app.get('/health/ready')
def health_ready(session: Session = Depends(get_session)):
    session.execute(select(ContentItem.id).limit(1))
    return {'status': 'ready'}


@app.get('/health/worker')
def health_worker(session: Session = Depends(get_session)):
    rows = session.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.last_seen_at.desc()).limit(5)).all()
    return {
        'workers': [
            {
                'worker_id': row.worker_id,
                'model_version': row.model_version,
                'last_seen_at': row.last_seen_at.isoformat(),
            }
            for row in rows
        ]
    }


@app.post('/api/v1/ingest/content', response_model=IngestResult, status_code=201)
def ingest_content(
    payload: IngestContent,
    x_ingest_key: str = Header(default='', alias='X-Ingest-Key'),
    session: Session = Depends(get_session),
):
    settings = get_settings()
    if not settings.ingest_api_key:
        raise HTTPException(status_code=503, detail='爬虫入库密钥尚未配置')
    if not secrets.compare_digest(x_ingest_key, settings.ingest_api_key):
        raise HTTPException(status_code=401, detail='无效的爬虫入库密钥')
    content_key, group_id, magnet, info_hash = build_content_key_and_group(payload)
    existing = session.scalar(content_query().where(ContentItem.content_key == content_key))
    if existing:
        existing.title_raw = payload.title_raw or existing.title_raw
        existing.title_clean = payload.title_clean or existing.title_clean
        existing.source_url = payload.source_url or existing.source_url
        existing.magnet_uri = magnet or existing.magnet_uri
        existing.info_hash = info_hash or existing.info_hash
        session.commit()
        return IngestResult(
            content_id=existing.id,
            created=False,
            duplicate=True,
            prediction_job_id=None,
            content=content_read(get_content_or_404(session, existing.id), active_model_version(session)),
        )

    inspected = []
    for input_item in payload.media:
        path, meta = validate_media_path(input_item.source_path, inspect=True)
        inspected.append((input_item, path, meta))

    content = ContentItem(
        content_key=content_key,
        content_group_id=group_id,
        source=payload.source,
        source_url=payload.source_url,
        title_raw=payload.title_raw,
        title_clean=payload.title_clean,
        magnet_uri=magnet,
        info_hash=info_hash,
    )
    session.add(content)
    session.flush()
    for input_item, path, meta in inspected:
        session.add(MediaAsset(
            content_id=content.id,
            source_path=str(path),
            ordinal=input_item.ordinal,
            mime_type=meta['mime_type'] if meta else 'image/jpeg',
            file_size=meta['file_size'] if meta else path.stat().st_size,
            width=meta['width'] if meta else None,
            height=meta['height'] if meta else None,
            sha256=meta['sha256'] if meta else '',
        ))
    job = queue_prediction_job(session, content)
    session.commit()
    refreshed = get_content_or_404(session, content.id)
    return IngestResult(
        content_id=refreshed.id,
        created=True,
        duplicate=False,
        prediction_job_id=job.id,
        content=content_read(refreshed, active_model_version(session)),
    )


@app.get('/api/v1/contents', response_model=ContentPage)
def list_contents(
    label: int | None = None,
    unlabeled: bool = False,
    watched: bool = False,
    status: str | None = None,
    q: str | None = None,
    limit: int = 24,
    cursor: str | None = None,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
):
    limit = min(max(limit, 1), 100)
    # 喜欢/不喜欢：最近标注优先；全部：模型分；未标注/已看过：入库时间
    sort_by_label_time = label in (0, 1) and not unlabeled and not watched
    sort_by_score = not unlabeled and not watched and not sort_by_label_time
    model_version = active_model_version(session)
    score_expr = None
    time_expr = ContentItem.created_at

    if sort_by_score:
        latest_prob = (
            select(Prediction.probability)
            .where(Prediction.content_id == ContentItem.id)
            .where(Prediction.model_version == model_version)
            .order_by(Prediction.created_at.desc())
            .limit(1)
            .scalar_subquery()
        )
        score_expr = func.coalesce(latest_prob, -1.0)
        query = content_query().order_by(score_expr.desc(), ContentItem.id.desc())
    elif sort_by_label_time:
        latest_label_at = (
            select(func.max(LabelEvent.created_at))
            .where(LabelEvent.content_id == ContentItem.id)
            .where(LabelEvent.label == label)
            .where(LabelEvent.source != 'undo')
            .scalar_subquery()
        )
        time_expr = func.coalesce(latest_label_at, ContentItem.updated_at, ContentItem.created_at)
        query = content_query().order_by(time_expr.desc(), ContentItem.id.desc())
    else:
        query = content_query().order_by(ContentItem.created_at.desc(), ContentItem.id.desc())

    if watched:
        # Permanent archive: any content with a watched event.
        query = query.where(watched_exists_clause())
    elif unlabeled:
        query = query.where(ContentItem.current_label.is_(None))
        query = query.where(~watched_exists_clause())
    elif label in (0, 1):
        query = query.where(ContentItem.current_label == label)
        query = query.where(~watched_exists_clause())
    if status:
        query = query.where(ContentItem.status == status)

    for token in parse_search_tokens(q):
        pattern = f'%{token}%'
        query = query.where(
            or_(
                func.normalize_title(ContentItem.title_clean).like(pattern),
                func.normalize_title(ContentItem.title_raw).like(pattern),
            )
        )

    if cursor:
        if sort_by_score:
            score, content_id = parse_score_cursor(cursor)
            query = query.where(
                (score_expr < score)
                | ((score_expr == score) & (ContentItem.id < content_id))
            )
        else:
            moment, content_id = parse_cursor(cursor)
            query = query.where(
                (time_expr < moment)
                | ((time_expr == moment) & (ContentItem.id < content_id))
            )

    rows = list(session.scalars(query.limit(limit + 1)).all())
    has_more = len(rows) > limit
    page = rows[:limit]
    watched_ids = watched_content_ids(session, [item.id for item in page])
    labeled_times = labeled_at_map(session, page)
    items = [
        content_read(
            item,
            model_version,
            is_watched=item.id in watched_ids,
            labeled_at=labeled_times.get(item.id),
        )
        for item in page
    ]
    if has_more and page:
        if sort_by_score:
            next_cursor = make_score_cursor(page[-1].id, items[-1].probability)
        elif sort_by_label_time:
            last = page[-1]
            labeled_at = session.scalar(
                select(func.max(LabelEvent.created_at)).where(
                    LabelEvent.content_id == last.id,
                    LabelEvent.label == label,
                    LabelEvent.source != 'undo',
                )
            )
            moment = labeled_at or last.updated_at or last.created_at
            next_cursor = make_time_cursor(moment, last.id)
        else:
            next_cursor = make_cursor(page[-1])
    else:
        next_cursor = None
    return ContentPage(items=items, next_cursor=next_cursor)


@app.get('/api/v1/contents/{content_id}', response_model=ContentRead)
def get_content(content_id: str, session: Session = Depends(get_session), _user: User = Depends(require_user)):
    content = get_content_or_404(session, content_id)
    return content_read(
        content,
        active_model_version(session),
        is_watched=content.id in watched_content_ids(session, [content.id]),
        labeled_at=labeled_at_map(session, [content]).get(content.id),
    )


@app.get('/api/v1/contents/{content_id}/media/{media_id}')
def get_media(content_id: str, media_id: str, session: Session = Depends(get_session), _user: User = Depends(require_user)):
    media = session.scalar(select(MediaAsset).where(MediaAsset.id == media_id, MediaAsset.content_id == content_id))
    if media is None:
        raise HTTPException(status_code=404, detail='图片不存在')
    path = Path(media.source_path)
    if not path.is_file():
        media.status = 'missing'
        session.commit()
        raise HTTPException(status_code=410, detail='原始图片已缺失')
    # Prefer real file type over possibly wrong DB mime (legacy import hardcodes image/jpeg).
    suffix = path.suffix.lower()
    mime_by_suffix = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    media_type = mime_by_suffix.get(suffix) or media.mime_type or 'application/octet-stream'
    # Named files may be mislabeled (.gif that is actually JPEG); sniff magic bytes.
    try:
        with path.open('rb') as source:
            header = source.read(16)
        if header.startswith(b'\xff\xd8\xff'):
            media_type = 'image/jpeg'
        elif header.startswith(b'\x89PNG\r\n\x1a\n'):
            media_type = 'image/png'
        elif header.startswith((b'GIF87a', b'GIF89a')):
            media_type = 'image/gif'
        elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':
            media_type = 'image/webp'
    except OSError:
        pass
    return FileResponse(path, media_type=media_type)


@app.get('/api/v1/feed', response_model=list[ContentRead])
def get_feed(
    mode: str = 'mixed',
    limit: int = 20,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
):
    if mode not in {'mixed', 'score', 'uncertain', 'newest', 'random'}:
        raise HTTPException(status_code=422, detail='不支持的推荐模式')
    limit = min(max(limit, 1), 50)
    model_version = active_model_version(session)
    rows = feed_contents(session, limit, mode)
    watched_ids = watched_content_ids(session, [item.id for item in rows])
    labeled_times = labeled_at_map(session, rows)
    return [
        content_read(
            item,
            model_version,
            is_watched=item.id in watched_ids,
            labeled_at=labeled_times.get(item.id),
        )
        for item in rows
    ]


@app.post('/api/v1/contents/{content_id}/label', response_model=LabelResultRead)
def label_content(
    content_id: str,
    payload: LabelCreate,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
    idempotency_key: str | None = Header(default=None, alias='Idempotency-Key'),
):
    request_hash = request_body_hash({
        'method': 'POST',
        'path': f'/api/v1/contents/{content_id}/label',
        'body': payload.model_dump(),
    })
    legacy_request_hash = request_body_hash(payload.model_dump())
    if idempotency_key:
        existing = get_idempotent_response(
            session,
            user.id,
            idempotency_key,
            request_hash,
            legacy_request_hash=legacy_request_hash,
            resource_id=content_id,
        )
        if existing:
            return LabelResultRead.model_validate(existing.response_body)
    content = get_content_or_404(session, content_id)
    try:
        event = apply_label(
            session,
            content,
            payload.label,
            user_id=user.id,
            model_version=payload.model_version,
            probability_at_label=payload.probability_at_label,
        )
    except LabelConflictError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    maybe_queue_training_snapshot(session)
    session.flush()
    model_version = active_model_version(session)
    labeled = get_content_or_404(session, content_id)
    result = LabelResultRead(
        **content_read(
            labeled,
            model_version,
            labeled_at=event.created_at,
        ).model_dump(),
        label_event_id=event.id,
    )
    if idempotency_key:
        save_idempotent_response(session, user.id, idempotency_key, request_hash, 200, result.model_dump(mode='json'))
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        if not idempotency_key:
            raise
        existing = get_idempotent_response(
            session,
            user.id,
            idempotency_key,
            request_hash,
            legacy_request_hash=legacy_request_hash,
            resource_id=content_id,
        )
        if existing is None:
            raise
        return LabelResultRead.model_validate(existing.response_body)
    return result


@app.put('/api/v1/contents/{content_id}/watched', response_model=ContentRead)
def update_watched(
    content_id: str,
    payload: WatchedUpdate,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    content = get_content_or_404(session, content_id)
    is_watched = set_content_watched(session, content, payload.watched)
    session.commit()
    refreshed = get_content_or_404(session, content_id)
    return content_read(
        refreshed,
        active_model_version(session),
        is_watched=is_watched,
        labeled_at=labeled_at_map(session, [refreshed]).get(refreshed.id),
    )


@app.post('/api/v1/contents/{content_id}/events', status_code=204)
def content_event(
    content_id: str,
    payload: EventCreate,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    content = get_content_or_404(session, content_id)
    # Prefer PUT /watched for toggles; keep event path as one-way mark without clearing labels.
    if payload.event_type == 'watched':
        set_content_watched(session, content, True)
    else:
        session.add(ViewEvent(content_id=content_id, event_type=payload.event_type))
    session.commit()
    return Response(status_code=204)


@app.get('/api/v1/labels/history', response_model=list[LabelEventRead])
def label_history(
    content_id: str | None = None,
    limit: int = 100,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
):
    limit = min(max(limit, 1), 500)
    query = select(LabelEvent).order_by(LabelEvent.created_at.desc()).limit(limit)
    if content_id:
        query = query.where(LabelEvent.content_id == content_id)
    return list(session.scalars(query).all())


@app.post('/api/v1/labels/{event_id}/undo', response_model=ContentRead)
def undo_label_event(
    event_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    event = session.get(LabelEvent, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail='标签事件不存在')
    try:
        content = undo_label(session, event, user_id=user.id)
    except LabelConflictError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    restored = get_content_or_404(session, content.id)
    return content_read(
        restored,
        active_model_version(session),
        labeled_at=labeled_at_map(session, [restored]).get(restored.id),
    )


@app.get('/api/v1/jobs', response_model=list[JobRead])
def list_jobs(limit: int = 50, session: Session = Depends(get_session), _user: User = Depends(require_user)):
    limit = min(max(limit, 1), 100)
    return session.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit)).all()


@app.get('/api/v1/jobs/stats', response_model=JobStatsRead)
def job_stats(session: Session = Depends(get_session), _user: User = Depends(require_user)):
    def count_jobs(status: str) -> int:
        return int(session.scalar(
            select(func.count()).select_from(Job).where(Job.job_type == 'predict', Job.status == status)
        ) or 0)

    active = session.scalar(select(ModelVersion).where(ModelVersion.status == 'active'))
    return JobStatsRead(
        contents_total=int(session.scalar(select(func.count()).select_from(ContentItem)) or 0),
        predictions_total=int(session.scalar(select(func.count()).select_from(Prediction)) or 0),
        scored_contents=int(session.scalar(select(func.count(func.distinct(Prediction.content_id)))) or 0),
        predict_pending=count_jobs('pending'),
        predict_running=count_jobs('running'),
        predict_succeeded=count_jobs('succeeded'),
        predict_failed=count_jobs('failed'),
        active_model_version=active.version if active else None,
    )


@app.get('/api/v1/models', response_model=list[ModelRead])
def list_models(session: Session = Depends(get_session), _user: User = Depends(require_user)):
    return session.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()


@app.post('/api/v1/models', response_model=ModelRead, status_code=201)
def register_model(
    payload: ModelRegister,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    checkpoint = Path(payload.checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise HTTPException(status_code=422, detail='模型文件不存在')
    if session.scalar(select(ModelVersion).where(ModelVersion.version == payload.version)):
        raise HTTPException(status_code=409, detail='模型版本已存在')
    model = ModelVersion(
        version=payload.version,
        checkpoint_path=str(checkpoint),
        decision_threshold=payload.decision_threshold,
        temperature=payload.temperature,
        metrics=payload.metrics,
        data_manifest_hash=payload.data_manifest_hash,
    )
    session.add(model)
    session.commit()
    return model


@app.post('/api/v1/models/{model_id}/activate', response_model=ModelRead)
def activate_model(
    model_id: str,
    force: bool = False,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    model = session.get(ModelVersion, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail='模型不存在')
    if not Path(model.checkpoint_path).is_file():
        raise HTTPException(status_code=422, detail='模型文件已缺失')
    active = session.scalar(select(ModelVersion).where(ModelVersion.status == 'active'))
    if active and isinstance(model.metrics, dict) and isinstance(active.metrics, dict):
        cand = model.metrics.get('pr_auc')
        act = active.metrics.get('pr_auc')
        if isinstance(cand, (int, float)) and isinstance(act, (int, float)) and cand < act - 0.02 and not force:
            raise HTTPException(status_code=409, detail='候选 PR-AUC 低于 active 超过 0.02，请使用 force=true 覆盖')
    for row in session.scalars(select(ModelVersion).where(ModelVersion.status == 'active')).all():
        row.status = 'archived'
    model.status = 'active'
    model.activated_at = utcnow()
    queue_missing_predictions_for_model(session, model.version)
    session.commit()
    return model


@app.post('/api/v1/models/{model_id}/reject', response_model=ModelRead)
def reject_model(
    model_id: str,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    model = session.get(ModelVersion, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail='模型不存在')
    if model.status == 'active':
        raise HTTPException(status_code=409, detail='不能直接拒绝 active 模型，请先回滚')
    model.status = 'rejected'
    session.commit()
    return model


@app.post('/api/v1/models/{model_id}/rollback', response_model=ModelRead)
def rollback_model(
    model_id: str,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    model = session.get(ModelVersion, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail='模型不存在')
    if not Path(model.checkpoint_path).is_file():
        raise HTTPException(status_code=422, detail='模型文件已缺失')
    for row in session.scalars(select(ModelVersion).where(ModelVersion.status == 'active')).all():
        row.status = 'archived'
    model.status = 'active'
    model.activated_at = utcnow()
    queue_missing_predictions_for_model(session, model.version)
    session.commit()
    return model


@app.get('/api/v1/training/status', response_model=TrainingStatusRead)
def training_status(session: Session = Depends(get_session), _user: User = Depends(require_user)):
    latest = session.scalar(select(TrainingSnapshot).order_by(TrainingSnapshot.created_at.desc()))
    label_query = select(LabelEvent).where(
        LabelEvent.source == 'explicit_web',
        ~exists(
            select(SnapshotLabelEvent.event_id)
            .where(SnapshotLabelEvent.event_id == LabelEvent.id)
        ),
    )
    count = len(session.scalars(label_query).all())
    threshold = get_settings().training_label_threshold
    return TrainingStatusRead(
        labels_since_last_snapshot=count,
        threshold=threshold,
        ready_for_snapshot=count >= threshold,
        latest_snapshot=latest,
    )


@app.get('/api/v1/training/snapshots', response_model=list[TrainingSnapshotRead])
def list_training_snapshots(session: Session = Depends(get_session), _user: User = Depends(require_user)):
    return session.scalars(select(TrainingSnapshot).order_by(TrainingSnapshot.created_at.desc())).all()


@app.post('/api/v1/training/snapshots', response_model=TrainingSnapshotRead, status_code=201)
def build_training_snapshot(
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    return create_snapshot(session)


@app.get('/api/v1/training/snapshots/{snapshot_id}/download')
def download_training_snapshot(
    snapshot_id: str,
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
):
    snapshot = session.get(TrainingSnapshot, snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail='训练快照不存在')
    path = Path(snapshot.archive_path)
    if not path.is_file():
        raise HTTPException(status_code=410, detail='训练包文件已缺失')
    return FileResponse(path, filename=path.name, media_type='application/zip')


@app.post('/api/v1/training/candidates/import', response_model=ModelRead, status_code=201)
async def import_training_candidate(
    archive: UploadFile = File(...),
    version: str | None = Form(default=None),
    session: Session = Depends(get_session),
    _user: User = Depends(require_user),
    _: None = Depends(enforce_csrf),
):
    return await import_candidate(session, archive, version)


@app.get('/api/v1/training/candidates/{model_id}/comparison', response_model=CandidateComparison)
def compare_candidate(model_id: str, session: Session = Depends(get_session), _user: User = Depends(require_user)):
    candidate = session.get(ModelVersion, model_id)
    if candidate is None or candidate.status != 'candidate':
        raise HTTPException(status_code=404, detail='候选模型不存在')
    active = session.scalar(select(ModelVersion).where(ModelVersion.status == 'active'))
    warnings = []
    hard_blocks = []
    if not Path(candidate.checkpoint_path).is_file():
        hard_blocks.append('候选 checkpoint 文件缺失')
    candidate_pr_auc = candidate.metrics.get('pr_auc') if isinstance(candidate.metrics, dict) else None
    active_pr_auc = active.metrics.get('pr_auc') if active and isinstance(active.metrics, dict) else None
    if isinstance(candidate_pr_auc, (int, float)) and isinstance(active_pr_auc, (int, float)) and candidate_pr_auc < active_pr_auc - 0.02:
        warnings.append('候选模型的 PR-AUC 比当前模型低超过 0.02')
    if candidate.data_manifest_hash == '':
        warnings.append('候选包未提供数据 manifest 哈希')
    return CandidateComparison(candidate=candidate, active=active, warnings=warnings, hard_blocks=hard_blocks)


FRONTEND_DIST = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'


def frontend_file(full_path: str) -> Path | None:
    root = FRONTEND_DIST.resolve()
    candidate = (root / full_path).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate if candidate.is_file() else None


if FRONTEND_DIST.is_dir():
    app.mount('/assets', StaticFiles(directory=FRONTEND_DIST / 'assets'), name='frontend-assets')

    @app.get('/')
    def spa_index():
        return FileResponse(FRONTEND_DIST / 'index.html')

    @app.get('/{full_path:path}')
    def spa_fallback(full_path: str, request: Request):
        if full_path.startswith('api/') or full_path.startswith('health'):
            raise HTTPException(status_code=404, detail='Not Found')
        candidate = frontend_file(full_path)
        if candidate is None and '..' in Path(full_path).parts:
            raise HTTPException(status_code=404, detail='Not Found')
        if candidate is not None:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / 'index.html')
