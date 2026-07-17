import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ContentItem(Base):
    __tablename__ = 'content_items'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    content_group_id: Mapped[str] = mapped_column(String(128), index=True)
    source: Mapped[str] = mapped_column(String(64), default='crawler')
    source_url: Mapped[str] = mapped_column(Text, default='')
    title_raw: Mapped[str] = mapped_column(Text, default='')
    title_clean: Mapped[str] = mapped_column(Text, default='')
    magnet_uri: Mapped[str] = mapped_column(Text, default='')
    info_hash: Mapped[str] = mapped_column(String(40), default='', index=True)
    status: Mapped[str] = mapped_column(String(24), default='ready', index=True)
    dataset_role: Mapped[str] = mapped_column(String(32), default='production', index=True)
    current_label: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    media: Mapped[list['MediaAsset']] = relationship(back_populates='content', cascade='all, delete-orphan')
    labels: Mapped[list['LabelEvent']] = relationship(back_populates='content', cascade='all, delete-orphan')
    predictions: Mapped[list['Prediction']] = relationship(back_populates='content', cascade='all, delete-orphan')


class User(Base):
    __tablename__ = 'users'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuthSession(Base):
    __tablename__ = 'auth_sessions'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MediaAsset(Base):
    __tablename__ = 'media_assets'
    __table_args__ = (UniqueConstraint('content_id', 'ordinal', name='uq_media_content_ordinal'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content_id: Mapped[str] = mapped_column(ForeignKey('content_items.id', ondelete='CASCADE'), index=True)
    source_path: Mapped[str] = mapped_column(Text)
    ordinal: Mapped[int] = mapped_column(Integer)
    mime_type: Mapped[str] = mapped_column(String(64), default='image/jpeg')
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), default='')
    status: Mapped[str] = mapped_column(String(24), default='ready')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    content: Mapped[ContentItem] = relationship(back_populates='media')


class LabelEvent(Base):
    __tablename__ = 'label_events'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content_id: Mapped[str] = mapped_column(ForeignKey('content_items.id', ondelete='CASCADE'), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    label: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), default='explicit_web')
    supersedes_event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    probability_at_label: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    content: Mapped[ContentItem] = relationship(back_populates='labels')


class ViewEvent(Base):
    __tablename__ = 'view_events'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content_id: Mapped[str] = mapped_column(ForeignKey('content_items.id', ondelete='CASCADE'), index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prediction(Base):
    __tablename__ = 'predictions'
    __table_args__ = (UniqueConstraint('content_id', 'model_version', name='uq_prediction_content_model'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content_id: Mapped[str] = mapped_column(ForeignKey('content_items.id', ondelete='CASCADE'), index=True)
    model_version: Mapped[str] = mapped_column(String(64), default='unassigned')
    probability: Mapped[float] = mapped_column(Float)
    decision_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    prediction: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    content: Mapped[ContentItem] = relationship(back_populates='predictions')


class ModelVersion(Base):
    __tablename__ = 'model_versions'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(24), default='candidate', index=True)
    checkpoint_path: Mapped[str] = mapped_column(Text)
    decision_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    temperature: Mapped[float] = mapped_column(Float, default=1.0)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    data_manifest_hash: Mapped[str] = mapped_column(String(64), default='')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrainingSnapshot(Base):
    __tablename__ = 'training_snapshots'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(24), default='ready', index=True)
    label_cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    positive_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, default=0)
    manifest_hash: Mapped[str] = mapped_column(String(64), default='')
    archive_path: Mapped[str] = mapped_column(Text, default='')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SnapshotLabelEvent(Base):
    __tablename__ = 'snapshot_label_events'

    event_id: Mapped[str] = mapped_column(
        ForeignKey('label_events.id', ondelete='CASCADE'), primary_key=True
    )
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey('training_snapshots.id', ondelete='CASCADE'), index=True
    )


class Job(Base):
    __tablename__ = 'jobs'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(24), default='pending', index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default='')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class IdempotencyRecord(Base):
    __tablename__ = 'idempotency_records'
    __table_args__ = (UniqueConstraint('user_id', 'key', name='uq_idempotency_user_key'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    status_code: Mapped[int] = mapped_column(Integer)
    response_body: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WorkerHeartbeat(Base):
    __tablename__ = 'worker_heartbeats'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    worker_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    model_version: Mapped[str] = mapped_column(String(64), default='')
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
