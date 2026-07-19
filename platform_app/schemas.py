from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MediaInput(BaseModel):
    source_path: str
    ordinal: int = Field(ge=1, le=5)


class IngestContent(BaseModel):
    content_key: str | None = Field(default=None, min_length=4, max_length=128)
    content_group_id: str | None = None
    source: str = 'crawler'
    source_url: str = ''
    title_raw: str = ''
    title_clean: str = ''
    magnet_uri: str = ''
    info_hash: str = ''
    media: list[MediaInput] = Field(min_length=1, max_length=5)


class MediaRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    ordinal: int
    mime_type: str
    width: int | None
    height: int | None
    status: str


class ContentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    content_key: str
    title_clean: str
    source_url: str
    magnet_uri: str
    status: str
    current_label: int | None
    is_watched: bool = False
    created_at: datetime
    labeled_at: datetime | None = None
    media: list[MediaRead] = []
    probability: float | None = None
    decision_threshold: float | None = None
    model_version: str | None = None
    dataset_role: str = 'production'


class IngestResult(BaseModel):
    content_id: str
    created: bool
    duplicate: bool
    prediction_job_id: str | None = None
    content: ContentRead


class LabelCreate(BaseModel):
    label: Literal[0, 1]
    model_version: str | None = None
    probability_at_label: float | None = Field(default=None, ge=0, le=1)


class LabelResultRead(ContentRead):
    label_event_id: str


class LabelEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    content_id: str
    user_id: str | None
    label: int
    source: str
    supersedes_event_id: str | None
    model_version: str | None
    probability_at_label: float | None
    created_at: datetime


class EventCreate(BaseModel):
    event_type: Literal['view', 'skip', 'watched', 'open_source', 'copy_magnet', 'open_magnet']


class WatchedUpdate(BaseModel):
    watched: bool


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    job_type: str
    status: str
    attempts: int
    last_error: str
    created_at: datetime


class JobStatsRead(BaseModel):
    contents_total: int
    predictions_total: int
    scored_contents: int
    predict_pending: int
    predict_running: int
    predict_succeeded: int
    predict_failed: int
    active_model_version: str | None = None


class ModelRegister(BaseModel):
    version: str = Field(min_length=1, max_length=64)
    checkpoint_path: str
    decision_threshold: float = Field(default=0.5, ge=0, le=1)
    temperature: float = Field(default=1.0, gt=0, le=20)
    metrics: dict = Field(default_factory=dict)
    data_manifest_hash: str = ''


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    version: str
    status: str
    checkpoint_path: str
    decision_threshold: float
    temperature: float
    metrics: dict
    data_manifest_hash: str = ''
    created_at: datetime
    activated_at: datetime | None


class SetupAdmin(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    password: str = Field(min_length=12, max_length=256)


class LoginRequest(BaseModel):
    username: str
    password: str


class SessionRead(BaseModel):
    username: str


class TrainingSnapshotRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    status: str
    label_cutoff_at: datetime
    sample_count: int
    positive_count: int
    negative_count: int
    manifest_hash: str
    created_at: datetime


class TrainingStatusRead(BaseModel):
    labels_since_last_snapshot: int
    threshold: int
    ready_for_snapshot: bool
    latest_snapshot: TrainingSnapshotRead | None


class CandidateComparison(BaseModel):
    candidate: ModelRead
    active: ModelRead | None
    warnings: list[str]
    hard_blocks: list[str] = []


class CursorPage(BaseModel):
    items: list[Any]
    next_cursor: str | None = None


class ContentPage(BaseModel):
    items: list[ContentRead]
    next_cursor: str | None = None
