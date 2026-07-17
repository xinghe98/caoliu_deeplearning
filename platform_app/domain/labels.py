from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ContentItem, LabelEvent, User


def latest_label_event(session: Session, content_id: str) -> LabelEvent | None:
    return session.scalar(
        select(LabelEvent)
        .where(LabelEvent.content_id == content_id)
        .order_by(LabelEvent.created_at.desc(), LabelEvent.id.desc())
    )


def apply_label(
    session: Session,
    content: ContentItem,
    label: int,
    *,
    user_id: str | None = None,
    source: str = 'explicit_web',
    model_version: str | None = None,
    probability_at_label: float | None = None,
) -> LabelEvent:
    previous = latest_label_event(session, content.id)
    event = LabelEvent(
        content_id=content.id,
        user_id=user_id,
        label=label,
        source=source,
        supersedes_event_id=previous.id if previous else None,
        model_version=model_version,
        probability_at_label=probability_at_label,
    )
    session.add(event)
    content.current_label = label
    return event


def undo_label(session: Session, event: LabelEvent, *, user_id: str | None = None) -> ContentItem:
    content = session.get(ContentItem, event.content_id)
    if content is None:
        raise ValueError('内容不存在')
    latest = latest_label_event(session, content.id)
    if latest is None or latest.id != event.id:
        raise ValueError('只能撤销最新标签事件')
    if event.source == 'undo':
        raise ValueError('撤销事件不能再次撤销')
    restored_label = None
    if event.supersedes_event_id:
        previous = session.get(LabelEvent, event.supersedes_event_id)
        restored_label = previous.label if previous and previous.label in (0, 1) else None
    content.current_label = restored_label
    session.add(LabelEvent(
        content_id=content.id,
        user_id=user_id,
        # -1 is an audit-only sentinel for restoring the unlabeled state. It is
        # never copied to ContentItem.current_label or training manifests.
        label=restored_label if restored_label is not None else -1,
        source='undo',
        supersedes_event_id=event.id,
        model_version=event.model_version,
        probability_at_label=event.probability_at_label,
    ))
    return content
