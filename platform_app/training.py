"""Create reproducible, self-contained training ZIP archives."""

import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .domain.splits import stable_split
from .models import ContentItem, LabelEvent, SnapshotLabelEvent, TrainingSnapshot, utcnow


def _manifest_rows(session: Session, include_external: bool = False):
    contents = session.scalars(
        select(ContentItem)
        .options(selectinload(ContentItem.media))
        .where(ContentItem.current_label.is_not(None), ContentItem.status == 'ready')
        .order_by(ContentItem.id)
    ).all()
    rows = []
    for content in contents:
        if content.dataset_role == 'external_test' and not include_external:
            # External test stays locked outside gradient packages by default.
            continue
        valid_media = [
            media for media in content.media
            if Path(media.source_path).is_file() and media.status == 'ready'
        ]
        if not valid_media:
            continue
        rows.append({
            'content_id': content.id,
            'content_group_id': content.content_group_id,
            'title': content.title_clean or content.title_raw,
            'label': content.current_label,
            'split': stable_split(content.content_group_id, content.dataset_role),
            'dataset_role': content.dataset_role,
            'download_link': content.magnet_uri,
            'source_url': content.source_url,
            'media': [
                {
                    'source_path': media.source_path,
                    'ordinal': media.ordinal,
                }
                for media in valid_media
            ],
        })
    return rows


def _build_manifest_payload(rows: list[dict]) -> tuple[bytes, bytes, str, dict]:
    manifest_rows = []
    for row in rows:
        images = []
        for media in row['media']:
            suffix = Path(media['source_path']).suffix.lower() or '.jpg'
            images.append(f"images/{row['content_id']}/image_{media['ordinal']:02d}{suffix}")
        manifest_rows.append(
            {key: value for key, value in row.items() if key != 'media'}
            | {'image_paths': ';'.join(images)}
        )
    fields = [
        'content_id', 'content_group_id', 'title', 'label', 'split',
        'dataset_role', 'download_link', 'source_url', 'image_paths',
    ]
    manifest_text = io.StringIO(newline='')
    writer = csv.DictWriter(manifest_text, fieldnames=fields)
    writer.writeheader()
    writer.writerows(manifest_rows)
    manifest_bytes = manifest_text.getvalue().encode('utf-8-sig')
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    split_manifest_text = io.StringIO(newline='')
    split_writer = csv.DictWriter(
        split_manifest_text,
        fieldnames=['content_id', 'content_group_id', 'split', 'label'],
    )
    split_writer.writeheader()
    for row in manifest_rows:
        split_writer.writerow({
            'content_id': row['content_id'],
            'content_group_id': row['content_group_id'],
            'split': row['split'],
            'label': row['label'],
        })
    split_bytes = split_manifest_text.getvalue().encode('utf-8-sig')
    split_summary = {
        name: sum(row['split'] == name for row in manifest_rows)
        for name in ('train', 'validation', 'production_shadow_test', 'external_test')
    }
    return manifest_bytes, split_bytes, manifest_hash, split_summary


def _write_snapshot_archive(
    *,
    archive_path: Path,
    snapshot_id: str,
    rows: list[dict],
    manifest_bytes: bytes,
    split_bytes: bytes,
    manifest_hash: str,
    split_summary: dict,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_name(archive_path.name + '.tmp')
    sha_entries = {
        'manifest.csv': hashlib.sha256(manifest_bytes).hexdigest(),
        'split_manifest.csv': hashlib.sha256(split_bytes).hexdigest(),
    }
    try:
        with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr('manifest.csv', manifest_bytes)
            archive.writestr('split_manifest.csv', split_bytes)
            config_bytes = json.dumps({
                'snapshot_id': snapshot_id,
                'manifest_hash': manifest_hash,
                'split_summary': split_summary,
                'image_root': 'images',
            }, ensure_ascii=False, indent=2).encode('utf-8')
            archive.writestr('config.json', config_bytes)
            sha_entries['config.json'] = hashlib.sha256(config_bytes).hexdigest()
            readme = (
                '# 训练快照\n\n'
                '使用 `manifest.csv` 与 `split_manifest.csv` 固定数据切分；不得在远端重新随机划分。\n'
                '`external_test` 与 `production_shadow_test` 不得进入训练梯度。\n'
            ).encode('utf-8')
            archive.writestr('README_TRAINING.md', readme)
            sha_entries['README_TRAINING.md'] = hashlib.sha256(readme).hexdigest()
            for row in rows:
                for media in row['media']:
                    source = Path(media['source_path'])
                    suffix = source.suffix.lower() or '.jpg'
                    arcname = f"images/{row['content_id']}/image_{media['ordinal']:02d}{suffix}"
                    archive.write(source, arcname)
                    sha_entries[arcname] = hashlib.sha256(source.read_bytes()).hexdigest()
            archive.writestr('SHA256SUMS.json', json.dumps(sha_entries, indent=2, ensure_ascii=False))
        os.replace(temp_path, archive_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _abandon_snapshot(session: Session, snapshot_id: str) -> None:
    """Release claimed events so a failed package can be rebuilt."""
    session.execute(
        delete(SnapshotLabelEvent).where(SnapshotLabelEvent.snapshot_id == snapshot_id)
    )
    snapshot = session.get(TrainingSnapshot, snapshot_id)
    if snapshot is not None:
        session.delete(snapshot)
    session.commit()


def create_snapshot(session: Session) -> TrainingSnapshot:
    if session.get_bind().dialect.name != 'sqlite':
        raise RuntimeError('训练快照目前仅支持 SQLite，以保证标签边界一致性')

    # Phase 1: short write lock — freeze label boundary and claim membership.
    session.connection().exec_driver_sql('BEGIN IMMEDIATE')
    cutoff = utcnow()
    included_event_ids = list(session.scalars(
        select(LabelEvent.id).where(
            ~exists(
                select(SnapshotLabelEvent.event_id)
                .where(SnapshotLabelEvent.event_id == LabelEvent.id)
            )
        )
    ).all())
    rows = _manifest_rows(session, include_external=False)
    positive_count = sum(row['label'] == 1 for row in rows)
    negative_count = sum(row['label'] == 0 for row in rows)
    if positive_count == 0 or negative_count == 0:
        raise HTTPException(status_code=422, detail='训练包至少需要一条喜欢和一条不喜欢标签')

    manifest_bytes, split_bytes, manifest_hash, split_summary = _build_manifest_payload(rows)
    snapshot = TrainingSnapshot(
        status='building',
        label_cutoff_at=cutoff,
        sample_count=len(rows),
        positive_count=positive_count,
        negative_count=negative_count,
        manifest_hash=manifest_hash,
    )
    session.add(snapshot)
    session.flush()
    session.add_all([
        SnapshotLabelEvent(event_id=event_id, snapshot_id=snapshot.id)
        for event_id in included_event_ids
    ])
    snapshot_id = snapshot.id
    session.commit()

    output_dir = get_settings().platform_data_dir / 'training_snapshots'
    archive_path = output_dir / f'training_snapshot_{snapshot_id}_{manifest_hash[:12]}.zip'

    # Phase 2: heavy ZIP I/O outside the SQLite write lock.
    try:
        _write_snapshot_archive(
            archive_path=archive_path,
            snapshot_id=snapshot_id,
            rows=rows,
            manifest_bytes=manifest_bytes,
            split_bytes=split_bytes,
            manifest_hash=manifest_hash,
            split_summary=split_summary,
        )
    except Exception:
        session.rollback()
        _abandon_snapshot(session, snapshot_id)
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        raise

    # Phase 3: publish archive path under a short write transaction.
    session.connection().exec_driver_sql('BEGIN IMMEDIATE')
    snapshot = session.get(TrainingSnapshot, snapshot_id)
    if snapshot is None:
        raise RuntimeError('训练快照在打包期间丢失')
    snapshot.archive_path = str(archive_path)
    snapshot.status = 'ready'
    session.commit()
    session.refresh(snapshot)
    return snapshot
