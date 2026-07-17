"""Create reproducible, self-contained training ZIP archives."""

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import exists, select
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
            'media': valid_media,
        })
    return rows


def create_snapshot(session: Session) -> TrainingSnapshot:
    if session.get_bind().dialect.name != 'sqlite':
        raise RuntimeError('训练快照目前仅支持 SQLite，以保证标签边界一致性')
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
    snapshot = TrainingSnapshot(label_cutoff_at=cutoff)
    session.add(snapshot)
    session.flush()
    rows = _manifest_rows(session, include_external=False)
    positive_count = sum(row['label'] == 1 for row in rows)
    negative_count = sum(row['label'] == 0 for row in rows)
    if positive_count == 0 or negative_count == 0:
        raise HTTPException(status_code=422, detail='训练包至少需要一条喜欢和一条不喜欢标签')

    manifest_rows = []
    for row in rows:
        images = []
        for media in row['media']:
            suffix = Path(media.source_path).suffix.lower() or '.jpg'
            images.append(f"images/{row['content_id']}/image_{media.ordinal:02d}{suffix}")
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

    snapshot.sample_count = len(rows)
    snapshot.positive_count = positive_count
    snapshot.negative_count = negative_count
    snapshot.manifest_hash = manifest_hash
    session.add_all([
        SnapshotLabelEvent(event_id=event_id, snapshot_id=snapshot.id)
        for event_id in included_event_ids
    ])
    output_dir = get_settings().platform_data_dir / 'training_snapshots'
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f'training_snapshot_{snapshot.id}_{manifest_hash[:12]}.zip'
    split_summary = {
        name: sum(row['split'] == name for row in manifest_rows)
        for name in ('train', 'validation', 'production_shadow_test', 'external_test')
    }
    sha_entries = {
        'manifest.csv': hashlib.sha256(manifest_bytes).hexdigest(),
        'split_manifest.csv': hashlib.sha256(split_bytes).hexdigest(),
    }
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('manifest.csv', manifest_bytes)
        archive.writestr('split_manifest.csv', split_bytes)
        config_bytes = json.dumps({
            'snapshot_id': snapshot.id,
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
                source = Path(media.source_path)
                suffix = source.suffix.lower() or '.jpg'
                arcname = f"images/{row['content_id']}/image_{media.ordinal:02d}{suffix}"
                archive.write(source, arcname)
                sha_entries[arcname] = hashlib.sha256(source.read_bytes()).hexdigest()
        archive.writestr('SHA256SUMS.json', json.dumps(sha_entries, indent=2, ensure_ascii=False))
    snapshot.archive_path = str(archive_path)
    session.commit()
    return snapshot
