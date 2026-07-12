"""Import historical dataset folders into the platform database."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from platform_app.config import clear_settings_cache, get_settings
from platform_app.database import Base, SessionLocal, configure_engine, get_engine
from platform_app.domain.keys import canonical_key, content_group_id, extract_info_hash_from_magnet, normalize_info_hash
from platform_app.domain.labels import apply_label
from platform_app.domain.media import inspect_image
from platform_app.models import ContentItem, MediaAsset


def discover_image_paths(folder: Path) -> list[Path]:
    paths = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp'):
        paths.extend(folder.glob(pattern))
    return sorted(paths)[:5]


def scan_dataset(root: Path, folder_name: str) -> list[dict]:
    csv_path = root / folder_name / 'index.csv'
    if not csv_path.is_file():
        return []
    rows = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = (row.get('video_id') or '').strip()
            title = (row.get('title') or '').strip()
            download_link = (row.get('download_link') or '').strip()
            label_raw = row.get('label')
            label = None
            if label_raw not in (None, ''):
                try:
                    label = int(float(label_raw))
                except ValueError:
                    label = 'invalid'
            media_dir = root / folder_name / video_id
            images = discover_image_paths(media_dir) if media_dir.is_dir() else []
            info_hash = extract_info_hash_from_magnet(download_link)
            try:
                key = canonical_key(source_url=f'legacy://{folder_name}/{video_id}', info_hash=info_hash, magnet_uri=download_link)
            except ValueError:
                key = f'legacy:{folder_name}:{video_id}'
            group = content_group_id(info_hash=info_hash, magnet_uri=download_link, download_link=download_link, title=title, content_key=key)
            issue = None
            if not images:
                issue = 'missing_images'
            elif label == 'invalid':
                issue = 'invalid_label'
            rows.append({
                'folder': folder_name,
                'video_id': video_id,
                'title': title,
                'download_link': download_link,
                'label': None if label == 'invalid' else label,
                'info_hash': info_hash,
                'content_key': key,
                'content_group_id': group,
                'images': images,
                'issue': issue,
                'dataset_role': 'external_test' if folder_name == '数据集3' else 'historical',
            })
    return rows


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['folder', 'video_id', 'content_key', 'label', 'issue', 'image_count', 'title']
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'folder': row['folder'],
                'video_id': row['video_id'],
                'content_key': row['content_key'],
                'label': row['label'],
                'issue': row['issue'] or '',
                'image_count': len(row['images']),
                'title': row['title'][:200],
            })


def apply_rows(session: Session, rows: list[dict]) -> dict:
    created = 0
    labeled = 0
    skipped = 0
    for row in rows:
        if row['issue'] == 'missing_images':
            skipped += 1
            continue
        existing = session.scalar(select(ContentItem).where(ContentItem.content_key == row['content_key']))
        if existing:
            skipped += 1
            continue
        content = ContentItem(
            content_key=row['content_key'],
            content_group_id=row['content_group_id'],
            source='historical_import',
            source_url=f"legacy://{row['folder']}/{row['video_id']}",
            title_raw=row['title'],
            title_clean=row['title'],
            magnet_uri=row['download_link'],
            info_hash=normalize_info_hash(row['info_hash']),
            dataset_role=row['dataset_role'],
            status='ready',
        )
        session.add(content)
        session.flush()
        for ordinal, image_path in enumerate(row['images'], start=1):
            try:
                meta = inspect_image(image_path)
            except Exception:
                meta = {
                    'mime_type': 'image/jpeg',
                    'file_size': image_path.stat().st_size,
                    'width': None,
                    'height': None,
                    'sha256': '',
                }
            session.add(MediaAsset(
                content_id=content.id,
                source_path=str(image_path.resolve()),
                ordinal=ordinal,
                mime_type=meta['mime_type'],
                file_size=meta['file_size'],
                width=meta['width'],
                height=meta['height'],
                sha256=meta['sha256'],
            ))
        if row['label'] in (0, 1):
            apply_label(session, content, row['label'], source='historical_import')
            labeled += 1
        from platform_app.services import queue_prediction_job

        queue_prediction_job(session, content)
        created += 1
    session.commit()
    return {'created': created, 'labeled': labeled, 'skipped': skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='迁移遗留数据集到平台数据库')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--folders', nargs='*', default=['数据集1', '数据集2', '数据集3', '数据集4', '数据集5', 'downloads'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.dry_run and not args.apply:
        args.dry_run = True

    clear_settings_cache()
    configure_engine()
    Base.metadata.create_all(bind=get_engine())

    all_rows: list[dict] = []
    for folder in args.folders:
        all_rows.extend(scan_dataset(args.root, folder))

    report_path = args.report or (get_settings().platform_data_dir / 'migration_report.csv')
    write_report(report_path, all_rows)
    print(f'扫描 {len(all_rows)} 条，报告: {report_path}')
    print(f"缺图: {sum(1 for row in all_rows if row['issue'] == 'missing_images')}")
    print(f"无效标签: {sum(1 for row in all_rows if row['issue'] == 'invalid_label')}")
    print(f"有标签: {sum(1 for row in all_rows if row['label'] in (0, 1))}")

    if args.apply:
        with SessionLocal() as session:
            stats = apply_rows(session, all_rows)
        print(f"已导入: {stats}")
        with SessionLocal() as session:
            stats2 = apply_rows(session, all_rows)
        print(f"幂等复跑: {stats2}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
