from pathlib import Path

from platform_app.database import SessionLocal
from platform_app.migrate_legacy import apply_rows, scan_dataset, write_report
from tests.conftest import make_image


def test_migrate_legacy_dry_run_and_idempotent_apply(platform_env, tmp_path):
    root = tmp_path / 'datasets'
    d1 = root / '数据集1' / 'video_01'
    d3 = root / '数据集3' / 'video_01'
    make_image(d1 / 'image_01.jpg')
    make_image(d3 / 'image_01.jpg')
    (root / '数据集1' / 'index.csv').write_text(
        'video_id,title,label,download_link\n'
        'video_01,pos,1,magnet:?xt=urn:btih:ffffffffffffffffffffffffffffffffffffffff\n',
        encoding='utf-8-sig',
    )
    (root / '数据集3' / 'index.csv').write_text(
        'video_id,title,label,download_link\n'
        'video_01,ext,0,magnet:?xt=urn:btih:1111111111111111111111111111111111111111\n',
        encoding='utf-8-sig',
    )
    rows = scan_dataset(root, '数据集1') + scan_dataset(root, '数据集3')
    assert len(rows) == 2
    assert any(row['dataset_role'] == 'external_test' for row in rows)
    report = tmp_path / 'report.csv'
    write_report(report, rows)
    assert report.is_file()
    with SessionLocal() as session:
        stats1 = apply_rows(session, rows)
        stats2 = apply_rows(session, rows)
    assert stats1['created'] == 2
    assert stats2['created'] == 0
