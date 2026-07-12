import csv
import json
import zipfile
from pathlib import Path

import pytest
from dataset import load_training_package
from pack_candidate import pack_candidate
from tests.conftest import make_image


def test_load_training_package_stable_split(tmp_path):
    root = tmp_path / 'pkg'
    images = root / 'images' / 'c1'
    make_image(images / 'image_01.jpg')
    make_image(root / 'images' / 'c2' / 'image_01.jpg')
    with (root / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'content_id', 'content_group_id', 'title', 'label', 'split', 'dataset_role',
            'download_link', 'source_url', 'image_paths',
        ])
        writer.writeheader()
        writer.writerow({
            'content_id': 'c1', 'content_group_id': 'g1', 'title': 'a', 'label': 1,
            'split': 'train', 'dataset_role': 'production', 'download_link': '',
            'source_url': '', 'image_paths': 'images/c1/image_01.jpg',
        })
        writer.writerow({
            'content_id': 'c2', 'content_group_id': 'g2', 'title': 'b', 'label': 0,
            'split': 'validation', 'dataset_role': 'production', 'download_link': '',
            'source_url': '', 'image_paths': 'images/c2/image_01.jpg',
        })
    with (root / 'split_manifest.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['content_id', 'content_group_id', 'split', 'label'])
        writer.writeheader()
        writer.writerow({'content_id': 'c1', 'content_group_id': 'g1', 'split': 'train', 'label': 1})
        writer.writerow({'content_id': 'c2', 'content_group_id': 'g2', 'split': 'validation', 'label': 0})
    zip_path = tmp_path / 'pkg.zip'
    with zipfile.ZipFile(zip_path, 'w') as archive:
        for path in root.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    df1 = load_training_package(zip_path)
    df2 = load_training_package(zip_path)
    assert sorted(df1['split'].tolist()) == sorted(df2['split'].tolist())
    assert set(df1['split']) == {'train', 'validation'}
    extracted_root = Path(df1.root)
    df1.close()
    df2.close()
    assert not extracted_root.exists()


@pytest.mark.parametrize('member', ['../escape.txt', '/escape.txt'])
def test_training_package_rejects_unsafe_zip_paths(tmp_path, member):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as package:
        package.writestr(member, 'bad')
    with pytest.raises(ValueError, match='非法路径'):
        load_training_package(archive)


def test_training_package_rejects_invalid_manifest(tmp_path):
    root = tmp_path / 'invalid'
    root.mkdir()
    (root / 'manifest.csv').write_text(
        'content_id,label,split,image_paths\na,2,unknown,missing.jpg\na,0,train,missing.jpg\n',
        encoding='utf-8-sig',
    )
    with pytest.raises(ValueError, match='content_id'):
        load_training_package(root)


def test_pack_candidate(tmp_path):
    import json
    import torch
    src = tmp_path / 'out'
    src.mkdir()
    torch.save({'model_state_dict': {'a': torch.tensor(1.0)}, 'decision_threshold': 0.5}, src / 'best_model.pth')
    (src / 'evaluation_report.json').write_text(json.dumps({'validation': {'pr_auc': 0.8}}), encoding='utf-8')
    out = tmp_path / 'candidate_run1.zip'
    pack_candidate(src, out, run_id='run1')
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
    assert 'best_model.pth' in names
    assert 'model_manifest.json' in names
    assert 'SHA256SUMS.json' in names


def test_pack_candidate_replaces_existing_manifest_once(tmp_path):
    import torch
    src = tmp_path / 'out'
    src.mkdir()
    torch.save({'model_state_dict': {'a': torch.tensor(1.0)}}, src / 'best_model.pth')
    (src / 'evaluation_report.json').write_text(json.dumps({'validation': {'pr_auc': 0.8}}), encoding='utf-8')
    (src / 'model_manifest.json').write_text(json.dumps({'custom': 'keep', 'version': 'old'}), encoding='utf-8')
    out = tmp_path / 'candidate.zip'
    pack_candidate(src, out, run_id='new')
    with zipfile.ZipFile(out) as archive:
        assert archive.namelist().count('model_manifest.json') == 1
        manifest = json.loads(archive.read('model_manifest.json'))
    assert manifest['custom'] == 'keep'
    assert manifest['version'] == 'new'
