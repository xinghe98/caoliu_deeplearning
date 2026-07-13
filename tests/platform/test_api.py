import io
import json
import zipfile
from pathlib import Path

import torch

from platform_app.database import SessionLocal
from platform_app.models import Job
from tests.conftest import make_image


def test_auth_and_protected_feed(client):
    denied = client.get('/api/v1/feed')
    assert denied.status_code == 401
    setup = client.post('/api/v1/auth/setup', json={
        'username': 'owner',
        'password': 'change-this-to-a-long-password',
    })
    assert setup.status_code == 201
    session = client.get('/api/v1/auth/session')
    assert session.status_code == 200
    assert session.json()['username'] == 'owner'


def test_ingest_duplicate_and_bad_path(auth_client, platform_env):
    image = make_image(platform_env['media_root'] / 'a.jpg')
    payload = {
        'content_key': 'btih:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'source_url': 'http://example.com/1',
        'title_raw': 't1',
        'title_clean': 't1',
        'magnet_uri': 'magnet:?xt=urn:btih:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'info_hash': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'media': [{'source_path': str(image), 'ordinal': 1}],
    }
    first = auth_client.post(
        '/api/v1/ingest/content',
        json=payload,
        headers={'X-Ingest-Key': 'test-ingest-key'},
    )
    assert first.status_code == 201
    body = first.json()
    assert body['created'] is True
    assert body['duplicate'] is False
    assert body['prediction_job_id']
    second = auth_client.post(
        '/api/v1/ingest/content',
        json=payload,
        headers={'X-Ingest-Key': 'test-ingest-key'},
    )
    assert second.status_code == 201
    assert second.json()['duplicate'] is True
    bad = auth_client.post(
        '/api/v1/ingest/content',
        json={
            **payload,
            'content_key': 'btih:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            'info_hash': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            'magnet_uri': 'magnet:?xt=urn:btih:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            'media': [{'source_path': str(Path('C:/Windows/System32/drivers/etc/hosts')), 'ordinal': 1}],
        },
        headers={'X-Ingest-Key': 'test-ingest-key'},
    )
    assert bad.status_code in {403, 422}


def test_label_history_undo_idempotency_and_feed(auth_client, platform_env):
    image = make_image(platform_env['media_root'] / 'b.jpg')
    ingested = auth_client.post(
        '/api/v1/ingest/content',
        json={
            'content_key': 'btih:cccccccccccccccccccccccccccccccccccccccc',
            'title_clean': 'item',
            'magnet_uri': 'magnet:?xt=urn:btih:cccccccccccccccccccccccccccccccccccccccc',
            'info_hash': 'cccccccccccccccccccccccccccccccccccccccc',
            'media': [{'source_path': str(image), 'ordinal': 1}],
        },
        headers={'X-Ingest-Key': 'test-ingest-key'},
    ).json()
    content_id = ingested['content_id']
    feed = auth_client.get('/api/v1/feed')
    assert any(item['id'] == content_id for item in feed.json())
    labeled = auth_client.post(
        f'/api/v1/contents/{content_id}/label',
        json={'label': 1},
        headers={'Idempotency-Key': 'label-1'},
    )
    assert labeled.status_code == 200
    label_event_id = labeled.json()['label_event_id']
    again = auth_client.post(
        f'/api/v1/contents/{content_id}/label',
        json={'label': 1},
        headers={'Idempotency-Key': 'label-1'},
    )
    assert again.status_code == 200
    assert again.json()['label_event_id'] == label_event_id
    history = auth_client.get(f'/api/v1/labels/history?content_id={content_id}')
    assert len(history.json()) == 1
    event_id = history.json()[0]['id']
    assert label_event_id == event_id
    feed_after = auth_client.get('/api/v1/feed')
    assert all(item['id'] != content_id for item in feed_after.json())
    undo = auth_client.post(f'/api/v1/labels/{event_id}/undo')
    assert undo.status_code == 200
    assert undo.json()['current_label'] is None


def test_training_snapshot_and_candidate_import(auth_client, platform_env):
    pos = make_image(platform_env['media_root'] / 'pos.jpg', color=(200, 20, 20))
    neg = make_image(platform_env['media_root'] / 'neg.jpg', color=(20, 20, 200))
    for key, image, label in [
        ('dddddddddddddddddddddddddddddddddddddddd', pos, 1),
        ('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', neg, 0),
    ]:
        content_id = auth_client.post(
            '/api/v1/ingest/content',
            json={
                'content_key': f'btih:{key}',
                'title_clean': key,
                'magnet_uri': f'magnet:?xt=urn:btih:{key}',
                'info_hash': key,
                'media': [{'source_path': str(image), 'ordinal': 1}],
            },
            headers={'X-Ingest-Key': 'test-ingest-key'},
        ).json()['content_id']
        auth_client.post(f'/api/v1/contents/{content_id}/label', json={'label': label})
    with SessionLocal() as session:
        queued_job = Job(job_type='export_training_snapshot', status='pending', payload={'marker': 'other-worker'})
        session.add(queued_job)
        session.commit()
        queued_job_id = queued_job.id
    snapshot = auth_client.post('/api/v1/training/snapshots')
    assert snapshot.status_code == 201
    with SessionLocal() as session:
        assert session.get(Job, queued_job_id).status == 'pending'
    snapshot_id = snapshot.json()['id']
    download = auth_client.get(f'/api/v1/training/snapshots/{snapshot_id}/download')
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        names = set(archive.namelist())
        assert 'manifest.csv' in names
        assert 'split_manifest.csv' in names
        assert 'SHA256SUMS.json' in names

    # candidate package
    cand_dir = platform_env['data_dir'] / 'cand_src'
    cand_dir.mkdir()
    ckpt = {
        'model_state_dict': {'x': torch.tensor([1.0])},
        'decision_threshold': 0.6,
        'temperature': 1.2,
        'data_manifest_hash': 'abc',
    }
    torch.save(ckpt, cand_dir / 'best_model.pth')
    (cand_dir / 'evaluation_report.json').write_text(
        json.dumps({'validation': {'pr_auc': 0.9, 'precision': 0.8, 'recall': 0.7}}),
        encoding='utf-8',
    )
    zip_path = platform_env['data_dir'] / 'candidate.zip'
    with zipfile.ZipFile(zip_path, 'w') as archive:
        archive.write(cand_dir / 'best_model.pth', 'best_model.pth')
        archive.write(cand_dir / 'evaluation_report.json', 'evaluation_report.json')
    with zip_path.open('rb') as handle:
        imported = auth_client.post(
            '/api/v1/training/candidates/import',
            files={'archive': ('candidate.zip', handle, 'application/zip')},
            data={'version': 'cand-1'},
        )
    assert imported.status_code == 201
    model_id = imported.json()['id']
    comparison = auth_client.get(f'/api/v1/training/candidates/{model_id}/comparison')
    assert comparison.status_code == 200


def test_candidate_path_traversal_rejected(auth_client, platform_env):
    zip_path = platform_env['data_dir'] / 'evil.zip'
    with zipfile.ZipFile(zip_path, 'w') as archive:
        archive.writestr('../evil.pth', b'nope')
        archive.writestr('best_model.pth', b'nope')
        archive.writestr('evaluation_report.json', b'{}')
    with zip_path.open('rb') as handle:
        response = auth_client.post(
            '/api/v1/training/candidates/import',
            files={'archive': ('evil.zip', handle, 'application/zip')},
        )
    assert response.status_code == 422


def test_invalid_content_cursor_returns_422(auth_client):
    for cursor in ('missing-separator', '|item', 'not-a-date|item'):
        response = auth_client.get('/api/v1/contents', params={'cursor': cursor})
        assert response.status_code == 422


def test_contents_cursor_pagination(auth_client, platform_env):
    for index in range(5):
        image = make_image(platform_env['media_root'] / f'page_{index}.jpg')
        response = auth_client.post(
            '/api/v1/ingest/content',
            headers={'X-Ingest-Key': 'test-ingest-key'},
            json={
                'content_key': f'url:page-{index}',
                'source_url': f'https://example.com/page-{index}',
                'title_raw': f'page {index}',
                'title_clean': f'page {index}',
                'media': [{'source_path': str(image), 'ordinal': 1}],
            },
        )
        assert response.status_code == 201

    first = auth_client.get('/api/v1/contents', params={'limit': 2})
    assert first.status_code == 200
    body = first.json()
    assert len(body['items']) == 2
    assert body['next_cursor']

    second = auth_client.get('/api/v1/contents', params={'limit': 2, 'cursor': body['next_cursor']})
    assert second.status_code == 200
    body2 = second.json()
    assert len(body2['items']) == 2
    ids = {item['id'] for item in body['items']} | {item['id'] for item in body2['items']}
    assert len(ids) == 4
