import io
import json
import threading
import zipfile
from pathlib import Path

import torch
from sqlalchemy import delete, select

from platform_app.database import SessionLocal
from platform_app.domain.labels import LabelConflictError, apply_label
from platform_app.models import ContentItem, Job, LabelEvent, ModelVersion, Prediction
from platform_app.training import create_snapshot
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
    with SessionLocal() as session:
        events = session.scalars(
            select(LabelEvent)
            .where(LabelEvent.content_id == content_id)
            .order_by(LabelEvent.created_at)
        ).all()
        assert len(events) == 2
        assert events[0].id == event_id
        assert events[1].source == 'undo'
        assert events[1].label == -1


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
    status = auth_client.get('/api/v1/training/status')
    assert status.status_code == 200
    assert status.json()['labels_since_last_snapshot'] == 0
    second_snapshot = auth_client.post('/api/v1/training/snapshots')
    assert second_snapshot.status_code == 201
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
    assert imported.json()['version'] == 'cand-1'
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


def test_label_version_cas_rejects_stale_update(auth_client, platform_env):
    from sqlalchemy import update

    image = make_image(platform_env['media_root'] / 'cas.jpg')
    content_id = auth_client.post(
        '/api/v1/ingest/content',
        headers={'X-Ingest-Key': 'test-ingest-key'},
        json={
            'content_key': 'url:cas-label',
            'source_url': 'https://example.com/cas-label',
            'title_clean': 'cas',
            'media': [{'source_path': str(image), 'ordinal': 1}],
        },
    ).json()['content_id']

    assert auth_client.post(f'/api/v1/contents/{content_id}/label', json={'label': 1}).status_code == 200

    with SessionLocal() as session:
        stale = session.execute(
            update(ContentItem)
            .where(ContentItem.id == content_id, ContentItem.label_version == 0)
            .values(current_label=0, label_version=1)
        )
        assert stale.rowcount == 0

        content = session.get(ContentItem, content_id)
        event = apply_label(session, content, 0)
        session.commit()
        assert content.current_label == 0
        assert content.label_version == 2
        assert event.supersedes_event_id is not None

    with SessionLocal() as session:
        events = session.scalars(
            select(LabelEvent)
            .where(LabelEvent.content_id == content_id)
            .order_by(LabelEvent.created_at, LabelEvent.id)
        ).all()
        assert len(events) == 2
        assert events[0].label == 1
        assert events[1].label == 0
        assert events[1].supersedes_event_id == events[0].id
        content = session.get(ContentItem, content_id)
        assert content.current_label == 0
        assert content.label_version == 2


def test_parallel_labels_keep_single_event_chain(auth_client, platform_env):
    image = make_image(platform_env['media_root'] / 'parallel.jpg')
    content_id = auth_client.post(
        '/api/v1/ingest/content',
        headers={'X-Ingest-Key': 'test-ingest-key'},
        json={
            'content_key': 'url:parallel-label',
            'source_url': 'https://example.com/parallel-label',
            'title_clean': 'parallel',
            'media': [{'source_path': str(image), 'ordinal': 1}],
        },
    ).json()['content_id']

    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(label: int) -> None:
        with SessionLocal() as session:
            content = session.get(ContentItem, content_id)
            barrier.wait(timeout=5)
            try:
                apply_label(session, content, label)
                session.commit()
                with lock:
                    outcomes.append('ok')
            except LabelConflictError:
                session.rollback()
                with lock:
                    outcomes.append('conflict')
            except Exception:
                session.rollback()
                with lock:
                    outcomes.append('error')

    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(0,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert 'error' not in outcomes
    assert outcomes.count('ok') >= 1

    with SessionLocal() as session:
        events = session.scalars(
            select(LabelEvent)
            .where(LabelEvent.content_id == content_id)
            .order_by(LabelEvent.created_at, LabelEvent.id)
        ).all()
        assert len(events) == outcomes.count('ok')
        roots = [event for event in events if event.supersedes_event_id is None]
        assert len(roots) == 1
        content = session.get(ContentItem, content_id)
        assert content.current_label in (0, 1)
        assert content.label_version == len(events)


def test_label_while_snapshot_zip_builds(auth_client, platform_env, monkeypatch):
    pos = make_image(platform_env['media_root'] / 'snap_pos.jpg', color=(200, 20, 20))
    neg = make_image(platform_env['media_root'] / 'snap_neg.jpg', color=(20, 20, 200))
    extra = make_image(platform_env['media_root'] / 'snap_extra.jpg', color=(20, 200, 20))
    labeled_ids = []
    for key, image, label in [
        ('f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1', pos, 1),
        ('f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2', neg, 0),
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
        labeled_ids.append(content_id)

    extra_id = auth_client.post(
        '/api/v1/ingest/content',
        json={
            'content_key': 'btih:f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3',
            'title_clean': 'extra',
            'magnet_uri': 'magnet:?xt=urn:btih:f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3',
            'info_hash': 'f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3',
            'media': [{'source_path': str(extra), 'ordinal': 1}],
        },
        headers={'X-Ingest-Key': 'test-ingest-key'},
    ).json()['content_id']

    release_zip = threading.Event()
    label_done = threading.Event()
    original_write = __import__('platform_app.training', fromlist=['_write_snapshot_archive'])._write_snapshot_archive

    def slow_write(**kwargs):
        label_done.wait(timeout=5)
        release_zip.set()
        return original_write(**kwargs)

    monkeypatch.setattr('platform_app.training._write_snapshot_archive', slow_write)

    errors = []

    def run_snapshot():
        try:
            with SessionLocal() as session:
                create_snapshot(session)
        except Exception as exc:  # pragma: no cover - surfaced via errors list
            errors.append(exc)

    worker = threading.Thread(target=run_snapshot)
    worker.start()
    # Give phase-1 commit a moment, then label while ZIP would still be held open previously.
    threading.Event().wait(0.2)
    labeled = auth_client.post(f'/api/v1/contents/{extra_id}/label', json={'label': 1})
    assert labeled.status_code == 200, labeled.text
    label_done.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert errors == []

    with SessionLocal() as session:
        content = session.get(ContentItem, extra_id)
        assert content is not None
        assert content.current_label == 1


def test_idempotency_key_is_scoped_to_content(auth_client, platform_env):
    content_ids = []
    for index in range(2):
        image = make_image(platform_env['media_root'] / f'idempotency_{index}.jpg')
        response = auth_client.post(
            '/api/v1/ingest/content',
            headers={'X-Ingest-Key': 'test-ingest-key'},
            json={
                'content_key': f'url:idempotency-{index}',
                'source_url': f'https://example.com/idempotency-{index}',
                'title_clean': f'item {index}',
                'media': [{'source_path': str(image), 'ordinal': 1}],
            },
        )
        content_ids.append(response.json()['content_id'])

    first = auth_client.post(
        f'/api/v1/contents/{content_ids[0]}/label',
        json={'label': 1},
        headers={'Idempotency-Key': 'same-key'},
    )
    second = auth_client.post(
        f'/api/v1/contents/{content_ids[1]}/label',
        json={'label': 1},
        headers={'Idempotency-Key': 'same-key'},
    )
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()['detail'] == 'Idempotency-Key 已用于不同请求'


def test_active_model_controls_content_score_and_queues_missing(auth_client, platform_env):
    content_ids = []
    for index in range(2):
        image = make_image(platform_env['media_root'] / f'model_{index}.jpg')
        response = auth_client.post(
            '/api/v1/ingest/content',
            headers={'X-Ingest-Key': 'test-ingest-key'},
            json={
                'content_key': f'url:model-{index}',
                'source_url': f'https://example.com/model-{index}',
                'title_clean': f'model item {index}',
                'media': [{'source_path': str(image), 'ordinal': 1}],
            },
        )
        content_ids.append(response.json()['content_id'])

    old_path = platform_env['data_dir'] / 'old.pth'
    new_path = platform_env['data_dir'] / 'new.pth'
    old_path.write_bytes(b'old')
    new_path.write_bytes(b'new')
    with SessionLocal() as session:
        session.execute(delete(Job))
        session.execute(delete(ModelVersion))
        old = ModelVersion(version='old', status='active', checkpoint_path=str(old_path))
        new = ModelVersion(version='new', status='candidate', checkpoint_path=str(new_path))
        session.add_all([old, new])
        session.add_all([
            Prediction(content_id=content_ids[0], model_version='old', probability=0.1, prediction=0),
            Prediction(content_id=content_ids[0], model_version='new', probability=0.9, prediction=1),
            Prediction(content_id=content_ids[1], model_version='old', probability=0.2, prediction=0),
        ])
        session.commit()
        new_id = new.id
        old_id = old.id

    assert auth_client.get(f'/api/v1/contents/{content_ids[0]}').json()['probability'] == 0.1
    activated = auth_client.post(f'/api/v1/models/{new_id}/activate')
    assert activated.status_code == 200
    content = auth_client.get(f'/api/v1/contents/{content_ids[0]}').json()
    assert content['model_version'] == 'new'
    assert content['probability'] == 0.9
    with SessionLocal() as session:
        queued_ids = {
            job.payload['content_id']
            for job in session.scalars(select(Job).where(Job.status == 'pending')).all()
        }
        assert content_ids[1] in queued_ids

    rolled_back = auth_client.post(f'/api/v1/models/{old_id}/rollback')
    assert rolled_back.status_code == 200
    assert auth_client.get(f'/api/v1/contents/{content_ids[0]}').json()['probability'] == 0.1


def test_ingest_fails_closed_without_key(client, monkeypatch):
    from platform_app.config import clear_settings_cache

    monkeypatch.setenv('INGEST_API_KEY', '')
    clear_settings_cache()
    response = client.post('/api/v1/ingest/content', json={
        'content_key': 'url:unconfigured-key',
        'media': [{'source_path': 'missing.jpg', 'ordinal': 1}],
    })
    assert response.status_code == 503


def test_frontend_file_rejects_parent_traversal(tmp_path, monkeypatch):
    import platform_app.main as main

    dist = tmp_path / 'dist'
    dist.mkdir()
    inside = dist / 'asset.txt'
    inside.write_text('ok', encoding='utf-8')
    outside = tmp_path / 'secret.txt'
    outside.write_text('secret', encoding='utf-8')
    monkeypatch.setattr(main, 'FRONTEND_DIST', dist)

    assert main.frontend_file('asset.txt') == inside
    assert main.frontend_file('../secret.txt') is None
