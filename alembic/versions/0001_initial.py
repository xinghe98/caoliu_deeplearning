"""empty message

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-11
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    op.create_table(
        'content_items',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('content_key', sa.String(length=128), nullable=False),
        sa.Column('content_group_id', sa.String(length=128), nullable=False),
        sa.Column('source', sa.String(length=64), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=False),
        sa.Column('title_raw', sa.Text(), nullable=False),
        sa.Column('title_clean', sa.Text(), nullable=False),
        sa.Column('magnet_uri', sa.Text(), nullable=False),
        sa.Column('info_hash', sa.String(length=40), nullable=False),
        sa.Column('status', sa.String(length=24), nullable=False),
        sa.Column('dataset_role', sa.String(length=32), nullable=False),
        sa.Column('current_label', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_content_items_content_key', 'content_items', ['content_key'], unique=True)
    op.create_index('ix_content_items_content_group_id', 'content_items', ['content_group_id'])
    op.create_index('ix_content_items_info_hash', 'content_items', ['info_hash'])
    op.create_index('ix_content_items_status', 'content_items', ['status'])
    op.create_index('ix_content_items_dataset_role', 'content_items', ['dataset_role'])
    op.create_index('ix_content_items_current_label', 'content_items', ['current_label'])

    op.create_table(
        'auth_sessions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_auth_sessions_user_id', 'auth_sessions', ['user_id'])
    op.create_index('ix_auth_sessions_token_hash', 'auth_sessions', ['token_hash'], unique=True)
    op.create_index('ix_auth_sessions_expires_at', 'auth_sessions', ['expires_at'])

    op.create_table(
        'media_assets',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('content_id', sa.String(length=36), sa.ForeignKey('content_items.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_path', sa.Text(), nullable=False),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('mime_type', sa.String(length=64), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=24), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('content_id', 'ordinal', name='uq_media_content_ordinal'),
    )
    op.create_index('ix_media_assets_content_id', 'media_assets', ['content_id'])

    op.create_table(
        'label_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('content_id', sa.String(length=36), sa.ForeignKey('content_items.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('label', sa.Integer(), nullable=False),
        sa.Column('source', sa.String(length=32), nullable=False),
        sa.Column('supersedes_event_id', sa.String(length=36), nullable=True),
        sa.Column('model_version', sa.String(length=64), nullable=True),
        sa.Column('probability_at_label', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_label_events_content_id', 'label_events', ['content_id'])
    op.create_index('ix_label_events_user_id', 'label_events', ['user_id'])

    op.create_table(
        'view_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('content_id', sa.String(length=36), sa.ForeignKey('content_items.id', ondelete='CASCADE'), nullable=False),
        sa.Column('event_type', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_view_events_content_id', 'view_events', ['content_id'])
    op.create_index('ix_view_events_event_type', 'view_events', ['event_type'])

    op.create_table(
        'predictions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('content_id', sa.String(length=36), sa.ForeignKey('content_items.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_version', sa.String(length=64), nullable=False),
        sa.Column('probability', sa.Float(), nullable=False),
        sa.Column('decision_threshold', sa.Float(), nullable=False),
        sa.Column('prediction', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('content_id', 'model_version', name='uq_prediction_content_model'),
    )
    op.create_index('ix_predictions_content_id', 'predictions', ['content_id'])

    op.create_table(
        'model_versions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('version', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=24), nullable=False),
        sa.Column('checkpoint_path', sa.Text(), nullable=False),
        sa.Column('decision_threshold', sa.Float(), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('data_manifest_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_model_versions_version', 'model_versions', ['version'], unique=True)
    op.create_index('ix_model_versions_status', 'model_versions', ['status'])

    op.create_table(
        'training_snapshots',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('status', sa.String(length=24), nullable=False),
        sa.Column('label_cutoff_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sample_count', sa.Integer(), nullable=False),
        sa.Column('positive_count', sa.Integer(), nullable=False),
        sa.Column('negative_count', sa.Integer(), nullable=False),
        sa.Column('manifest_hash', sa.String(length=64), nullable=False),
        sa.Column('archive_path', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_training_snapshots_status', 'training_snapshots', ['status'])

    op.create_table(
        'jobs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('job_type', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=24), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('locked_by', sa.String(length=64), nullable=True),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_jobs_job_type', 'jobs', ['job_type'])
    op.create_index('ix_jobs_status', 'jobs', ['status'])

    op.create_table(
        'idempotency_records',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('key', sa.String(length=128), nullable=False),
        sa.Column('request_hash', sa.String(length=64), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=False),
        sa.Column('response_body', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('user_id', 'key', name='uq_idempotency_user_key'),
    )
    op.create_index('ix_idempotency_records_user_id', 'idempotency_records', ['user_id'])

    op.create_table(
        'worker_heartbeats',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('worker_id', sa.String(length=64), nullable=False),
        sa.Column('model_version', sa.String(length=64), nullable=False),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_worker_heartbeats_worker_id', 'worker_heartbeats', ['worker_id'], unique=True)


def downgrade() -> None:
    op.drop_table('worker_heartbeats')
    op.drop_table('idempotency_records')
    op.drop_table('jobs')
    op.drop_table('training_snapshots')
    op.drop_table('model_versions')
    op.drop_table('predictions')
    op.drop_table('view_events')
    op.drop_table('label_events')
    op.drop_table('media_assets')
    op.drop_table('auth_sessions')
    op.drop_table('content_items')
    op.drop_table('users')
