"""Track which label events have been included in a snapshot.

Revision ID: 0002_snapshot_label_events
Revises: 0001_initial
Create Date: 2026-07-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0002_snapshot_label_events'
down_revision: Union[str, None] = '0001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'snapshot_label_events',
        sa.Column(
            'event_id', sa.String(length=36),
            sa.ForeignKey('label_events.id', ondelete='CASCADE'), primary_key=True,
        ),
        sa.Column(
            'snapshot_id', sa.String(length=36),
            sa.ForeignKey('training_snapshots.id', ondelete='CASCADE'), nullable=False,
        ),
    )
    op.create_index(
        'ix_snapshot_label_events_snapshot_id',
        'snapshot_label_events',
        ['snapshot_id'],
    )
    op.execute(sa.text("""
        INSERT INTO snapshot_label_events (event_id, snapshot_id)
        SELECT label_events.id, (
            SELECT training_snapshots.id
            FROM training_snapshots
            WHERE training_snapshots.label_cutoff_at >= label_events.created_at
            ORDER BY training_snapshots.label_cutoff_at ASC
            LIMIT 1
        )
        FROM label_events
        WHERE EXISTS (
            SELECT 1 FROM training_snapshots
            WHERE training_snapshots.label_cutoff_at >= label_events.created_at
        )
    """))


def downgrade() -> None:
    undo_count = op.get_bind().execute(
        sa.text("SELECT COUNT(*) FROM label_events WHERE source = 'undo'")
    ).scalar_one()
    if undo_count:
        raise RuntimeError('存在不可变撤销事件，不能安全降级')
    op.drop_index('ix_snapshot_label_events_snapshot_id', table_name='snapshot_label_events')
    op.drop_table('snapshot_label_events')
