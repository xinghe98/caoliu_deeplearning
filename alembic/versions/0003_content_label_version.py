"""Add optimistic concurrency version for content labels.

Revision ID: 0003_content_label_version
Revises: 0002_snapshot_label_events
Create Date: 2026-07-18
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision: str = '0003_content_label_version'
down_revision: Union[str, None] = '0002_snapshot_label_events'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in inspect(bind).get_columns('content_items')}
    if 'label_version' not in columns:
        op.add_column(
            'content_items',
            sa.Column('label_version', sa.Integer(), server_default='0', nullable=False),
        )


def downgrade() -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in inspect(bind).get_columns('content_items')}
    if 'label_version' in columns:
        op.drop_column('content_items', 'label_version')
