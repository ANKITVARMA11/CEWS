"""Count how often each organization spelling is seen

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the mention counter used to pick an organization's display name."""
    with op.batch_alter_table("organization_aliases") as batch:
        batch.add_column(
            sa.Column("mention_count", sa.Integer(), nullable=False, server_default="0")
        )


def downgrade() -> None:
    """Remove the mention counter."""
    with op.batch_alter_table("organization_aliases") as batch:
        batch.drop_column("mention_count")
