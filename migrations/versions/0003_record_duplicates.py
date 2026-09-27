"""Mark records that duplicate another source's copy of the same work.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("source_records") as batch:
        batch.add_column(sa.Column("duplicate_of_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("duplicate_reason", sa.String(length=32), nullable=True))
        batch.create_foreign_key(
            "fk_source_records_duplicate_of_id_source_records",
            "source_records",
            ["duplicate_of_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index("ix_source_records_duplicate_of_id", "source_records", ["duplicate_of_id"])


def downgrade() -> None:
    op.drop_index("ix_source_records_duplicate_of_id", table_name="source_records")
    with op.batch_alter_table("source_records") as batch:
        batch.drop_constraint(
            "fk_source_records_duplicate_of_id_source_records", type_="foreignkey"
        )
        batch.drop_column("duplicate_reason")
        batch.drop_column("duplicate_of_id")
