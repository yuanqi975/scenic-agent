"""Add durable retry metadata to background tasks."""

from alembic import op


revision = "0002_task_reliability"
down_revision = "0001_runtime_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE async_tasks ADD COLUMN IF NOT EXISTS attempt_count integer NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE async_tasks ADD COLUMN IF NOT EXISTS max_attempts integer NOT NULL DEFAULT 5")
    op.execute("ALTER TABLE async_tasks ADD COLUMN IF NOT EXISTS last_error text")
    op.execute("CREATE INDEX IF NOT EXISTS async_tasks_retry_idx ON async_tasks(park_id,status,updated_at)")


def downgrade() -> None:
    # Keep retry history during rollback; the columns are harmless to older code.
    pass
