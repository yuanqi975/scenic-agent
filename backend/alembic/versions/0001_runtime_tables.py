"""Add runtime tables without changing the imported scenic dataset."""
from alembic import op

revision = "0001_runtime_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = [
        "CREATE TABLE IF NOT EXISTS admin_users (id bigserial primary key, email text unique not null, password_hash text not null, is_active boolean not null default true, created_at timestamptz not null default now())",
        "CREATE TABLE IF NOT EXISTS conversations (conversation_id text primary key, park_id text not null, visitor_id text, title text, summary text, created_at timestamptz not null default now(), updated_at timestamptz not null default now())",
        "CREATE TABLE IF NOT EXISTS conversation_messages (id bigserial primary key, conversation_id text not null references conversations(conversation_id), role text not null, content text not null, citations jsonb not null default '[]', created_at timestamptz not null default now())",
        "CREATE TABLE IF NOT EXISTS agent_runs (run_id text primary key, conversation_id text, park_id text not null, intent text not null, agent_name text not null, retrieved_document_ids jsonb not null default '[]', cache_hit boolean not null default false, latency_ms integer, status text not null, error text, created_at timestamptz not null default now())",
        "CREATE TABLE IF NOT EXISTS knowledge_versions (version text primary key, park_id text not null, status text not null, created_at timestamptz not null default now())",
        "CREATE TABLE IF NOT EXISTS knowledge_reviews (id bigserial primary key, candidate_id text not null, park_id text not null, action text not null, reviewer text not null, created_at timestamptz not null default now())",
        "CREATE TABLE IF NOT EXISTS async_tasks (task_id text primary key, park_id text not null, task_type text not null, status text not null, payload jsonb not null default '{}', created_at timestamptz not null default now(), updated_at timestamptz not null default now())",
        "CREATE TABLE IF NOT EXISTS system_settings (key text primary key, value text not null, updated_at timestamptz not null default now())",
        "CREATE INDEX IF NOT EXISTS documents_park_updated_idx ON documents(park_id, updated_at DESC)",
        "CREATE INDEX IF NOT EXISTS documents_content_fts_idx ON documents USING gin (to_tsvector('simple', content))",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    # Runtime tables are intentionally retained; dropping them could remove audit data.
    pass
