CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS parks (park_id text PRIMARY KEY, payload jsonb NOT NULL, created_at timestamptz NOT NULL);
CREATE TABLE IF NOT EXISTS attractions (park_id text NOT NULL REFERENCES parks(park_id), attraction_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, attraction_id), UNIQUE (park_id, attraction_id));
CREATE TABLE IF NOT EXISTS facilities (park_id text NOT NULL REFERENCES parks(park_id), facility_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, facility_id));
CREATE TABLE IF NOT EXISTS routes (park_id text NOT NULL REFERENCES parks(park_id), route_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, route_id));
CREATE TABLE IF NOT EXISTS faqs (park_id text NOT NULL REFERENCES parks(park_id), faq_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, faq_id));
CREATE TABLE IF NOT EXISTS notices (park_id text NOT NULL REFERENCES parks(park_id), notice_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, notice_id));
CREATE TABLE IF NOT EXISTS feedbacks (park_id text NOT NULL REFERENCES parks(park_id), feedback_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, feedback_id));
CREATE TABLE IF NOT EXISTS feedback_candidates (park_id text NOT NULL REFERENCES parks(park_id), candidate_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, candidate_id));
CREATE TABLE IF NOT EXISTS documents (park_id text NOT NULL REFERENCES parks(park_id), document_id text NOT NULL, source_type text NOT NULL, source_id text NOT NULL, content text NOT NULL, metadata jsonb NOT NULL, knowledge_version text NOT NULL, updated_at timestamptz NOT NULL, PRIMARY KEY (park_id, document_id), UNIQUE (park_id, document_id));
CREATE TABLE IF NOT EXISTS document_chunks (park_id text NOT NULL REFERENCES parks(park_id), document_id text NOT NULL, chunk_index integer NOT NULL DEFAULT 0, content text NOT NULL, embedding halfvec(2560), updated_at timestamptz NOT NULL, PRIMARY KEY (park_id, document_id, chunk_index), FOREIGN KEY (park_id, document_id) REFERENCES documents(park_id, document_id));
CREATE TABLE IF NOT EXISTS evaluation_questions (park_id text NOT NULL REFERENCES parks(park_id), question_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, question_id));
CREATE TABLE IF NOT EXISTS async_tasks (task_id text PRIMARY KEY, park_id text NOT NULL, task_type text NOT NULL, status text NOT NULL, payload jsonb NOT NULL DEFAULT '{}', attempt_count integer NOT NULL DEFAULT 0, max_attempts integer NOT NULL DEFAULT 5, last_error text, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS document_chunks_park_idx ON document_chunks(park_id);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx ON document_chunks USING hnsw (embedding halfvec_cosine_ops);
