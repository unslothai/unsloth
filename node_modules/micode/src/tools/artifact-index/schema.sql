-- src/tools/artifact-index/schema.sql
-- Artifact Index Schema for SQLite + FTS5
-- NOTE: FTS tables are standalone (not content-linked) and manually synced by code

-- Plans table
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    title TEXT,
    file_path TEXT UNIQUE NOT NULL,
    overview TEXT,
    approach TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ledgers table - with file operation tracking
CREATE TABLE IF NOT EXISTS ledgers (
    id TEXT PRIMARY KEY,
    session_name TEXT,
    file_path TEXT UNIQUE NOT NULL,
    goal TEXT,
    state_now TEXT,
    key_decisions TEXT,
    files_read TEXT,
    files_modified TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Milestone artifacts table
CREATE TABLE IF NOT EXISTS milestone_artifacts (
    id TEXT PRIMARY KEY,
    milestone_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    source_session_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags TEXT,
    payload TEXT NOT NULL,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 virtual tables for full-text search (standalone, manually synced)
CREATE VIRTUAL TABLE IF NOT EXISTS plans_fts USING fts5(
    id,
    title,
    overview,
    approach
);

CREATE VIRTUAL TABLE IF NOT EXISTS ledgers_fts USING fts5(
    id,
    session_name,
    goal,
    state_now,
    key_decisions
);

CREATE VIRTUAL TABLE IF NOT EXISTS milestone_artifacts_fts USING fts5(
    id,
    milestone_id,
    artifact_type,
    payload,
    tags,
    source_session_id
);
