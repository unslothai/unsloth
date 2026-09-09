-- SPDX-License-Identifier: AGPL-3.0-only

-- Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

-- Frozen sqlite_master from mu/base~1 (cc0cdab40e), populated only with sqlite in tests.

PRAGMA journal_mode=WAL;

CREATE TABLE api_usage_events (
            id TEXT NOT NULL PRIMARY KEY,
            subject TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        ) WITHOUT ROWID
        ;

CREATE TABLE app_settings (
            key TEXT NOT NULL PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

CREATE TABLE chat_attachment_inventory (
            message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            attachment_id TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            content_type TEXT,
            size_bytes INTEGER,
            PRIMARY KEY(message_id, attachment_id)
        ) WITHOUT ROWID
        ;

CREATE TABLE chat_attachment_inventory_state (
            singleton INTEGER NOT NULL PRIMARY KEY CHECK(singleton = 1),
            inventory_version INTEGER NOT NULL DEFAULT 0,
            dirty INTEGER NOT NULL DEFAULT 1,
            backfilled_at INTEGER NOT NULL
        );

CREATE TABLE chat_attachment_tombstones (
            thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            message_id TEXT NOT NULL,
            attachment_id TEXT NOT NULL,
            deleted_at INTEGER NOT NULL,
            PRIMARY KEY(thread_id, message_id, attachment_id)
        ) WITHOUT ROWID
    ;

CREATE TABLE chat_clear_operations (
            id TEXT NOT NULL PRIMARY KEY,
            active_research_run_ids_json TEXT NOT NULL,
            deleted_thread_ids_json TEXT NOT NULL DEFAULT '[]',
            cleared_at INTEGER NOT NULL,
            reapable_image_ids_json TEXT,
            caches_cleared_at INTEGER
        ) WITHOUT ROWID
        ;

CREATE TABLE chat_generation_events (
            run_id TEXT NOT NULL REFERENCES chat_generation_runs(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY(run_id, seq)
        ) WITHOUT ROWID
        ;

CREATE TABLE chat_generation_runs (
            id TEXT NOT NULL PRIMARY KEY,
            owner_subject TEXT NOT NULL,
            thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            user_message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            assistant_message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            request_hash TEXT NOT NULL,
            request_json TEXT NOT NULL,
            worker_token TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN (
                'queued', 'running', 'cancelling', 'cancelled', 'completed', 'failed'
            )),
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            last_event_seq INTEGER NOT NULL DEFAULT 0,
            finish_reason TEXT,
            error_message TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER
        );

CREATE TABLE chat_legacy_imports (
            legacy_thread_id TEXT NOT NULL PRIMARY KEY,
            imported_at INTEGER NOT NULL
        ) WITHOUT ROWID
        ;

CREATE TABLE chat_messages (
            id TEXT NOT NULL PRIMARY KEY,
            thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            parent_id TEXT,
            role TEXT NOT NULL,
            content_json TEXT NOT NULL,
            attachments_json TEXT,
            metadata_json TEXT,
            created_at INTEGER NOT NULL
        );

CREATE TABLE chat_projects (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            instructions TEXT,
            root_path TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

CREATE TABLE chat_settings (
            key TEXT NOT NULL PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

CREATE TABLE chat_settings_quarantine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            quarantined_at TEXT NOT NULL
        );

CREATE TABLE chat_thread_tombstones (
            id TEXT NOT NULL PRIMARY KEY,
            deleted_at INTEGER NOT NULL
        ) WITHOUT ROWID
        ;

CREATE TABLE chat_threads (
            id TEXT NOT NULL PRIMARY KEY,
            title TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_id TEXT,
            model_gguf_variant TEXT,
            pair_id TEXT,
            project_id TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER,
            openai_code_exec_container_id TEXT,
            anthropic_code_exec_container_id TEXT,
            forked_from_thread_id TEXT,
            forked_from_message_id TEXT,
            settings_json TEXT, settings_seqs TEXT,
            FOREIGN KEY(project_id) REFERENCES chat_projects(id) ON DELETE CASCADE
        );

CREATE TABLE credential_secrets (
            credential_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            nonce BLOB NOT NULL,
            ciphertext BLOB NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (credential_kind, scope_id)
        ) WITHOUT ROWID
        ;

CREATE TABLE prompt_entries (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

CREATE TABLE prompt_lists (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            items_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

CREATE TABLE research_document_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
            step_position INTEGER,
            source_key TEXT NOT NULL,
            document_id TEXT,
            chunk_id TEXT,
            filename TEXT NOT NULL,
            page INTEGER,
            score REAL,
            snippet TEXT,
            fetched_at INTEGER NOT NULL,
            UNIQUE(run_id, source_key)
        );

CREATE TABLE research_events (
            run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY(run_id, seq)
        ) WITHOUT ROWID
        ;

CREATE TABLE research_plan_steps (
            run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            title TEXT NOT NULL,
            query TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            result_json TEXT,
            started_at INTEGER,
            completed_at INTEGER,
            PRIMARY KEY(run_id, position)
        ) WITHOUT ROWID
        ;

CREATE TABLE research_runs (
            id TEXT NOT NULL PRIMARY KEY,
            owner_subject TEXT NOT NULL,
            thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            user_message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            assistant_message_id TEXT REFERENCES chat_messages(id) ON DELETE SET NULL,
            status TEXT NOT NULL CHECK(status IN (
                'planning', 'awaiting_approval', 'queued', 'running', 'paused',
                'cancelling', 'cancelled', 'completed', 'failed'
            )),
            plan_json TEXT,
            plan_revision INTEGER NOT NULL DEFAULT 0,
            plan_hash TEXT,
            config_json TEXT NOT NULL,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_expires_at INTEGER,
            heartbeat_at INTEGER,
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            report_text TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            next_event_seq INTEGER NOT NULL DEFAULT 1
        );

CREATE TABLE research_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
            step_position INTEGER,
            url TEXT NOT NULL,
            title TEXT,
            snippet TEXT,
            fetched_at INTEGER NOT NULL,
            UNIQUE(run_id, url)
        );

CREATE TABLE research_thread_claims (
            owner_subject TEXT NOT NULL,
            thread_id TEXT NOT NULL PRIMARY KEY REFERENCES chat_threads(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL
        ) WITHOUT ROWID
        ;

CREATE TABLE scan_folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE ,
            created_at TEXT NOT NULL
        );

CREATE TABLE training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            loss REAL,
            learning_rate REAL,
            grad_norm REAL,
            eval_loss REAL,
            epoch REAL,
            num_tokens INTEGER,
            elapsed_seconds REAL,
            UNIQUE(run_id, step)
        );

CREATE TABLE training_runs (
            id TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'running',
            model_name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            total_steps INTEGER,
            final_step INTEGER,
            final_loss REAL,
            output_dir TEXT,
            error_message TEXT,
            duration_seconds REAL,
            loss_sparkline TEXT,
            display_name TEXT,
            resume_blocked INTEGER NOT NULL DEFAULT 0,
            resumed_from_run_id TEXT
        );

CREATE INDEX idx_api_usage_events_created_at ON api_usage_events(created_at);

CREATE INDEX idx_api_usage_events_subject_created_at ON api_usage_events(subject, created_at);

CREATE INDEX idx_chat_generation_runs_owner_thread_status ON chat_generation_runs(owner_subject, thread_id, status);

CREATE INDEX idx_chat_generation_runs_thread_status ON chat_generation_runs(thread_id, status);

CREATE INDEX idx_chat_messages_thread_id_created_at ON chat_messages(thread_id, created_at);

CREATE INDEX idx_chat_projects_archived_updated_at ON chat_projects(archived, updated_at);

CREATE INDEX idx_chat_threads_model_type_created_at ON chat_threads(model_type, created_at);

CREATE INDEX idx_chat_threads_pair_id ON chat_threads(pair_id);

CREATE INDEX idx_chat_threads_project_id ON chat_threads(project_id);

CREATE INDEX idx_metrics_run_id ON training_metrics(run_id);

CREATE INDEX idx_prompt_entries_created_at ON prompt_entries(created_at);

CREATE INDEX idx_prompt_lists_created_at ON prompt_lists(created_at);

CREATE INDEX idx_research_document_sources_run ON research_document_sources(run_id, id);

CREATE INDEX idx_research_runs_lease ON research_runs(status, lease_expires_at);

CREATE INDEX idx_research_runs_owner_thread_status ON research_runs(owner_subject, thread_id, status);

CREATE INDEX idx_research_sources_run ON research_sources(run_id, id);

CREATE TRIGGER chat_attachment_inventory_dirty_delete
        AFTER DELETE ON chat_messages
        BEGIN
            INSERT INTO chat_attachment_inventory_state
                (singleton, inventory_version, dirty, backfilled_at)
            VALUES (1, 0, 1, 0)
            ON CONFLICT(singleton) DO UPDATE SET dirty = 1;
        END;

CREATE TRIGGER chat_attachment_inventory_dirty_insert
        AFTER INSERT ON chat_messages
        BEGIN
            INSERT INTO chat_attachment_inventory_state
                (singleton, inventory_version, dirty, backfilled_at)
            VALUES (1, 0, 1, 0)
            ON CONFLICT(singleton) DO UPDATE SET dirty = 1;
        END;

CREATE TRIGGER chat_attachment_inventory_dirty_update
    AFTER UPDATE OF attachments_json, content_json ON chat_messages
    BEGIN
        INSERT INTO chat_attachment_inventory_state
            (singleton, inventory_version, dirty, backfilled_at)
        VALUES (1, 0, 1, 0)
        ON CONFLICT(singleton) DO UPDATE SET dirty = 1;
    END;

CREATE TRIGGER reject_second_active_chat_generation_insert
        BEFORE INSERT ON chat_generation_runs
        WHEN NEW.status IN ('queued','running','cancelling')
         AND EXISTS (
             SELECT 1 FROM chat_generation_runs
             WHERE thread_id = NEW.thread_id
               AND status IN ('queued','running','cancelling')
         )
        BEGIN
            SELECT RAISE(ABORT, 'thread already has an active chat generation');
        END;

CREATE TRIGGER reject_second_active_chat_generation_update
        BEFORE UPDATE OF thread_id, status ON chat_generation_runs
        WHEN NEW.status IN ('queued','running','cancelling')
         AND EXISTS (
             SELECT 1 FROM chat_generation_runs
             WHERE thread_id = NEW.thread_id
               AND id != NEW.id
               AND status IN ('queued','running','cancelling')
         )
        BEGIN
            SELECT RAISE(ABORT, 'thread already has an active chat generation');
        END;

CREATE TRIGGER tombstone_chat_generation_run_id
        BEFORE DELETE ON chat_generation_runs
        BEGIN
            INSERT OR REPLACE INTO app_settings (key, value_json, updated_at)
            VALUES (
                'chat-generation-run-tombstone:' || OLD.id,
                'true',
                CAST(strftime('%s', 'now') AS INTEGER) * 1000
            );
        END;
