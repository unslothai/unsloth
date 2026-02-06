// src/tools/artifact-index/index.ts
import { Database } from "bun:sqlite";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const DEFAULT_DB_DIR = join(homedir(), ".config", "opencode", "artifact-index");
const DB_NAME = "context.db";

export interface PlanRecord {
  id: string;
  title?: string;
  filePath: string;
  overview?: string;
  approach?: string;
}

export interface LedgerRecord {
  id: string;
  sessionName?: string;
  filePath: string;
  goal?: string;
  stateNow?: string;
  keyDecisions?: string;
  filesRead?: string;
  filesModified?: string;
}

export interface MilestoneArtifactRecord {
  id: string;
  milestoneId: string;
  artifactType: string;
  sourceSessionId?: string;
  createdAt?: string;
  tags?: string[];
  payload: string;
}

export interface SearchResult {
  type: "plan" | "ledger";
  id: string;
  filePath: string;
  title?: string;
  summary?: string;
  score: number;
}

export interface MilestoneArtifactSearchResult {
  type: "milestone";
  id: string;
  milestoneId: string;
  artifactType: string;
  sourceSessionId?: string;
  createdAt?: string;
  tags: string[];
  payload: string;
  score: number;
}

export class ArtifactIndex {
  private db: Database | null = null;
  private dbPath: string;

  constructor(dbDir: string = DEFAULT_DB_DIR) {
    this.dbPath = join(dbDir, DB_NAME);
  }

  async initialize(): Promise<void> {
    // Ensure directory exists
    const dir = dirname(this.dbPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    this.db = new Database(this.dbPath);

    // Load and execute schema
    const schemaPath = join(dirname(import.meta.path), "schema.sql");
    let schema: string;

    try {
      schema = readFileSync(schemaPath, "utf-8");
    } catch {
      // Fallback: inline schema for when bundled
      schema = this.getInlineSchema();
    }

    // Execute schema - use exec for multi-statement support
    this.db.exec(schema);
  }

  private getInlineSchema(): string {
    return `
      CREATE TABLE IF NOT EXISTS plans (
        id TEXT PRIMARY KEY,
        title TEXT,
        file_path TEXT UNIQUE NOT NULL,
        overview TEXT,
        approach TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
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
      CREATE VIRTUAL TABLE IF NOT EXISTS plans_fts USING fts5(id, title, overview, approach);
      CREATE VIRTUAL TABLE IF NOT EXISTS ledgers_fts USING fts5(id, session_name, goal, state_now, key_decisions);
      CREATE VIRTUAL TABLE IF NOT EXISTS milestone_artifacts_fts USING fts5(
        id,
        milestone_id,
        artifact_type,
        payload,
        tags,
        source_session_id
      );
    `;
  }

  async indexPlan(record: PlanRecord): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    // Check for existing record by file_path to clean up old FTS entry
    const existing = this.db
      .query<{ id: string }, [string]>(`SELECT id FROM plans WHERE file_path = ?`)
      .get(record.filePath);
    if (existing) {
      this.db.run(`DELETE FROM plans_fts WHERE id = ?`, [existing.id]);
    }

    this.db.run(
      `
      INSERT INTO plans (id, title, file_path, overview, approach, indexed_at)
      VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
      ON CONFLICT(file_path) DO UPDATE SET
        id = excluded.id,
        title = excluded.title,
        overview = excluded.overview,
        approach = excluded.approach,
        indexed_at = CURRENT_TIMESTAMP
    `,
      [record.id, record.title ?? null, record.filePath, record.overview ?? null, record.approach ?? null],
    );

    this.db.run(
      `
      INSERT INTO plans_fts (id, title, overview, approach)
      VALUES (?, ?, ?, ?)
    `,
      [record.id, record.title ?? null, record.overview ?? null, record.approach ?? null],
    );
  }

  async indexLedger(record: LedgerRecord): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    // Check for existing record by file_path to clean up old FTS entry
    const existing = this.db
      .query<{ id: string }, [string]>(`SELECT id FROM ledgers WHERE file_path = ?`)
      .get(record.filePath);
    if (existing) {
      this.db.run(`DELETE FROM ledgers_fts WHERE id = ?`, [existing.id]);
    }

    this.db.run(
      `
      INSERT INTO ledgers (id, session_name, file_path, goal, state_now, key_decisions, files_read, files_modified, indexed_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
      ON CONFLICT(file_path) DO UPDATE SET
        id = excluded.id,
        session_name = excluded.session_name,
        goal = excluded.goal,
        state_now = excluded.state_now,
        key_decisions = excluded.key_decisions,
        files_read = excluded.files_read,
        files_modified = excluded.files_modified,
        indexed_at = CURRENT_TIMESTAMP
    `,
      [
        record.id,
        record.sessionName ?? null,
        record.filePath,
        record.goal ?? null,
        record.stateNow ?? null,
        record.keyDecisions ?? null,
        record.filesRead ?? null,
        record.filesModified ?? null,
      ],
    );

    this.db.run(
      `
      INSERT INTO ledgers_fts (id, session_name, goal, state_now, key_decisions)
      VALUES (?, ?, ?, ?, ?)
    `,
      [
        record.id,
        record.sessionName ?? null,
        record.goal ?? null,
        record.stateNow ?? null,
        record.keyDecisions ?? null,
      ],
    );
  }

  async search(query: string, limit: number = 10): Promise<SearchResult[]> {
    if (!this.db) throw new Error("Database not initialized");

    const results: SearchResult[] = [];
    const escapedQuery = this.escapeFtsQuery(query);

    // Search plans
    const plans = this.db
      .query<{ id: string; file_path: string; title: string; rank: number }, [string, number]>(`
      SELECT p.id, p.file_path, p.title, rank
      FROM plans_fts
      JOIN plans p ON plans_fts.id = p.id
      WHERE plans_fts MATCH ?
      ORDER BY rank
      LIMIT ?
    `)
      .all(escapedQuery, limit);

    for (const row of plans) {
      results.push({
        type: "plan",
        id: row.id,
        filePath: row.file_path,
        title: row.title,
        score: -row.rank,
      });
    }

    // Search ledgers
    const ledgers = this.db
      .query<{ id: string; file_path: string; session_name: string; goal: string; rank: number }, [string, number]>(`
      SELECT l.id, l.file_path, l.session_name, l.goal, rank
      FROM ledgers_fts
      JOIN ledgers l ON ledgers_fts.id = l.id
      WHERE ledgers_fts MATCH ?
      ORDER BY rank
      LIMIT ?
    `)
      .all(escapedQuery, limit);

    for (const row of ledgers) {
      results.push({
        type: "ledger",
        id: row.id,
        filePath: row.file_path,
        title: row.session_name,
        summary: row.goal,
        score: -row.rank,
      });
    }

    // Sort all results by score descending
    results.sort((a, b) => b.score - a.score);

    return results.slice(0, limit);
  }

  async indexMilestoneArtifact(record: MilestoneArtifactRecord): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    const tags = JSON.stringify(record.tags ?? []);
    const createdAt = record.createdAt ?? new Date().toISOString();
    const existing = this.db.query("SELECT id FROM milestone_artifacts WHERE id = ?").get(record.id) as
      | { id: string }
      | undefined;

    if (existing) {
      this.db.run("DELETE FROM milestone_artifacts_fts WHERE id = ?", [existing.id]);
    }

    this.db.run(
      `INSERT INTO milestone_artifacts (
          id,
          milestone_id,
          artifact_type,
          source_session_id,
          created_at,
          tags,
          payload,
          indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
          milestone_id = excluded.milestone_id,
          artifact_type = excluded.artifact_type,
          source_session_id = excluded.source_session_id,
          created_at = excluded.created_at,
          tags = excluded.tags,
          payload = excluded.payload,
          indexed_at = CURRENT_TIMESTAMP`,
      [
        record.id,
        record.milestoneId,
        record.artifactType,
        record.sourceSessionId ?? null,
        createdAt,
        tags,
        record.payload,
      ],
    );

    this.db.run(
      `INSERT INTO milestone_artifacts_fts (
          id,
          milestone_id,
          artifact_type,
          payload,
          tags,
          source_session_id
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      [record.id, record.milestoneId, record.artifactType, record.payload, tags, record.sourceSessionId ?? ""],
    );
  }

  async searchMilestoneArtifacts(
    query: string,
    options: { milestoneId?: string; artifactType?: string; limit?: number } = {},
  ): Promise<MilestoneArtifactSearchResult[]> {
    if (!this.db) throw new Error("Database not initialized");

    const escapedQuery = this.escapeFtsQuery(query);
    const milestoneId = options.milestoneId ?? null;
    const artifactType = options.artifactType ?? null;
    const limit = options.limit ?? 10;

    const rows = this.db
      .query(
        `SELECT
          milestone_artifacts.id,
          milestone_artifacts.milestone_id,
          milestone_artifacts.artifact_type,
          milestone_artifacts.source_session_id,
          milestone_artifacts.created_at,
          milestone_artifacts.tags,
          milestone_artifacts.payload,
          milestone_artifacts_fts.rank
        FROM milestone_artifacts_fts
        JOIN milestone_artifacts ON milestone_artifacts.id = milestone_artifacts_fts.id
        WHERE milestone_artifacts_fts MATCH ?
          AND (? IS NULL OR milestone_artifacts.milestone_id = ?)
          AND (? IS NULL OR milestone_artifacts.artifact_type = ?)
        ORDER BY milestone_artifacts_fts.rank
        LIMIT ?`,
      )
      .all(escapedQuery, milestoneId, milestoneId, artifactType, artifactType, limit) as Array<{
      id: string;
      milestone_id: string;
      artifact_type: string;
      source_session_id: string | null;
      created_at: string | null;
      tags: string | null;
      payload: string;
      rank: number;
    }>;

    return rows.map((row) => ({
      type: "milestone",
      id: row.id,
      milestoneId: row.milestone_id,
      artifactType: row.artifact_type,
      sourceSessionId: row.source_session_id ?? undefined,
      createdAt: row.created_at ?? undefined,
      tags: row.tags ? JSON.parse(row.tags) : [],
      payload: row.payload,
      score: -row.rank,
    }));
  }

  private escapeFtsQuery(query: string): string {
    // Escape special FTS5 characters and wrap terms in quotes
    return query
      .replace(/['"]/g, "")
      .split(/\s+/)
      .filter((term) => term.length > 0)
      .map((term) => `"${term}"`)
      .join(" OR ");
  }

  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

// Singleton instance for global use
let globalIndex: ArtifactIndex | null = null;

export async function getArtifactIndex(): Promise<ArtifactIndex> {
  if (!globalIndex) {
    globalIndex = new ArtifactIndex();
    await globalIndex.initialize();
  }
  return globalIndex;
}
