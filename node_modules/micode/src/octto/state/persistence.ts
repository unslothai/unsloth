// src/octto/state/persistence.ts
import { existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";

import { STATE_DIR } from "../constants";
import type { BrainstormState } from "./types";

export interface StatePersistence {
  save: (state: BrainstormState) => Promise<void>;
  load: (sessionId: string) => Promise<BrainstormState | null>;
  delete: (sessionId: string) => Promise<void>;
  list: () => Promise<string[]>;
}

function validateSessionId(sessionId: string): void {
  if (!/^[a-zA-Z0-9_-]+$/.test(sessionId)) {
    throw new Error(`Invalid session ID: ${sessionId}`);
  }
}

export function createStatePersistence(baseDir = STATE_DIR): StatePersistence {
  function getFilePath(sessionId: string): string {
    validateSessionId(sessionId);
    return join(baseDir, `${sessionId}.json`);
  }

  function ensureDir(): void {
    if (!existsSync(baseDir)) {
      mkdirSync(baseDir, { recursive: true });
    }
  }

  return {
    async save(state: BrainstormState): Promise<void> {
      ensureDir();
      const filePath = getFilePath(state.session_id);
      state.updated_at = Date.now();
      await Bun.write(filePath, JSON.stringify(state, null, 2));
    },

    async load(sessionId: string): Promise<BrainstormState | null> {
      const filePath = getFilePath(sessionId);
      if (!existsSync(filePath)) {
        return null;
      }
      const content = await Bun.file(filePath).text();
      return JSON.parse(content) as BrainstormState;
    },

    async delete(sessionId: string): Promise<void> {
      const filePath = getFilePath(sessionId);
      if (existsSync(filePath)) {
        rmSync(filePath);
      }
    },

    async list(): Promise<string[]> {
      if (!existsSync(baseDir)) {
        return [];
      }
      const files = readdirSync(baseDir);
      return files.filter((f) => f.endsWith(".json")).map((f) => f.replace(".json", ""));
    },
  };
}
