// src/tools/pty/types.ts
import type { RingBuffer } from "./buffer";

export type PTYStatus = "running" | "exited" | "killed";

export interface PTYSession {
  id: string;
  title: string;
  command: string;
  args: string[];
  workdir: string;
  env?: Record<string, string>;
  status: PTYStatus;
  exitCode?: number;
  pid: number;
  createdAt: Date;
  parentSessionId: string;
  buffer: RingBuffer;
  process: import("bun-pty").IPty;
}

export interface PTYSessionInfo {
  id: string;
  title: string;
  command: string;
  args: string[];
  workdir: string;
  status: PTYStatus;
  exitCode?: number;
  pid: number;
  createdAt: Date;
  lineCount: number;
}

export interface SpawnOptions {
  command: string;
  args?: string[];
  workdir?: string;
  env?: Record<string, string>;
  title?: string;
  parentSessionId: string;
}

export interface ReadResult {
  lines: string[];
  totalLines: number;
  offset: number;
  hasMore: boolean;
}

export interface SearchMatch {
  lineNumber: number;
  text: string;
}

export interface SearchResult {
  matches: SearchMatch[];
  totalMatches: number;
  totalLines: number;
  offset: number;
  hasMore: boolean;
}
