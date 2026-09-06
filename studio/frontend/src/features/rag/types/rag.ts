// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string | null;
  createdAt?: string | null;
  documentCount?: number;
}

/** Index status: pending -> running -> completed | failed. */
export type DocumentStatus = "pending" | "running" | "completed" | "failed";

export interface RagDocument {
  id: string;
  filename: string;
  status: DocumentStatus;
  error?: string | null;
  numChunks?: number | null;
  kbId?: string | null;
  threadId?: string | null;
  projectId?: string | null;
  linkedFolderId?: string | null;
  managed: boolean;
  createdAt?: string | null;
}

export function isLinkedFolderManaged(document: RagDocument): boolean {
  return Boolean(document.managed || document.linkedFolderId);
}

/** RagDocument enriched for the global uploaded-files list (settings Data tab). */
export interface UploadedDocument extends RagDocument {
  sizeBytes?: number | null;
  kbName?: string | null;
  projectName?: string | null;
}

export interface DocumentUploadResult {
  documentId: string;
  jobId: string;
  filename: string;
}

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type TerminalJobStatus = Extract<
  JobStatus,
  "completed" | "failed" | "cancelled"
>;

export function terminalJobStatus(status: JobStatus): TerminalJobStatus | null {
  return status === "completed" || status === "failed" || status === "cancelled"
    ? status
    : null;
}

export interface IndexJob {
  id: string;
  documentId: string;
  status: JobStatus;
  stage?: string | null;
  progress?: number | null;
  error?: string | null;
  numChunks?: number | null;
}

/** One SSE frame from /jobs/{jobId}/events. */
export interface JobEvent {
  type: "progress" | "complete" | "error";
  stage?: string | null;
  progress?: number | null;
  error?: string | null;
  num_chunks?: number | null;
}

export type LinkedFolderScopeType = "knowledge_base" | "project";

export interface LinkedFolderScope {
  type: LinkedFolderScopeType;
  id: string;
}

export type LinkedFolderStatus = "idle" | "syncing" | "error";

/** A local directory whose durable access grant is held by the desktop backend. */
export interface LinkedFolder {
  id: string;
  displayName: string;
  scopeType: LinkedFolderScopeType;
  scopeId: string;
  scopeName?: string | null;
  status: LinkedFolderStatus;
  documentCount?: number;
  lastSyncedAt?: string | null;
  error?: string | null;
  activeJobId?: string | null;
  createdAt?: string | null;
}

export function linkedFolderSourcesChanged(
  previous: LinkedFolder[] | null,
  current: LinkedFolder[],
): boolean {
  if (!previous) return false;
  const previousById = new Map(previous.map((folder) => [folder.id, folder]));
  if (
    previous.length !== current.length ||
    previous.some((folder) => !current.some((row) => row.id === folder.id))
  ) {
    return true;
  }
  return current.some((folder) => {
    const prior = previousById.get(folder.id);
    return (
      prior !== undefined &&
      (prior.documentCount !== folder.documentCount ||
        prior.lastSyncedAt !== folder.lastSyncedAt)
    );
  });
}

export type FolderSyncMode = "sync" | "rebuild";

/** Aggregate job for discovering and indexing all changes in a linked folder. */
export interface FolderSyncJob {
  id: string;
  linkedFolderId: string;
  mode: FolderSyncMode;
  status: JobStatus;
  stage?: string | null;
  progress?: number | null;
  discoveredFiles?: number;
  processedFiles?: number;
  indexedFiles?: number;
  removedFiles?: number;
  failedFiles?: number;
  error?: string | null;
}

/** One SSE frame from /linked-folder-jobs/{jobId}/events. */
export interface FolderSyncJobEvent extends Partial<FolderSyncJob> {
  type: "progress" | "complete" | "error";
}

export function retainActiveFolderJobs(
  folders: LinkedFolder[],
  jobs: Record<string, FolderSyncJob>,
): Record<string, FolderSyncJob> {
  const retained: Record<string, FolderSyncJob> = {};
  for (const folder of folders) {
    const job = jobs[folder.id];
    if (job && folder.activeJobId === job.id) retained[folder.id] = job;
  }
  return retained;
}

/** Coords 0..1, top-left origin. */
export interface PdfRegion {
  pageIndex: number;
  pageNumber: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PreviewTarget {
  documentId: string;
  filename: string;
  mediaKind: "pdf" | "text";
  targetPage?: number | null;
  pdfRegions: PdfRegion[];
  text?: string | null;
}

export const RAG_UPLOAD_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.html,.htm";

export const SOURCE_CODE_EXTENSIONS = [
  // Programming and scripting languages
  ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".hh", ".ipp", ".inl", ".cu", ".cuh",
  ".cs", ".vb", ".vbs", ".fs", ".fsi", ".fsx", ".csproj",
  ".go", ".rs", ".zig", ".odin", ".nim", ".nims", ".nimble", ".cr", ".d", ".v", ".sv", ".svh", ".vhd", ".vhdl", ".asm", ".s",
  ".java", ".kt", ".kts", ".scala", ".groovy", ".gradle", ".sbt", ".clj", ".cljs", ".cljc",
  ".js", ".jsx", ".mjs", ".cjs",
  ".ts", ".tsx", ".mts", ".cts",
  ".py", ".pyi", ".pyx", ".pxd", ".ipynb",
  ".rb", ".php", ".swift", ".lua", ".r", ".pl", ".pm", ".t", ".sh", ".bash", ".zsh", ".fish", ".bat", ".cmd", ".ps1", ".psm1",
  ".dart", ".ex", ".exs", ".erl", ".hrl", ".hs", ".lhs", ".ml", ".mli",
  ".sql", ".prc", ".tab",
  // Stylesheets, web templates, markup
  ".css", ".scss", ".sass", ".less", ".styl", ".svg", ".vue", ".svelte", ".astro",
  ".pug", ".jade", ".haml", ".slim", ".ejs", ".erb", ".hbs", ".handlebars", ".mustache", ".njk", ".jinja", ".jinja2", ".j2", ".twig", ".liquid",
  ".cshtml", ".razor", ".aspx", ".jsp", ".tpl", ".qml",
  // Structured data and config
  ".csv", ".tsv", ".psv", ".json", ".jsonl", ".ndjson", ".jsonc", ".json5", ".geojson", ".har", ".avsc",
  ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".cnf", ".properties", ".plist", ".edn", ".ron", ".cue", ".lock",
  // Documentation and text
  ".rst", ".adoc", ".asciidoc", ".org", ".textile", ".wiki", ".tex", ".latex", ".sty", ".cls", ".bib", ".rmd", ".qmd",
  ".srt", ".vtt", ".sbv", ".ass", ".ssa", ".sub", ".lrc",
  ".po", ".pot", ".strings", ".resx", ".xliff", ".xlf", ".log",
];

const KNOWN_EXACT_NAMES = new Set([
  "dockerfile",
  "makefile",
  "gemfile",
  "rakefile",
  "procfile",
  "vagrantfile",
  "cmakelists.txt",
]);

export const RAG_SOURCE_UPLOAD_ACCEPT = `${RAG_UPLOAD_ACCEPT},${SOURCE_CODE_EXTENSIONS.join(",")}`;

const ACCEPTED_SOURCE_EXTS = new Set(
  RAG_SOURCE_UPLOAD_ACCEPT.split(",").map((ext) => ext.trim().toLowerCase()),
);

// `accept` only filters the picker, so a drop can carry anything, including an
// extension-less folder entry the backend would reject.
export function isSupportedSourceName(name: string): boolean {
  const lower = name.toLowerCase();
  const base = lower.split(/[\\/]/).pop() ?? lower;
  if (KNOWN_EXACT_NAMES.has(base)) return true;
  const dot = base.lastIndexOf(".");
  if (dot <= 0) return false;
  return ACCEPTED_SOURCE_EXTS.has(base.slice(dot));
}
