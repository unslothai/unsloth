// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Rows from GET /api/inference/audio/catalog. Relative imports only, so node --test and the
// catalog check load it directly.

import type { ModelOption } from "../model-picker/components/model-selector/types.ts";
import {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_REPO,
  AUDIO_CPP_UNOFFERED_FOLDERS,
  audioCppDisplayName,
  isAudioCppFolderId,
} from "./audio-cpp-catalog.ts";

export type AudioCatalogStatus =
  | "supported"
  | "community"
  | "experimental"
  | "testing"
  | "wip";

export interface AudioCatalogRow {
  id: string;
  repo_id: string;
  folder: string;
  source: "umbrella" | "community";
  display_name: string;
  family: string | null;
  task: string;
  workflows: readonly string[];
  description: string | null;
  languages: readonly string[] | null;
  status: AudioCatalogStatus | null;
  size_bytes: number | null;
  downloaded: boolean;
  unsupported_reason: string | null;
}

export interface AudioCatalogResponse {
  source: "hub" | "saved" | "cache";
  fetched_at: number;
  runtime_specs: boolean;
  models: AudioCatalogRow[];
}

/** Same shape as page-models.ts declares; kept structural so neither file imports the other. */
export interface PageCatalogEntry {
  workflows: readonly string[];
  supported: boolean;
}

export type AudioCatalogEntry = PageCatalogEntry & AudioCatalogRow;

export function catalogById(
  rows: readonly AudioCatalogRow[],
): Map<string, AudioCatalogEntry> {
  const map = new Map<string, AudioCatalogEntry>();
  for (const row of rows) {
    map.set(row.id.trim().toLowerCase(), {
      ...row,
      supported: row.unsupported_reason == null && row.workflows.length > 0,
    });
  }
  return map;
}

/** A More row has no Hub pipeline tag, so a catalog ASR pick names its own: the page then loads
 *  it as speech-to-text instead of speech. */
export function withCatalogPipelineTag<
  M extends { pipelineTag?: string | null },
>(byId: ReadonlyMap<string, { task?: string | null }>, id: string, meta: M): M {
  if (meta.pipelineTag || byId.get(id.trim().toLowerCase())?.task !== "asr")
    return meta;
  return { ...meta, pipelineTag: "automatic-speech-recognition" };
}

const STATUS_ORDER: readonly string[] = [
  "supported",
  "community",
  "testing",
  "experimental",
  "wip",
];

function statusRank(status: string | null): number {
  const index = STATUS_ORDER.indexOf(status ?? "supported");
  return index < 0 ? STATUS_ORDER.length : index;
}

const SEED_FOLDERS = new Set(
  AUDIO_CPP_MODELS.filter((m) => isAudioCppFolderId(m.id)).map((m) =>
    audioCppDisplayName(m.id),
  ),
);
const SEED_IDS = new Set(AUDIO_CPP_MODELS.map((m) => m.id.toLowerCase()));

/** Supported rows the bundled seed does not list: new umbrella folders and Community repos. */
export function catalogMoreRows(
  rows: readonly AudioCatalogRow[],
): AudioCatalogRow[] {
  return rows
    .filter((row) => row.unsupported_reason == null && row.workflows.length > 0)
    .filter((row) =>
      row.source === "community"
        ? !SEED_IDS.has(row.id.toLowerCase())
        : !SEED_FOLDERS.has(row.folder) &&
          !Object.hasOwn(AUDIO_CPP_UNOFFERED_FOLDERS, row.folder),
    )
    .sort(
      (a, b) =>
        statusRank(a.status) - statusRank(b.status) ||
        a.display_name.localeCompare(b.display_name),
    );
}

/** A "More models" row; the picker reads `sizeBytes` for its size column. */
export function catalogModelOption(
  row: AudioCatalogRow,
): ModelOption & { sizeBytes: number | null } {
  // Named by folder like the seed rows, and a Community repo by its full id: two repos can
  // share one spec display name, or one repo name.
  const name = row.folder || row.id;
  const about = audioRowDescription(row);
  return {
    id: row.id,
    name,
    isGguf: true,
    description:
      row.display_name && row.display_name !== name
        ? [row.display_name, about].filter(Boolean).join(" · ")
        : about,
    sizeBytes: row.size_bytes,
  };
}

const WORKFLOW_LABELS: Readonly<Record<string, string>> = {
  speak: "Text to Speech",
  clone: "Clone",
  edit: "Edit",
  convert: "Convert",
  music: "Music",
  separate: "Separate",
  transcribe: "Transcribe",
};

const STATUS_LABELS: Readonly<Record<string, string>> = {
  community: "Community",
  testing: "Testing",
  experimental: "Experimental",
  wip: "Work in progress",
};

let languageNames: Intl.DisplayNames | null | undefined;

function languageName(code: string): string {
  if (languageNames === undefined) {
    try {
      languageNames = new Intl.DisplayNames(["en"], { type: "language" });
    } catch {
      languageNames = null;
    }
  }
  try {
    const name = languageNames?.of(code);
    if (name && name.toLowerCase() !== code.toLowerCase()) return name;
  } catch {
    // Not a valid BCP 47 tag; show the code.
  }
  return code.toUpperCase();
}

/** One line for the row tooltip, e.g. "Text to Speech, Clone · English, Chinese +3 · Experimental". */
export function audioRowDescription(row: AudioCatalogRow): string {
  const parts: string[] = [];
  const pages = row.workflows.map((w) => WORKFLOW_LABELS[w]).filter(Boolean);
  if (pages.length > 0) parts.push(pages.join(", "));
  const codes = (row.languages ?? []).filter(
    (code) => code && code.toLowerCase() !== "auto",
  );
  if (codes.length > 0) {
    const names = codes.slice(0, 3).map(languageName).join(", ");
    parts.push(codes.length > 3 ? `${names} +${codes.length - 3}` : names);
  }
  const status = row.status ? STATUS_LABELS[row.status] : undefined;
  if (status) parts.push(status);
  return parts.join(" · ");
}

/** Nightly check verdicts for the umbrella repo's folders. A folder in neither list is only a
 *  warning (the backend catalog lists it); a stale UNOFFERED entry is a failure. */
export function umbrellaFolderFindings(
  folders: ReadonlySet<string>,
  seedFolders: ReadonlySet<string> = SEED_FOLDERS,
  unoffered: Readonly<Record<string, string>> = AUDIO_CPP_UNOFFERED_FOLDERS,
): { warnings: string[]; failures: string[] } {
  const warnings: string[] = [];
  const failures: string[] = [];
  for (const folder of folders) {
    if (!seedFolders.has(folder) && !Object.hasOwn(unoffered, folder)) {
      warnings.push(
        `${AUDIO_CPP_REPO}: '${folder}/' is not in the bundled seed; Unsloth Studio lists it from /api/inference/audio/catalog`,
      );
    }
  }
  for (const folder of Object.keys(unoffered)) {
    if (!folders.has(folder)) {
      failures.push(
        `${AUDIO_CPP_REPO}: AUDIO_CPP_UNOFFERED_FOLDERS lists '${folder}', which the repo no longer has`,
      );
    }
  }
  return { warnings, failures };
}

export interface AudioCatalogSnapshot {
  rows: readonly AudioCatalogRow[];
  byId: ReadonlyMap<string, AudioCatalogEntry>;
  source: AudioCatalogResponse["source"] | "static";
}

const STATIC_SNAPSHOT: AudioCatalogSnapshot = {
  rows: [],
  byId: new Map(),
  source: "static",
};

/** A module-level cache for `useSyncExternalStore`, refetched after `ttlMs` (the server's own
 *  TTL). A failed fetch keeps the last answer, which starts as the static seed, and retries
 *  after `retryMs`. */
export function createAudioCatalogStore(
  fetchCatalog: () => Promise<AudioCatalogResponse>,
  { ttlMs = 300_000, retryMs = 30_000, now = () => Date.now() } = {},
) {
  let snapshot = STATIC_SNAPSHOT;
  let nextFetchAt = 0;
  let inflight: Promise<void> | null = null;
  const listeners = new Set<() => void>();
  return {
    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getSnapshot: (): AudioCatalogSnapshot => snapshot,
    refresh(): Promise<void> {
      if (inflight || now() < nextFetchAt) return inflight ?? Promise.resolve();
      inflight = fetchCatalog()
        .then((response) => {
          const rows = Array.isArray(response?.models) ? response.models : [];
          snapshot = { rows, byId: catalogById(rows), source: response.source };
          nextFetchAt = now() + ttlMs;
          for (const listener of listeners) listener();
        })
        .catch(() => {
          nextFetchAt = now() + retryMs;
        })
        .finally(() => {
          inflight = null;
        });
      return inflight;
    },
  };
}
