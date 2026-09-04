// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MemoryRecord = {
  id: string;
  namespace_id?: string;
  kind: string;
  status: string;
  title: string;
  body: string;
  provenance: string;
  source_episode_id?: string | null;
  created_at?: string;
  updated_at?: string;
  hits?: number;
  explicit?: boolean;
  vote?: { decision: string; reason: string };
};

export type MemorySummary = {
  db_path: string;
  archive_count: number;
  compiled_count: number;
  contradiction_count: number;
  records: {
    total: number;
    by_status: Record<string, number>;
    by_kind: Record<string, number>;
    by_provenance: Record<string, number>;
  };
  adapters: {
    shadow: number;
    promoted: number;
    discarded: number;
    promoted_id: string | null;
  };
  last_inject: {
    episode_id: string;
    contact: string;
    standing_chars: number;
    retrieve_chars: number;
    trajectory_chars: number;
    total_chars: number;
    compiled_ids?: string;
    retrieved_ids?: string;
  } | null;
};

export type OperatorItem = {
  id?: string | null;
  kind?: string;
  title?: string;
  decision?: string | null;
  reason?: string | null;
  applied?: string | null;
  error?: string;
};

export type AdapterRow = {
  id: string;
  status: string;
  recipe?: string;
  backend?: string;
  pack_id?: string;
  path?: string;
  metrics?: string | Record<string, unknown> | null;
};

export type PackRow = {
  id: string;
  n_train?: number;
  n_holdout?: number;
  include_sim?: boolean | number;
  created_at?: string;
};

export type CompactReport = {
  dry_run: boolean;
  [key: string]: unknown;
};

export type WorkspaceTab =
  | "inbox"
  | "notebook"
  | "standing"
  | "archive"
  | "sidecar"
  | "hygiene";
