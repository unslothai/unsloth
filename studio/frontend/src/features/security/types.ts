// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type RemoteCodeSeverity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

export interface RemoteCodeSnippetRow {
  number: number;
  text: string;
  isMatch: boolean;
  matchStart?: number;
  matchEnd?: number;
}

export interface RemoteCodeFinding {
  severity: RemoteCodeSeverity;
  file: string;
  check: string;
  evidence?: string;
  line?: number | null;
  snippet?: RemoteCodeSnippetRow[];
}

export interface UnsafeFile {
  path: string;
  level: string;
}

export type RemoteCodeBlockCause =
  | "unsafe_files"
  | "critical_findings"
  | "unscanned_weights"
  | "managed_refused"
  | "not_on_source"
  | "unscannable";

export interface RemoteCodeScan {
  blockCause: RemoteCodeBlockCause | null;
  // The active model source, and the repo it lacks when that is the block.
  blockSource: string;
  blockTarget: string | null;
  requiresTrustRemoteCode: boolean;
  approvable: boolean;
  maxSeverity: RemoteCodeSeverity | null;
  fingerprint: string | null;
  findings: RemoteCodeFinding[];
  findingsSummary: string;
  modelName: string;
  createdByScan: boolean;
  // Every repo the scan first cached (a LoRA scan pulls adapter and base); a decline purges each.
  scanCreatedRepos: string[];
  unsafeFiles: UnsafeFile[]; // non-empty => hard block
  securityBlocked: boolean;
  // Same commit + fingerprint already approved by this user: skip the dialog.
  alreadyApproved: boolean;
  provider: string | null;
}
