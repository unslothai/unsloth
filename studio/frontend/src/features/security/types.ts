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

export interface RemoteCodeScan {
  requiresTrustRemoteCode: boolean;
  approvable: boolean;
  maxSeverity: RemoteCodeSeverity | null;
  fingerprint: string | null;
  findings: RemoteCodeFinding[];
  findingsSummary: string;
  modelName: string;
  createdByScan: boolean;
  // Supersedes createdByScan, which tracks only the primary repo.
  scanCreatedRepos: string[];
  unsafeFiles: UnsafeFile[]; // non-empty => hard block
  securityBlocked: boolean;
  // Same commit + fingerprint already approved by this user: skip the dialog.
  alreadyApproved: boolean;
  provider: string | null;
}
