// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type RemoteCodeSeverity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

/** One line of the code window shown for a finding. */
export interface RemoteCodeSnippetRow {
  number: number; // 1-based source line
  text: string;
  isMatch: boolean; // the flagged line
  matchStart?: number; // column span of the match, when known
  matchEnd?: number;
}

export interface RemoteCodeFinding {
  severity: RemoteCodeSeverity;
  file: string;
  check: string;
  evidence?: string;
  line?: number | null; // 1-based line of the match
  snippet?: RemoteCodeSnippetRow[]; // surrounding code (+/- 3 lines)
}

/**
 * Result of the backend remote-code scan for a model that ships custom
 * (`auto_map`) Python. Drives the consent dialog: the findings are shown to the
 * user and the fingerprint pins their approval to this exact code version.
 */
export interface RemoteCodeScan {
  requiresTrustRemoteCode: boolean;
  approvable: boolean; // false for CRITICAL -> the user cannot override
  maxSeverity: RemoteCodeSeverity | null;
  fingerprint: string | null;
  findings: RemoteCodeFinding[];
  findingsSummary: string;
  modelName: string;
  // True when our scan is what first downloaded this repo into the HF cache, so a
  // decline may safely purge it (a model the user already had stays put).
  createdByScan: boolean;
}
