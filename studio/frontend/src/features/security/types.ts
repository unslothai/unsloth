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

/** A repo file Hugging Face's security scan flagged (e.g. a malicious pickle). */
export interface UnsafeFile {
  path: string;
  level: string; // Hugging Face level: "unsafe" | "suspicious" | "malicious"
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
  // Files Hugging Face's security scan flagged as unsafe (malicious pickles etc.).
  // When non-empty the load is a hard block (approvable is false).
  unsafeFiles: UnsafeFile[];
  // True when the load is blocked specifically by the malware/unsafe-file scan.
  securityBlocked: boolean;
}
