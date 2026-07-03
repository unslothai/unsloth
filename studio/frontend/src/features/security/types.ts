// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type RemoteCodeSeverity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

/** One line of the code window shown for a finding. */
export interface RemoteCodeSnippetRow {
  number: number; // 1-based source line
  text: string;
  isMatch: boolean; // the flagged line
  matchStart?: number; // match column span, when known
  matchEnd?: number;
}

export interface RemoteCodeFinding {
  severity: RemoteCodeSeverity;
  file: string;
  check: string;
  evidence?: string;
  line?: number | null; // 1-based
  snippet?: RemoteCodeSnippetRow[];
}

/** A repo file Hugging Face's security scan flagged (e.g. a malicious pickle). */
export interface UnsafeFile {
  path: string;
  level: string; // HF level: unsafe | suspicious | malicious
}

/** Backend remote-code scan result; drives the consent dialog and pins approval to the scanned code version. */
export interface RemoteCodeScan {
  requiresTrustRemoteCode: boolean;
  approvable: boolean; // false for CRITICAL (no override)
  maxSeverity: RemoteCodeSeverity | null;
  fingerprint: string | null;
  findings: RemoteCodeFinding[];
  findingsSummary: string;
  modelName: string;
  // True when our scan first downloaded this repo, so a decline may purge it.
  createdByScan: boolean;
  // Every repo the scan first cached (a LoRA scan pulls adapter + base); a decline
  // purges each. Supersedes createdByScan, which tracks only the primary.
  scanCreatedRepos: string[];
  unsafeFiles: UnsafeFile[]; // files HF flagged unsafe; non-empty => hard block
  securityBlocked: boolean; // blocked specifically by the malware gate
  // This user already approved this exact code (same commit + fingerprint): skip the dialog.
  alreadyApproved: boolean;
  provider: string | null; // HF org for the "from <provider>" tag; null when unattributable
}
