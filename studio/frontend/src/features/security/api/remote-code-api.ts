// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  RemoteCodeFinding,
  RemoteCodeScan,
  RemoteCodeSeverity,
  RemoteCodeSnippetRow,
  UnsafeFile,
} from "../types";

interface SnippetRowResponse {
  number: number;
  text: string;
  is_match: boolean;
  match_start?: number;
  match_end?: number;
}

interface RemoteCodeScanResponse {
  requires_trust_remote_code?: boolean;
  has_remote_code?: boolean;
  approvable?: boolean;
  max_severity?: string | null;
  fingerprint?: string | null;
  findings?: Array<{
    severity: string;
    file: string;
    check: string;
    evidence?: string;
    line?: number | null;
    snippet?: SnippetRowResponse[];
  }>;
  findings_summary?: string;
  model_name?: string;
  created_by_scan?: boolean;
  scan_created_repos?: string[];
  unsafe_files?: Array<{ path?: string; level?: string }>;
  security_blocked?: boolean;
  already_approved?: boolean;
  provider?: string | null;
}

/** Scan a model's auto_map code for the consent dialog (backend reads config + repo
 *  .py, never loads the model). Token rides in the POST body, never the URL. */
export async function getRemoteCodeScan(
  modelName: string,
  hfToken?: string | null,
): Promise<RemoteCodeScan> {
  const response = await authFetch(`/api/models/remote-code-scan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName, hf_token: hfToken ?? null }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  const data = (await response.json()) as RemoteCodeScanResponse;
  const findings: RemoteCodeFinding[] = (data.findings ?? []).map((f) => ({
    severity: f.severity as RemoteCodeSeverity,
    file: f.file,
    check: f.check,
    evidence: f.evidence,
    line: f.line ?? null,
    snippet: (f.snippet ?? []).map(
      (r): RemoteCodeSnippetRow => ({
        number: r.number,
        text: r.text,
        isMatch: r.is_match,
        matchStart: r.match_start,
        matchEnd: r.match_end,
      }),
    ),
  }));
  const unsafeFiles: UnsafeFile[] = (data.unsafe_files ?? []).map((u) => ({
    path: u.path ?? "",
    level: u.level ?? "unsafe",
  }));
  return {
    requiresTrustRemoteCode: Boolean(
      data.requires_trust_remote_code ?? data.has_remote_code,
    ),
    approvable: data.approvable ?? true,
    maxSeverity: (data.max_severity as RemoteCodeSeverity | null) ?? null,
    fingerprint: data.fingerprint ?? null,
    findings,
    findingsSummary: data.findings_summary ?? "",
    modelName: data.model_name ?? modelName,
    createdByScan: Boolean(data.created_by_scan),
    // Fall back to the primary flag for an older backend.
    scanCreatedRepos:
      data.scan_created_repos ??
      (data.created_by_scan ? [data.model_name ?? modelName] : []),
    unsafeFiles,
    securityBlocked: Boolean(data.security_blocked),
    alreadyApproved: Boolean(data.already_approved),
    provider: data.provider ?? null,
  };
}

/** Decline cleanup: purge what the scan downloaded. Fire-and-forget; the backend only
 *  removes a metadata-only cache entry it created (never weights, a loaded model, or a local path). */
export async function discardRemoteCodeDownload(modelName: string): Promise<void> {
  try {
    await authFetch(`/api/models/discard-remote-code`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName }),
    });
  } catch {
    // Best-effort cleanup; ignore failures.
  }
}
