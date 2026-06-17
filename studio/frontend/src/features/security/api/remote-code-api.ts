// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  RemoteCodeFinding,
  RemoteCodeScan,
  RemoteCodeSeverity,
  RemoteCodeSnippetRow,
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
}

/** Encode a repo id for a `{model_name:path}` route, preserving the org slash. */
function encodeModelPath(modelName: string): string {
  return modelName.split("/").map(encodeURIComponent).join("/");
}

/**
 * Statically scan a model's custom (`auto_map`) code so the consent dialog can
 * show findings before the user enables trust_remote_code. Code-free on the
 * backend: it reads config.json and scans the repo .py, never loading the model.
 */
export async function getRemoteCodeScan(
  modelName: string,
  hfToken?: string | null,
): Promise<RemoteCodeScan> {
  const qs = hfToken ? `?hf_token=${encodeURIComponent(hfToken)}` : "";
  const response = await authFetch(
    `/api/models/remote-code-scan/${encodeModelPath(modelName)}${qs}`,
  );
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
  };
}
