// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type LlmCompressorConsentKind = "shadow" | "workspace";

export interface LlmCompressorExportProbe {
  ready: boolean;
  needs_consent: boolean;
  consent_kind: LlmCompressorConsentKind | null;
  install_summary: string | null;
  workspace_install_command: string;
  shadow_path: string;
  autoinstall_disabled: boolean;
  shadow_disabled: boolean;
  offline: boolean;
  blocked_reason: string | null;
  python_executable: string;
  has_pip: boolean;
  has_uv: boolean;
}

export async function fetchLlmCompressorProbe(): Promise<LlmCompressorExportProbe> {
  const response = await authFetch("/api/export/llm-compressor-probe");
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as LlmCompressorExportProbe;
}
