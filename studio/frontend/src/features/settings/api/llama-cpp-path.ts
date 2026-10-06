// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export interface LlamaCppPathSettings {
  path: string | null;
  source: "default" | "studio" | "environment";
  editable: boolean;
  available: boolean;
  resolvedBinary: string | null;
  environmentVariable: string | null;
  reloadRequired: boolean;
}

interface ApiLlamaCppPathSettings {
  path: string | null;
  source: LlamaCppPathSettings["source"];
  editable: boolean;
  available: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  resolved_binary: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  environment_variable: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  reload_required: boolean;
}

function fromApi(value: ApiLlamaCppPathSettings): LlamaCppPathSettings {
  return {
    path: value.path,
    source: value.source,
    editable: value.editable,
    available: value.available,
    resolvedBinary: value.resolved_binary,
    environmentVariable: value.environment_variable,
    reloadRequired: value.reload_required,
  };
}

export async function loadLlamaCppPathSettings(): Promise<LlamaCppPathSettings> {
  const response = await authFetch("/api/settings/llama-cpp-path");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to load the llama.cpp folder"),
    );
  }
  return fromApi(await response.json());
}

export async function updateLlamaCppPathSettings(
  path: string | null,
): Promise<LlamaCppPathSettings> {
  const response = await authFetch("/api/settings/llama-cpp-path", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to update the llama.cpp folder"),
    );
  }
  return fromApi(await response.json());
}
