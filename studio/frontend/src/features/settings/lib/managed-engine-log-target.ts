// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MANAGED_ENGINE_LOG_FAMILIES = ["sglang", "vllm"] as const;
export type ManagedEngineLogFamily = (typeof MANAGED_ENGINE_LOG_FAMILIES)[number];

export function isManagedEngineLogFamily(value: unknown): value is ManagedEngineLogFamily {
  return MANAGED_ENGINE_LOG_FAMILIES.some((family) => family === value);
}

export function managedEngineFailureLogTarget(detail: unknown): {
  family: ManagedEngineLogFamily;
  path: string;
} | null {
  if (!detail || typeof detail !== "object") return null;
  const { code, log_family, log_path } = detail as Record<string, unknown>;
  if (
    code !== "managed_engine_load_failed" ||
    !isManagedEngineLogFamily(log_family) ||
    typeof log_path !== "string" ||
    !log_path.trim()
  ) {
    return null;
  }
  return { family: log_family, path: log_path };
}
