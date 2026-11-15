// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NpuStatus } from "./api";

export function npuVersionTag(
  version: string | null | undefined,
): string | null {
  return version ? `v${version.replace(/^v/, "")}` : null;
}

/** Both runtime versions, or null unless an AMD NPU is detected and FastFlowLM and Lemonade are both installed. */
export function npuEngineVersions(
  status: Pick<NpuStatus, "hardware" | "versions"> | null | undefined,
): { fastflowlm: string; lemonade: string } | null {
  if (!status?.hardware?.present) return null;
  const fastflowlm = npuVersionTag(status.versions?.fastflowlm);
  const lemonade = npuVersionTag(status.versions?.lemonade);
  return fastflowlm && lemonade ? { fastflowlm, lemonade } : null;
}
