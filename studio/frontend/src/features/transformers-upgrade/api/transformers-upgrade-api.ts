// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  ModelCachePin,
  TransformersUpgradeCheck,
  TransformersUpgradeInfo,
} from "../types";

interface TransformersUpgradeCheckResponse {
  // biome-ignore lint/style/useNamingConvention: API schema
  requires_transformers_upgrade?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  transformers_upgrade?: TransformersUpgradeInfo | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  requires_trust_remote_code?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  latest_tier_active?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  forces_16bit?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  install_breaks_exact_resume?: boolean;
}

/** Pre-load half of the consent gate for callers without chat's `/validate`. The token rides
 * in the body; `options` pins the snapshot (and resume run) to answer about. */
export async function checkTransformersUpgrade(
  modelName: string,
  hfToken?: string | null,
  options?: ModelCachePin & { resumeRunId?: string | null },
): Promise<TransformersUpgradeCheck> {
  const response = await authFetch(
    "/api/inference/transformers-upgrade-check",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_name: modelName,
        hf_token: hfToken ?? null,
        prefer_local_cache: options?.preferLocalCache ?? false,
        model_local_path: options?.modelLocalPath ?? null,
        model_snapshot_path: options?.modelSnapshotPath ?? null,
        model_snapshot_repo_id: options?.modelSnapshotRepoId ?? null,
        resume_run_id: options?.resumeRunId ?? null,
      }),
    },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  const data = (await response.json()) as TransformersUpgradeCheckResponse;
  const upgrade = data.requires_transformers_upgrade
    ? (data.transformers_upgrade ?? null)
    : null;
  return {
    upgrade,
    requiresTrustRemoteCode: Boolean(data.requires_trust_remote_code),
    latestTierActive: Boolean(data.latest_tier_active),
    forces16Bit: Boolean(data.forces_16bit),
    installBreaksExactResume: Boolean(data.install_breaks_exact_resume),
  };
}

interface InstallLatestTransformersResponse {
  success: boolean;
  version: string;
  message: string;
  /** Set even on a structured failure, so callers can restore their model state. */
  model_unloaded?: boolean;
  /** On a version mismatch, the newer release Retry should use. */
  latest_version?: string | null;
}

/** Synchronous, can take minutes. `forceCancelActive` carries the user's "stop N chats" answer;
 * without it the install 409s on those chats. */
export async function installLatestTransformers(
  version: string,
  forceCancelActive = false,
): Promise<InstallLatestTransformersResponse> {
  const response = await authFetch(
    "/api/inference/install-latest-transformers",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version, force_cancel_active: forceCancelActive }),
    },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as InstallLatestTransformersResponse;
}
