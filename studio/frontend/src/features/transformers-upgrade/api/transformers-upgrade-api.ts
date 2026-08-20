


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

/** Ask whether loading `modelName` needs a newer transformers than any installed sidecar.
 *
 * The pre-load half of the consent gate for callers that do not run chat's `/validate`
 * (the Train tab). The token rides in the POST body, never the URL, like the scan route.
 * `options` carries the cache pin, so the answer describes the snapshot the load will
 * open rather than the repo's current config, and for a resume the run it precedes. */
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
  /** The server unloaded the active chat model before the swap (set even on a
   *  structured failure, so callers can restore their model state). */
  model_unloaded?: boolean;
  /** On a version-mismatch failure: the release that superseded the requested
   *  one, so Retry can use it. */
  latest_version?: string | null;
}

/** Consented install of the latest transformers into the sidecar; synchronous, can take minutes.
 *
 * `forceCancelActive` carries the answer the user already gave the model swap's "stop N
 * chats" prompt: without it the install 409s while those chats run, and nothing between the
 * two dialogs stops them. Only ever true after that confirmation. */
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
