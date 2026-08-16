import { platformRequest } from "./client";
import {
  type PlatformApiToken,
  type PlatformCreatedApiToken,
  type PlatformLangfuseConfig,
  type PlatformLangfuseInput,
  type PlatformOperationsStatus,
  type PlatformUsageStats,
  mapCreatedPlatformApiToken,
  mapPlatformApiTokens,
  mapPlatformLangfuseConfig,
  mapPlatformOperationsStatus,
  mapPlatformUsageStats,
  normalizePlatformLangfuseInput,
} from "./operations-types";

export interface PlatformStatsRange {
  fromDate?: string;
  toDate?: string;
  signal?: AbortSignal;
}

export async function getPlatformOperationsStatus(
  signal?: AbortSignal,
): Promise<PlatformOperationsStatus> {
  return mapPlatformOperationsStatus(
    await platformRequest<unknown>("/system/status", { signal }),
  );
}

export async function getPlatformUsageStats({
  fromDate,
  toDate,
  signal,
}: PlatformStatsRange = {}): Promise<PlatformUsageStats> {
  return mapPlatformUsageStats(
    await platformRequest<unknown>("/system/stats", {
      query: { from_date: fromDate, to_date: toDate },
      signal,
    }),
  );
}

export async function listPlatformApiTokens(
  signal?: AbortSignal,
): Promise<PlatformApiToken[]> {
  return mapPlatformApiTokens(
    await platformRequest<unknown>("/system/tokens", { signal }),
  );
}

export async function createPlatformApiToken(
  signal?: AbortSignal,
): Promise<PlatformCreatedApiToken> {
  return mapCreatedPlatformApiToken(
    await platformRequest<unknown>("/system/tokens", {
      method: "POST",
      signal,
    }),
  );
}

export async function revokePlatformApiToken(
  revokeKey: string,
  signal?: AbortSignal,
): Promise<void> {
  const key = revokeKey.trim();
  if (!key) throw new TypeError("İptal edilecek API token'ı eksik.");
  await platformRequest<unknown>(`/system/tokens/${encodeURIComponent(key)}`, {
    method: "DELETE",
    signal,
  });
}

/** Duplicate Go alias retained as an API-only compatibility contract. */
export async function listPlatformSystemKeysAlias(
  signal?: AbortSignal,
): Promise<PlatformApiToken[]> {
  return mapPlatformApiTokens(
    await platformRequest<unknown>("/system/keys", { signal }),
  );
}

export async function createPlatformSystemKeyAlias(
  signal?: AbortSignal,
): Promise<PlatformCreatedApiToken> {
  return mapCreatedPlatformApiToken(
    await platformRequest<unknown>("/system/keys", {
      method: "POST",
      signal,
    }),
  );
}

export async function revokePlatformSystemKeyAlias(
  revokeKey: string,
  signal?: AbortSignal,
): Promise<void> {
  const key = revokeKey.trim();
  if (!key) throw new TypeError("İptal edilecek API anahtarı eksik.");
  await platformRequest<unknown>(`/system/keys/${encodeURIComponent(key)}`, {
    method: "DELETE",
    signal,
  });
}

export async function getPlatformLangfuseConfig(
  signal?: AbortSignal,
): Promise<PlatformLangfuseConfig | null> {
  return mapPlatformLangfuseConfig(
    await platformRequest<unknown>("/langfuse/api-key", { signal }),
  );
}

async function savePlatformLangfuseConfig(
  method: "POST" | "PUT",
  input: PlatformLangfuseInput,
  signal?: AbortSignal,
): Promise<PlatformLangfuseConfig | null> {
  const normalized = normalizePlatformLangfuseInput(input);
  return mapPlatformLangfuseConfig(
    await platformRequest<unknown>("/langfuse/api-key", {
      method,
      json: {
        host: normalized.host,
        public_key: normalized.publicKey,
        secret_key: normalized.secretKey,
      },
      signal,
    }),
  );
}

export function createPlatformLangfuseConfig(
  input: PlatformLangfuseInput,
  signal?: AbortSignal,
): Promise<PlatformLangfuseConfig | null> {
  return savePlatformLangfuseConfig("POST", input, signal);
}

export function updatePlatformLangfuseConfig(
  input: PlatformLangfuseInput,
  signal?: AbortSignal,
): Promise<PlatformLangfuseConfig | null> {
  return savePlatformLangfuseConfig("PUT", input, signal);
}

export async function deletePlatformLangfuseConfig(
  signal?: AbortSignal,
): Promise<void> {
  await platformRequest<unknown>("/langfuse/api-key", {
    method: "DELETE",
    signal,
  });
}
