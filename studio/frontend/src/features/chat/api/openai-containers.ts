// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Wrappers for the three OpenAI shell-tool container management
 * endpoints exposed by the backend (studio/backend/routes/inference.py).
 * Each one proxies to OpenAI's /v1/containers REST surface using the
 * user's encrypted API key. Backend rejects any base URL that isn't
 * api.openai.com — the shell tool only exists on the managed cloud.
 */

import { authFetch } from "@/features/auth";
import { encryptProviderApiKey } from "./providers-api";

export interface OpenAIContainerSummary {
  id: string;
  name?: string | null;
  createdAt?: number | null;
  lastActiveAt?: number | null;
  expiresAfterMinutes?: number | null;
  status?: string | null;
}

interface RawSummary {
  id: string;
  name?: string | null;
  created_at?: number | null;
  last_active_at?: number | null;
  expires_after_minutes?: number | null;
  status?: string | null;
}

function fromRaw(raw: RawSummary): OpenAIContainerSummary {
  return {
    id: raw.id,
    name: raw.name ?? null,
    createdAt: raw.created_at ?? null,
    lastActiveAt: raw.last_active_at ?? null,
    expiresAfterMinutes: raw.expires_after_minutes ?? null,
    status: raw.status ?? null,
  };
}

async function parseError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    /* fall through */
  }
  return `HTTP ${response.status}`;
}

interface AuthInputs {
  apiKey: string;
  baseUrl: string | null;
}

async function buildAuthBody(auth: AuthInputs) {
  return {
    encrypted_api_key: await encryptProviderApiKey(auth.apiKey),
    provider_base_url: auth.baseUrl,
  };
}

export async function listOpenAIContainers(
  auth: AuthInputs,
): Promise<OpenAIContainerSummary[]> {
  const response = await authFetch(
    "/api/inference/external/openai/containers/list",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(await buildAuthBody(auth)),
    },
  );
  if (!response.ok) throw new Error(await parseError(response));
  const body = (await response.json()) as { containers?: RawSummary[] };
  return (body.containers ?? []).map(fromRaw);
}

export async function createOpenAIContainer(
  auth: AuthInputs,
  params: { name: string; ttlMinutes: number },
): Promise<OpenAIContainerSummary> {
  const response = await authFetch(
    "/api/inference/external/openai/containers/create",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        ...(await buildAuthBody(auth)),
        name: params.name,
        ttl_minutes: params.ttlMinutes,
      }),
    },
  );
  if (!response.ok) throw new Error(await parseError(response));
  const raw = (await response.json()) as RawSummary;
  return fromRaw(raw);
}

export async function deleteOpenAIContainer(
  auth: AuthInputs,
  containerId: string,
): Promise<void> {
  const response = await authFetch(
    "/api/inference/external/openai/containers/delete",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        ...(await buildAuthBody(auth)),
        container_id: containerId,
      }),
    },
  );
  // 404 = container already gone (deleted elsewhere, or expired-then-purged).
  // Treat as idempotent success so a stale list entry doesn't surface as a
  // confusing error — the caller will refresh and the entry will disappear.
  if (!response.ok && response.status !== 204 && response.status !== 404) {
    throw new Error(await parseError(response));
  }
}
