// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Wrappers for the backend's three OpenAI shell-tool container endpoints (routes/inference.py).
 *  Each proxies to OpenAI's /v1/containers using a saved provider key or encrypted request
 *  override. The backend rejects any base URL but api.openai.com. */

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
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
  return readFastApiError(response, "HTTP");
}

interface AuthInputs {
  providerId: string;
  apiKey?: string | null;
  baseUrl: string | null;
}

async function buildAuthBody(auth: AuthInputs) {
  return {
    provider_id: auth.providerId,
    ...(auth.apiKey
      ? { encrypted_api_key: await encryptProviderApiKey(auth.apiKey) }
      : {}),
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
  // 404 means the container is already gone; treat as idempotent success so a stale list entry does
  // not surface as a confusing error.
  if (!response.ok && response.status !== 204 && response.status !== 404) {
    throw new Error(await parseError(response));
  }
}
