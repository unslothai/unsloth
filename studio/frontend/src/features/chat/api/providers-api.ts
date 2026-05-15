// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import forge from "node-forge";
import { authFetch } from "@/features/auth";

export interface ProviderRegistryEntry {
  provider_type: string;
  display_name: string;
  base_url: string;
  default_models: string[];
  supports_streaming: boolean;
  supports_vision: boolean;
  supports_tool_calling: boolean;
  /** remote = fetch /models; curated = huge catalogs — UI uses defaults + manual IDs only */
  model_list_mode?: "remote" | "curated";
}

export interface ProviderConfig {
  id: string;
  provider_type: string;
  display_name: string;
  base_url: string;
  is_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface ProviderModelInfo {
  id: string;
  display_name: string;
  context_length?: number | null;
  owned_by?: string | null;
}

export interface ProviderTestResult {
  success: boolean;
  message: string;
  models_count?: number | null;
}

function parseErrorText(status: number, body: unknown): string {
  if (
    body &&
    typeof body === "object" &&
    "detail" in body &&
    typeof body.detail === "string"
  ) {
    return body.detail;
  }
  if (
    body &&
    typeof body === "object" &&
    "message" in body &&
    typeof body.message === "string"
  ) {
    return body.message;
  }
  return `Request failed (${status})`;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(parseErrorText(response.status, body));
  }
  return body as T;
}

export function isProviderKeyRotationError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const normalized = error.message.toLowerCase();
  return (
    normalized.includes("public key may have changed") ||
    normalized.includes("server key may have changed")
  );
}

let cachedPublicKeyPem: string | null = null;
let cachedForgeKey: forge.pki.rsa.PublicKey | null = null;

export function clearProviderPublicKeyCache(): void {
  cachedPublicKeyPem = null;
  cachedForgeKey = null;
}

async function importProviderPublicKey(
  forceRefresh = false,
): Promise<forge.pki.rsa.PublicKey> {
  if (!forceRefresh && cachedForgeKey) {
    return cachedForgeKey;
  }
  const response = await authFetch("/api/providers/public-key");
  const body = await parseJsonOrThrow<{ public_key: string }>(response);
  const publicKeyPem = body.public_key?.trim();
  if (!publicKeyPem) {
    throw new Error("Provider public key is missing.");
  }
  if (!forceRefresh && cachedPublicKeyPem === publicKeyPem && cachedForgeKey) {
    return cachedForgeKey;
  }
  const forgeKey = forge.pki.publicKeyFromPem(publicKeyPem);
  cachedPublicKeyPem = publicKeyPem;
  cachedForgeKey = forgeKey;
  return forgeKey;
}

export async function encryptProviderApiKey(
  plaintextApiKey: string,
  forceRefresh = false,
): Promise<string> {
  const key = await importProviderPublicKey(forceRefresh);
  const encrypted = key.encrypt(plaintextApiKey, "RSA-OAEP", {
    md: forge.md.sha256.create(),
    mgf1: { md: forge.md.sha256.create() },
  });
  return forge.util.encode64(encrypted);
}

export async function listProviderRegistry(): Promise<ProviderRegistryEntry[]> {
  const response = await authFetch("/api/providers/registry");
  return parseJsonOrThrow<ProviderRegistryEntry[]>(response);
}

export async function listProviderConfigs(): Promise<ProviderConfig[]> {
  const response = await authFetch("/api/providers/");
  return parseJsonOrThrow<ProviderConfig[]>(response);
}

export async function createProviderConfig(payload: {
  providerType: string;
  displayName: string;
  baseUrl?: string | null;
}): Promise<ProviderConfig> {
  const response = await authFetch("/api/providers/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider_type: payload.providerType,
      display_name: payload.displayName,
      base_url: payload.baseUrl ?? null,
    }),
  });
  return parseJsonOrThrow<ProviderConfig>(response);
}

export async function deleteProviderConfig(providerId: string): Promise<void> {
  const response = await authFetch(`/api/providers/${providerId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }
}

export async function updateProviderConfig(
  providerId: string,
  payload: {
    displayName?: string;
    baseUrl?: string | null;
    isEnabled?: boolean;
  },
): Promise<ProviderConfig> {
  const response = await authFetch(`/api/providers/${providerId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...(payload.displayName === undefined ? {} : { display_name: payload.displayName }),
      ...(payload.baseUrl === undefined ? {} : { base_url: payload.baseUrl }),
      ...(payload.isEnabled === undefined ? {} : { is_enabled: payload.isEnabled }),
    }),
  });
  return parseJsonOrThrow<ProviderConfig>(response);
}

async function withApiKeyEncryptionRetry<T>(
  plaintextApiKey: string,
  call: (encryptedApiKey: string | null) => Promise<T>,
): Promise<T> {
  // Empty key (local providers): skip RSA round-trip and let the backend omit auth.
  if (!plaintextApiKey) {
    return await call(null);
  }
  try {
    const encrypted = await encryptProviderApiKey(plaintextApiKey, false);
    return await call(encrypted);
  } catch (error) {
    if (!isProviderKeyRotationError(error)) {
      throw error;
    }
    clearProviderPublicKeyCache();
    const encrypted = await encryptProviderApiKey(plaintextApiKey, true);
    return await call(encrypted);
  }
}

export async function testProviderConnection(payload: {
  providerType: string;
  apiKey: string;
  baseUrl?: string | null;
}): Promise<ProviderTestResult> {
  return withApiKeyEncryptionRetry(payload.apiKey, async (encryptedApiKey) => {
    const response = await authFetch("/api/providers/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider_type: payload.providerType,
        encrypted_api_key: encryptedApiKey,
        base_url: payload.baseUrl ?? null,
      }),
    });
    return parseJsonOrThrow<ProviderTestResult>(response);
  });
}

export async function listProviderModels(payload: {
  providerType: string;
  apiKey: string;
  baseUrl?: string | null;
}): Promise<ProviderModelInfo[]> {
  return withApiKeyEncryptionRetry(payload.apiKey, async (encryptedApiKey) => {
    const response = await authFetch("/api/providers/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider_type: payload.providerType,
        encrypted_api_key: encryptedApiKey,
        base_url: payload.baseUrl ?? null,
      }),
    });
    return parseJsonOrThrow<ProviderModelInfo[]>(response);
  });
}
