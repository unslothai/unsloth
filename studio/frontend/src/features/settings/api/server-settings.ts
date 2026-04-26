// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";

export type ActiveSslSource =
  | "default"
  | "cli_no_ssl"
  | "cli_paths"
  | "cli_self_signed"
  | "env_paths"
  | "env_self_signed"
  | "db_paths"
  | "db_self_signed";

export interface ServerSslSettings {
  ssl_enabled: boolean;
  ssl_self_signed: boolean;
  ssl_certfile: string | null;
  ssl_keyfile: string | null;
  active_scheme: "http" | "https";
  active_port: number | null;
  active_source: ActiveSslSource;
  restart_supported: boolean;
}

export interface ServerSslUpdate {
  ssl_enabled: boolean;
  ssl_self_signed: boolean;
  ssl_certfile: string | null;
  ssl_keyfile: string | null;
}

export async function fetchServerSettings(): Promise<ServerSslSettings> {
  const res = await authFetch("/api/settings/server");
  if (!res.ok) {
    throw new Error("Failed to load server settings");
  }
  return res.json();
}

export async function updateServerSettings(
  payload: ServerSslUpdate,
): Promise<ServerSslSettings> {
  const res = await authFetch("/api/settings/server", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || "Failed to update server settings");
  }
  return res.json();
}

export interface TestCertificateRequest {
  ssl_certfile: string;
  ssl_keyfile: string;
}

export async function testCertificate(
  payload: TestCertificateRequest,
): Promise<{ ok: boolean; error: string | null }> {
  const res = await authFetch("/api/settings/server/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    return { ok: false, error: detail || `HTTP ${res.status}` };
  }
  return res.json();
}

export async function restartServer(): Promise<{
  restarting: boolean;
  supported: boolean;
}> {
  const res = await authFetch("/api/settings/server/restart", {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error("Failed to restart server");
  }
  return res.json();
}
