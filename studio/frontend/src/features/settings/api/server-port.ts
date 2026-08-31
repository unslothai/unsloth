// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function parseCustomServerPort(value: string): number | null {
  if (!/^\d+$/.test(value.trim())) return null;
  const port = Number(value);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? port : null;
}

export async function loadServerPort(): Promise<number | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<number | null>("get_server_port");
}

export async function updateServerPort(port: number | null): Promise<number | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<number | null>("set_server_port", { port });
}
