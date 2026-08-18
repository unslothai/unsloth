// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A chat-settings endpoint a test can hold open, so it can act while hydration
// is still in flight. `puts` records what the store wrote back.

export const settingsHttp = {
  settings: {} as Record<string, unknown>,
  puts: [] as Record<string, unknown>[],
  /** Resolve to let a held GET complete. */
  release: null as (() => void) | null,
  hold(): void {
    settingsHttp.gate = new Promise<void>((resolve) => {
      settingsHttp.release = resolve;
    });
  },
  gate: null as Promise<void> | null,
};

export async function authFetch(
  _url: string,
  init?: { method?: string; body?: string },
): Promise<Response> {
  if (init?.method === "PUT") {
    settingsHttp.puts.push(JSON.parse(init.body ?? "{}"));
  } else if (settingsHttp.gate) {
    await settingsHttp.gate;
  }
  return {
    ok: true,
    status: 200,
    json: async () => ({ settings: settingsHttp.settings }),
  } as Response;
}
