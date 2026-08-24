// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A chat-settings endpoint a test can hold open, so it can act while hydration
// is still in flight. `puts` records what the store wrote back.

export const settingsHttp = {
  settings: {} as Record<string, unknown>,
  /** Optional one-response-per-GET sequence for stale-read race tests. */
  getResponses: [] as Array<
    Record<string, unknown> | Promise<Record<string, unknown>>
  >,
  gets: 0,
  beforeConditionalApply: null as (() => void) | null,
  puts: [] as Record<string, unknown>[],
  /** One-shot failures for ordinary PUT ordering/retry tests. */
  putFailures: [] as Array<{ status: number; detail?: unknown }>,
  /** Resolve to let a held GET complete. */
  release: null as (() => void) | null,
  hold(): void {
    settingsHttp.gate = new Promise<void>((resolve) => {
      settingsHttp.release = resolve;
    });
  },
  gate: null as Promise<void> | null,
};

function matchesExpected(current: unknown, expected: unknown): boolean {
  if (expected && typeof expected === "object" && !Array.isArray(expected)) {
    if (!current || typeof current !== "object" || Array.isArray(current)) {
      return false;
    }
    return Object.entries(expected).every(
      ([key, value]) =>
        key in current &&
        matchesExpected((current as Record<string, unknown>)[key], value),
    );
  }
  return Object.is(current, expected);
}

function pathExists(current: unknown, path: string[]): boolean {
  let node = current;
  for (const segment of path) {
    if (
      node == null ||
      typeof node !== "object" ||
      Array.isArray(node) ||
      !(segment in node)
    ) {
      return false;
    }
    node = (node as Record<string, unknown>)[segment];
  }
  return true;
}

function deepMerge(
  current: Record<string, unknown>,
  patch: Record<string, unknown>,
): Record<string, unknown> {
  const merged = { ...current };
  for (const [key, value] of Object.entries(patch)) {
    const existing = merged[key];
    merged[key] =
      existing &&
      typeof existing === "object" &&
      !Array.isArray(existing) &&
      value &&
      typeof value === "object" &&
      !Array.isArray(value)
        ? deepMerge(
            existing as Record<string, unknown>,
            value as Record<string, unknown>,
          )
        : value;
  }
  return merged;
}

function nextPutFailureResponse(): Response | null {
  const failure = settingsHttp.putFailures.shift();
  if (!failure) return null;
  return {
    ok: false,
    status: failure.status,
    json: async () => ({ detail: failure.detail ?? "temporary failure" }),
  } as Response;
}

export async function authFetch(
  url: string,
  init?: { method?: string; body?: string },
): Promise<Response> {
  let responseSettings = settingsHttp.settings;
  let applied: boolean | undefined;
  if (url.endsWith("/compare-and-set") && init?.method === "POST") {
    const request = JSON.parse(init.body ?? "{}") as {
      expected: Record<string, unknown>;
      expectedAbsent?: string[];
      expectedAbsentPaths?: string[][];
      patch: Record<string, unknown>;
    };
    settingsHttp.beforeConditionalApply?.();
    settingsHttp.beforeConditionalApply = null;
    responseSettings = settingsHttp.settings;
    applied =
      (request.expectedAbsent ?? []).every(
        (key) => !(key in settingsHttp.settings),
      ) &&
      (request.expectedAbsentPaths ?? []).every(
        (path) => !pathExists(settingsHttp.settings, path),
      ) &&
      matchesExpected(settingsHttp.settings, request.expected);
    if (applied) {
      settingsHttp.puts.push(request.patch);
      settingsHttp.settings = deepMerge(settingsHttp.settings, request.patch);
      responseSettings = settingsHttp.settings;
    }
  } else if (init?.method === "PUT") {
    settingsHttp.puts.push(JSON.parse(init.body ?? "{}"));
    const failureResponse = nextPutFailureResponse();
    if (failureResponse) return failureResponse;
  } else {
    if (settingsHttp.gate) {
      await settingsHttp.gate;
    }
    settingsHttp.gets += 1;
    if (settingsHttp.getResponses.length > 0) {
      responseSettings = await (settingsHttp.getResponses.shift() ??
        settingsHttp.settings);
    }
  }
  return {
    ok: true,
    status: 200,
    json: async () => ({
      settings: responseSettings,
      ...(applied !== undefined ? { applied } : {}),
    }),
  } as Response;
}
