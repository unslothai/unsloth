// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * API helpers for the local Codex SDK provider.
 *
 * The backend exposes two endpoints under ``/api/codex``:
 *
 * - ``GET /api/codex/status`` returns ``{installed, logged_in, version,
 *   cli_path, sdk_importable, supported_models}``. The chat-providers
 *   dialog consults this BEFORE surfacing the "codex" entry in the
 *   picker -- if ``installed`` is false the provider stays hidden, if
 *   ``logged_in`` is false we render a "Sign in to Codex" button in
 *   place of the API-key field.
 *
 * - ``POST /api/codex/login`` runs ``codex auth login --device-auth``
 *   under the hood and streams SSE events. The first event is always
 *   ``{type: "device_url", url}`` so the UI can window.open it; later
 *   events forward CLI log lines so the user can watch progress.
 */

import { authFetch } from "@/features/auth";

export interface CodexStatus {
  installed: boolean;
  logged_in: boolean;
  cli_path: string | null;
  sdk_importable: boolean;
  version: string | null;
  supported_models: string[];
}

export interface CodexLoginEvent {
  type: "device_url" | "log" | "error" | "done";
  url?: string;
  line?: string;
  message?: string;
  ok?: boolean;
  return_code?: number;
}

const DEFAULT_STATUS: CodexStatus = {
  installed: false,
  logged_in: false,
  cli_path: null,
  sdk_importable: false,
  version: null,
  supported_models: [],
};

/**
 * Fetch the current Codex availability snapshot. Network failures are
 * swallowed and reported as ``installed=false`` because every caller
 * either uses this to gate UI surfacing (the right answer on error is
 * "hide the entry") or kicks off a chat (the right answer on error is
 * "fall back to a different provider"). Throwing here would force
 * every consumer to wrap the call in a try/catch.
 */
export async function fetchCodexStatus(): Promise<CodexStatus> {
  try {
    const response = await authFetch("/api/codex/status");
    if (!response.ok) {
      return DEFAULT_STATUS;
    }
    const body = (await response.json()) as Partial<CodexStatus>;
    return {
      installed: Boolean(body.installed),
      logged_in: Boolean(body.logged_in),
      cli_path: typeof body.cli_path === "string" ? body.cli_path : null,
      sdk_importable: Boolean(body.sdk_importable),
      version: typeof body.version === "string" ? body.version : null,
      supported_models: Array.isArray(body.supported_models)
        ? body.supported_models.filter(
            (value): value is string => typeof value === "string",
          )
        : [],
    };
  } catch {
    return DEFAULT_STATUS;
  }
}

/**
 * Open a Codex device-auth login stream and yield each parsed event.
 *
 * Returns an async generator the caller drives in a for-await loop --
 * the dialog reads the first ``device_url`` event to know what URL to
 * window.open, then collects the remaining ``log`` lines into the
 * visible progress area until the ``done`` sentinel arrives.
 *
 * The generator handles abort signals: when the dialog closes mid-
 * flow, the caller passes an AbortSignal that tears down the SSE
 * stream cleanly. Without this, the long-running login subprocess
 * would keep pumping lines into a torn-down React tree.
 */
export async function* streamCodexDeviceLogin(
  signal?: AbortSignal,
): AsyncGenerator<CodexLoginEvent, void, void> {
  const response = await authFetch("/api/codex/login", {
    method: "POST",
    signal,
  });
  if (!response.ok || !response.body) {
    yield {
      type: "error",
      message: `codex login request failed: HTTP ${response.status}`,
    };
    yield { type: "done", ok: false };
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newlineIdx;
      // SSE frames are separated by blank lines; within each frame the
      // payload sits on a single ``data: {...}`` line. Strip both the
      // SSE prefix and the [DONE] sentinel before JSON.parse.
      while ((newlineIdx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, newlineIdx);
        buffer = buffer.slice(newlineIdx + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const body = line.slice("data:".length).trim();
          if (!body || body === "[DONE]") continue;
          try {
            yield JSON.parse(body) as CodexLoginEvent;
          } catch {
            // Skip any malformed line. The CLI shouldn't ever produce
            // these, but it costs nothing to defend against.
          }
        }
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}
