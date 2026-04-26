// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ApiScheme, setApiBase } from "@/lib/api-base";

interface ProbeResult {
  scheme: ApiScheme;
  port: number;
  instanceId: string | null;
}

interface HealthPayload {
  status?: string;
  service?: string;
  instance_id?: string | null;
}

/**
 * Read /api/health once. Returns the parsed payload + which scheme
 * answered. The HTTPS probe runs first so an SSL backend is detected
 * even when the page itself is on HTTP (e.g. just-flipped-on flow).
 * The HTTP probe is the fallback for backends running plain HTTP.
 */
export async function fetchBackendHealth(options: {
  port: number;
  host?: string;
  timeoutMs?: number;
}): Promise<ProbeResult | null> {
  const host = options.host ?? defaultProbeHost();
  const timeoutMs = options.timeoutMs ?? 1500;
  for (const scheme of ["https", "http"] as ApiScheme[]) {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      const url = `${scheme}://${host}:${options.port}/api/health`;
      const res = await fetch(url, { signal: ctrl.signal });
      clearTimeout(t);
      if (!res.ok) {
        continue;
      }
      const json = (await res.json()) as HealthPayload;
      if (json.status !== "healthy" || json.service !== "Unsloth UI Backend") {
        continue;
      }
      return {
        scheme,
        port: options.port,
        instanceId: json.instance_id ?? null,
      };
    } catch {
      // try next scheme
    }
  }
  return null;
}

/**
 * Default polling host. In browser mode this must match whatever the user
 * has open in the address bar — self-signed cert exceptions are tracked
 * per-hostname by browsers, so probing 127.0.0.1 from a localhost tab
 * (or vice-versa) gets blocked even when the cert SAN covers both.
 *
 * Tauri keeps 127.0.0.1 because its webview origin (`tauri://...`) is
 * unrelated to the backend address.
 */
function defaultProbeHost(): string {
  if (typeof window === "undefined") {
    return "127.0.0.1";
  }
  const isTauri = "__TAURI__" in window;
  if (isTauri) {
    return "127.0.0.1";
  }
  return window.location.hostname || "127.0.0.1";
}

/**
 * Race http:// and https:// health probes against the same port until one
 * responds 200. Used after a server restart triggered by the settings UI to
 * detect the new scheme without needing the answer up front.
 *
 * Each scheme is probed once per cycle. The poll exits early on the first
 * success so the UI can switch over fast; on timeout it returns null and
 * the caller surfaces the recovery instructions.
 */
export async function pollUntilBackendReady(options: {
  port: number;
  host?: string;
  timeoutMs?: number;
  intervalMs?: number;
  signal?: AbortSignal;
  /**
   * The instance_id of the backend before the restart was triggered.
   * Polling will only return a healthy probe whose `instance_id`
   * differs from this value — that's the unambiguous signal that the
   * Python process was actually replaced. When undefined or null
   * (e.g. we couldn't read it pre-restart, or the backend version
   * doesn't expose it), the poller falls back to the legacy "saw the
   * server unreachable at least once" heuristic.
   */
  previousInstanceId?: string | null;
}): Promise<ProbeResult | null> {
  const host = options.host ?? defaultProbeHost();
  // First-time restarts on a fresh box can spend 30-60s in matplotlib font
  // cache build, hardware detect, and the GGUF pre-cache thread before the
  // /api/health route is reachable. 90s covers that with headroom.
  const timeoutMs = options.timeoutMs ?? 90_000;
  const intervalMs = options.intervalMs ?? 1_000;
  const deadline = Date.now() + timeoutMs;

  const previousInstanceId = options.previousInstanceId ?? null;
  // Fallback heuristic for backends that don't expose instance_id, or
  // when we couldn't read it before the restart.
  let sawDown = false;

  while (Date.now() < deadline) {
    if (options.signal?.aborted) {
      return null;
    }

    const probe = await fetchBackendHealth({
      port: options.port,
      host,
      timeoutMs: Math.min(intervalMs, 800),
    });

    if (probe) {
      const idShifted =
        previousInstanceId !== null &&
        probe.instanceId !== null &&
        probe.instanceId !== previousInstanceId;
      if (idShifted) {
        return probe;
      }
      // Same instance_id (or backend can't tell us) — only believe in
      // the restart if we previously saw the server go down.
      if (
        sawDown &&
        (probe.instanceId === null || previousInstanceId === null)
      ) {
        return probe;
      }
    } else {
      sawDown = true;
    }

    await new Promise((r) => setTimeout(r, intervalMs));
  }

  return null;
}

/**
 * Apply the discovered scheme to the API client. Returns whether the
 * caller should reload — only true when the page itself is on the wrong
 * scheme (HTTP↔HTTPS flipped during the restart) and a redirect is the
 * only way to get there. When the scheme is unchanged we just update
 * the API base; in-flight `authFetch` requests parked on the restart
 * gate will resume against the new server without a page bounce, so
 * modals and other in-memory UI state stay intact.
 */
export function applyDiscoveredBackend(result: ProbeResult): {
  reloaded: boolean;
} {
  setApiBase(result.port, result.scheme);
  if (typeof window === "undefined") {
    return { reloaded: false };
  }
  const isTauri = "__TAURI__" in window;
  if (isTauri) {
    return { reloaded: false }; // Tauri uses absolute URLs, no reload needed.
  }

  const currentScheme =
    window.location.protocol === "https:" ? "https" : "http";
  if (currentScheme !== result.scheme) {
    const target = `${result.scheme}://${window.location.host}${window.location.pathname}${window.location.search}${window.location.hash}`;
    window.location.replace(target);
    return { reloaded: true };
  }
  return { reloaded: false };
}
