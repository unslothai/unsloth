// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** How the stream openers call authFetch, narrowed to what this helper forwards. */
export type StreamFetcher = (
  url: string,
  init: RequestInit,
  options?: { retryNetworkErrors?: boolean },
) => Promise<Response>;

/**
 * Open an event stream over POST, retrying once as GET on 405.
 *
 * POST is the verb that survives a Cloudflare quick tunnel, which holds a streamed GET
 * until it closes. Every current route answers both, so the retry never fires against a
 * matching backend. It exists because the desktop app ships its own SPA but updates the
 * Python backend in a separate step, so a newer UI can meet a backend that only
 * registered GET and would otherwise 405 on every stream until the user finishes the
 * backend update. That pairing is always loopback, where a streamed GET is fine.
 *
 * Only 405 retries: these routes answer 404 for an unknown job, and retrying that would
 * double every miss.
 */
export async function openStreamResponse(
  fetcher: StreamFetcher,
  url: string,
  init: RequestInit = {},
  options?: { retryNetworkErrors?: boolean },
): Promise<Response> {
  const response = await fetcher(url, { ...init, method: "POST" }, options);
  if (response.status !== 405) {
    return response;
  }
  return fetcher(url, { ...init, method: "GET" }, options);
}
