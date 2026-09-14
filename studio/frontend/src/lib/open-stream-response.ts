// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type StreamFetcher = (
  url: string,
  init: RequestInit,
  options?: { retryNetworkErrors?: boolean },
) => Promise<Response>;

/**
 * Open an event stream over POST, retrying once as GET on 405.
 *
 * Quick tunnels hold a streamed GET until it closes, so POST is the verb that works.
 * The retry covers version skew only: the desktop app ships its own SPA but updates the
 * Python backend separately, so a newer UI can meet a GET-only backend. That pairing is
 * always loopback, where a streamed GET is fine.
 *
 * 405 only. These routes answer 404 for an unknown job, and retrying would double misses.
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
