// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type StreamFetcher = (
  url: string,
  init: RequestInit,
  options?: { retryNetworkErrors?: boolean },
) => Promise<Response>;

/**
 * Quick tunnels hold a streamed GET until it closes, so POST is used. The GET retry on 405 only
 * covers a newer desktop UI with an older loopback backend. Not on 404, which would double misses.
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
