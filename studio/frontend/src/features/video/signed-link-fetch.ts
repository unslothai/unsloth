// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Fetch a clip through its signed link, minting a fresh one once if the server refuses it: a
 * restart changes the signing secret, so a link the player already holds can die at any time.
 */
export async function fetchWithFreshLink(
  url: string,
  mint: () => Promise<string>,
  fetchImpl: (url: string) => Promise<Response> = (target) => fetch(target),
): Promise<Response> {
  const response = await fetchImpl(url);
  if (response.status !== 401 && response.status !== 403) return response;
  void response.body?.cancel();
  return fetchImpl(await mint());
}
