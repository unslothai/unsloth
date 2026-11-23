// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Split out of chat-api so it imports without the auth barrel; bounded so the picker's expander
// never hangs on "Loading variants".

import {
  type PollSignal,
  disposableTimeoutSignal,
  pollSignal,
  withAbort,
} from "@/features/hub/lib/abort-signals";

/** Matches the Hub client's bound on the same listing (features/hub/inventory/api.ts). */
export const GGUF_VARIANTS_TIMEOUT_MS = 30_000;

export interface GgufVariantsRequestOptions {
  /** Answer from disk without Hub discovery or authorization probes. */
  localOnly?: boolean;
  preferLocalCache?: boolean;
  includeCacheLocations?: boolean;
  localPath?: string | null;
  signal?: AbortSignal;
}

/** Query for GET /api/models/gguf-variants. A Hub already known to be unreachable is asked for the
 *  cached answer instead of a remote listing that cannot arrive, as the Hub client does. */
export function ggufVariantsQuery(
  repoId: string,
  options: GgufVariantsRequestOptions | undefined,
  offline: boolean,
): URLSearchParams {
  const localOnly = options?.localOnly === true || offline;
  const params = new URLSearchParams({ repo_id: repoId });
  // Chat resolves logical quants across remembered folders. Media callers opt out.
  if (options?.includeCacheLocations !== false) {
    params.set("include_cache_locations", "true");
  }
  if (options?.preferLocalCache || localOnly) {
    params.set("prefer_local_cache", "true");
  }
  const localPath = options?.localPath?.trim();
  if (localPath) {
    params.set("local_path", localPath);
  }
  if (localOnly) {
    params.set("offline", "true");
  }
  return params;
}

/** Signal every variant request carries: the caller's abort (the expander drops the request when
 *  its row collapses) folded with the timeout. Callers MUST dispose once settled. */
export function ggufVariantsAbort(signal?: AbortSignal): PollSignal {
  return signal
    ? pollSignal(signal, GGUF_VARIANTS_TIMEOUT_MS)
    : disposableTimeoutSignal(GGUF_VARIANTS_TIMEOUT_MS);
}

/** Settles on the bound regardless: on a 401 authFetch awaits a shared refresh with no signal. */
export function runBoundedVariantsRequest<T>(
  signal: AbortSignal | undefined,
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const abort = ggufVariantsAbort(signal);
  let started: Promise<T>;
  try {
    started = request(abort.signal);
  } catch (err) {
    abort.dispose();
    return Promise.reject(err);
  }
  return withAbort(started, abort.signal).finally(() => abort.dispose());
}
