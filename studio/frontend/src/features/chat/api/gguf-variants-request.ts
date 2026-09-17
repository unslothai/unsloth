// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How the chat side asks for a repo's GGUF quants. Split out of chat-api so it imports without
// the auth barrel. The picker's expander and chat auto-load both block on this listing, so an
// unbounded request leaves the expander on "Loading variants" forever.

import {
  type PollSignal,
  disposableTimeoutSignal,
  pollSignal,
  withAbort,
} from "@/features/hub/lib/abort-signals";

/** Matches the Hub client's bound on the same listing (features/hub/inventory/api.ts). */
export const GGUF_VARIANTS_TIMEOUT_MS = 30_000;

export interface GgufVariantsRequestOptions {
  preferLocalCache?: boolean;
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
  const params = new URLSearchParams({ repo_id: repoId });
  if (options?.preferLocalCache || offline) {
    params.set("prefer_local_cache", "true");
  }
  const localPath = options?.localPath?.trim();
  if (localPath) {
    params.set("local_path", localPath);
  }
  if (offline) {
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

/** Runs a listing under the bound and settles on it whatever the request is doing. Handing the
 *  signal to fetch alone is not enough: on a 401 authFetch awaits a shared session refresh that
 *  carries no signal, so the listing could still hang there. */
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
