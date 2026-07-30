// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, hasAuthToken } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { useCallback, useEffect, useRef, useState } from "react";

// Keyed to one exact version, so a new update never pairs with older notes.
export interface ReleaseNotes {
  version: string;
  markdown: string | null;
  matched: boolean;
  truncated: boolean;
  source: string | null;
  releaseNotesUrl: string | null;
  // Set when the lookup itself failed, as opposed to a version with no notes.
  error: string | null;
}

export type ReleaseNotesState = "idle" | "loading" | "ready" | "error";

// Desktop auto-auth installs its token after first paint, so a startup popup can
// ask before one exists. Wait briefly rather than fail.
const AUTH_POLL_MS = 250;
const AUTH_POLL_LIMIT = 40;

interface UseReleaseNotesOptions {
  version: string | null | undefined;
  enabled?: boolean;
}

type ApiObject = Record<string, unknown>;

function stringOrNull(value: ApiObject, key: string): string | null {
  const field = value[key];
  return typeof field === "string" && field.length > 0 ? field : null;
}

function toReleaseNotes(value: unknown, version: string): ReleaseNotes | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const payload = value as ApiObject;
  const notesVersion = stringOrNull(payload, "version");
  // A response for another version is not usable here.
  if (notesVersion !== version) {
    return null;
  }
  const markdown = stringOrNull(payload, "markdown");
  return {
    version,
    markdown,
    matched: payload.matched === true && markdown !== null,
    truncated: payload.truncated === true,
    source: stringOrNull(payload, "source"),
    releaseNotesUrl: stringOrNull(payload, "release_notes_url"),
    error: stringOrNull(payload, "error"),
  };
}

async function fetchReleaseNotes(
  version: string,
  refresh = false,
): Promise<ReleaseNotes | null> {
  const query = `version=${encodeURIComponent(version)}${refresh ? "&refresh=true" : ""}`;
  // authFetch, not fetch: an expired token is refreshed and retried.
  const res = await authFetch(apiUrl(`/api/studio/release-notes?${query}`));
  if (!res.ok) {
    throw new Error(`Release notes request failed: ${res.status}`);
  }

  return toReleaseNotes(await res.json(), version);
}

export function useReleaseNotes({
  version,
  enabled = true,
}: UseReleaseNotesOptions) {
  const [state, setState] = useState<ReleaseNotesState>("idle");
  const [notes, setNotes] = useState<ReleaseNotes | null>(null);
  // Version the current state belongs to; a change invalidates it.
  const requestedVersionRef = useRef<string | null>(null);
  // Identifies one request, so an earlier response cannot overwrite a later one.
  const requestIdRef = useRef(0);

  const load = useCallback((target: string, refresh = false) => {
    requestedVersionRef.current = target;
    requestIdRef.current += 1;
    const requestId = requestIdRef.current;
    setState("loading");
    setNotes(null);
    fetchReleaseNotes(target, refresh)
      .then((next) => {
        // A newer request owns the state now.
        if (requestIdRef.current !== requestId) {
          return;
        }
        setNotes(next);
        // A reported failure is retryable; "no notes for this version" is not.
        const failed = !next || (!next.matched && next.error !== null);
        setState(failed ? "error" : "ready");
      })
      .catch(() => {
        if (requestIdRef.current === requestId) {
          setNotes(null);
          setState("error");
        }
      });
  }, []);

  useEffect(() => {
    if (!enabled || !version || requestedVersionRef.current === version) {
      return;
    }
    if (hasAuthToken()) {
      load(version);
      return;
    }
    let attempts = 0;
    const timer = window.setInterval(() => {
      attempts += 1;
      if (hasAuthToken() || attempts >= AUTH_POLL_LIMIT) {
        window.clearInterval(timer);
        // Out of patience: load anyway so the panel settles on retry.
        load(version);
      }
    }, AUTH_POLL_MS);
    return () => window.clearInterval(timer);
  }, [enabled, version, load]);

  const retry = useCallback(() => {
    if (version) {
      requestedVersionRef.current = null;
      // Bypass the cached remote failure, or retry waits for it to expire.
      load(version, true);
    }
  }, [version, load]);

  // Never hand back another version's notes: state lags `version` by a render.
  const matchesVersion = notes !== null && notes.version === version;
  return {
    state: notes !== null && !matchesVersion ? "loading" : state,
    notes: matchesVersion ? notes : null,
    retry,
  };
}
