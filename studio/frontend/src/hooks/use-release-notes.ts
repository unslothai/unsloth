// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken, hasAuthToken } from "@/features/auth";
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

// Desktop auto-auth installs its token after first paint, so a popup shown at
// startup can ask before one exists. Wait briefly instead of recording a
// failure the user would have to clear by hand.
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
  // Defensive: a response for another version is not usable here.
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
  const token = getAuthToken();
  if (!token) {
    return null;
  }

  const headers = new Headers();
  headers.set("Authorization", `Bearer ${token}`);
  const query = `version=${encodeURIComponent(version)}${refresh ? "&refresh=true" : ""}`;
  const res = await fetch(apiUrl(`/api/studio/release-notes?${query}`), {
    headers,
  });
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

  const load = useCallback((target: string, refresh = false) => {
    requestedVersionRef.current = target;
    setState("loading");
    setNotes(null);
    fetchReleaseNotes(target, refresh)
      .then((next) => {
        // A newer request owns the state now.
        if (requestedVersionRef.current !== target) {
          return;
        }
        setNotes(next);
        // A reported failure is retryable; "no notes for this version" is not.
        const failed = !next || (!next.matched && next.error !== null);
        setState(failed ? "error" : "ready");
      })
      .catch(() => {
        if (requestedVersionRef.current === target) {
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
      // Bypass the cached remote failure, or retry cannot recover until it
      // expires.
      load(version, true);
    }
  }, [version, load]);

  // Never hand back another version's notes: on the render where `version`
  // changes, state still describes the previous one until the effect runs.
  const matchesVersion = notes !== null && notes.version === version;
  return {
    state: notes !== null && !matchesVersion ? "loading" : state,
    notes: matchesVersion ? notes : null,
    retry,
  };
}
