// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useCallback, useEffect, useRef, useState } from "react";

export interface LlamaUpdateChangeLink {
  label: string;
  url: string;
}

export interface LlamaUpdateChange {
  summary: string;
  links: LlamaUpdateChangeLink[];
}

interface LlamaUpdateChangelog {
  matched: boolean;
  installedTag: string | null;
  latestTag: string | null;
  changes: LlamaUpdateChange[];
  totalChanges: number;
  truncated: boolean;
  releaseUrl: string | null;
  error: string | null;
}

// "unavailable" is a definitive answer, not a failure to get one: this pair can
// never be compared, so a Retry would spend two lookups on the same conclusion.
export type LlamaUpdateChangelogState =
  | "idle"
  | "loading"
  | "ready"
  | "unavailable"
  | "error";

// Predates the itemised body format; tracks a repo with per-release notes.
const PERMANENT_ERRORS = new Set([
  "notes_not_itemised",
  "notes_not_comparable",
]);

function githubUrl(value: unknown): string | null {
  return typeof value === "string" && value.startsWith("https://github.com/")
    ? value
    : null;
}

function parseChangelog(value: unknown): LlamaUpdateChangelog | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const payload = value as Record<string, unknown>;
  const rawChanges = Array.isArray(payload.changes) ? payload.changes : [];
  const changes = rawChanges.flatMap((raw): LlamaUpdateChange[] => {
    if (!raw || typeof raw !== "object") {
      return [];
    }
    const item = raw as Record<string, unknown>;
    if (typeof item.summary !== "string" || item.summary.length === 0) {
      return [];
    }
    const rawLinks = Array.isArray(item.links) ? item.links : [];
    const links = rawLinks.flatMap((rawLink): LlamaUpdateChangeLink[] => {
      if (!rawLink || typeof rawLink !== "object") {
        return [];
      }
      const link = rawLink as Record<string, unknown>;
      const url = githubUrl(link.url);
      return typeof link.label === "string" && link.label && url
        ? [{ label: link.label, url }]
        : [];
    });
    return [{ summary: item.summary, links }];
  });
  return {
    matched: payload.matched === true,
    installedTag:
      typeof payload.installed_tag === "string" ? payload.installed_tag : null,
    latestTag:
      typeof payload.latest_tag === "string" ? payload.latest_tag : null,
    changes,
    totalChanges:
      typeof payload.total_changes === "number"
        ? Math.max(0, payload.total_changes)
        : changes.length,
    truncated: payload.truncated === true,
    releaseUrl: githubUrl(payload.release_url),
    error: typeof payload.error === "string" ? payload.error : null,
  };
}

export function useLlamaUpdateChangelog({
  enabled,
  installedTag,
  latestTag,
}: {
  enabled: boolean;
  installedTag: string | null | undefined;
  latestTag: string | null | undefined;
}) {
  const [state, setState] = useState<LlamaUpdateChangelogState>("idle");
  const [changelog, setChangelog] = useState<LlamaUpdateChangelog | null>(null);
  const requestKeyRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);
  const key =
    installedTag && latestTag ? `${installedTag}\0${latestTag}` : null;

  const load = useCallback(
    (refresh = false) => {
      if (!(key && installedTag && latestTag)) {
        return;
      }
      requestKeyRef.current = key;
      requestIdRef.current += 1;
      const requestId = requestIdRef.current;
      setState("loading");
      setChangelog(null);
      // Name the pair being displayed: another surface's forced status check
      // advances the backend's shared memo, and the check below rejects the
      // answer that would come back about a target this banner has not adopted.
      const query = new URLSearchParams();
      query.set("installed_tag", installedTag);
      query.set("latest_tag", latestTag);
      if (refresh) {
        query.set("force_refresh", "true");
      }
      authFetch(`/api/llama/update-changelog?${query}`)
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`Changelog request failed: ${response.status}`);
          }
          const parsed = parseChangelog(await response.json());
          if (
            !parsed ||
            parsed.installedTag !== installedTag ||
            parsed.latestTag !== latestTag
          ) {
            throw new Error("Changelog response no longer matches this update");
          }
          if (requestIdRef.current !== requestId) {
            return;
          }
          setChangelog(parsed);
          setState(
            parsed.matched
              ? "ready"
              : parsed.error && PERMANENT_ERRORS.has(parsed.error)
                ? "unavailable"
                : "error",
          );
        })
        .catch(() => {
          if (requestIdRef.current !== requestId) {
            return;
          }
          setChangelog(null);
          setState("error");
        });
    },
    [installedTag, key, latestTag],
  );

  useEffect(() => {
    if (!(enabled && key) || requestKeyRef.current === key) {
      return;
    }
    load();
  }, [enabled, key, load]);

  const retry = useCallback(() => {
    requestKeyRef.current = null;
    load(true);
  }, [load]);

  return { state, changelog, retry };
}
