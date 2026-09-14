// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useSyncExternalStore } from "react";

export type SkillRecord = {
  name: string;
  description: string;
  source: "agents" | "claude" | "bundled";
  enabled: boolean;
  valid: boolean;
  shadowed: boolean;
  shadowed_by?: "agents" | "claude" | "bundled" | null;
  error?: string | null;
  license?: string | null;
  compatibility?: string | null;
  metadata?: Record<string, string> | null;
  allowed_tools?: string | null;
};

type SkillsSnapshot = {
  skills: readonly SkillRecord[];
  loading: boolean;
  initialized: boolean;
  error: string | null;
};

const EMPTY_SNAPSHOT: SkillsSnapshot = {
  skills: [],
  loading: false,
  initialized: false,
  error: null,
};
let snapshot = EMPTY_SNAPSHOT;
let requestGeneration = 0;
let lastFetchedAt = 0;
let pending: Promise<readonly SkillRecord[]> | null = null;
const listeners = new Set<() => void>();
const channel =
  typeof BroadcastChannel === "undefined"
    ? null
    : new BroadcastChannel("unsloth-agent-skills");

function publish(next: SkillsSnapshot): void {
  snapshot = next;
  for (const listener of listeners) listener();
}

async function parseResponse<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String(body.detail)
        : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body as T;
}

export function subscribeSkills(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getSkillsSnapshot(): SkillsSnapshot {
  return snapshot;
}

export function listSkills(force = false): Promise<readonly SkillRecord[]> {
  if (pending && !force) return pending;
  const generation = ++requestGeneration;
  publish({ ...snapshot, loading: true, error: null });
  const request = authFetch("/api/skills")
    .then((response) => parseResponse<SkillRecord[]>(response))
    .then((skills) => {
      if (!Array.isArray(skills)) {
        throw new Error("Could not load Agent Skills.");
      }
      return skills;
    })
    .then((skills) => {
      if (generation === requestGeneration) {
        lastFetchedAt = Date.now();
        publish({ skills, loading: false, initialized: true, error: null });
      }
      return skills;
    })
    .catch((error: unknown) => {
      if (generation === requestGeneration) {
        lastFetchedAt = Date.now();
        publish({
          ...snapshot,
          loading: false,
          initialized: true,
          error:
            error instanceof Error
              ? error.message
              : "Could not load Agent Skills.",
        });
      }
      throw error;
    })
    .finally(() => {
      if (pending === request) pending = null;
    });
  pending = request;
  return request;
}

export async function setSkillEnabled(
  name: string,
  enabled: boolean,
): Promise<SkillRecord> {
  const response = await authFetch(
    `/api/skills/${encodeURIComponent(name)}/enabled`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    },
  );
  const updated = await parseResponse<SkillRecord>(response);
  requestGeneration += 1;
  publish({
    skills: snapshot.skills.map((skill) =>
      skill.name === updated.name && !skill.shadowed
        ? { ...skill, enabled: updated.enabled }
        : skill,
    ),
    loading: false,
    initialized: true,
    error: null,
  });
  channel?.postMessage("changed");
  void import("../utils/refresh-context-usage").then(
    ({ refreshContextUsage }) => refreshContextUsage({ invalidate: true }),
  );
  return updated;
}

// A skill name per the Agent Skills spec (lowercase, digits, single hyphens, 1-64 chars), and
// only when the name ends where a word would: trailing punctuation is fine (`@probe-alpha.`),
// but `@example.com`, `@3pm`, `@probe_alpha` and `@Probe` are not mentions, so they never
// trigger a catalog re-read on send.
export const SKILL_MENTION_PATTERN =
  /(^|\s)@([a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)(?=$|\s|[.,;:!?)\]'"]+(?:$|\s))/g;

// How long a pre-send catalog re-read may hold the request. authFetch has no deadline of its
// own, so without this a hung /api/skills would stall every send that names an unknown skill.
const SETTLE_TIMEOUT_MS = 3000;

// Settle the catalog before a request decides tool enablement from it: finish any fetch
// already in flight, and re-read the folders when the text names a skill the snapshot has
// never seen (a pasted @mention gets no bare-@ keystroke to refresh on).
export async function settleSkillsForText(text: string): Promise<void> {
  const deadline = new Promise<void>((resolve) =>
    setTimeout(resolve, SETTLE_TIMEOUT_MS),
  );
  if (pending) await Promise.race([pending.catch(() => undefined), deadline]);
  // Usable entries only: a mention of a skill the snapshot holds as invalid, shadowed or
  // disabled re-reads too, since the file may have been repaired or re-enabled since.
  const known = new Set(
    snapshot.skills
      .filter((skill) => skill.valid && !skill.shadowed && skill.enabled)
      .map((skill) => skill.name),
  );
  let stale = !snapshot.initialized;
  for (const match of text.matchAll(SKILL_MENTION_PATTERN)) {
    const name = match[2] ?? "";
    if (!known.has(name)) {
      stale = true;
      break;
    }
  }
  if (stale) {
    await Promise.race([listSkills(true).catch(() => undefined), deadline]);
  }
}

// Skills are files the user edits while Studio is open, so the places that surface the
// catalog (the dialog, an @ mention) re-read it instead of trusting the page-load snapshot.
// Throttled: a burst of @ keystrokes costs one request.
export function refreshSkillsCatalog(maxAgeMs = 1500): void {
  if (pending || Date.now() - lastFetchedAt < maxAgeMs) return;
  void listSkills(true).catch(() => undefined);
}

export function useSkillsCatalog(): SkillsSnapshot {
  const value = useSyncExternalStore(
    subscribeSkills,
    getSkillsSnapshot,
    () => EMPTY_SNAPSHOT,
  );
  useEffect(() => {
    if (!value.initialized && !value.loading)
      void listSkills().catch(() => undefined);
  }, [value.initialized, value.loading]);
  return value;
}

channel?.addEventListener("message", () => {
  void listSkills(true).catch(() => undefined);
  void import("../utils/refresh-context-usage").then(
    ({ refreshContextUsage }) => refreshContextUsage({ invalidate: true }),
  );
});
