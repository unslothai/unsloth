// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useSyncExternalStore } from "react";

export type SkillRecord = {
  name: string;
  description: string;
  source: "agents" | "claude";
  enabled: boolean;
  valid: boolean;
  shadowed: boolean;
  shadowed_by?: "agents" | "claude" | null;
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
      if (generation === requestGeneration) {
        publish({ skills, loading: false, initialized: true, error: null });
      }
      return skills;
    })
    .catch((error: unknown) => {
      if (generation === requestGeneration) {
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
