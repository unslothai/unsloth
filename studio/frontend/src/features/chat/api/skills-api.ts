// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export interface AgentSkill {
  name: string;
  description: string;
  enabled: boolean;
  bundled: boolean;
  license?: string;
  compatibility?: string;
  metadata?: Record<string, string>;
}

let enabledSkillsCache: AgentSkill[] | null = null;
let enabledSkillsRequest: Promise<AgentSkill[]> | null = null;
let catalogRevision = 0;
let catalogChannel: BroadcastChannel | null | undefined;
const catalogListeners = new Set<() => void>();
const CATALOG_CHANNEL_NAME = "unsloth.skills.changed";

function resetSkillCatalogCache(): void {
  catalogRevision += 1;
  enabledSkillsCache = null;
  enabledSkillsRequest = null;
}

function notifySkillCatalogChanged(): void {
  resetSkillCatalogCache();
  for (const listener of catalogListeners) {
    listener();
  }
}

function getCatalogChannel(): BroadcastChannel | null {
  if (catalogChannel !== undefined) {
    return catalogChannel;
  }
  if (
    typeof window === "undefined" ||
    typeof BroadcastChannel === "undefined"
  ) {
    catalogChannel = null;
    return null;
  }
  catalogChannel = new BroadcastChannel(CATALOG_CHANNEL_NAME);
  (catalogChannel as { unref?: () => void }).unref?.();
  catalogChannel.onmessage = notifySkillCatalogChanged;
  return catalogChannel;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Request failed (${response.status})`);
  }
  return body as T;
}

function cacheEnabledSkills(skills: AgentSkill[]): void {
  enabledSkillsCache = skills.filter((skill) => skill.enabled);
}

export async function listSkills(): Promise<AgentSkill[]> {
  getCatalogChannel();
  while (true) {
    const revision = catalogRevision;
    const response = await authFetch("/api/skills");
    const body = await parseJsonOrThrow<{ skills: AgentSkill[] }>(response);
    if (revision !== catalogRevision) {
      continue;
    }
    cacheEnabledSkills(body.skills);
    return body.skills;
  }
}

export async function enabledSkills(): Promise<AgentSkill[]> {
  getCatalogChannel();
  if (enabledSkillsCache !== null) {
    return enabledSkillsCache;
  }
  if (enabledSkillsRequest) {
    return enabledSkillsRequest;
  }
  const request = listSkills()
    .then((skills) => skills.filter((skill) => skill.enabled))
    .finally(() => {
      if (enabledSkillsRequest === request) {
        enabledSkillsRequest = null;
      }
    });
  enabledSkillsRequest = request;
  return request;
}

export async function importSkillBundle(
  file: File,
  replace = false,
): Promise<AgentSkill> {
  const form = new FormData();
  form.append("file", file);
  const response = await authFetch(
    `/api/skills/import?replace=${replace ? "true" : "false"}`,
    { method: "POST", body: form },
  );
  const body = await parseJsonOrThrow<{ skill: AgentSkill }>(response);
  clearSkillCatalogCache();
  return body.skill;
}

export async function setSkillEnabled(
  name: string,
  enabled: boolean,
): Promise<AgentSkill> {
  const response = await authFetch(
    `/api/skills/${encodeURIComponent(name)}/enabled`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    },
  );
  const body = await parseJsonOrThrow<{ skill: AgentSkill }>(response);
  clearSkillCatalogCache();
  return body.skill;
}

export async function deleteSkill(name: string): Promise<void> {
  const response = await authFetch(`/api/skills/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Delete failed (${response.status})`);
  }
  clearSkillCatalogCache();
}

export async function hasEnabledSkills(): Promise<boolean> {
  try {
    return (await enabledSkills()).length > 0;
  } catch {
    return false;
  }
}

export function clearSkillCatalogCache(): void {
  notifySkillCatalogChanged();
  getCatalogChannel()?.postMessage("changed");
}

export function subscribeSkillCatalogChanges(listener: () => void): () => void {
  getCatalogChannel();
  catalogListeners.add(listener);
  return () => catalogListeners.delete(listener);
}
