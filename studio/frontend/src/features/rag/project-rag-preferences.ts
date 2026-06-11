// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RagSource } from "@/features/chat/stores/chat-runtime-store";

const PROJECT_RAG_SOURCE_PREFIX = "unsloth_project_rag_source:";

function key(projectId: string): string {
  return `${PROJECT_RAG_SOURCE_PREFIX}${projectId}`;
}

export function loadProjectRagSource(projectId: string): RagSource | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(key(projectId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as RagSource;
    if (parsed?.type === "thread") return { type: "thread" };
    if (parsed?.type === "project" && parsed.projectId === projectId) {
      return { type: "project", projectId };
    }
    if (parsed?.type === "kb" && typeof parsed.kbId === "string") {
      return { type: "kb", kbId: parsed.kbId };
    }
  } catch {
  }
  return null;
}

export function saveProjectRagSource(projectId: string, source: RagSource): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key(projectId), JSON.stringify(source));
  } catch {
  }
}

export function hasProjectRagSourcePreference(projectId: string): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(key(projectId)) !== null;
  } catch {
    return false;
  }
}
