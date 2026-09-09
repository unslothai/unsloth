// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";

export interface ProjectInstructionLayer {
  path: string;
  scope: string;
  content: string;
  truncated: boolean;
  bytesRead: number;
}

export interface ProjectInstructions {
  layers: ProjectInstructionLayer[];
  combined: string;
  truncated: boolean;
  issues: Array<{ path: string; reason: string }>;
  precedence: string;
  bytesRead: number;
}

export interface ProjectSkill {
  name: string;
  description: string;
  path: string;
  size: number;
  modifiedNs: number;
  ambiguous: boolean;
}

export interface ProjectSkills {
  skills: ProjectSkill[];
  issues: Array<{ path: string; reason: string }>;
  truncated: boolean;
}

export interface ProjectInitResult {
  status: "created" | "already_exists";
  created: boolean;
  path: "AGENTS.md" | string;
  instructions: ProjectInstructions;
}

export const PROJECT_GUIDANCE_UPDATED_EVENT =
  "unsloth-project-guidance-updated";

function projectGuidancePath(
  projectId: string,
  resource: "instructions" | "skills" | "init",
): string {
  return `/api/agent/projects/${encodeURIComponent(projectId)}/${resource}`;
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(input, init);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      formatApiErrorBody(body) ??
        `Project guidance request failed (${response.status})`,
    );
  }
  return body as T;
}

function notifyProjectGuidanceUpdated(projectId: string): void {
  if (
    typeof window === "undefined" ||
    typeof window.dispatchEvent !== "function"
  ) {
    return;
  }
  window.dispatchEvent(
    new CustomEvent(PROJECT_GUIDANCE_UPDATED_EVENT, {
      detail: { projectId },
    }),
  );
}

export function subscribeProjectGuidanceUpdated(
  projectId: string,
  listener: () => void,
): () => void {
  if (
    typeof window === "undefined" ||
    typeof window.addEventListener !== "function"
  ) {
    return () => {
      // No browser event target is available during server rendering.
    };
  }
  const handle = (event: Event) => {
    const detail = (event as CustomEvent<{ projectId?: string }>).detail;
    if (detail?.projectId === projectId) {
      listener();
    }
  };
  window.addEventListener(PROJECT_GUIDANCE_UPDATED_EVENT, handle);
  return () =>
    window.removeEventListener(PROJECT_GUIDANCE_UPDATED_EVENT, handle);
}

export function getProjectInstructions(
  projectId: string,
): Promise<ProjectInstructions> {
  return request(projectGuidancePath(projectId, "instructions"));
}

export function getProjectSkills(projectId: string): Promise<ProjectSkills> {
  return request(projectGuidancePath(projectId, "skills"));
}

export async function initializeProjectAgents(
  projectId: string,
): Promise<ProjectInitResult> {
  const result = await request<ProjectInitResult>(
    projectGuidancePath(projectId, "init"),
    { method: "POST" },
  );
  notifyProjectGuidanceUpdated(projectId);
  return result;
}
