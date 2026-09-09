// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";

export const PROJECT_HOOK_EVENTS = [
  "SessionStart",
  "SessionEnd",
  "UserPromptSubmit",
  "PreToolUse",
  "PermissionRequest",
  "PostToolUse",
  "PreCompact",
  "PostCompact",
  "SubagentStart",
  "SubagentStop",
  "Stop",
] as const;

export type ProjectHookEvent = (typeof PROJECT_HOOK_EVENTS)[number];

export interface ProjectHookHandler {
  id: string;
  type: "command";
  command: string;
  commandWindows: string | null;
  timeout: number;
  statusMessage: string | null;
  additionalContextLimit: number;
  async: boolean;
  /** Saved user preference for this handler. */
  enabled: boolean;
  /** Effective state after exact-file trust and runtime policy are applied. */
  active: boolean;
}

export interface ProjectHookGroup {
  matcher: string | null;
  hooks: ProjectHookHandler[];
}

export interface ProjectHooks {
  execution?: { available: boolean; backend: string | null; reason: string | null };
  workspaceAvailable?: boolean;
  workspaceRevision: number | null;
  sourcePath: string;
  exists: boolean;
  contentHash: string | null;
  description: string | null;
  groupCount: number;
  handlerCount: number;
  trusted: boolean;
  revision: number;
  storedTrust?: ProjectHooksStoredTrust | null;
  hooks: Record<ProjectHookEvent, ProjectHookGroup[]>;
}

export interface ProjectHooksStoredTrust {
  contentHash: string;
  revision: number;
}

export interface ProjectHooksTrustState {
  storedTrust: ProjectHooksStoredTrust | null;
  revision: number;
}

export interface ProjectHooksTrustIdentity {
  projectId: string;
  workspaceRevision: number;
  contentHash: string;
  revision: number;
}

type ProjectHooksTrustRequest = {
  workspaceRevision: number;
  contentHash: string;
  revision: number;
};

type ProjectHooksRevisionRequest = {
  revision: number;
};

type ProjectHookHandlerRequest = ProjectHooksTrustRequest & {
  enabled: boolean;
};

function projectHooksPath(projectId: string, suffix = ""): string {
  const base = `/api/agent/projects/${encodeURIComponent(projectId)}/hooks`;
  return suffix ? `${base}/${suffix}` : base;
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(input, init);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      formatApiErrorBody(body) ??
        `Project hooks request failed (${response.status})`,
    );
  }
  return body as T;
}

function post<T>(input: string, body: unknown): Promise<T> {
  return request<T>(input, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getProjectHooks(projectId: string): Promise<ProjectHooks> {
  return request(projectHooksPath(projectId), { cache: "no-store" });
}

export function getProjectHooksTrustState(
  projectId: string,
): Promise<ProjectHooksTrustState> {
  return request(projectHooksPath(projectId, "trust"), { cache: "no-store" });
}

export function trustProjectHooks(
  projectId: string,
  input: ProjectHooksTrustRequest,
): Promise<ProjectHooks> {
  return post(projectHooksPath(projectId, "trust"), input);
}

export function revokeProjectHooks(
  projectId: string,
  input: ProjectHooksRevisionRequest,
): Promise<ProjectHooks> {
  return post(projectHooksPath(projectId, "revoke"), input);
}

export function setProjectHookHandlerEnabled(
  projectId: string,
  handlerId: string,
  input: ProjectHookHandlerRequest,
): Promise<ProjectHooks> {
  return post(
    projectHooksPath(projectId, `handlers/${encodeURIComponent(handlerId)}`),
    input,
  );
}

const FORMAT_CONTROL = /\p{Cf}/u;

function visibleUnicodeEscape(codePoint: number): string {
  const hexadecimal = codePoint.toString(16).toUpperCase();
  return codePoint <= 0xffff
    ? `\\u${hexadecimal.padStart(4, "0")}`
    : `\\u{${hexadecimal}}`;
}

function visibleProjectHookCharacter(
  codePoint: number,
  character: string,
): string {
  if (codePoint === 0x5c) {
    return "\\\\";
  }
  if (codePoint === 0x0d) {
    return "\\r\n";
  }
  if (codePoint === 0x0a) {
    return "\\n\n";
  }
  if (codePoint === 0x09) {
    return "\\t";
  }
  if (
    codePoint < 0x20 ||
    (codePoint >= 0x7f && codePoint <= 0x9f) ||
    (codePoint >= 0xd800 && codePoint <= 0xdfff) ||
    FORMAT_CONTROL.test(character)
  ) {
    return visibleUnicodeEscape(codePoint);
  }
  return character;
}

/** Make untrusted hook text reviewable without invisible terminal controls. */
export function visibleProjectHookText(value: string | null): string {
  if (value === null) {
    return "";
  }
  let visible = "";
  for (let index = 0; index < value.length; ) {
    const codePoint = value.codePointAt(index);
    if (codePoint === undefined) {
      break;
    }
    const character = String.fromCodePoint(codePoint);
    index += character.length;

    if (codePoint === 0x0d && value.codePointAt(index) === 0x0a) {
      index += 1;
      visible += "\\r\\n\n";
      continue;
    }
    visible += visibleProjectHookCharacter(codePoint, character);
  }
  return visible;
}

export function projectHooksTrustIdentityMatches(
  identity: ProjectHooksTrustIdentity,
  current: {
    projectId: string;
    hooks: ProjectHooks | null;
  },
): boolean {
  return (
    current.hooks?.exists === true &&
    current.projectId === identity.projectId &&
    current.hooks.workspaceRevision === identity.workspaceRevision &&
    current.hooks.contentHash === identity.contentHash &&
    current.hooks.revision === identity.revision &&
    current.hooks.trusted === false
  );
}

/** Mount-scoped sequencing for reads and mutations that return full snapshots. */
export class ProjectHooksRequestGuard {
  private revision = 0;
  private mounted = false;

  activate(): void {
    this.mounted = true;
    this.revision += 1;
  }

  begin(): number {
    this.revision += 1;
    return this.revision;
  }

  accepts(revision: number): boolean {
    return this.mounted && revision === this.revision;
  }

  retire(): void {
    this.mounted = false;
    this.revision += 1;
  }
}
