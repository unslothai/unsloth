// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type AgentWorkspaceQueryValue =
  | string
  | number
  | boolean
  | null
  | undefined;

const AGENT_WORKSPACE_PROJECTS_PATH = "/api/agent-workspace/projects";
const GIT_OBJECT_ID_PATTERN = /^[0-9a-f]{40,64}$/i;
const CONNECTOR_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const REQUEST_DIGEST_PATTERN = /^[0-9a-f]{64}$/;

export function agentWorkspaceRequestPath(
  projectId: string,
  resource = "",
  query?: Readonly<Record<string, AgentWorkspaceQueryValue>>,
): string {
  const normalizedProjectId = projectId.trim();
  if (!normalizedProjectId) {
    throw new Error("Project id is required.");
  }
  const normalizedResource = resource.replace(/^\/+|\/+$/g, "");
  const path = `${AGENT_WORKSPACE_PROJECTS_PATH}/${encodeURIComponent(normalizedProjectId)}${normalizedResource ? `/${normalizedResource}` : ""}`;
  if (!query) return path;

  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) continue;
    params.set(key, String(value));
  }
  const encoded = params.toString();
  return encoded ? `${path}?${encoded}` : path;
}

export function agentWorkspaceJsonRequest(
  method: "POST" | "PUT" | "PATCH" | "DELETE",
  payload?: unknown,
): RequestInit {
  if (payload === undefined) return { method };
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

export interface AgentBackgroundAgentRequestPayload {
  instruction: string;
  runtime: AgentBackgroundRuntimeSelection;
  planId?: string;
  planTaskId?: string;
  worktreeId?: string;
  cleanupWorktreeOnCancel?: boolean;
  delegationPolicy?: AgentDelegationPolicy;
  start?: boolean;
}

export interface AgentDelegationPolicy {
  enabled: boolean;
  maxChildren: number;
  maxParallelChildren: number;
  maxDepth: 1;
  totalChildOutputTokens: number;
  totalChildToolCalls: number;
  totalChildWallSeconds: number;
}

export type AgentBackgroundRuntimeKind = "local" | "provider";
export type AgentBackgroundPermissionMode = "off" | "full";

export interface AgentBackgroundRuntimeSelection {
  kind: AgentBackgroundRuntimeKind;
  model: string;
  providerId?: string;
  permissionMode: AgentBackgroundPermissionMode;
  reasoningEffort?: string;
  maxOutputTokens: number;
}

function agentBackgroundRuntime(
  runtime: AgentBackgroundRuntimeSelection | null | undefined,
): AgentBackgroundRuntimeSelection {
  if (!runtime || typeof runtime !== "object") {
    throw new Error("A background agent runtime selection is required.");
  }
  const kind = runtime.kind;
  const model = runtime.model.trim();
  const providerId = runtime.providerId?.trim() || undefined;
  const reasoningEffort = runtime.reasoningEffort?.trim() || undefined;
  if (kind !== "local" && kind !== "provider") {
    throw new Error("Background agent runtime kind is invalid.");
  }
  if (!model || model.length > 512 || /[\0\r\n]/.test(model)) {
    throw new Error("Background agent model is invalid.");
  }
  if (runtime.permissionMode !== "off" && runtime.permissionMode !== "full") {
    throw new Error(
      "Background agents cannot use Ask or Auto without an interactive approval stream.",
    );
  }
  if (
    reasoningEffort &&
    (reasoningEffort.length > 64 || /[\0\r\n]/.test(reasoningEffort))
  ) {
    throw new Error("Background agent reasoning effort is invalid.");
  }
  if (
    !Number.isInteger(runtime.maxOutputTokens) ||
    runtime.maxOutputTokens < 1 ||
    runtime.maxOutputTokens > 32_768
  ) {
    throw new Error("Background agent output token limit is invalid.");
  }
  if (kind === "provider" && !providerId) {
    throw new Error("A saved provider connection is required.");
  }
  if (providerId && (providerId.length > 256 || /[\0\r\n]/.test(providerId))) {
    throw new Error("Background agent provider connection is invalid.");
  }
  if (kind === "local" && providerId) {
    throw new Error("A local runtime cannot name an external provider.");
  }
  return {
    kind,
    model,
    providerId,
    permissionMode: runtime.permissionMode,
    reasoningEffort,
    maxOutputTokens: runtime.maxOutputTokens,
  };
}

export function agentBackgroundAgentRequest(
  projectId: string,
  payload: AgentBackgroundAgentRequestPayload,
): { path: string; init: RequestInit } {
  const instruction = payload.instruction.trim();
  if (!instruction) {
    throw new Error("Agent task instructions are required.");
  }
  return {
    path: agentWorkspaceRequestPath(projectId, "background/agent"),
    init: agentWorkspaceJsonRequest("POST", {
      instruction,
      runtime: agentBackgroundRuntime(payload.runtime),
      planId: payload.planId || undefined,
      planTaskId: payload.planTaskId || undefined,
      worktreeId: payload.worktreeId || undefined,
      cleanupWorktreeOnCancel: payload.cleanupWorktreeOnCancel ?? false,
      delegationPolicy: payload.delegationPolicy ?? {
        enabled: false,
        maxChildren: 4,
        maxParallelChildren: 2,
        maxDepth: 1,
        totalChildOutputTokens: 32_768,
        totalChildToolCalls: 100,
        totalChildWallSeconds: 3_600,
      },
      start: payload.start ?? false,
    }),
  };
}

export function agentWorktreeMergeRequest(
  projectId: string,
  worktreeId: string,
  expectedTargetHead: string,
): { path: string; init: RequestInit } {
  const normalizedWorktreeId = worktreeId.trim();
  const normalizedHead = expectedTargetHead.trim();
  if (!normalizedWorktreeId) {
    throw new Error("Worktree id is required.");
  }
  if (!GIT_OBJECT_ID_PATTERN.test(normalizedHead)) {
    throw new Error("Expected target head is invalid.");
  }
  return {
    path: agentWorkspaceRequestPath(
      projectId,
      `worktrees/${encodeURIComponent(normalizedWorktreeId)}/merge`,
    ),
    init: agentWorkspaceJsonRequest("POST", {
      expectedTargetHead: normalizedHead,
    }),
  };
}

function preparedOperationId(value: string, label: string): string {
  const normalized = value.trim();
  if (!normalized || normalized.length > 256) {
    throw new Error(`${label} is invalid.`);
  }
  return normalized;
}

function confirmationToken(value: string): string {
  if (value.length < 32 || value.length > 256) {
    throw new Error("Confirmation token is invalid.");
  }
  return value;
}

export function agentPreparedCommitRequest(
  projectId: string,
  payload: { ownedPaths: string[]; message: string },
): { path: string; init: RequestInit } {
  const message = payload.message.trim();
  const ownedPaths = [...new Set(payload.ownedPaths)];
  if (!message) {
    throw new Error("Commit message is required.");
  }
  if (
    ownedPaths.length === 0 ||
    ownedPaths.length > 5_000 ||
    ownedPaths.some((path) => !path || path.includes("\0"))
  ) {
    throw new Error("Select one or more valid changed paths.");
  }
  return {
    path: agentWorkspaceRequestPath(projectId, "git/commits/prepare"),
    init: agentWorkspaceJsonRequest("POST", { ownedPaths, message }),
  };
}

export function agentPreparedCommitConfirmationRequest(
  projectId: string,
  preparationId: string,
  rawConfirmationToken: string,
): { path: string; init: RequestInit } {
  const id = preparedOperationId(preparationId, "Commit preparation id");
  return {
    path: agentWorkspaceRequestPath(
      projectId,
      `git/commits/preparations/${encodeURIComponent(id)}/confirm`,
    ),
    init: agentWorkspaceJsonRequest("POST", {
      confirmationToken: confirmationToken(rawConfirmationToken),
    }),
  };
}

export interface AgentPullRequestHandoffPayload {
  serverId: string;
  owner: string;
  repository: string;
  base: string;
  head: string;
  draft: boolean;
}

export function agentPullRequestHandoffRequest(
  projectId: string,
  payload: AgentPullRequestHandoffPayload,
): { path: string; init: RequestInit } {
  const serverId = payload.serverId.trim();
  const owner = payload.owner.trim();
  const repository = payload.repository.trim();
  const base = payload.base.trim();
  const head = payload.head.trim();
  if (!CONNECTOR_ID_PATTERN.test(serverId)) {
    throw new Error("GitHub connector id is invalid.");
  }
  if (!owner || !repository || !base || !head) {
    throw new Error("GitHub owner, repository, base, and head are required.");
  }
  return {
    path: agentWorkspaceRequestPath(
      projectId,
      "review/pull-request-handoff/prepare",
    ),
    init: agentWorkspaceJsonRequest("POST", {
      serverId,
      owner,
      repository,
      base,
      head,
      draft: payload.draft,
    }),
  };
}

export function agentPullRequestHandoffConfirmationRequest(
  projectId: string,
  handoffId: string,
  payload: {
    serverId: string;
    confirmationToken: string;
    expectedRequestDigest: string;
  },
): { path: string; init: RequestInit } {
  const id = preparedOperationId(handoffId, "Pull request handoff id");
  const serverId = payload.serverId.trim();
  if (!CONNECTOR_ID_PATTERN.test(serverId)) {
    throw new Error("GitHub connector id is invalid.");
  }
  if (!REQUEST_DIGEST_PATTERN.test(payload.expectedRequestDigest)) {
    throw new Error("Pull request preview digest is invalid.");
  }
  return {
    path: agentWorkspaceRequestPath(
      projectId,
      `review/pull-request-handoff/${encodeURIComponent(id)}/confirm`,
    ),
    init: agentWorkspaceJsonRequest("POST", {
      serverId,
      confirmationToken: confirmationToken(payload.confirmationToken),
      expectedRequestDigest: payload.expectedRequestDigest,
    }),
  };
}
