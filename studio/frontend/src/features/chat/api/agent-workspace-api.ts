// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";
import {
  type AgentBackgroundAgentRequestPayload,
  type AgentPullRequestHandoffPayload,
  agentBackgroundAgentRequest,
  agentPreparedCommitConfirmationRequest,
  agentPreparedCommitRequest,
  agentPullRequestHandoffConfirmationRequest,
  agentPullRequestHandoffRequest,
  agentWorkspaceJsonRequest,
  agentWorkspaceRequestPath,
  agentWorktreeMergeRequest,
} from "./agent-workspace-requests";

export type {
  AgentBackgroundPermissionMode,
  AgentBackgroundRuntimeKind,
  AgentBackgroundRuntimeSelection,
} from "./agent-workspace-requests";

export interface AgentWorkspaceCapabilities {
  instructions: boolean;
  repositoryMap: boolean;
  verification: boolean;
  plans: boolean;
  background: boolean;
  git: boolean;
  worktrees: boolean;
  review: boolean;
}

export interface AgentWorkspaceOverview {
  projectId: string;
  workspaceKind: "managed" | "folder" | string;
  available: boolean;
  error: string | null;
  isGitRepository: boolean;
  capabilities: AgentWorkspaceCapabilities;
}

export interface AgentProjectContextSnapshot {
  id: string;
  expiresAt: number;
}

export interface AgentInstructionLayer {
  path: string;
  scope: string;
  content: string;
  truncated: boolean;
  bytesRead: number;
}

export interface AgentInstructions {
  layers: AgentInstructionLayer[];
  combined: string;
  truncated: boolean;
  issues: Array<{ path: string; reason: string }>;
  precedence: string;
  bytesRead: number;
}

export interface AgentRepositoryEntry {
  path: string;
  size: number;
  modifiedNs: number;
}

export interface AgentRepositoryMap {
  source: "git" | "filesystem" | string;
  entries: AgentRepositoryEntry[];
  fileCount: number;
  pathsScanned: number;
  bytesIncluded: number;
  skipped: Record<string, number>;
  truncated: boolean;
  truncationReasons: string[];
  limits: {
    maxPaths: number;
    maxTotalBytes: number;
    maxFileBytes: number;
    previewBytes: number;
  };
}

export type AgentVerificationKind = "test" | "lint" | "build" | "custom";

export interface AgentVerificationCheck {
  name: string;
  kind: AgentVerificationKind;
  command: string;
  required: boolean;
  timeoutSeconds: number;
  logLimitBytes: number;
}

export interface AgentVerificationConfig {
  projectId: string;
  checks: AgentVerificationCheck[];
  requireForGoalCompletion: boolean;
  revision: number;
  updatedAt: number | null;
  shellContract?: string;
}

export interface AgentVerificationResult extends AgentVerificationCheck {
  status: "passed" | "failed" | "cancelled" | "timed_out" | string;
  exitCode: number | null;
  output: string;
  outputBytes: number;
  outputTruncated: boolean;
  startedAt: number;
  completedAt: number;
  durationMs: number;
}

export interface AgentVerificationRun {
  id: string;
  projectId: string;
  worktreeId: string | null;
  status: "running" | "passed" | "failed" | "cancelled" | string;
  configRevision: number;
  sourceFingerprint: string;
  finalFingerprint: string | null;
  currentFingerprint?: string | null;
  results: AgentVerificationResult[];
  startedAt: number;
  completedAt: number | null;
  changedDuringRun?: boolean;
  evidenceComplete?: boolean;
  unverifiable?: boolean;
  stale: boolean;
}

export interface AgentGitStatus {
  head: string;
  branch: string | null;
  detached: boolean;
  clean: boolean;
  counts: {
    staged: number;
    unstaged: number;
    untracked: number;
    conflicted: number;
  };
  files: Array<{ code: string; path: string }>;
  truncated: boolean;
}

export interface AgentGitDiff {
  staged: boolean;
  diff: string;
  truncated: boolean;
}

export type AgentPlanStatus = "active" | "blocked" | "completed" | "cancelled";
export type AgentPlanTaskStatus =
  | "pending"
  | "running"
  | "blocked"
  | "completed"
  | "cancelled";

export interface AgentPlanTask {
  id: string;
  planId: string;
  position: number;
  title: string;
  status: AgentPlanTaskStatus;
  blocker: string | null;
  verification: Array<Record<string, unknown>>;
  createdAt: number;
  updatedAt: number;
}

export interface AgentPlan {
  id: string;
  projectId: string;
  title: string;
  goalSnapshot: string | null;
  goalUpdatedAt: number | null;
  status: AgentPlanStatus;
  revision: number;
  tasks: AgentPlanTask[];
  createdAt: number;
  updatedAt: number;
}

export type AgentBackgroundStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed"
  | "interrupted";

export interface AgentBackgroundRuntimeSnapshot {
  kind: "local" | "provider" | string;
  model: string;
  providerId?: string | null;
  providerType?: string | null;
  permissionMode: "off" | "full" | "ask" | "auto" | string;
  reasoningEffort?: string | null;
  maxOutputTokens: number;
}

export interface AgentBackgroundTask {
  id: string;
  projectId: string;
  kind: "verification" | "agent" | string;
  payload: {
    selectedNames?: string[] | null;
    instruction?: string;
    cleanupWorktreeOnCancel?: boolean;
    worktreeId?: string | null;
    runtime?: AgentBackgroundRuntimeSnapshot | null;
  } & Record<string, unknown>;
  goalSnapshot: string | null;
  goalStatusSnapshot: string | null;
  goalUpdatedAt: number | null;
  planId: string | null;
  planRevision: number | null;
  planTaskId: string | null;
  planSnapshot: AgentPlan | null;
  worktreeId: string | null;
  status: AgentBackgroundStatus;
  attempt: number;
  parentTaskId: string | null;
  result: AgentVerificationRun | Record<string, unknown> | null;
  error: string | null;
  cancelRequested: boolean;
  createdAt: number;
  updatedAt: number;
  startedAt: number | null;
  completedAt: number | null;
  appExitPolicy: "interrupt" | string;
  appExitContract: {
    activeTaskState: "interrupted" | string;
    managedCommandsSurvive: boolean;
    adapterMustHonorCancellation: boolean;
  };
}

export interface AgentWorktreeMerge {
  status: "checking" | "merged" | "conflict" | "failed" | string;
  targetBranch: string;
  expectedTargetHead: string;
  sourceHead: string;
  resultHead: string | null;
  startedAt: number;
  completedAt: number | null;
  primaryWorkspaceChanged: boolean;
  conflicts: string[];
  conflictsTruncated?: boolean;
}

export interface AgentWorktree {
  id: string;
  projectId: string;
  branch: string;
  baseRef: string;
  backgroundTaskId: string | null;
  status: "active" | "removed" | string;
  merge?: AgentWorktreeMerge | null;
  createdAt: number;
  updatedAt: number;
}

export interface AgentReviewSummary {
  projectId: string;
  goal: string | null;
  goalStatus: string | null;
  git: AgentGitStatus | null;
  gitError: string | null;
  diff: AgentGitDiff | null;
  plans: AgentPlan[];
  verification: AgentVerificationRun[];
  limits: { diffBytes: number; verificationRuns: number };
  projectRoot: "<project_root>" | string;
}

export interface AgentPullRequestDraft {
  title: string;
  body: string;
  localOnly: true;
  submitted: false;
}

export interface AgentPreparedCommitFile {
  code: string;
  path: string;
  oldPath?: string;
}

export interface AgentPreparedCommit {
  id: string;
  projectId: string;
  status: "awaiting_confirmation" | "confirmed" | string;
  branch: string;
  baseHead: string;
  message: string;
  ownedPaths: string[];
  sourceFingerprint: string;
  createdAt: number;
  expiresAt: number;
  confirmationToken?: string;
  files?: AgentPreparedCommitFile[];
  diff?: string;
  diffTruncated?: boolean;
  commitSha?: string;
  refName?: string;
  confirmedAt?: number;
}

export interface AgentPullRequestHandoffPreview {
  id: string;
  confirmationToken: string;
  requestDigest: string;
  expiresAt: number;
  connector: { id: string; displayName: string };
  request: {
    owner: string;
    repo: string;
    base: string;
    head: string;
    title: string;
    body: string;
    draft: boolean;
    maintainer_can_modify: boolean;
  };
  submitted: false;
}

export interface AgentPullRequestHandoffResult {
  id: string;
  requestDigest: string;
  connector: { id: string; displayName: string };
  submitted: true;
  connectorResult: string;
  connectorResultTruncated: boolean;
}

export class AgentWorkspaceRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "AgentWorkspaceRequestError";
    this.status = status;
  }
}

export function agentWorkspaceMutationOutcomeUnknown(error: unknown): boolean {
  return !(error instanceof AgentWorkspaceRequestError && error.status < 500);
}

async function parseAgentResponse<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new AgentWorkspaceRequestError(
      formatApiErrorBody(body) ??
        `Workspace request failed (${response.status})`,
      response.status,
    );
  }
  return body as T;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return parseAgentResponse<T>(await authFetch(path, init));
}

export function getAgentWorkspace(
  projectId: string,
): Promise<AgentWorkspaceOverview> {
  return request(agentWorkspaceRequestPath(projectId, "workspace"));
}

export function createAgentProjectContextSnapshot(
  projectId: string,
  query = "",
): Promise<AgentProjectContextSnapshot> {
  return request(
    agentWorkspaceRequestPath(projectId, "context-snapshots"),
    agentWorkspaceJsonRequest("POST", { query }),
  );
}

export function getAgentInstructions(
  projectId: string,
  target?: string,
): Promise<AgentInstructions> {
  return request(
    agentWorkspaceRequestPath(projectId, "instructions", { target }),
  );
}

export function getAgentRepositoryMap(
  projectId: string,
  options: { maxPaths?: number; maxTotalBytes?: number } = {},
): Promise<AgentRepositoryMap> {
  return request(
    agentWorkspaceRequestPath(projectId, "repository-map", {
      max_paths: options.maxPaths,
      max_total_bytes: options.maxTotalBytes,
    }),
  );
}

export function getAgentVerificationConfig(
  projectId: string,
): Promise<AgentVerificationConfig> {
  return request(agentWorkspaceRequestPath(projectId, "verification"));
}

export function saveAgentVerificationConfig(
  projectId: string,
  checks: AgentVerificationCheck[],
  requireForGoalCompletion: boolean,
  expectedRevision: number,
): Promise<AgentVerificationConfig> {
  return request(
    agentWorkspaceRequestPath(projectId, "verification"),
    agentWorkspaceJsonRequest("PUT", {
      checks,
      requireForGoalCompletion,
      expectedRevision,
    }),
  );
}

export function runAgentVerification(
  projectId: string,
  configRevision: number,
  selectedNames?: string[],
): Promise<AgentVerificationRun> {
  return request(
    agentWorkspaceRequestPath(projectId, "verify"),
    agentWorkspaceJsonRequest("POST", { selectedNames, configRevision }),
  );
}

export async function listAgentVerificationRuns(
  projectId: string,
  limit = 20,
): Promise<AgentVerificationRun[]> {
  const result = await request<{ runs: AgentVerificationRun[] }>(
    agentWorkspaceRequestPath(projectId, "verifications", { limit }),
  );
  return result.runs;
}

export function cancelAgentVerification(
  projectId: string,
  runId: string,
): Promise<{ cancelRequested: boolean }> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `verifications/${encodeURIComponent(runId)}/cancel`,
    ),
    agentWorkspaceJsonRequest("POST"),
  );
}

export function getAgentGitStatus(projectId: string): Promise<AgentGitStatus> {
  return request(agentWorkspaceRequestPath(projectId, "git/status"));
}

export function getAgentGitDiff(
  projectId: string,
  options: { staged?: boolean; maxBytes?: number } = {},
): Promise<AgentGitDiff> {
  return request(
    agentWorkspaceRequestPath(projectId, "git/diff", {
      staged: options.staged,
      max_bytes: options.maxBytes,
    }),
  );
}

export function createAgentCheckpoint(
  projectId: string,
  ownedPaths: string[],
): Promise<{ id: string; ownedPaths: string[]; sourceFingerprint: string }> {
  return request(
    agentWorkspaceRequestPath(projectId, "git/checkpoints"),
    agentWorkspaceJsonRequest("POST", { ownedPaths }),
  );
}

export function rollbackAgentCheckpoint(
  projectId: string,
  checkpointId: string,
  expectedCurrentFingerprint: string,
): Promise<{
  checkpointId: string;
  restoredPaths: string[];
  fingerprint: string;
}> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `git/checkpoints/${encodeURIComponent(checkpointId)}/rollback`,
    ),
    agentWorkspaceJsonRequest("POST", { expectedCurrentFingerprint }),
  );
}

export function prepareAgentCommit(
  projectId: string,
  ownedPaths: string[],
  message: string,
): Promise<AgentPreparedCommit> {
  const spec = agentPreparedCommitRequest(projectId, {
    ownedPaths,
    message,
  });
  return request(spec.path, spec.init);
}

export function confirmAgentPreparedCommit(
  projectId: string,
  preparationId: string,
  confirmationToken: string,
): Promise<AgentPreparedCommit> {
  const spec = agentPreparedCommitConfirmationRequest(
    projectId,
    preparationId,
    confirmationToken,
  );
  return request(spec.path, spec.init);
}

export async function listAgentPlans(projectId: string): Promise<AgentPlan[]> {
  const result = await request<{ plans: AgentPlan[] }>(
    agentWorkspaceRequestPath(projectId, "plans"),
  );
  return result.plans;
}

export function createAgentPlan(
  projectId: string,
  payload: {
    title: string;
    tasks: Array<{
      title: string;
      status?: AgentPlanTaskStatus;
      blocker?: string | null;
      verification?: Array<Record<string, unknown>>;
    }>;
  },
): Promise<AgentPlan> {
  return request(
    agentWorkspaceRequestPath(projectId, "plans"),
    agentWorkspaceJsonRequest("POST", payload),
  );
}

export function updateAgentPlan(
  projectId: string,
  plan: Pick<AgentPlan, "id" | "revision">,
  status: AgentPlanStatus,
): Promise<AgentPlan> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `plans/${encodeURIComponent(plan.id)}`,
    ),
    agentWorkspaceJsonRequest("PATCH", {
      status,
      expectedRevision: plan.revision,
    }),
  );
}

export function updateAgentPlanTask(
  projectId: string,
  plan: Pick<AgentPlan, "id" | "revision">,
  taskId: string,
  patch: {
    status?: AgentPlanTaskStatus;
    blocker?: string | null;
    verification?: Array<Record<string, unknown>>;
  },
): Promise<AgentPlan> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `plans/${encodeURIComponent(plan.id)}/tasks/${encodeURIComponent(taskId)}`,
    ),
    agentWorkspaceJsonRequest("PATCH", {
      ...patch,
      expectedRevision: plan.revision,
    }),
  );
}

export async function listAgentBackgroundTasks(
  projectId: string,
  limit = 100,
): Promise<AgentBackgroundTask[]> {
  const result = await request<{ tasks: AgentBackgroundTask[] }>(
    agentWorkspaceRequestPath(projectId, "background", { limit }),
  );
  return result.tasks;
}

export function queueAgentVerification(
  projectId: string,
  configRevision: number,
  selectedNames?: string[],
  start = true,
): Promise<AgentBackgroundTask> {
  return request(
    agentWorkspaceRequestPath(projectId, "background/verification"),
    agentWorkspaceJsonRequest("POST", {
      selectedNames,
      configRevision,
      start,
    }),
  );
}

export function queueAgentTask(
  projectId: string,
  payload: AgentBackgroundAgentRequestPayload,
): Promise<AgentBackgroundTask> {
  const spec = agentBackgroundAgentRequest(projectId, payload);
  return request(spec.path, spec.init);
}

function backgroundMutation(
  projectId: string,
  taskId: string,
  action: "start" | "cancel" | "retry",
): Promise<AgentBackgroundTask> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `background/${encodeURIComponent(taskId)}/${action}`,
    ),
    agentWorkspaceJsonRequest("POST"),
  );
}

export const startAgentBackgroundTask = (projectId: string, taskId: string) =>
  backgroundMutation(projectId, taskId, "start");
export const cancelAgentBackgroundTask = (projectId: string, taskId: string) =>
  backgroundMutation(projectId, taskId, "cancel");
export const retryAgentBackgroundTask = (projectId: string, taskId: string) =>
  backgroundMutation(projectId, taskId, "retry");

export async function listAgentWorktrees(
  projectId: string,
): Promise<AgentWorktree[]> {
  const result = await request<{ worktrees: AgentWorktree[] }>(
    agentWorkspaceRequestPath(projectId, "worktrees"),
  );
  return result.worktrees;
}

export function createAgentWorktree(
  projectId: string,
  payload: {
    branch?: string;
    baseRef?: string;
    backgroundTaskId?: string;
  } = {},
): Promise<AgentWorktree> {
  return request(
    agentWorkspaceRequestPath(projectId, "worktrees"),
    agentWorkspaceJsonRequest("POST", {
      branch: payload.branch || undefined,
      baseRef: payload.baseRef || "HEAD",
      backgroundTaskId: payload.backgroundTaskId || undefined,
    }),
  );
}

export function cleanupAgentWorktree(
  projectId: string,
  worktreeId: string,
): Promise<AgentWorktree> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `worktrees/${encodeURIComponent(worktreeId)}`,
    ),
    agentWorkspaceJsonRequest("DELETE"),
  );
}

export function mergeAgentWorktree(
  projectId: string,
  worktreeId: string,
  expectedTargetHead: string,
): Promise<AgentWorktree> {
  const spec = agentWorktreeMergeRequest(
    projectId,
    worktreeId,
    expectedTargetHead,
  );
  return request(spec.path, spec.init);
}

export function getAgentReview(projectId: string): Promise<AgentReviewSummary> {
  return request(agentWorkspaceRequestPath(projectId, "review"));
}

export function createAgentPullRequestDraft(
  projectId: string,
  payload: { title?: string; bodyNote?: string } = {},
): Promise<AgentPullRequestDraft> {
  return request(
    agentWorkspaceRequestPath(projectId, "review/pull-request-draft"),
    agentWorkspaceJsonRequest("POST", {
      title: payload.title || "",
      bodyNote: payload.bodyNote || "",
    }),
  );
}

export function prepareAgentPullRequestHandoff(
  projectId: string,
  payload: AgentPullRequestHandoffPayload,
): Promise<AgentPullRequestHandoffPreview> {
  const spec = agentPullRequestHandoffRequest(projectId, payload);
  return request(spec.path, spec.init);
}

export function confirmAgentPullRequestHandoff(
  projectId: string,
  handoffId: string,
  payload: {
    serverId: string;
    confirmationToken: string;
    expectedRequestDigest: string;
  },
): Promise<AgentPullRequestHandoffResult> {
  const spec = agentPullRequestHandoffConfirmationRequest(
    projectId,
    handoffId,
    payload,
  );
  return request(spec.path, spec.init);
}
