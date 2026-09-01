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
  AgentDelegationPolicy,
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
  memory?: boolean;
  dreaming?: boolean;
  graphs?: boolean;
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

export interface AgentMemoryEntry {
  path: string;
  scope: "organization" | "project" | "agent" | "session" | string;
  version: number;
  hash: string;
  bytes: number;
  updatedAt: number | null;
  updatedBy: string | null;
  sourceSessionId: string | null;
  sourceTranscriptIds: string[];
  dreamId: string | null;
  content?: string;
  snippet?: string;
}

export interface AgentMemoryTranscript {
  id: string;
  title: string;
  updatedAt: number;
  archived: boolean;
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
  kind: "verification" | "agent" | "dream" | string;
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
  retryOfTaskId: string | null;
  rootTaskId: string | null;
  delegationRole: "explorer" | "implementer" | "verifier" | "reviewer" | null;
  delegationDepth: number;
  delegationBudget: {
    maxOutputTokens: number;
    maxToolCalls: number;
    wallSeconds: number;
  } | null;
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

export type AgentGraphNodeType =
  | "input"
  | "loop"
  | "model"
  | "tool"
  | "condition"
  | "approval"
  | "output";

export interface AgentGraphNode {
  id: string;
  type: AgentGraphNodeType;
  label?: string;
  config: Record<string, unknown>;
  retryPolicy?: {
    maxAttempts: number;
    backoffMs: number;
    retryOn: Array<"error" | "timeout">;
  };
}

export interface AgentGraphEdge {
  from: string;
  to: string;
  when?: "true" | "false" | "default";
}

export interface AgentGraphDocument {
  name: string;
  description: string;
  metadata?: Record<string, unknown>;
  inputSchema: Record<string, unknown>;
  outputSchema: Record<string, unknown>;
  nodes: AgentGraphNode[];
  edges: AgentGraphEdge[];
  permissions: { allowedToolServerIds?: string[] };
  limits: {
    maxNodes: number;
    maxRunSeconds: number;
    maxOutputBytes: number;
    maxIterations: number;
    maxOutputTokens: number;
  };
}

export interface AgentGraphSummary {
  id: string;
  projectId: string;
  name: string;
  description: string;
  currentRevision: number;
  createdAt: number;
  updatedAt: number;
}

export interface AgentGraphRevision extends AgentGraphDocument {
  graphId: string;
  projectId: string;
  revision: number;
  createdAt: number;
}

export interface AgentGraphRevisionSummary {
  graphId: string;
  projectId: string;
  revision: number;
  name: string;
  description: string;
  createdAt: number;
}

export type AgentGraphRunStatus =
  | "queued"
  | "running"
  | "pausing"
  | "paused"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed"
  | "interrupted"
  | string;

export interface AgentGraphRun {
  id: string;
  projectId: string;
  graphId: string;
  revision: number;
  input: Record<string, unknown>;
  output: unknown;
  error: string | null;
  currentNodeId: string | null;
  status: AgentGraphRunStatus;
  attempt: number;
  retryOfRunId: string | null;
  idempotencyKey: string | null;
  iterationCount: number;
  reservedOutputTokens: number;
  pauseRequested: boolean;
  cancelRequested: boolean;
  createdAt: number;
  updatedAt: number;
  startedAt: number | null;
  completedAt: number | null;
}

export interface AgentGraphNodeExecution {
  id: string;
  runId: string;
  nodeId: string;
  nodeType: AgentGraphNodeType;
  attempt: number;
  status: string;
  input: unknown;
  output: unknown;
  checkpoint: Record<string, unknown> | null;
  error: string | null;
  createdAt: number;
  startedAt: number | null;
  completedAt: number | null;
}

export interface AgentGraphEvent {
  id: string;
  runId: string;
  sequence: number;
  type: string;
  nodeId: string | null;
  payload: Record<string, unknown>;
  createdAt: number;
}

export interface AgentGraphApproval {
  id: string;
  projectId: string;
  runId: string;
  nodeId: string;
  title: string;
  description: string;
  status: "pending" | "approved" | "rejected" | string;
  decision: string | null;
  createdAt: number;
  updatedAt: number;
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

export interface AgentDreamProposal {
  id: string;
  path: string;
  scope: string;
  operation: "create" | "replace" | "delete" | string;
  content: string;
  expectedHash: string | null;
  prevalence: { transcripts: number; selected: number; ratio: number };
  rationale: string;
  examples: Array<{
    threadId: string;
    messageId: string;
    excerpt: string;
  }>;
  sourceTranscriptIds: string[];
  decision: "pending" | "accepted" | "rejected" | string;
  acceptedEntry?: AgentMemoryEntry;
  deletedEntry?: { path: string; deleted: boolean; previousHash: string };
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

export async function listAgentMemoryEntries(
  projectId: string,
  options: { query?: string; includeContent?: boolean } = {},
): Promise<AgentMemoryEntry[]> {
  const result = await request<{ entries: AgentMemoryEntry[] }>(
    agentWorkspaceRequestPath(projectId, "memory", {
      query: options.query,
      include_content: options.includeContent,
    }),
  );
  return result.entries;
}

export async function listAgentMemoryTranscripts(
  projectId: string,
  limit = 20,
): Promise<AgentMemoryTranscript[]> {
  const result = await request<{ transcripts: AgentMemoryTranscript[] }>(
    agentWorkspaceRequestPath(projectId, "memory/transcripts", { limit }),
  );
  return result.transcripts;
}

export function saveAgentMemoryEntry(
  projectId: string,
  payload: { path: string; content: string; expectedHash?: string | null },
): Promise<AgentMemoryEntry> {
  return request(
    agentWorkspaceRequestPath(projectId, "memory/entry"),
    agentWorkspaceJsonRequest("PUT", {
      path: payload.path,
      content: payload.content,
      expectedHash: payload.expectedHash || undefined,
    }),
  );
}

export async function listAgentDreams(
  projectId: string,
  limit = 20,
): Promise<AgentBackgroundTask[]> {
  const result = await request<{ dreams: AgentBackgroundTask[] }>(
    agentWorkspaceRequestPath(projectId, "memory/dreams", { limit }),
  );
  return result.dreams;
}

export function queueAgentDream(
  projectId: string,
  threadIds: string[],
  instructions = "",
): Promise<AgentBackgroundTask> {
  return request(
    agentWorkspaceRequestPath(projectId, "memory/dreams"),
    agentWorkspaceJsonRequest("POST", { threadIds, instructions, start: true }),
  );
}

export function decideAgentDreamProposal(
  projectId: string,
  dreamId: string,
  proposalId: string,
  decision: "accept" | "reject",
  expectedHash?: string | null,
): Promise<{ dream: AgentBackgroundTask; proposal: AgentDreamProposal }> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `memory/dreams/${encodeURIComponent(dreamId)}/proposals/${encodeURIComponent(proposalId)}`,
    ),
    agentWorkspaceJsonRequest("POST", {
      decision,
      expectedHash: expectedHash || undefined,
    }),
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

export async function getAgentBackgroundTaskTree(
  projectId: string,
  taskId: string,
): Promise<{
  rootTaskId: string;
  tasks: AgentBackgroundTask[];
  truncated: boolean;
}> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `background/${encodeURIComponent(taskId)}/tree`,
    ),
  );
}

export function queueAgentChild(
  projectId: string,
  parentTaskId: string,
  payload: {
    role: "explorer" | "implementer" | "verifier" | "reviewer";
    instruction: string;
    budget: {
      maxOutputTokens: number;
      maxToolCalls: number;
      wallSeconds: number;
    };
    worktreeId: string;
    cleanupWorktreeOnCancel?: boolean;
    start?: boolean;
  },
): Promise<AgentBackgroundTask> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `background/${encodeURIComponent(parentTaskId)}/children`,
    ),
    agentWorkspaceJsonRequest("POST", payload),
  );
}

export function cancelAgentChild(
  projectId: string,
  parentTaskId: string,
  childTaskId: string,
): Promise<AgentBackgroundTask> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `background/${encodeURIComponent(parentTaskId)}/children/${encodeURIComponent(childTaskId)}/cancel`,
    ),
    agentWorkspaceJsonRequest("POST"),
  );
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

export async function listAgentGraphs(
  projectId: string,
): Promise<AgentGraphSummary[]> {
  const result = await request<{ graphs: AgentGraphSummary[] }>(
    agentWorkspaceRequestPath(projectId, "graphs"),
  );
  return result.graphs;
}

export async function getAgentGraph(
  projectId: string,
  graphId: string,
  revision?: number,
): Promise<{ graph: AgentGraphSummary; revision: AgentGraphRevision }> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `graphs/${encodeURIComponent(graphId)}`,
      {
        revision,
      },
    ),
  );
}

export function createAgentGraph(
  projectId: string,
  document: AgentGraphDocument,
): Promise<AgentGraphSummary> {
  return request(
    agentWorkspaceRequestPath(projectId, "graphs"),
    agentWorkspaceJsonRequest("POST", document),
  );
}

export function validateAgentGraph(
  projectId: string,
  document: AgentGraphDocument,
): Promise<{ valid: true; document: AgentGraphDocument }> {
  return request(
    agentWorkspaceRequestPath(projectId, "graphs/validate"),
    agentWorkspaceJsonRequest("POST", document),
  );
}

export async function listAgentGraphRevisions(
  projectId: string,
  graphId: string,
  limit = 100,
): Promise<AgentGraphRevisionSummary[]> {
  const result = await request<{ revisions: AgentGraphRevisionSummary[] }>(
    agentWorkspaceRequestPath(
      projectId,
      `graphs/${encodeURIComponent(graphId)}/revisions`,
      { limit },
    ),
  );
  return result.revisions;
}

export function updateAgentGraph(
  projectId: string,
  graph: Pick<AgentGraphSummary, "id" | "currentRevision">,
  document: AgentGraphDocument,
): Promise<AgentGraphSummary> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `graphs/${encodeURIComponent(graph.id)}`,
    ),
    agentWorkspaceJsonRequest("PUT", {
      ...document,
      expectedRevision: graph.currentRevision,
    }),
  );
}

export async function deleteAgentGraph(
  projectId: string,
  graphId: string,
): Promise<void> {
  await request(
    agentWorkspaceRequestPath(
      projectId,
      `graphs/${encodeURIComponent(graphId)}`,
    ),
    { method: "DELETE" },
  );
}

export async function listAgentGraphRuns(
  projectId: string,
  graphId: string,
  limit = 50,
): Promise<AgentGraphRun[]> {
  const result = await request<{ runs: AgentGraphRun[] }>(
    agentWorkspaceRequestPath(
      projectId,
      `graphs/${encodeURIComponent(graphId)}/runs`,
      { limit },
    ),
  );
  return result.runs;
}

export function startAgentGraphRun(
  projectId: string,
  graphId: string,
  payload: {
    input: Record<string, unknown>;
    revision?: number;
    idempotencyKey?: string;
    start?: boolean;
  },
): Promise<AgentGraphRun> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `graphs/${encodeURIComponent(graphId)}/runs`,
    ),
    agentWorkspaceJsonRequest("POST", payload),
  );
}

export async function getAgentGraphRun(
  projectId: string,
  runId: string,
): Promise<{
  run: AgentGraphRun;
  nodes: AgentGraphNodeExecution[];
  approvals: AgentGraphApproval[];
}> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `graph-runs/${encodeURIComponent(runId)}`,
    ),
  );
}

export async function listAgentGraphEvents(
  projectId: string,
  runId: string,
  after = 0,
): Promise<AgentGraphEvent[]> {
  const result = await request<{ events: AgentGraphEvent[] }>(
    agentWorkspaceRequestPath(
      projectId,
      `graph-runs/${encodeURIComponent(runId)}/events`,
      { after },
    ),
  );
  return result.events;
}

function graphRunMutation(
  projectId: string,
  runId: string,
  action: "start" | "pause" | "resume" | "cancel" | "retry",
): Promise<AgentGraphRun> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `graph-runs/${encodeURIComponent(runId)}/${action}`,
    ),
    agentWorkspaceJsonRequest("POST"),
  );
}

export const pauseAgentGraphRun = (projectId: string, runId: string) =>
  graphRunMutation(projectId, runId, "pause");
export const startQueuedAgentGraphRun = (projectId: string, runId: string) =>
  graphRunMutation(projectId, runId, "start");
export const resumeAgentGraphRun = (projectId: string, runId: string) =>
  graphRunMutation(projectId, runId, "resume");
export const cancelAgentGraphRun = (projectId: string, runId: string) =>
  graphRunMutation(projectId, runId, "cancel");
export const retryAgentGraphRun = (projectId: string, runId: string) =>
  graphRunMutation(projectId, runId, "retry");

export function decideAgentGraphApproval(
  projectId: string,
  runId: string,
  approvalId: string,
  decision: "approved" | "rejected",
): Promise<AgentGraphApproval> {
  return request(
    agentWorkspaceRequestPath(
      projectId,
      `graph-runs/${encodeURIComponent(runId)}/approvals/${encodeURIComponent(approvalId)}`,
    ),
    agentWorkspaceJsonRequest("POST", { decision }),
  );
}

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
