// SPDX-License-Identifier: AGPL-3.0-only

export type ResearchRunStatus =
  | "planning"
  | "awaiting_approval"
  | "queued"
  | "running"
  | "paused"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed";

export type ResearchPhase = "planning" | "decision" | "synthesis" | "unknown";
export type ResearchAction = "search" | "fetch";

export interface ResearchPlanStep {
  title: string;
  query: string;
}

export interface ResearchPlan {
  title: string;
  steps: ResearchPlanStep[];
}

export interface ResearchEvidenceSource {
  kind: "knowledge_base";
  chunkId?: string | null;
  documentId?: string | null;
  filename: string;
  page?: number | null;
  score?: number | null;
  snippet?: string;
}

export interface ResearchStepResult {
  action?: ResearchAction;
  input?: string;
  sourceCount?: number;
  sourceUrls?: string[];
  evidenceSources?: ResearchEvidenceSource[];
  excerpt?: string;
  error?: string;
}

export interface ResearchStepSnapshot extends ResearchPlanStep {
  position: number;
  input?: string;
  status: "pending" | "queued" | "running" | "completed" | "failed";
  result?: ResearchStepResult | null;
  startedAt?: number | null;
  completedAt?: number | null;
}

export interface ResearchSource {
  id?: string | number;
  stepPosition?: number | null;
  title: string;
  url: string;
  snippet?: string | null;
  fetchedAt?: number;
}

export interface ResearchInferenceRequest {
  model: string;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  enableThinking?: boolean;
  reasoningEffort?: string;
}

export interface ResearchBudgets {
  maxSteps: number;
  maxSources: number;
  modelTimeoutSeconds: number;
  toolTimeoutSeconds: number;
}

export interface ResearchWebsitePolicy {
  allowedDomains: string[];
  blockedDomains: string[];
}

export interface CreateResearchRunInput {
  threadId: string;
  userMessageId: string;
  assistantMessageId?: string;
  inferenceRequest: ResearchInferenceRequest;
  ragScope?: Record<string, unknown>;
  budgets?: Partial<ResearchBudgets>;
  websitePolicy?: ResearchWebsitePolicy;
}

export interface ResearchRun {
  id: string;
  threadId: string;
  userMessageId: string;
  assistantMessageId?: string | null;
  status: ResearchRunStatus;
  plan: ResearchPlan | null;
  planRevision: number;
  planHash: string | null;
  steps: ResearchStepSnapshot[];
  sources: ResearchSource[];
  config?: {
    model?: string;
    inferenceRequest?: Record<string, unknown>;
    ragScope?: Record<string, unknown> | null;
    budgets?: ResearchBudgets;
    websitePolicy?: ResearchWebsitePolicy;
  };
  cancelRequested?: boolean;
  retryCount?: number;
  error?: string | null;
  report?: string | null;
  lastEventSeq: number;
  createdAt: number;
  updatedAt: number;
  startedAt?: number | null;
  completedAt?: number | null;
  heartbeatAt?: number | null;
}

export type ResearchEventType =
  | "run.created"
  | "run.started"
  | "plan.ready"
  | "run.approved"
  | "reasoning.updated"
  | "step.started"
  | "source.added"
  | "step.completed"
  | "step.failed"
  | "report.updated"
  | "run.cancelRequested"
  | "run.cancelled"
  | "run.retried"
  | "run.completed"
  | "run.failed";

export interface ResearchEventData {
  run: ResearchRun;
  createdAt: number;
  attempt?: number;
  status?: ResearchRunStatus;
  phase?: ResearchPhase;
  callId?: string;
  reasoningDelta?: string;
  reasoningOffset?: number;
  position?: number;
  stepPosition?: number;
  title?: string;
  action?: ResearchAction;
  input?: string;
  url?: string;
  snippet?: string;
  fetchedAt?: number;
  sourceCount?: number;
  error?: string | null;
  delta?: string;
  offset?: number;
  length?: number;
  report?: string;
  plan?: ResearchPlan;
  planRevision?: number;
  planHash?: string;
}

export interface ResearchEvent {
  id: number;
  event: ResearchEventType;
  createdAt: number;
  data: ResearchEventData;
  run: ResearchRun;
}

export interface ResearchMessageMetadata {
  researchRunId?: string;
  researchRun?: ResearchRun;
  researchStatus?: ResearchRunStatus;
  researchPlanRevision?: number;
  serverManaged?: boolean;
  serverRevision?: number;
  reasoningDuration?: number;
}
