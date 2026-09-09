// SPDX-License-Identifier: AGPL-3.0-only

import { create } from "zustand";
// eslint-disable-next-line no-restricted-imports -- Avoid the auth barrel's React login page.
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth/session";
import { followResearchRun, type ResearchRunUpdate } from "../api/research-api";
import type {
  ResearchAction,
  ResearchEvent,
  ResearchEvidenceSource,
  ResearchPhase,
  ResearchPlan,
  ResearchRun,
  ResearchSource,
} from "../types/research";

export type ResearchConnectionState =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected";

export interface ResearchActivity {
  id: string;
  seq: number;
  attempt: number;
  kind: "status" | "reasoning" | "plan" | "step" | "report";
  createdAt: number;
  title: string;
  detail?: string;
  state?: "running" | "complete" | "failed" | "cancelled" | "action";
  phase?: ResearchPhase;
  reasoning?: string;
  /** Plan step titles published as the planner writes them, before the plan is parseable. */
  previewLabels?: string[];
  /** True when a phase.started opened this row, so phase.ended is what closes it. */
  bracketed?: boolean;
  plan?: ResearchPlan;
  stepPosition?: number;
  action?: ResearchAction;
  input?: string;
  sources?: ResearchSource[];
  evidenceSources?: ResearchEvidenceSource[];
  excerpt?: string;
}

export interface ResearchSession {
  run: ResearchRun;
  activities: ResearchActivity[];
  lastAppliedSeq: number;
  following: boolean;
  connection: ResearchConnectionState;
  error: string | null;
}

export interface ResearchPlanReviewState {
  revision: number;
  open: boolean;
  editing: boolean;
  draft: ResearchPlan;
}

interface ResearchRunState {
  sessions: Record<string, ResearchSession>;
  latestRunByThreadId: Record<string, string>;
  claimedThreadIds: Record<string, boolean>;
  activityOpenByRunId: Record<string, Record<string, boolean>>;
  planReviewByRunId: Record<string, ResearchPlanReviewState>;
  openRunId: string | null;
  ingest: (run: ResearchRun, event?: ResearchEvent) => void;
  setThreadClaimed: (threadId: string, claimed: boolean) => void;
  setFollowing: (
    runId: string,
    following: boolean,
    connection?: ResearchConnectionState,
  ) => void;
  setConnectionError: (runId: string, error: string | null) => void;
  openPanel: (runId: string) => void;
  closePanel: () => void;
  setActivityOpen: (runId: string, activityId: string, open: boolean) => void;
  setPlanReviewOpen: (runId: string, open: boolean) => void;
  setPlanReviewEditing: (runId: string, editing: boolean) => void;
  setPlanReviewDraft: (runId: string, draft: ResearchPlan) => void;
}

export const terminalResearchStatuses: ReadonlySet<string> = new Set([
  "completed",
  "failed",
  "cancelled",
]);
const terminalStatuses = terminalResearchStatuses;

export function isSettledResearchRun(
  run: ResearchRun,
  lastAppliedSeq: number,
): boolean {
  return terminalStatuses.has(run.status) && lastAppliedSeq >= run.lastEventSeq;
}

function statusActivity(event: ResearchEvent): ResearchActivity | null {
  const attempt = event.data.attempt ?? 0;
  const base = {
    id: `event-${event.id}`,
    seq: event.id,
    attempt,
    kind: "status" as const,
    createdAt: event.createdAt,
  };
  switch (event.event) {
    case "run.created":
      return { ...base, title: "Research requested", state: "complete" };
    case "run.started":
      return event.data.status === "planning"
        ? null
        : {
            ...base,
            title:
              event.data.resumed || attempt > 0
                ? "Research resumed"
                : "Research started",
            state: "complete",
          };
    case "run.approved":
      return { ...base, title: "Plan approved", state: "complete" };
    case "run.cancelRequested":
      return { ...base, title: "Stopping research safely", state: "running" };
    case "run.cancelled":
      return { ...base, title: "Research cancelled", state: "cancelled" };
    case "run.retried":
      return {
        ...base,
        title: `Started attempt ${attempt + 1}`,
        detail: "Previous activity is preserved below.",
        state: "complete",
      };
    case "run.rebound":
      return {
        ...base,
        title: "Moved to a new question",
        detail: "Activity from the stopped question is preserved below.",
        state: "complete",
      };
    case "run.completed":
      return { ...base, title: "Research completed", state: "complete" };
    case "run.failed":
      return {
        ...base,
        title: "Research failed",
        detail: event.data.error ?? undefined,
        state: "failed",
      };
    default:
      return null;
  }
}

function findLastActivityIndex(
  activities: ResearchActivity[],
  predicate: (activity: ResearchActivity) => boolean,
): number {
  for (let index = activities.length - 1; index >= 0; index -= 1) {
    if (predicate(activities[index])) return index;
  }
  return -1;
}

export function researchPhaseTitle(phase: ResearchPhase | undefined): string {
  switch (phase) {
    case "planning":
      return "Planning an approach";
    case "synthesis_audit":
      return "Checking the evidence";
    case "synthesis":
    case "synthesis_recovery":
      return "Writing the report";
    case "decision":
      return "Choosing the next step";
    default:
      return "Working";
  }
}

/** Rows for one model call are keyed by its callId so the phase bracket and any streamed
 *  reasoning for that call land on the same activity. */
function phaseActivityId(attempt: number, callId: string): string {
  return `reasoning-${attempt}-${callId}`;
}

/** What the run is doing right now, for the collapsed card: the live activity, qualified by the
 *  latest thing it has produced so a long call still shows movement. */
export function runningResearchActivityTitle(
  activities: ResearchActivity[] | undefined,
): string | null {
  if (!activities) return null;
  const index = findLastActivityIndex(
    activities,
    (activity) => activity.state === "running",
  );
  if (index < 0) return null;
  const activity = activities[index];
  const latestLabel = activity.previewLabels?.at(-1);
  return latestLabel ? `${activity.title} · ${latestLabel}` : activity.title;
}

export function stepResultDetail(sourceCount: number, action?: string): string {
  if (sourceCount > 0) {
    return `${sourceCount} ${sourceCount === 1 ? "source" : "sources"} found`;
  }
  // A fetch records an excerpt of one page and never collects sources, so a count would read as a
  // failure. A search that returns nothing is a real outcome, not a styled success.
  return action === "fetch" ? "Page read" : "No usable results";
}

/** Header summary. Counts are omitted until they exist, so a run that has not searched yet reads
 *  as the work it is doing rather than "0 sources, 0 actions". */
export function researchProgressSummary(
  run: ResearchRun,
  elapsed: string,
): string {
  const documentCount = new Set(
    (run.documentSources ?? []).map(
      (source) => source.documentId ?? source.filename,
    ),
  ).size;
  const sourceCount = run.sources.length + documentCount;
  const finishedSteps = run.steps.filter(
    (step) => step.status === "completed" || step.status === "failed",
  ).length;
  const activeStep = run.steps.find((step) => step.status === "running");
  const parts = [elapsed];
  if (sourceCount > 0) {
    parts.push(`${sourceCount} ${sourceCount === 1 ? "source" : "sources"}`);
  }
  // No fraction: run.steps only holds actions already started, and the agent stops when the
  // evidence is enough rather than at the plan's length, so any denominator would be invented.
  if (activeStep) {
    parts.push(`step ${activeStep.position + 1}`);
  } else if (finishedSteps > 0) {
    parts.push(`${finishedSteps} ${finishedSteps === 1 ? "step" : "steps"}`);
  } else if (run.status === "planning") {
    parts.push("building the plan");
  } else if (run.status === "awaiting_approval") {
    parts.push("waiting for you");
  }
  return parts.join(" · ");
}


function syncPlanReviewState(
  current: ResearchPlanReviewState | undefined,
  run: ResearchRun,
): ResearchPlanReviewState | undefined {
  if (!run.plan || run.status !== "awaiting_approval") return current;
  if (current?.revision === run.planRevision) return current;
  return {
    revision: run.planRevision,
    open: true,
    editing: false,
    draft: run.plan,
  };
}

function reduceActivity(
  activities: ResearchActivity[],
  event: ResearchEvent,
): ResearchActivity[] {
  const next = [...activities];
  const attempt = event.data.attempt ?? 0;
  // A retry deletes the old attempt's step rows while its events survive, and the stream attaches
  // the live snapshot to replayed history, so run.steps only describes its own attempt.
  const snapshotIsSameAttempt = attempt === (event.run.retryCount ?? 0);

  // runs recorded before phase events carry no phase.ended, so close their rows as before.
  if (!event.event.startsWith("phase.") && event.event !== "reasoning.updated") {
    const unbracketed = findLastActivityIndex(
      next,
      (activity) =>
        activity.kind === "reasoning" &&
        activity.state === "running" &&
        !activity.bracketed,
    );
    if (unbracketed >= 0) {
      next[unbracketed] = { ...next[unbracketed], state: "complete" };
    }
  }

  if (
    event.event === "phase.started" ||
    event.event === "phase.progress" ||
    event.event === "phase.ended"
  ) {
    const phase = event.data.phase ?? "unknown";
    const callId = event.data.callId ?? `${phase}-${event.id}`;
    const id = phaseActivityId(attempt, callId);
    const existingIndex = next.findIndex((activity) => activity.id === id);
    if (event.event === "phase.progress") {
      const label = event.data.label?.trim();
      if (existingIndex < 0 || !label) return next;
      const existing = next[existingIndex];
      if (existing.previewLabels?.includes(label)) return next;
      next[existingIndex] = {
        ...existing,
        seq: event.id,
        previewLabels: [...(existing.previewLabels ?? []), label],
      };
      return next;
    }
    if (event.event === "phase.ended") {
      if (existingIndex >= 0) {
        next[existingIndex] = {
          ...next[existingIndex],
          seq: event.id,
          state: "complete",
        };
      }
      return next;
    }
    if (existingIndex >= 0) return next;
    // phase.ended is best-effort (_note_phase swallows append failures), so a new phase also closes
    // the previous one; otherwise a dropped end leaves that row spinning all run.
    const stale = findLastActivityIndex(
      next,
      (activity) =>
        activity.kind === "reasoning" &&
        activity.state === "running" &&
        activity.id !== id,
    );
    if (stale >= 0) {
      next[stale] = { ...next[stale], state: "complete" };
    }
    next.push({
      id,
      seq: event.id,
      attempt,
      kind: "reasoning",
      createdAt: event.createdAt,
      title: researchPhaseTitle(phase),
      phase,
      state: "running",
      bracketed: true,
      stepPosition: event.data.stepPosition,
    });
    return next;
  }

  if (event.event === "reasoning.updated") {
    const phase = event.data.phase ?? "unknown";
    const callId = event.data.callId ?? `${phase}-${event.id}`;
    const id = phaseActivityId(attempt, callId);
    const existingIndex = next.findIndex((activity) => activity.id === id);
    const delta = event.data.reasoningDelta ?? "";
    if (existingIndex >= 0) {
      const existing = next[existingIndex];
      next[existingIndex] = {
        ...existing,
        seq: event.id,
        reasoning: `${existing.reasoning ?? ""}${delta}`,
        state: "running",
      };
    } else {
      // a new unbracketed call ends the previous one; phase.ended closes bracketed rows.
      const stale = findLastActivityIndex(
        next,
        (activity) =>
          activity.kind === "reasoning" &&
          activity.state === "running" &&
          !activity.bracketed,
      );
      if (stale >= 0) {
        next[stale] = { ...next[stale], state: "complete" };
      }
      next.push({
        id,
        seq: event.id,
        attempt,
        kind: "reasoning",
        createdAt: event.createdAt,
        title: researchPhaseTitle(phase),
        phase,
        reasoning: delta,
        state: "running",
        stepPosition: event.data.stepPosition,
      });
    }
    return next;
  }

  if (event.event === "plan.ready") {
    next.push({
      id: `plan-${attempt}-${event.data.planRevision ?? event.id}`,
      seq: event.id,
      attempt,
      kind: "plan",
      createdAt: event.createdAt,
      title: "Research plan ready",
      plan: event.data.plan ?? event.run.plan ?? undefined,
      state: "action",
    });
    return next;
  }

  if (event.event === "run.approved") {
    const planIndex = findLastActivityIndex(
      next,
      (activity) =>
        activity.kind === "plan" &&
        activity.attempt === attempt &&
        activity.state === "action",
    );
    if (planIndex >= 0) {
      next[planIndex] = {
        ...next[planIndex],
        seq: event.id,
        state: "complete",
      };
    }
  }

  if (event.event === "step.started") {
    const action = event.data.action ?? "search";
    const activity: ResearchActivity = {
      id: `step-${attempt}-${event.data.stepPosition ?? event.id}`,
      seq: event.id,
      attempt,
      kind: "step",
      createdAt: event.createdAt,
      title:
        event.data.title ??
        (action === "fetch" ? "Reading a page" : "Searching the web"),
      detail: action === "fetch" ? "Reading page" : "Web search",
      state: "running",
      stepPosition: event.data.stepPosition ?? event.data.position,
      action,
      input: event.data.input,
      sources: [],
    };
    const existingIndex = next.findIndex((item) => item.id === activity.id);
    if (existingIndex >= 0) next[existingIndex] = activity;
    else next.push(activity);
    return next;
  }

  if (event.event === "source.added") {
    const stepPosition = event.data.stepPosition ?? event.data.position;
    const index = findLastActivityIndex(
      next,
      (activity) =>
        activity.kind === "step" &&
        activity.attempt === attempt &&
        activity.stepPosition === stepPosition,
    );
    if (index >= 0 && event.data.url) {
      const activity = next[index];
      const source: ResearchSource = {
        id: `${event.id}`,
        stepPosition,
        url: event.data.url,
        title: event.data.title ?? event.data.url,
        snippet: event.data.snippet,
        fetchedAt: event.data.fetchedAt,
      };
      next[index] = {
        ...activity,
        sources: [...(activity.sources ?? []), source],
      };
    }
    return next;
  }

  if (event.event === "step.completed" || event.event === "step.failed") {
    const stepPosition = event.data.stepPosition ?? event.data.position;
    const index = findLastActivityIndex(
      next,
      (activity) =>
        activity.kind === "step" &&
        activity.attempt === attempt &&
        activity.stepPosition === stepPosition,
    );
    if (index >= 0) {
      const activity = next[index];
      const snapshot = snapshotIsSameAttempt
        ? event.run.steps.find((step) => step.position === stepPosition)
        : undefined;
      next[index] = {
        ...activity,
        seq: event.id,
        state: event.event === "step.failed" ? "failed" : "complete",
        detail:
          event.event === "step.failed"
            ? (event.data.error ?? "The tool could not complete this action.")
            : stepResultDetail(
                event.data.sourceCount ?? activity.sources?.length ?? 0,
                event.data.action ?? activity.action,
              ),
        evidenceSources:
          snapshot?.result?.evidenceSources ?? activity.evidenceSources,
        excerpt: snapshot?.result?.excerpt ?? activity.excerpt,
      };
    }
    return next;
  }

  if (event.event === "report.updated") {
    // a live synthesis row already says this; only pre-phase-event runs need their own.
    const synthesisIndex = findLastActivityIndex(
      next,
      (activity) =>
        activity.kind === "reasoning" &&
        activity.attempt === attempt &&
        (activity.phase === "synthesis" || activity.phase === "synthesis_recovery"),
    );
    if (synthesisIndex >= 0) {
      const existing = next[synthesisIndex];
      next[synthesisIndex] = {
        ...existing,
        seq: event.id,
        state: existing.bracketed ? existing.state : "running",
      };
      return next;
    }
    const id = `report-${attempt}`;
    const index = next.findIndex((activity) => activity.id === id);
    if (index >= 0) {
      next[index] = { ...next[index], seq: event.id, state: "running" };
    } else {
      next.push({
        id,
        seq: event.id,
        attempt,
        kind: "report",
        createdAt: event.createdAt,
        title: "Writing the report",
        state: "running",
      });
    }
    return next;
  }

  if (
    event.event === "run.completed" ||
    event.event === "run.failed" ||
    event.event === "run.cancelled"
  ) {
    const terminalState =
      event.event === "run.completed"
        ? "complete"
        : event.event === "run.failed"
          ? "failed"
          : "cancelled";
    for (let index = 0; index < next.length; index += 1) {
      const activity = next[index];
      if (activity.attempt === attempt && activity.state === "running") {
        next[index] = { ...activity, seq: event.id, state: terminalState };
      }
    }
  }

  if (
    event.event === "run.started" &&
    event.data.resumed &&
    snapshotIsSameAttempt
  ) {
    for (let index = next.length - 1; index >= 0; index -= 1) {
      const activity = next[index];
      // A worker killed mid-call never wrote its phase.ended; the resume closes that row.
      if (
        activity.kind === "reasoning" &&
        activity.attempt === attempt &&
        activity.state === "running"
      ) {
        next[index] = { ...activity, seq: event.id, state: "complete" };
        continue;
      }
      if (activity.kind !== "step" || activity.attempt !== attempt) continue;
      const snapshot = event.run.steps.find(
        (step) => step.position === activity.stepPosition,
      );
      if (snapshot?.status !== "completed" && snapshot?.status !== "failed") {
        next.splice(index, 1);
        continue;
      }
      next[index] = {
        ...activity,
        seq: event.id,
        state: snapshot.status === "failed" ? "failed" : "complete",
        evidenceSources: snapshot.result?.evidenceSources,
        excerpt: snapshot.result?.excerpt,
      };
    }
  }

  const status = statusActivity(event);
  if (status) next.push(status);
  return next;
}

export const useResearchRunStore = create<ResearchRunState>((set) => ({
  sessions: {},
  latestRunByThreadId: {},
  claimedThreadIds: {},
  activityOpenByRunId: {},
  planReviewByRunId: {},
  openRunId: null,
  ingest: (run, event) =>
    set((state) => {
      const previous = state.sessions[run.id];
      if (event && previous && event.id <= previous.lastAppliedSeq)
        return state;
      if (
        !event &&
        previous &&
        (run.lastEventSeq < previous.run.lastEventSeq ||
          run.updatedAt < previous.run.updatedAt)
      ) {
        return state;
      }
      const activities = event
        ? reduceActivity(previous?.activities ?? [], event)
        : (previous?.activities ?? []);
      const lastAppliedSeq = event?.id ?? previous?.lastAppliedSeq ?? 0;
      const settled = isSettledResearchRun(run, lastAppliedSeq);
      const session: ResearchSession = {
        run,
        activities,
        lastAppliedSeq,
        following: settled ? false : (previous?.following ?? false),
        connection: settled ? "idle" : (previous?.connection ?? "idle"),
        error: settled ? null : (previous?.error ?? null),
      };
      const currentLatestId = state.latestRunByThreadId[run.threadId];
      const currentLatestRun = currentLatestId
        ? state.sessions[currentLatestId]?.run
        : undefined;
      const shouldBecomeLatest =
        !currentLatestRun ||
        currentLatestRun.id === run.id ||
        run.createdAt >= currentLatestRun.createdAt;
      const planReview = syncPlanReviewState(
        state.planReviewByRunId[run.id],
        run,
      );
      // Claimed means spent: a finished run is the chat's one research. A run still going keeps the
      // toggle lit and a stopped one can be re-pointed, so neither takes the toggle away.
      const claimed = shouldBecomeLatest
        ? run.status === "completed" || run.status === "failed"
        : Boolean(state.claimedThreadIds[run.threadId]);
      return {
        sessions: { ...state.sessions, [run.id]: session },
        claimedThreadIds:
          state.claimedThreadIds[run.threadId] === claimed
            ? state.claimedThreadIds
            : { ...state.claimedThreadIds, [run.threadId]: claimed },
        latestRunByThreadId: shouldBecomeLatest
          ? { ...state.latestRunByThreadId, [run.threadId]: run.id }
          : state.latestRunByThreadId,
        ...(planReview && planReview !== state.planReviewByRunId[run.id]
          ? {
              planReviewByRunId: {
                ...state.planReviewByRunId,
                [run.id]: planReview,
              },
            }
          : {}),
      };
    }),
  setThreadClaimed: (threadId, claimed) =>
    set((state) =>
      state.claimedThreadIds[threadId] === claimed
        ? state
        : {
            claimedThreadIds: {
              ...state.claimedThreadIds,
              [threadId]: claimed,
            },
          },
    ),
  setFollowing: (
    runId,
    following,
    connection = following ? "connected" : "idle",
  ) =>
    set((state) => {
      const session = state.sessions[runId];
      if (!session) return state;
      if (
        session.following === following &&
        session.connection === connection
      ) {
        return state;
      }
      return {
        sessions: {
          ...state.sessions,
          [runId]: { ...session, following, connection },
        },
      };
    }),
  setConnectionError: (runId, error) =>
    set((state) => {
      const session = state.sessions[runId];
      if (!session) return state;
      return {
        sessions: {
          ...state.sessions,
          [runId]: {
            ...session,
            error,
            connection: error ? "disconnected" : session.connection,
          },
        },
      };
    }),
  openPanel: (openRunId) => set({ openRunId }),
  closePanel: () => set({ openRunId: null }),
  setActivityOpen: (runId, activityId, open) =>
    set((state) => {
      const current = state.activityOpenByRunId[runId] ?? {};
      if (current[activityId] === open) return state;
      return {
        activityOpenByRunId: {
          ...state.activityOpenByRunId,
          [runId]: { ...current, [activityId]: open },
        },
      };
    }),
  setPlanReviewOpen: (runId, open) =>
    set((state) => {
      const current = state.planReviewByRunId[runId];
      if (!current || current.open === open) return state;
      return {
        planReviewByRunId: {
          ...state.planReviewByRunId,
          [runId]: { ...current, open },
        },
      };
    }),
  setPlanReviewEditing: (runId, editing) =>
    set((state) => {
      const current = state.planReviewByRunId[runId];
      if (!current || current.editing === editing) return state;
      return {
        planReviewByRunId: {
          ...state.planReviewByRunId,
          [runId]: { ...current, editing },
        },
      };
    }),
  setPlanReviewDraft: (runId, draft) =>
    set((state) => {
      const current = state.planReviewByRunId[runId];
      if (!current || current.draft === draft) return state;
      return {
        planReviewByRunId: {
          ...state.planReviewByRunId,
          [runId]: { ...current, draft },
        },
      };
    }),
}));

const ownedFollowers = new Map<string, AbortController>();
const externalFollowerStops = new Map<string, Set<() => void>>();
const pendingStreamEvents = new Map<
  string,
  {
    run: ResearchRun;
    event: ResearchEvent;
    timer: ReturnType<typeof setTimeout>;
  }
>();
const STREAM_EVENT_FLUSH_MS = 80;

function flushPendingStreamEvent(runId: string): void {
  const pending = pendingStreamEvents.get(runId);
  if (!pending) return;
  clearTimeout(pending.timer);
  pendingStreamEvents.delete(runId);
  useResearchRunStore.getState().ingest(pending.run, pending.event);
}

function canCoalesceStreamEvent(
  previous: ResearchEvent,
  next: ResearchEvent,
): boolean {
  if (previous.event !== next.event) return false;
  if (next.event === "report.updated") return true;
  return (
    next.event === "reasoning.updated" &&
    previous.data.callId === next.data.callId &&
    (previous.data.attempt ?? 0) === (next.data.attempt ?? 0)
  );
}

function compactReplayUpdates(
  updates: ResearchRunUpdate[],
): ResearchRunUpdate[] {
  const compacted: ResearchRunUpdate[] = [];
  for (const update of updates) {
    const event = update.event;
    const previous = compacted[compacted.length - 1];
    if (
      event &&
      previous?.event &&
      canCoalesceStreamEvent(previous.event, event)
    ) {
      const reasoningDelta =
        event.event === "reasoning.updated"
          ? `${previous.event.data.reasoningDelta ?? ""}${event.data.reasoningDelta ?? ""}`
          : undefined;
      compacted[compacted.length - 1] = {
        ...update,
        event: {
          ...event,
          createdAt: previous.event.createdAt,
          data: {
            ...previous.event.data,
            ...event.data,
            ...(reasoningDelta !== undefined ? { reasoningDelta } : {}),
          },
        },
      };
    } else {
      compacted.push(update);
    }
  }
  return compacted;
}

function hydrateResearchReplay(
  runId: string,
  updates: ResearchRunUpdate[],
  connection?: ResearchConnectionState,
): void {
  if (!updates.length) return;
  useResearchRunStore.setState((state) => {
    const previous = state.sessions[runId];
    if (!previous) return state;
    const compacted = compactReplayUpdates(
      updates.filter(
        (update) => update.event && update.event.id > previous.lastAppliedSeq,
      ),
    );
    let activities = previous.activities;
    let lastAppliedSeq = previous.lastAppliedSeq;
    let run = previous.run;
    for (const update of compacted) {
      if (!update.event || update.event.id <= lastAppliedSeq) continue;
      activities = reduceActivity(activities, update.event);
      lastAppliedSeq = update.event.id;
      if (
        update.run.lastEventSeq > run.lastEventSeq ||
        (update.run.lastEventSeq === run.lastEventSeq &&
          update.run.updatedAt >= run.updatedAt)
      ) {
        run = update.run;
      }
    }
    if (lastAppliedSeq === previous.lastAppliedSeq) return state;
    const planReview = syncPlanReviewState(
      state.planReviewByRunId[runId],
      run,
    );
    const settled = isSettledResearchRun(run, lastAppliedSeq);
    return {
      sessions: {
        ...state.sessions,
        [runId]: {
          ...previous,
          run,
          activities,
          lastAppliedSeq,
          following: settled ? false : previous.following,
          connection: settled ? "idle" : (connection ?? previous.connection),
          error: settled ? null : previous.error,
        },
      },
      ...(planReview && planReview !== state.planReviewByRunId[runId]
        ? {
            planReviewByRunId: {
              ...state.planReviewByRunId,
              [runId]: planReview,
            },
          }
        : {}),
    };
  });
}

export function ingestResearchUpdate(
  run: ResearchRun,
  event?: ResearchEvent,
): void {
  if (!event) {
    flushPendingStreamEvent(run.id);
    useResearchRunStore.getState().ingest(run);
    return;
  }
  if (event.event !== "reasoning.updated" && event.event !== "report.updated") {
    flushPendingStreamEvent(run.id);
    useResearchRunStore.getState().ingest(run, event);
    return;
  }

  const pending = pendingStreamEvents.get(run.id);
  if (pending && event.id <= pending.event.id) {
    return;
  }
  if (pending && canCoalesceStreamEvent(pending.event, event)) {
    const reasoningDelta =
      event.event === "reasoning.updated"
        ? `${pending.event.data.reasoningDelta ?? ""}${event.data.reasoningDelta ?? ""}`
        : undefined;
    pendingStreamEvents.set(run.id, {
      run,
      event: {
        ...event,
        createdAt: pending.event.createdAt,
        data: {
          ...pending.event.data,
          ...event.data,
          ...(reasoningDelta !== undefined ? { reasoningDelta } : {}),
        },
      },
      timer: pending.timer,
    });
    return;
  }
  flushPendingStreamEvent(run.id);
  pendingStreamEvents.set(run.id, {
    run,
    event,
    timer: setTimeout(
      () => flushPendingStreamEvent(run.id),
      STREAM_EVENT_FLUSH_MS,
    ),
  });
}

export function beginExternalResearchFollow(
  run: ResearchRun,
  stop: () => void,
): () => void {
  ingestResearchUpdate(run);
  useResearchRunStore.getState().openPanel(run.id);
  useResearchRunStore.getState().setConnectionError(run.id, null);
  const stops = externalFollowerStops.get(run.id) ?? new Set();
  stops.add(stop);
  externalFollowerStops.set(run.id, stops);
  // the store owns the stream, so a caller that stops reading cannot stall ingestion.
  ensureResearchRunFollowed(run.id, run);
  // The caller handed us the run it just created, so there is no history to restore.
  useResearchRunStore.getState().setFollowing(run.id, true, "connected");
  return () => {
    const currentStops = externalFollowerStops.get(run.id);
    currentStops?.delete(stop);
    if (currentStops?.size === 0) externalFollowerStops.delete(run.id);
  };
}

/** Yield the run each time the store applies something to it, until it settles or *signal*
 *  aborts. Independent of the event stream, so a slow consumer cannot stall ingestion. */
export async function* watchResearchRun(
  runId: string,
  options: { signal?: AbortSignal } = {},
): AsyncGenerator<ResearchRun> {
  const { signal } = options;
  let notify: (() => void) | null = null;
  let dirty = true;
  const wake = () => {
    dirty = true;
    notify?.();
  };
  const unsubscribe = useResearchRunStore.subscribe(wake);
  signal?.addEventListener("abort", wake, { once: true });
  try {
    while (!signal?.aborted) {
      const session = useResearchRunStore.getState().sessions[runId];
      // a follower that gave up never restarts, so surface it rather than park forever.
      if (session?.error && !ownedFollowers.has(runId)) {
        throw new Error(session.error);
      }
      // Nothing to watch: returning beats spinning the microtask queue on an absent session.
      if (!session) return;
      if (dirty) {
        dirty = false;
        yield session.run;
        if (isSettledResearchRun(session.run, session.lastAppliedSeq)) return;
        continue;
      }
      await new Promise<void>((resolve) => {
        if (dirty || signal?.aborted) {
          resolve();
          return;
        }
        notify = resolve;
      });
      notify = null;
    }
  } finally {
    unsubscribe();
    signal?.removeEventListener("abort", wake);
  }
}

export function ensureResearchRunFollowed(
  runId: string,
  initialRun?: ResearchRun,
): void {
  if (initialRun) ingestResearchUpdate(initialRun);
  const state = useResearchRunStore.getState();
  const session = state.sessions[runId];
  if (
    session &&
    isSettledResearchRun(session.run, session.lastAppliedSeq)
  ) {
    state.setConnectionError(runId, null);
    state.setFollowing(runId, false, "idle");
    return;
  }
  if (session?.error) return;
  if (state.sessions[runId]?.following || ownedFollowers.has(runId)) return;
  const controller = new AbortController();
  ownedFollowers.set(runId, controller);
  state.setFollowing(runId, true, "connecting");
  void (async () => {
    let replayThroughSeq = 0;
    let replaying = true;
    const replayUpdates: ResearchRunUpdate[] = [];
    const flushReplay = (markConnected = true) => {
      if (replayUpdates.length) {
        hydrateResearchReplay(
          runId,
          replayUpdates.splice(0),
          markConnected ? "connected" : undefined,
        );
      }
      replaying = false;
      if (markConnected) {
        useResearchRunStore.getState().setFollowing(runId, true, "connected");
      }
    };
    try {
      for await (const update of followResearchRun(runId, {
        initialRun,
        signal: controller.signal,
        replayFrom: session?.lastAppliedSeq ?? 0,
      })) {
        if (update.source === "snapshot") {
          const appliedSeq =
            useResearchRunStore.getState().sessions[runId]?.lastAppliedSeq ?? 0;
          if (!replaying && update.run.lastEventSeq > appliedSeq) {
            replaying = true;
            useResearchRunStore
              .getState()
              .setFollowing(runId, true, "reconnecting");
          }
          replayThroughSeq = Math.max(
            replayThroughSeq,
            update.run.lastEventSeq,
          );
          ingestResearchUpdate(update.run);
          if (replayThroughSeq === 0) flushReplay();
          continue;
        }
        if (replaying && update.event && update.event.id <= replayThroughSeq) {
          replayUpdates.push(update);
          if (update.event.id >= replayThroughSeq) flushReplay();
          continue;
        }
        if (replaying) flushReplay();
        ingestResearchUpdate(update.run, update.event);
        useResearchRunStore.getState().setFollowing(runId, true, "connected");
      }
      if (replaying) flushReplay();
      useResearchRunStore.getState().setConnectionError(runId, null);
    } catch (error) {
      if (!controller.signal.aborted) {
        useResearchRunStore
          .getState()
          .setConnectionError(
            runId,
            error instanceof Error
              ? error.message
              : "Research activity disconnected",
          );
      }
    } finally {
      if (replaying) flushReplay(false);
      flushPendingStreamEvent(runId);
      const stillOwnsFollow = ownedFollowers.get(runId) === controller;
      if (stillOwnsFollow)
        ownedFollowers.delete(runId);
      if (stillOwnsFollow) {
        const run = useResearchRunStore.getState().sessions[runId]?.run;
        useResearchRunStore
          .getState()
          .setFollowing(
            runId,
            false,
            terminalStatuses.has(run?.status ?? "") ? "idle" : "disconnected",
          );
      }
    }
  })();
}

export function resetResearchRunState(): void {
  for (const controller of ownedFollowers.values()) controller.abort();
  ownedFollowers.clear();
  for (const stops of externalFollowerStops.values()) {
    for (const stop of stops) stop();
  }
  externalFollowerStops.clear();
  for (const pending of pendingStreamEvents.values()) clearTimeout(pending.timer);
  pendingStreamEvents.clear();
  useResearchRunStore.setState({
    sessions: {},
    latestRunByThreadId: {},
    claimedThreadIds: {},
    activityOpenByRunId: {},
    planReviewByRunId: {},
    openRunId: null,
  });
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, resetResearchRunState);
}
