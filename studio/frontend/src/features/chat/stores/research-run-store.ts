// SPDX-License-Identifier: AGPL-3.0-only

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
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

const terminalStatuses = new Set(["completed", "failed", "cancelled"]);

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
  if (event.event !== "reasoning.updated") {
    const activeReasoningIndex = findLastActivityIndex(
      next,
      (activity) =>
        activity.kind === "reasoning" && activity.state === "running",
    );
    if (activeReasoningIndex >= 0) {
      next[activeReasoningIndex] = {
        ...next[activeReasoningIndex],
        state: "complete",
      };
    }
  }
  if (event.event === "reasoning.updated") {
    const phase = event.data.phase ?? "unknown";
    const callId = event.data.callId ?? `${phase}-${event.id}`;
    const id = `reasoning-${attempt}-${callId}`;
    const existingIndex = next.findIndex((activity) => activity.id === id);
    const delta = event.data.reasoningDelta ?? "";
    const title =
      phase === "planning"
        ? "Planning an approach"
        : phase === "synthesis"
          ? "Connecting the findings"
          : "Choosing the next step";
    if (existingIndex >= 0) {
      const existing = next[existingIndex];
      next[existingIndex] = {
        ...existing,
        seq: event.id,
        reasoning: `${existing.reasoning ?? ""}${delta}`,
        state: "running",
      };
    } else {
      const activeReasoningIndex = findLastActivityIndex(
        next,
        (activity) =>
          activity.kind === "reasoning" && activity.state === "running",
      );
      if (activeReasoningIndex >= 0) {
        next[activeReasoningIndex] = {
          ...next[activeReasoningIndex],
          state: "complete",
        };
      }
      next.push({
        id,
        seq: event.id,
        attempt,
        kind: "reasoning",
        createdAt: event.createdAt,
        title,
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
      const snapshot = event.run.steps.find(
        (step) => step.position === stepPosition,
      );
      next[index] = {
        ...activity,
        seq: event.id,
        state: event.event === "step.failed" ? "failed" : "complete",
        detail:
          event.event === "step.failed"
            ? (event.data.error ?? "The tool could not complete this action.")
            : `${event.data.sourceCount ?? activity.sources?.length ?? 0} sources found`,
        evidenceSources: snapshot?.result?.evidenceSources,
        excerpt: snapshot?.result?.excerpt,
      };
    }
    return next;
  }

  if (event.event === "report.updated") {
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

  if (event.event === "run.started" && event.data.resumed) {
    for (let index = next.length - 1; index >= 0; index -= 1) {
      const activity = next[index];
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
      return {
        sessions: { ...state.sessions, [run.id]: session },
        claimedThreadIds: state.claimedThreadIds[run.threadId]
          ? state.claimedThreadIds
          : { ...state.claimedThreadIds, [run.threadId]: true },
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
  useResearchRunStore.getState().setFollowing(run.id, true, "connected");
  const stops = externalFollowerStops.get(run.id) ?? new Set();
  stops.add(stop);
  externalFollowerStops.set(run.id, stops);
  return () => {
    const currentStops = externalFollowerStops.get(run.id);
    currentStops?.delete(stop);
    if (currentStops?.size === 0) externalFollowerStops.delete(run.id);
    flushPendingStreamEvent(run.id);
    const latest = useResearchRunStore.getState().sessions[run.id]?.run;
    useResearchRunStore
      .getState()
      .setFollowing(
        run.id,
        false,
        terminalStatuses.has(latest?.status ?? "") ? "idle" : "disconnected",
      );
  };
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

export function stopResearchRunFollower(runId: string): void {
  flushPendingStreamEvent(runId);
  ownedFollowers.get(runId)?.abort();
  ownedFollowers.delete(runId);
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
