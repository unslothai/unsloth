// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

import {
  PROJECT_VERIFICATION_MAX_CHECKS,
  PROJECT_VERIFICATION_MAX_COMMAND_BYTES,
  PROJECT_VERIFICATION_MAX_KIND_BYTES,
  PROJECT_VERIFICATION_MAX_LOG_LIMIT_BYTES,
  ProjectVerificationApiError,
  type ProjectVerificationCheck,
  type ProjectVerificationConfig,
  ProjectVerificationPollBackoff,
  ProjectVerificationRequestGuard,
  type ProjectVerificationResult,
  type ProjectVerificationRun,
  type ProjectVerificationRunMetadata,
  type ProjectVerificationRunSummary,
  cancelProjectVerificationRun,
  getProjectVerificationConfig,
  getProjectVerificationRun,
  listProjectVerificationRuns,
  projectVerificationChecksError,
  projectVerificationCommandNeedsExactPreview,
  projectVerificationConfigCanReplace,
  projectVerificationPollingEvidenceMarker,
  projectVerificationPollingEvidenceRegressed,
  projectVerificationRunHasDetails,
  projectVerificationRunIsActive,
  projectVerificationRunSummary,
  saveProjectVerificationConfig,
  startProjectVerification,
  subscribeProjectVerificationUpdated,
  verificationConflictRequiresRefresh,
  visibleProjectVerificationCommand,
  visibleProjectVerificationText,
} from "../api/project-verification-api";
import type { ProjectRecord as StoredProjectRecord } from "../types";

type ProjectRecord = StoredProjectRecord & { workspaceAvailable?: boolean };

const MAX_RENDERED_OUTPUT_CHARACTERS = 128_000;
const DEFAULT_TIMEOUT_SECONDS = 300;
const DEFAULT_LOG_LIMIT_BYTES = 256 * 1024;

let draftSequence = 0;

type DraftCheck = ProjectVerificationCheck & { draftId: string };
type Mutation = "save" | "start" | "cancel" | null;
type PollError = {
  projectId: string;
  runId: string;
  message: string;
  terminal: boolean;
};
type VerificationRunRecord =
  | ProjectVerificationRun
  | ProjectVerificationRunSummary;

function draftCheck(check: Partial<ProjectVerificationCheck> = {}): DraftCheck {
  draftSequence += 1;
  return {
    draftId: `verification-check-${draftSequence}`,
    name: check.name ?? "",
    kind: check.kind ?? "custom",
    command: check.command ?? "",
    required: check.required ?? true,
    timeoutSeconds: check.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS,
    logLimitBytes: check.logLimitBytes ?? DEFAULT_LOG_LIMIT_BYTES,
  };
}

function draftChecks(checks: ProjectVerificationCheck[]): DraftCheck[] {
  return checks.map((check) => draftCheck(check));
}

export function projectVerificationPublicChecks(
  checks: ReadonlyArray<ProjectVerificationCheck>,
): ProjectVerificationCheck[] {
  return checks.map((check) => ({
    name: check.name,
    kind: check.kind,
    command: check.command,
    required: check.required,
    timeoutSeconds: check.timeoutSeconds,
    logLimitBytes: check.logLimitBytes,
  }));
}

function checksMatch(
  left: DraftCheck[],
  right: ProjectVerificationCheck[],
): boolean {
  return (
    JSON.stringify(projectVerificationPublicChecks(left)) ===
    JSON.stringify(right)
  );
}

export function mergeRuns(
  current: ProjectVerificationRunSummary[],
  incoming: ProjectVerificationRunMetadata[],
  replaceTerminalHistory = false,
): ProjectVerificationRunSummary[] {
  const retainedCurrent = replaceTerminalHistory
    ? current.filter((run) => projectVerificationRunIsActive(run))
    : current;
  const byId = new Map(retainedCurrent.map((run) => [run.id, run]));
  for (const incomingRun of incoming) {
    const run = projectVerificationRunSummary(incomingRun);
    const previous = byId.get(run.id);
    if (!previous || projectVerificationRunCanReplace(previous, run)) {
      byId.set(run.id, run);
    }
  }
  return [...byId.values()]
    .sort((left, right) => {
      const leftActive = projectVerificationRunIsActive(left);
      const rightActive = projectVerificationRunIsActive(right);
      if (leftActive !== rightActive) {
        return leftActive ? -1 : 1;
      }
      const sequenceDifference =
        (right.historySequence ?? 0) - (left.historySequence ?? 0);
      if (sequenceDifference !== 0) {
        return sequenceDifference;
      }
      if (right.evidenceRevision !== left.evidenceRevision) {
        return right.evidenceRevision - left.evidenceRevision;
      }
      if (right.updatedAt !== left.updatedAt) {
        return right.updatedAt - left.updatedAt;
      }
      return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
    })
    .slice(0, 20);
}

export function projectVerificationRunCanReplace(
  previous: ProjectVerificationRunMetadata,
  next: ProjectVerificationRunMetadata,
): boolean {
  if (next.evidenceRevision !== previous.evidenceRevision) {
    return next.evidenceRevision > previous.evidenceRevision;
  }
  if (next.updatedAt !== previous.updatedAt) {
    return next.updatedAt > previous.updatedAt;
  }
  const previousActive = projectVerificationRunIsActive(previous);
  const nextActive = projectVerificationRunIsActive(next);
  return previousActive || !nextActive;
}

export function projectVerificationHistoryCanReplace(
  refreshGeneration: number,
  currentGeneration: number,
): boolean {
  return refreshGeneration === currentGeneration;
}

export function projectVerificationConfigFromRefresh(
  current: ProjectVerificationConfig | null,
  incoming: ProjectVerificationConfig,
  refreshOwnsRunState: boolean,
): ProjectVerificationConfig {
  if (refreshOwnsRunState) {
    return incoming;
  }
  return {
    ...incoming,
    activeRun:
      current?.projectId === incoming.projectId ? current.activeRun : null,
  };
}

export function projectVerificationRunsFromRefresh(
  incoming: ProjectVerificationRunMetadata[],
  refreshOwnsRunState: boolean,
): ProjectVerificationRunMetadata[] {
  return refreshOwnsRunState
    ? incoming
    : incoming.filter((run) => !projectVerificationRunIsActive(run));
}

export function projectVerificationAcceptedRun(
  merged: ProjectVerificationRunSummary[],
  incoming: ProjectVerificationRunMetadata,
): {
  run: ProjectVerificationRunSummary;
  incomingAccepted: boolean;
} | null {
  const run = merged.find((candidate) => candidate.id === incoming.id);
  if (!run) {
    return null;
  }
  return {
    run,
    incomingAccepted:
      run.evidenceRevision === incoming.evidenceRevision &&
      run.updatedAt === incoming.updatedAt &&
      run.status === incoming.status &&
      run.cancelRequested === incoming.cancelRequested,
  };
}

export function projectVerificationAcceptedPollMarker(
  current: number | undefined,
  accepted: ProjectVerificationRunMetadata,
): number {
  return current === undefined
    ? accepted.evidenceRevision
    : Math.max(current, accepted.evidenceRevision);
}

export function projectVerificationActiveRunAfterMerge(
  current: VerificationRunRecord | null,
  mergedActive: ProjectVerificationRunSummary | null,
  incoming: VerificationRunRecord,
  incomingAccepted: boolean,
): VerificationRunRecord | null {
  if (mergedActive === null) {
    return null;
  }
  if (incomingAccepted && mergedActive.id === incoming.id) {
    return incoming;
  }
  if (
    current?.id === mergedActive.id &&
    (current.evidenceRevision > mergedActive.evidenceRevision ||
      (current.evidenceRevision === mergedActive.evidenceRevision &&
        (current.updatedAt > mergedActive.updatedAt ||
          (current.updatedAt === mergedActive.updatedAt &&
            current.status === mergedActive.status &&
            current.cancelRequested === mergedActive.cancelRequested))))
  ) {
    return current;
  }
  return mergedActive;
}

export function projectVerificationRunCardKey(
  run: ProjectVerificationRunMetadata,
): string {
  return `${run.id}:${run.evidenceRevision}:${run.updatedAt}`;
}

export function projectVerificationDetailMatches(
  detail: ProjectVerificationRun,
  run: ProjectVerificationRunMetadata,
): boolean {
  return (
    detail.id === run.id &&
    detail.evidenceRevision >= run.evidenceRevision &&
    detail.updatedAt >= run.updatedAt
  );
}

export function visibleProjectVerificationPollError(
  pollError: PollError | null,
  projectId: string,
  activeRunId: string | null,
): string | null {
  if (
    !pollError ||
    pollError.projectId !== projectId ||
    (pollError.runId !== activeRunId &&
      !(pollError.terminal && activeRunId === null))
  ) {
    return null;
  }
  return pollError.message;
}

export function withoutProjectVerificationRun(
  runs: ProjectVerificationRunSummary[],
  runId: string,
): ProjectVerificationRunSummary[] {
  return runs.filter((run) => run.id !== runId);
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Project verification request failed.";
}

export function projectVerificationRefreshFailureMessage(
  result: PromiseSettledResult<unknown>,
  refreshOwnsState: boolean,
): string | null {
  if (result.status !== "rejected" || !refreshOwnsState) {
    return null;
  }
  return errorMessage(result.reason);
}

function statusClass(status: string): string {
  if (status === "passed") {
    return "text-emerald-600 dark:text-emerald-400";
  }
  if (["failed", "blocked", "timed_out", "interrupted"].includes(status)) {
    return "text-destructive";
  }
  if (status === "cancelled" || status === "cancelling") {
    return "text-amber-600 dark:text-amber-400";
  }
  return "text-muted-foreground";
}

function timestamp(value: number | null): string {
  return value === null ? "Not completed" : new Date(value).toLocaleString();
}

function currentActiveRun(
  config: ProjectVerificationConfig,
  runs: ProjectVerificationRunSummary[],
): VerificationRunRecord | null {
  const configured = config.activeRun;
  if (!configured) {
    return runs.find((run) => projectVerificationRunIsActive(run)) ?? null;
  }
  const listed = runs.find((run) => run.id === configured.id);
  if (!listed || !projectVerificationRunCanReplace(configured, listed)) {
    return projectVerificationRunIsActive(configured) ? configured : null;
  }
  if (!projectVerificationRunIsActive(listed)) {
    return null;
  }
  return projectVerificationActiveRunAfterMerge(
    configured,
    listed,
    listed,
    false,
  );
}

export function ProjectVerificationPanel({
  project,
}: {
  project: ProjectRecord;
}) {
  return (
    <ProjectVerificationPanelForProject key={project.id} project={project} />
  );
}

function ProjectVerificationPanelForProject({
  project,
}: {
  project: ProjectRecord;
}) {
  const [config, setConfig] = useState<ProjectVerificationConfig | null>(null);
  const [draft, setDraft] = useState<DraftCheck[]>([]);
  const [runs, setRuns] = useState<ProjectVerificationRunSummary[]>([]);
  const [activeRun, setActiveRun] = useState<VerificationRunRecord | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [profileFresh, setProfileFresh] = useState(false);
  const [mutation, setMutation] = useState<Mutation>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<PollError | null>(null);
  const requestGuard = useRef(new ProjectVerificationRequestGuard());
  const pollGuard = useRef(new ProjectVerificationRequestGuard());
  const mutationGuard = useRef(new ProjectVerificationRequestGuard());
  const configRef = useRef<ProjectVerificationConfig | null>(null);
  const draftRef = useRef<DraftCheck[]>([]);
  const runsRef = useRef<ProjectVerificationRunSummary[]>([]);
  const activeRunRef = useRef<VerificationRunRecord | null>(null);
  const configStateGenerationRef = useRef(0);
  const runStateGenerationRef = useRef(0);

  useEffect(() => {
    configRef.current = config;
  }, [config]);
  useEffect(() => {
    activeRunRef.current = activeRun;
  }, [activeRun]);

  const commitDraft = useCallback(
    (update: DraftCheck[] | ((current: DraftCheck[]) => DraftCheck[])) => {
      const next =
        typeof update === "function" ? update(draftRef.current) : update;
      draftRef.current = next;
      setDraft(next);
    },
    [],
  );

  const commitActiveRun = useCallback((next: VerificationRunRecord | null) => {
    activeRunRef.current = next;
    setActiveRun(next);
  }, []);

  const commitConfig = useCallback((next: ProjectVerificationConfig) => {
    if (!projectVerificationConfigCanReplace(configRef.current, next)) {
      return false;
    }
    configRef.current = next;
    configStateGenerationRef.current += 1;
    setConfig(next);
    setProfileFresh(true);
    return true;
  }, []);

  const applyConfig = useCallback(
    (next: ProjectVerificationConfig) => {
      const previous = configRef.current;
      const hasLocalDraft =
        previous?.projectId === next.projectId &&
        !checksMatch(draftRef.current, previous.checks);
      if (!commitConfig(next)) {
        return false;
      }
      if (!hasLocalDraft) {
        const nextDraft = draftChecks(next.checks);
        commitDraft(nextDraft);
      }
      return true;
    },
    [commitConfig, commitDraft],
  );

  const applyRuns = useCallback(
    (
      incoming: ProjectVerificationRunMetadata[],
      replaceTerminalHistory = false,
      markExternalUpdate = true,
    ) => {
      if (markExternalUpdate) {
        runStateGenerationRef.current += 1;
      }
      const currentProjectRuns = runsRef.current.filter(
        (run) => run.projectId === project.id,
      );
      const next = mergeRuns(
        currentProjectRuns,
        incoming.filter((run) => run.projectId === project.id),
        replaceTerminalHistory,
      );
      runsRef.current = next;
      setRuns(next);
      return next;
    },
    [project.id],
  );

  const removeRun = useCallback((runId: string) => {
    runStateGenerationRef.current += 1;
    const next = withoutProjectVerificationRun(runsRef.current, runId);
    runsRef.current = next;
    setRuns(next);
  }, []);

  const clearConfigActiveRun = useCallback((runId: string) => {
    const current = configRef.current;
    if (!current || current.activeRun?.id !== runId) {
      return;
    }
    const next = { ...current, activeRun: null };
    configRef.current = next;
    setConfig(next);
  }, []);

  const refresh = useCallback(async () => {
    const revision = requestGuard.current.begin();
    const configStateGeneration = configStateGenerationRef.current;
    const runStateGeneration = runStateGenerationRef.current;
    setLoading(true);
    setError(null);
    const [configResult, runsResult] = await Promise.allSettled([
      getProjectVerificationConfig(project.id),
      listProjectVerificationRuns(project.id),
    ]);
    if (!requestGuard.current.accepts(revision)) {
      return;
    }
    const refreshOwnsRunState = projectVerificationHistoryCanReplace(
      runStateGeneration,
      runStateGenerationRef.current,
    );
    const refreshOwnsConfigState =
      configStateGeneration === configStateGenerationRef.current;
    let currentConfig =
      configRef.current?.projectId === project.id ? configRef.current : null;
    if (configResult.status === "fulfilled") {
      applyConfig(
        projectVerificationConfigFromRefresh(
          currentConfig,
          configResult.value,
          refreshOwnsRunState,
        ),
      );
      currentConfig =
        configRef.current?.projectId === project.id ? configRef.current : null;
    } else if (refreshOwnsConfigState) {
      setProfileFresh(false);
    }
    let mergedRuns = runsRef.current;
    if (currentConfig?.activeRun) {
      mergedRuns = applyRuns([currentConfig.activeRun], false, false);
    }
    if (runsResult.status === "fulfilled") {
      mergedRuns = applyRuns(
        projectVerificationRunsFromRefresh(
          runsResult.value,
          refreshOwnsRunState,
        ),
        refreshOwnsRunState,
        false,
      );
    }
    if (refreshOwnsRunState) {
      if (currentConfig) {
        commitActiveRun(currentActiveRun(currentConfig, mergedRuns));
      } else if (runsResult.status === "fulfilled") {
        commitActiveRun(
          mergedRuns.find((run) => projectVerificationRunIsActive(run)) ?? null,
        );
      }
    }
    const failures = [
      projectVerificationRefreshFailureMessage(
        configResult,
        refreshOwnsConfigState,
      ),
      projectVerificationRefreshFailureMessage(runsResult, refreshOwnsRunState),
    ].filter((failure): failure is string => failure !== null);
    if (failures.length > 0) {
      setError(failures.join(" "));
    } else if (refreshOwnsRunState) {
      setPollError(null);
    }
    setLoading(false);
  }, [applyConfig, applyRuns, commitActiveRun, project.id]);

  useEffect(() => {
    const requests = requestGuard.current;
    const polls = pollGuard.current;
    const mutations = mutationGuard.current;
    requests.activate();
    polls.activate();
    mutations.activate();
    const unsubscribe = subscribeProjectVerificationUpdated(
      project.id,
      (detail) => {
        if (detail.config) {
          applyConfig(detail.config);
        }
        const nextRun = detail.run;
        if (nextRun) {
          const mergedRuns = applyRuns([nextRun]);
          const accepted = projectVerificationAcceptedRun(mergedRuns, nextRun);
          const latestActive = mergedRuns.find((run) =>
            projectVerificationRunIsActive(run),
          );
          commitActiveRun(
            projectVerificationActiveRunAfterMerge(
              activeRunRef.current,
              latestActive ?? null,
              nextRun,
              accepted?.incomingAccepted ?? false,
            ),
          );
          if (
            accepted?.incomingAccepted &&
            !projectVerificationRunIsActive(nextRun)
          ) {
            clearConfigActiveRun(nextRun.id);
          }
        }
      },
    );
    const initialRefresh = window.setTimeout(() => {
      refresh().catch(() => undefined);
    }, 0);
    return () => {
      window.clearTimeout(initialRefresh);
      unsubscribe();
      requests.retire();
      polls.retire();
      mutations.retire();
    };
  }, [
    applyConfig,
    applyRuns,
    clearConfigActiveRun,
    commitActiveRun,
    project.id,
    refresh,
  ]);

  const activeRunId =
    activeRun?.projectId === project.id &&
    projectVerificationRunIsActive(activeRun)
      ? activeRun.id
      : null;

  useEffect(() => {
    if (!activeRunId) {
      return;
    }
    let stopped = false;
    let timer: number | null = null;
    const polls = pollGuard.current;
    const backoff = new ProjectVerificationPollBackoff();
    const initialRun = activeRunRef.current;
    let afterEvidenceRevision =
      initialRun && projectVerificationRunHasDetails(initialRun)
        ? initialRun.evidenceRevision
        : undefined;
    const schedule = (delay: number) => {
      if (!stopped) {
        timer = window.setTimeout(() => void poll(), delay);
      }
    };
    const poll = async () => {
      const revision = polls.begin();
      try {
        const next = await getProjectVerificationRun(
          project.id,
          activeRunId,
          afterEvidenceRevision,
        );
        if (stopped || !polls.accepts(revision)) {
          return;
        }
        if (next === null) {
          setPollError((current) =>
            current?.projectId === project.id && current.runId === activeRunId
              ? null
              : current,
          );
          schedule(backoff.successDelay());
          return;
        }
        const mergedRuns = applyRuns([next]);
        const accepted = projectVerificationAcceptedRun(mergedRuns, next);
        if (accepted === null) {
          schedule(backoff.successDelay());
          return;
        }
        afterEvidenceRevision = projectVerificationAcceptedPollMarker(
          afterEvidenceRevision,
          accepted.run,
        );
        setPollError((current) =>
          current?.projectId === project.id && current.runId === activeRunId
            ? null
            : current,
        );
        const latestActive = mergedRuns.find((run) =>
          projectVerificationRunIsActive(run),
        );
        if (latestActive) {
          commitActiveRun(
            projectVerificationActiveRunAfterMerge(
              activeRunRef.current,
              latestActive,
              next,
              accepted.incomingAccepted,
            ),
          );
          if (latestActive.id === activeRunId) {
            schedule(backoff.successDelay());
          }
        } else {
          commitActiveRun(null);
          clearConfigActiveRun(activeRunId);
        }
      } catch (nextError) {
        if (stopped || !polls.accepts(revision)) {
          return;
        }
        const evidenceRegressed =
          projectVerificationPollingEvidenceRegressed(nextError);
        afterEvidenceRevision = projectVerificationPollingEvidenceMarker(
          afterEvidenceRevision,
          nextError,
        );
        if (evidenceRegressed) {
          removeRun(activeRunId);
          clearConfigActiveRun(activeRunId);
        }
        const delay = backoff.failureDelay(nextError);
        setPollError({
          projectId: project.id,
          runId: activeRunId,
          message: errorMessage(nextError),
          terminal: delay === null,
        });
        if (delay === null) {
          removeRun(activeRunId);
          commitActiveRun(null);
          clearConfigActiveRun(activeRunId);
          setProfileFresh(false);
          return;
        }
        schedule(delay);
      }
    };

    timer = window.setTimeout(() => void poll(), 0);
    return () => {
      stopped = true;
      polls.begin();
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [
    activeRunId,
    applyRuns,
    clearConfigActiveRun,
    commitActiveRun,
    project.id,
    removeRun,
  ]);

  const publicDraft = useMemo(
    () => projectVerificationPublicChecks(draft),
    [draft],
  );
  const validationError = useMemo(
    () => projectVerificationChecksError(publicDraft),
    [publicDraft],
  );
  const dirty = config ? !checksMatch(draft, config.checks) : false;
  const configCurrent = config?.projectId === project.id;
  const activeRunCurrent = activeRun?.projectId === project.id;
  const visibleRuns = runs.filter((run) => run.projectId === project.id);
  const workspaceAvailable =
    configCurrent &&
    project.workspaceAvailable !== false &&
    config?.workspaceAvailable !== false;
  const busy = loading || mutation !== null;
  const runActive =
    activeRunCurrent && projectVerificationRunIsActive(activeRun);
  const maySave = Boolean(
    config &&
      configCurrent &&
      workspaceAvailable &&
      !busy &&
      !runActive &&
      validationError === null &&
      (dirty || !config.active),
  );
  const mayRun = Boolean(
    configCurrent &&
      profileFresh &&
      config?.active &&
      config.execution.available &&
      workspaceAvailable &&
      !busy &&
      !dirty &&
      config.checks.length > 0 &&
      !runActive,
  );
  const visiblePollError = visibleProjectVerificationPollError(
    pollError,
    project.id,
    activeRunId,
  );
  const visibleError = error ?? visiblePollError;

  function updateCheck(
    draftId: string,
    patch: Partial<ProjectVerificationCheck>,
  ): void {
    commitDraft((current) =>
      current.map((check) =>
        check.draftId === draftId ? { ...check, ...patch } : check,
      ),
    );
  }

  async function save(): Promise<void> {
    if (!config || !maySave) {
      return;
    }
    const payload = publicDraft;
    const revision = mutationGuard.current.begin();
    setMutation("save");
    setError(null);
    try {
      const next = await saveProjectVerificationConfig(project.id, {
        checks: payload,
        expectedRevision: config.revision,
        workspaceRevision: config.workspaceRevision,
      });
      if (!mutationGuard.current.accepts(revision)) {
        return;
      }
      if (!commitConfig(next)) {
        return;
      }
      const nextDraft = draftChecks(next.checks);
      commitDraft(nextDraft);
    } catch (nextError) {
      if (!mutationGuard.current.accepts(revision)) {
        return;
      }
      if (
        nextError instanceof ProjectVerificationApiError &&
        nextError.status === 422
      ) {
        setError(`Verification profile rejected: ${nextError.message}`);
      } else if (verificationConflictRequiresRefresh(nextError)) {
        await refresh().catch(() => undefined);
        if (!mutationGuard.current.accepts(revision)) {
          return;
        }
        setError(
          "Verification settings or the workspace changed. The local draft was preserved. Review it against the refreshed revision before saving again.",
        );
      } else {
        setError(errorMessage(nextError));
      }
    } finally {
      if (mutationGuard.current.accepts(revision)) {
        setMutation(null);
      }
    }
  }

  async function run(): Promise<void> {
    if (!config || !mayRun) {
      return;
    }
    const revision = mutationGuard.current.begin();
    setMutation("start");
    setError(null);
    try {
      const next = await startProjectVerification(project.id, {
        configRevision: config.revision,
        workspaceRevision: config.workspaceRevision,
      });
      if (!mutationGuard.current.accepts(revision)) {
        return;
      }
      const mergedRuns = applyRuns([next]);
      const accepted = projectVerificationAcceptedRun(mergedRuns, next);
      const latestActive = mergedRuns.find((run) =>
        projectVerificationRunIsActive(run),
      );
      setPollError(null);
      const currentRun = projectVerificationActiveRunAfterMerge(
        activeRunRef.current,
        latestActive ?? null,
        next,
        accepted?.incomingAccepted ?? false,
      );
      commitActiveRun(currentRun);
      const currentConfig = configRef.current;
      if (currentConfig) {
        const nextConfig = {
          ...currentConfig,
          activeRun:
            currentRun && projectVerificationRunHasDetails(currentRun)
              ? currentRun
              : null,
        };
        configRef.current = nextConfig;
        setConfig(nextConfig);
      }
    } catch (nextError) {
      if (!mutationGuard.current.accepts(revision)) {
        return;
      }
      if (verificationConflictRequiresRefresh(nextError)) {
        setProfileFresh(false);
        await refresh().catch(() => undefined);
        if (!mutationGuard.current.accepts(revision)) {
          return;
        }
      }
      setError(errorMessage(nextError));
    } finally {
      if (mutationGuard.current.accepts(revision)) {
        setMutation(null);
      }
    }
  }

  async function cancel(): Promise<void> {
    if (!activeRun || activeRun.projectId !== project.id || mutation !== null) {
      return;
    }
    const revision = mutationGuard.current.begin();
    setMutation("cancel");
    setError(null);
    try {
      const next = await cancelProjectVerificationRun(project.id, activeRun.id);
      if (!mutationGuard.current.accepts(revision)) {
        return;
      }
      const mergedRuns = applyRuns([next]);
      const accepted = projectVerificationAcceptedRun(mergedRuns, next);
      const latestActive = mergedRuns.find((run) =>
        projectVerificationRunIsActive(run),
      );
      const currentRun = projectVerificationActiveRunAfterMerge(
        activeRunRef.current,
        latestActive ?? null,
        next,
        accepted?.incomingAccepted ?? false,
      );
      commitActiveRun(currentRun);
      if (currentRun === null) {
        clearConfigActiveRun(next.id);
      }
    } catch (nextError) {
      if (!mutationGuard.current.accepts(revision)) {
        return;
      }
      setError(errorMessage(nextError));
    } finally {
      if (mutationGuard.current.accepts(revision)) {
        setMutation(null);
      }
    }
  }

  return (
    <div className="mt-3 border-t border-border/70 pt-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-foreground">
            Project verification
          </p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Save and run the project&apos;s trusted test, lint, typecheck, and
            build commands.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {activeRun && activeRunCurrent && runActive ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={busy || activeRun.cancelRequested === true}
              onClick={() => void cancel()}
            >
              {mutation === "cancel" || activeRun.cancelRequested === true
                ? "Cancelling..."
                : "Cancel run"}
            </Button>
          ) : (
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={!mayRun}
              onClick={() => void run()}
            >
              {mutation === "start" ? "Starting..." : "Run checks"}
            </Button>
          )}
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={busy}
            onClick={() => void refresh()}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
      </div>

      <div className="mt-3 rounded-xl border border-amber-500/25 bg-amber-500/5 px-3 py-2 text-xs text-muted-foreground">
        Source freshness is unverified. These runs preserve bounded command
        evidence, but they do not yet certify that the project source stayed
        unchanged.
      </div>

      {configCurrent && !workspaceAvailable ? (
        <p className="mt-2 text-xs text-muted-foreground">
          Reconnect the project folder before editing or running verification.
        </p>
      ) : null}
      {config && configCurrent && !config.active && workspaceAvailable ? (
        <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
          This saved profile is not bound to the current workspace revision.
          Save it again before running checks.
        </p>
      ) : null}
      {config && configCurrent && !config.execution.available ? (
        <p className="mt-2 text-xs text-destructive" role="alert">
          {config.execution.reason ??
            "Secure supervised project commands are unavailable. No verification command will run."}
        </p>
      ) : null}
      {runActive ? (
        <p className="mt-2 text-xs text-muted-foreground">
          The saved profile is locked until this verification run finishes.
        </p>
      ) : null}
      {configCurrent && !profileFresh && !loading ? (
        <p className="mt-2 text-xs text-muted-foreground">
          Refresh the verification profile before starting a run. Editing stays
          available.
        </p>
      ) : null}
      {visibleError ? (
        <p className="mt-2 text-xs text-destructive" role="alert">
          {visibleError}
        </p>
      ) : null}

      {config && configCurrent ? (
        <div className="mt-3 space-y-3">
          {draft.map((check, index) => (
            <VerificationCheckEditor
              key={check.draftId}
              check={check}
              index={index}
              disabled={busy || runActive || !workspaceAvailable}
              onChange={(patch) => updateCheck(check.draftId, patch)}
              onRemove={() =>
                commitDraft((current) =>
                  current.filter((item) => item.draftId !== check.draftId),
                )
              }
            />
          ))}
          {draft.length === 0 ? (
            <p className="rounded-xl bg-background/70 px-3 py-3 text-xs text-muted-foreground">
              No verification checks are configured.
            </p>
          ) : null}
          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={
                busy ||
                runActive ||
                !workspaceAvailable ||
                draft.length >= PROJECT_VERIFICATION_MAX_CHECKS
              }
              onClick={() =>
                commitDraft((current) => [...current, draftCheck()])
              }
            >
              Add check
            </Button>
            <Button
              type="button"
              size="sm"
              disabled={!maySave}
              onClick={() => void save()}
            >
              {mutation === "save"
                ? "Saving..."
                : config.active
                  ? "Save profile"
                  : "Bind and save profile"}
            </Button>
            <span className="text-xs text-muted-foreground">
              Saved revision {config.revision}, workspace revision{" "}
              {config.workspaceRevision}
            </span>
          </div>
          {validationError ? (
            <p className="text-xs text-destructive" role="alert">
              {validationError}
            </p>
          ) : dirty ? (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              Save this draft before running verification.
            </p>
          ) : null}
        </div>
      ) : (
        <p className="mt-3 text-xs text-muted-foreground">
          {loading ? "Loading verification profile..." : "Profile unavailable."}
        </p>
      )}

      {activeRun && activeRunCurrent ? (
        <div className="mt-4">
          <p className="mb-2 text-xs font-semibold text-foreground">
            Active run
          </p>
          <VerificationRunCard
            run={activeRun}
            expanded={true}
            fetchDetailsOnOpen={false}
          />
        </div>
      ) : null}

      {visibleRuns.length > 0 ? (
        <div className="mt-4">
          <p className="mb-2 text-xs font-semibold text-foreground">
            Run history
          </p>
          <div className="space-y-2">
            {visibleRuns
              .filter((run) => run.id !== activeRun?.id)
              .slice(0, 5)
              .map((run) => (
                <VerificationRunCard key={run.id} run={run} />
              ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function VerificationCheckEditor({
  check,
  index,
  disabled,
  onChange,
  onRemove,
}: {
  check: DraftCheck;
  index: number;
  disabled: boolean;
  onChange: (patch: Partial<ProjectVerificationCheck>) => void;
  onRemove: () => void;
}) {
  const showExactCommandPreview = projectVerificationCommandNeedsExactPreview(
    check.command,
  );
  return (
    <div className="rounded-xl bg-background/70 px-3 py-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-medium text-foreground">Check {index + 1}</p>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          disabled={disabled}
          onClick={onRemove}
        >
          Remove
        </Button>
      </div>
      <div className="mt-2 grid gap-2 sm:grid-cols-2">
        <label
          className="text-xs text-muted-foreground"
          htmlFor={`${check.draftId}-name`}
        >
          Name
          <Input
            id={`${check.draftId}-name`}
            className="mt-1"
            value={check.name}
            disabled={disabled}
            onChange={(event) => onChange({ name: event.target.value })}
          />
        </label>
        <label
          className="text-xs text-muted-foreground"
          htmlFor={`${check.draftId}-kind`}
        >
          Kind
          <Input
            id={`${check.draftId}-kind`}
            className="mt-1"
            value={check.kind}
            disabled={disabled}
            maxLength={PROJECT_VERIFICATION_MAX_KIND_BYTES}
            placeholder="test"
            onChange={(event) => onChange({ kind: event.target.value })}
          />
        </label>
      </div>
      <label
        className="mt-2 block text-xs text-muted-foreground"
        htmlFor={`${check.draftId}-command`}
      >
        Command
        <Textarea
          id={`${check.draftId}-command`}
          className="mt-1 min-h-20 font-mono text-xs"
          value={check.command}
          disabled={disabled}
          dir="ltr"
          style={{ unicodeBidi: "isolate" }}
          maxLength={PROJECT_VERIFICATION_MAX_COMMAND_BYTES}
          onChange={(event) => onChange({ command: event.target.value })}
        />
      </label>
      {showExactCommandPreview ? (
        <div className="mt-2 rounded-md bg-muted/35 px-2 py-2">
          <p className="text-[11px] text-muted-foreground">
            Exact command preview. Every non-ASCII code point is escaped.
          </p>
          <pre
            aria-label="Exact escaped command preview"
            dir="ltr"
            className="mt-1 whitespace-pre-wrap break-words text-[11px] text-foreground"
            style={{ unicodeBidi: "isolate" }}
          >
            {visibleProjectVerificationCommand(check.command)}
          </pre>
        </div>
      ) : null}
      <div className="mt-2 grid gap-2 sm:grid-cols-2">
        <label
          className="text-xs text-muted-foreground"
          htmlFor={`${check.draftId}-timeout`}
        >
          Timeout seconds
          <Input
            id={`${check.draftId}-timeout`}
            className="mt-1"
            type="number"
            min={1}
            max={3600}
            value={check.timeoutSeconds}
            disabled={disabled}
            onChange={(event) =>
              onChange({ timeoutSeconds: Number(event.target.value) })
            }
          />
        </label>
        <label
          className="text-xs text-muted-foreground"
          htmlFor={`${check.draftId}-log-limit`}
        >
          Log limit bytes
          <Input
            id={`${check.draftId}-log-limit`}
            className="mt-1"
            type="number"
            min={1024}
            max={PROJECT_VERIFICATION_MAX_LOG_LIMIT_BYTES}
            step={1024}
            value={check.logLimitBytes}
            disabled={disabled}
            onChange={(event) =>
              onChange({ logLimitBytes: Number(event.target.value) })
            }
          />
        </label>
      </div>
      <label
        className="mt-3 flex items-center gap-2 text-xs text-muted-foreground"
        htmlFor={`${check.draftId}-required`}
      >
        <Checkbox
          id={`${check.draftId}-required`}
          checked={check.required}
          disabled={disabled}
          onCheckedChange={(checked) =>
            onChange({ required: checked === true })
          }
        />
        Required for the run to pass
      </label>
    </div>
  );
}

export function VerificationRunCard({
  run,
  expanded = false,
  fetchDetailsOnOpen = true,
}: {
  run: VerificationRunRecord;
  expanded?: boolean;
  fetchDetailsOnOpen?: boolean;
}) {
  return (
    <VerificationRunCardForMarker
      key={projectVerificationRunCardKey(run)}
      run={run}
      expanded={expanded}
      fetchDetailsOnOpen={fetchDetailsOnOpen}
    />
  );
}

function VerificationRunCardForMarker({
  run,
  expanded,
  fetchDetailsOnOpen,
}: {
  run: VerificationRunRecord;
  expanded: boolean;
  fetchDetailsOnOpen: boolean;
}) {
  const runProjectId = run.projectId;
  const runId = run.id;
  const runEvidenceRevision = run.evidenceRevision;
  const runUpdatedAt = run.updatedAt;
  const runHasDetails = projectVerificationRunHasDetails(run);
  const [open, setOpen] = useState(expanded);
  const [detail, setDetail] = useState<ProjectVerificationRun | null>(() =>
    runHasDetails ? run : null,
  );
  const [detailFailure, setDetailFailure] = useState<{
    attemptKey: string;
    message: string;
  } | null>(null);
  const detailGuard = useRef(new ProjectVerificationRequestGuard());
  const detailAttemptKey = useRef<string | null>(null);
  const attemptKey = `${runId}:${runEvidenceRevision}:${runUpdatedAt}`;
  const currentDetail =
    detail && projectVerificationDetailMatches(detail, run) ? detail : null;
  const exactRun = runHasDetails ? run : currentDetail;
  const detailError =
    detailFailure?.attemptKey === attemptKey ? detailFailure.message : null;
  const displayStatus = exactRun
    ? exactRun.cancelRequested && projectVerificationRunIsActive(exactRun)
      ? "cancelling"
      : exactRun.status
    : "Details not loaded";

  useEffect(() => {
    const guard = detailGuard.current;
    guard.activate();
    return () => guard.retire();
  }, []);

  useEffect(() => {
    if (!open) {
      detailAttemptKey.current = null;
      return;
    }
    if (runHasDetails) {
      detailAttemptKey.current = null;
      return;
    }
    if (
      !fetchDetailsOnOpen ||
      currentDetail !== null ||
      detailAttemptKey.current === attemptKey
    ) {
      return;
    }
    const guard = detailGuard.current;
    const revision = guard.begin();
    detailAttemptKey.current = attemptKey;
    getProjectVerificationRun(runProjectId, runId)
      .then((next) => {
        if (!guard.accepts(revision)) {
          return;
        }
        if (
          next === null ||
          next.projectId !== runProjectId ||
          next.id !== runId ||
          next.evidenceRevision < runEvidenceRevision ||
          next.updatedAt < runUpdatedAt
        ) {
          setDetailFailure({
            attemptKey,
            message: "Run details were not returned for this project.",
          });
          return;
        }
        setDetail(next);
      })
      .catch((nextError: unknown) => {
        if (guard.accepts(revision)) {
          setDetailFailure({
            attemptKey,
            message: errorMessage(nextError),
          });
        }
      });
    return () => {
      if (guard.accepts(revision)) {
        guard.begin();
      }
    };
  }, [
    attemptKey,
    currentDetail,
    fetchDetailsOnOpen,
    open,
    runEvidenceRevision,
    runHasDetails,
    runId,
    runProjectId,
    runUpdatedAt,
  ]);

  return (
    <details
      open={open}
      onToggle={(event) => {
        const nextOpen = event.currentTarget.open;
        if (nextOpen) {
          setDetailFailure(null);
        }
        setOpen(nextOpen);
      }}
      className="rounded-xl border border-border/70 bg-background/70 px-3 py-2"
    >
      <summary className="cursor-pointer text-xs text-foreground">
        <span className={`font-medium ${statusClass(displayStatus)}`}>
          {displayStatus}
        </span>{" "}
        <span className="text-muted-foreground">
          {timestamp(run.startedAt)}. Run {run.id}
        </span>
      </summary>
      {open ? (
        exactRun ? (
          <VerificationRunDetails run={exactRun} />
        ) : (
          <p
            className={
              detailError
                ? "mt-2 text-xs text-destructive"
                : "mt-2 text-xs text-muted-foreground"
            }
            role={detailError ? "alert" : undefined}
          >
            {detailError
              ? visibleProjectVerificationText(detailError)
              : "Loading run details..."}
          </p>
        )
      ) : null}
    </details>
  );
}

function VerificationRunDetails({ run }: { run: ProjectVerificationRun }) {
  return (
    <div className="mt-2 space-y-2">
      <p className="text-xs text-muted-foreground">
        Config revision {run.configRevision}, workspace revision{" "}
        {run.workspaceRevision}. Source freshness: unverified.
      </p>
      {run.error ? (
        <p className="text-xs text-destructive" role="alert">
          {visibleProjectVerificationText(run.error)}
        </p>
      ) : null}
      {run.results.length > 0 ? (
        run.results.map((result, index) => (
          <VerificationResultCard
            key={`${run.id}:${index}:${result.name}`}
            result={result}
          />
        ))
      ) : (
        <p className="text-xs text-muted-foreground">
          {projectVerificationRunIsActive(run)
            ? "Waiting for the first check..."
            : "No check results were recorded."}
        </p>
      )}
    </div>
  );
}

function VerificationResultCard({
  result,
}: {
  result: ProjectVerificationResult;
}) {
  const rawOutput = result.output ?? "";
  const visibleOutput = visibleProjectVerificationText(
    rawOutput.slice(0, MAX_RENDERED_OUTPUT_CHARACTERS),
  );
  const outputWasClipped =
    rawOutput.length > MAX_RENDERED_OUTPUT_CHARACTERS ||
    visibleOutput.length > MAX_RENDERED_OUTPUT_CHARACTERS;
  const renderedOutput = visibleOutput.slice(0, MAX_RENDERED_OUTPUT_CHARACTERS);
  return (
    <div className="rounded-lg bg-muted/35 px-3 py-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs font-medium text-foreground">
          {visibleProjectVerificationText(result.name)}{" "}
          <span className={statusClass(result.status)}>({result.status})</span>
        </p>
        <p className="text-[11px] text-muted-foreground">
          {result.exitCode === null
            ? "No exit code"
            : `Exit ${result.exitCode}`}
          {result.durationMs === null ? "" : `, ${result.durationMs} ms`}
        </p>
      </div>
      <pre
        dir="ltr"
        className="mt-1 whitespace-pre-wrap break-words text-[11px] text-muted-foreground"
        style={{ unicodeBidi: "isolate" }}
      >
        {visibleProjectVerificationCommand(result.command)}
      </pre>
      {renderedOutput ? (
        <pre
          dir="ltr"
          className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-md bg-background px-2 py-2 text-[11px] text-foreground"
          style={{ unicodeBidi: "isolate" }}
        >
          {renderedOutput}
        </pre>
      ) : null}
      {result.error ? (
        <p className="mt-1 text-xs text-destructive">
          {visibleProjectVerificationText(result.error)}
        </p>
      ) : null}
      {result.outputTruncated || outputWasClipped ? (
        <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-400">
          Output was truncated. {result.outputBytes} bytes were produced.
        </p>
      ) : null}
    </div>
  );
}
