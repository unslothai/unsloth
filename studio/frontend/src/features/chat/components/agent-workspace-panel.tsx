// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import {
  CheckCircle2,
  Clipboard,
  FileCode2,
  FileText,
  GitBranch,
  GitCompare,
  GitFork,
  ListChecks,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Square,
  Trash2,
} from "lucide-react";
import {
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type AgentBackgroundPermissionMode,
  type AgentBackgroundRuntimeKind,
  type AgentBackgroundRuntimeSelection,
  type AgentBackgroundTask,
  type AgentGitDiff,
  type AgentGitStatus,
  type AgentInstructions,
  type AgentPlan,
  type AgentPlanStatus,
  type AgentPlanTaskStatus,
  type AgentPreparedCommit,
  type AgentPullRequestDraft,
  type AgentPullRequestHandoffPreview,
  type AgentPullRequestHandoffResult,
  type AgentRepositoryMap,
  type AgentReviewSummary,
  type AgentVerificationCheck,
  type AgentVerificationConfig,
  type AgentVerificationRun,
  type AgentWorkspaceOverview,
  type AgentWorktree,
  agentWorkspaceMutationOutcomeUnknown,
  cancelAgentBackgroundTask,
  cleanupAgentWorktree,
  confirmAgentPreparedCommit,
  confirmAgentPullRequestHandoff,
  createAgentPlan,
  createAgentPullRequestDraft,
  createAgentWorktree,
  getAgentGitDiff,
  getAgentGitStatus,
  getAgentInstructions,
  getAgentRepositoryMap,
  getAgentReview,
  getAgentVerificationConfig,
  getAgentWorkspace,
  listAgentBackgroundTasks,
  listAgentPlans,
  listAgentVerificationRuns,
  listAgentWorktrees,
  mergeAgentWorktree,
  prepareAgentCommit,
  prepareAgentPullRequestHandoff,
  queueAgentTask,
  queueAgentVerification,
  retryAgentBackgroundTask,
  runAgentVerification,
  saveAgentVerificationConfig,
  startAgentBackgroundTask,
  updateAgentPlan,
  updateAgentPlanTask,
} from "../api/agent-workspace-api";
import {
  type AgentPullRequestSubmissionDisplay,
  BACKGROUND_AGENT_FULL_ACCESS_WARNING,
  BACKGROUND_AGENT_PERMISSION_POLICY,
  agentBackgroundActions,
  agentBackgroundSnapshot,
  agentPlanProgress,
  agentRepositoryMapSummary,
  agentStatusLabel,
  agentWorkspaceRequestIsCurrent,
  agentWorkspaceStatus,
  agentWorktreeMergeAction,
  backgroundAgentPermissionNeedsConfirmation,
  latestVerificationSummary,
  preparedCommitConfirmation,
  pullRequestHandoffCanSubmit,
  pullRequestHandoffConfirmation,
  pullRequestSubmissionDisplay,
  reconcileAgentBackgroundMutation,
  safeAgentWorkspaceError,
} from "./agent-workspace-state";

const MAP_PATH_LIMIT = 20_000;
const MAP_BYTE_LIMIT = 2 * 1024 * 1024;

function formatDate(value: number | null | undefined): string {
  if (!value) return "";
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function statusBadgeVariant(
  status: string,
): "secondary" | "outline" | "destructive" {
  if (["failed", "blocked", "interrupted", "timed_out"].includes(status)) {
    return "destructive";
  }
  if (["passed", "completed", "active"].includes(status)) return "secondary";
  return "outline";
}

function AgentSection({
  icon,
  title,
  detail,
  actions,
  children,
}: {
  icon: ReactNode;
  title: string;
  detail?: string;
  actions?: ReactNode;
  children: ReactNode;
}): ReactElement {
  return (
    <section className="rounded-[22px] border border-border/60 bg-card/35 px-4 py-4">
      <div className="flex min-w-0 items-start gap-3">
        <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
          {icon}
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-ui-14 font-semibold text-foreground">{title}</h2>
          {detail ? (
            <p className="mt-0.5 text-xs text-muted-foreground">{detail}</p>
          ) : null}
        </div>
        {actions ? (
          <div className="flex shrink-0 gap-1.5">{actions}</div>
        ) : null}
      </div>
      <div className="mt-3">{children}</div>
    </section>
  );
}

function Empty({ children }: { children: ReactNode }): ReactElement {
  return (
    <div className="rounded-xl bg-muted/35 px-3 py-4 text-center text-xs text-muted-foreground">
      {children}
    </div>
  );
}

function normalizedVerificationChecks(
  checks: AgentVerificationCheck[],
): AgentVerificationCheck[] {
  return checks
    .map((check) => ({
      ...check,
      name: check.name.trim(),
      command: check.command.trim(),
    }))
    .filter((check) => check.name.length > 0 && check.command.length > 0);
}

export function AgentWorkspacePanel({
  projectId,
}: {
  projectId: string;
}): ReactElement {
  const generationRef = useRef(0);
  const activeProjectIdRef = useRef(projectId);
  const [workspace, setWorkspace] = useState<AgentWorkspaceOverview | null>(
    null,
  );
  const [instructions, setInstructions] = useState<AgentInstructions | null>(
    null,
  );
  const [instructionTarget, setInstructionTarget] = useState("");
  const [repositoryMap, setRepositoryMap] = useState<AgentRepositoryMap | null>(
    null,
  );
  const [gitStatus, setGitStatus] = useState<AgentGitStatus | null>(null);
  const [gitDiff, setGitDiff] = useState<AgentGitDiff | null>(null);
  const [commitPathFilter, setCommitPathFilter] = useState("");
  const [selectedCommitPaths, setSelectedCommitPaths] = useState<Set<string>>(
    () => new Set(),
  );
  const [commitMessage, setCommitMessage] = useState("");
  const [preparedCommit, setPreparedCommit] =
    useState<AgentPreparedCommit | null>(null);
  const [confirmedCommit, setConfirmedCommit] =
    useState<AgentPreparedCommit | null>(null);
  const [commitConfirmationError, setCommitConfirmationError] = useState<
    string | null
  >(null);
  const [verificationChecks, setVerificationChecks] = useState<
    AgentVerificationCheck[]
  >([]);
  const [
    requireVerificationForGoalCompletion,
    setRequireVerificationForGoalCompletion,
  ] = useState(false);
  const [verificationConfigRevision, setVerificationConfigRevision] =
    useState(0);
  const [verificationRuns, setVerificationRuns] = useState<
    AgentVerificationRun[]
  >([]);
  const [plans, setPlans] = useState<AgentPlan[]>([]);
  const [backgroundTasks, setBackgroundTasks] = useState<AgentBackgroundTask[]>(
    [],
  );
  const [worktrees, setWorktrees] = useState<AgentWorktree[]>([]);
  const [review, setReview] = useState<AgentReviewSummary | null>(null);
  const [pullRequestDraft, setPullRequestDraft] =
    useState<AgentPullRequestDraft | null>(null);
  const [githubConnectorId, setGithubConnectorId] = useState("");
  const [githubOwner, setGithubOwner] = useState("");
  const [githubRepository, setGithubRepository] = useState("");
  const [githubBase, setGithubBase] = useState("main");
  const [githubHead, setGithubHead] = useState("");
  const [githubDraft, setGithubDraft] = useState(true);
  const [pullRequestHandoff, setPullRequestHandoff] =
    useState<AgentPullRequestHandoffPreview | null>(null);
  const [pullRequestSubmission, setPullRequestSubmission] =
    useState<AgentPullRequestSubmissionDisplay | null>(null);
  const [pullRequestSubmissionResult, setPullRequestSubmissionResult] =
    useState<AgentPullRequestHandoffResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [planTitle, setPlanTitle] = useState("");
  const [planTasks, setPlanTasks] = useState("");
  const [agentInstruction, setAgentInstruction] = useState("");
  const [agentRuntimeKind, setAgentRuntimeKind] =
    useState<AgentBackgroundRuntimeKind>("local");
  const [agentRuntimeModel, setAgentRuntimeModel] = useState("");
  const [agentRuntimeProviderId, setAgentRuntimeProviderId] = useState("");
  const [agentPermissionMode, setAgentPermissionMode] =
    useState<AgentBackgroundPermissionMode>("off");
  const [agentReasoningEffort, setAgentReasoningEffort] = useState("");
  const [agentMaxOutputTokens, setAgentMaxOutputTokens] = useState("8192");
  const [agentPlanId, setAgentPlanId] = useState("");
  const [agentPlanTaskId, setAgentPlanTaskId] = useState("");
  const [agentWorktreeId, setAgentWorktreeId] = useState("");
  const [cleanupWorktreeOnCancel, setCleanupWorktreeOnCancel] = useState(false);
  const [worktreeBranch, setWorktreeBranch] = useState("");
  const goalCompletionPolicyId = useId();
  const cleanupWorktreePolicyId = useId();
  const githubDraftId = useId();

  useEffect(() => {
    activeProjectIdRef.current = projectId;
  }, [projectId]);

  const loadDashboard = useCallback(
    async (reset = false) => {
      const requestProjectId = projectId;
      const generation = ++generationRef.current;
      const requestIsCurrent = () =>
        agentWorkspaceRequestIsCurrent({
          requestProjectId,
          activeProjectId: activeProjectIdRef.current,
          requestGeneration: generation,
          activeGeneration: generationRef.current,
        });
      setBusy(null);
      setLoading(true);
      setLoadError(null);
      if (reset) {
        setWorkspace(null);
        setInstructions(null);
        setRepositoryMap(null);
        setGitStatus(null);
        setGitDiff(null);
        setSelectedCommitPaths(new Set());
        setPreparedCommit(null);
        setConfirmedCommit(null);
        setCommitConfirmationError(null);
        setVerificationChecks([]);
        setRequireVerificationForGoalCompletion(false);
        setVerificationConfigRevision(0);
        setVerificationRuns([]);
        setPlans([]);
        setBackgroundTasks([]);
        setWorktrees([]);
        setReview(null);
        setPullRequestDraft(null);
        setPullRequestHandoff(null);
        setPullRequestSubmission(null);
        setPullRequestSubmissionResult(null);
      }
      try {
        const nextWorkspace = await getAgentWorkspace(requestProjectId);
        if (!requestIsCurrent()) return;
        setWorkspace(nextWorkspace);
        if (!nextWorkspace.available) {
          setInstructions(null);
          setRepositoryMap(null);
          setGitStatus(null);
          setVerificationChecks([]);
          setVerificationRuns([]);
          setPlans([]);
          setBackgroundTasks([]);
          setWorktrees([]);
          return;
        }
        const capabilities = nextWorkspace.capabilities;
        const results = await Promise.allSettled([
          capabilities.instructions
            ? getAgentInstructions(requestProjectId)
            : Promise.resolve(null),
          capabilities.repositoryMap
            ? getAgentRepositoryMap(requestProjectId, {
                maxPaths: MAP_PATH_LIMIT,
                maxTotalBytes: MAP_BYTE_LIMIT,
              })
            : Promise.resolve(null),
          capabilities.verification
            ? getAgentVerificationConfig(requestProjectId)
            : Promise.resolve(null),
          capabilities.verification
            ? listAgentVerificationRuns(requestProjectId, 10)
            : Promise.resolve([]),
          capabilities.git
            ? getAgentGitStatus(requestProjectId)
            : Promise.resolve(null),
          capabilities.plans
            ? listAgentPlans(requestProjectId)
            : Promise.resolve([]),
          capabilities.background
            ? listAgentBackgroundTasks(requestProjectId, 50)
            : Promise.resolve([]),
          capabilities.worktrees
            ? listAgentWorktrees(requestProjectId)
            : Promise.resolve([]),
        ] as const);
        if (!requestIsCurrent()) return;

        if (results[0].status === "fulfilled") {
          setInstructions(results[0].value);
        }
        if (results[1].status === "fulfilled") {
          setRepositoryMap(results[1].value);
        }
        if (results[2].status === "fulfilled" && results[2].value) {
          setVerificationChecks(results[2].value.checks);
          setRequireVerificationForGoalCompletion(
            results[2].value.requireForGoalCompletion,
          );
          setVerificationConfigRevision(results[2].value.revision);
        }
        if (results[3].status === "fulfilled") {
          setVerificationRuns(results[3].value);
        }
        if (results[4].status === "fulfilled") {
          setGitStatus(results[4].value);
        }
        if (results[5].status === "fulfilled") {
          setPlans(results[5].value);
        }
        if (results[6].status === "fulfilled") {
          setBackgroundTasks(results[6].value);
        }
        if (results[7].status === "fulfilled") {
          setWorktrees(results[7].value);
        }

        const sectionNames = [
          "instructions",
          "repository map",
          "verification policy",
          "verification runs",
          "Git status",
          "plans",
          "background tasks",
          "worktrees",
        ];
        const failures = results.flatMap((result, index) =>
          result.status === "rejected"
            ? [
                `${sectionNames[index] ?? "workspace section"}: ${safeAgentWorkspaceError(result.reason)}`,
              ]
            : [],
        );
        if (failures.length > 0) setLoadError(failures.join("; "));
      } catch (error) {
        if (!requestIsCurrent()) return;
        setLoadError(safeAgentWorkspaceError(error));
      } finally {
        if (requestIsCurrent()) setLoading(false);
      }
    },
    [projectId],
  );

  useEffect(() => {
    const timer = window.setTimeout(() => void loadDashboard(true), 0);
    return () => {
      window.clearTimeout(timer);
      generationRef.current += 1;
    };
  }, [loadDashboard]);

  const hasLiveBackgroundTask = backgroundTasks.some((task) =>
    ["queued", "running", "cancelling"].includes(task.status),
  );
  useEffect(() => {
    if (!hasLiveBackgroundTask) return;
    const generation = generationRef.current;
    const requestProjectId = projectId;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const nextTasks = await listAgentBackgroundTasks(
            requestProjectId,
            50,
          );
          if (
            cancelled ||
            !agentWorkspaceRequestIsCurrent({
              requestProjectId,
              activeProjectId: activeProjectIdRef.current,
              requestGeneration: generation,
              activeGeneration: generationRef.current,
            })
          ) {
            return;
          }
          setBackgroundTasks(nextTasks);
          const nextRuns = await listAgentVerificationRuns(
            requestProjectId,
            10,
          );
          if (
            cancelled ||
            !agentWorkspaceRequestIsCurrent({
              requestProjectId,
              activeProjectId: activeProjectIdRef.current,
              requestGeneration: generation,
              activeGeneration: generationRef.current,
            })
          ) {
            return;
          }
          setVerificationRuns(nextRuns);
        } catch {
          // The next poll or a project reload will reconcile transient errors.
        }
      })();
    }, 2_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [hasLiveBackgroundTask, projectId]);

  async function runAction<T>(
    key: string,
    action: () => Promise<T>,
    complete: (value: T) => void,
    success?: string,
  ): Promise<void> {
    if (busy) return;
    const generation = generationRef.current;
    const requestProjectId = projectId;
    const requestIsCurrent = () =>
      agentWorkspaceRequestIsCurrent({
        requestProjectId,
        activeProjectId: activeProjectIdRef.current,
        requestGeneration: generation,
        activeGeneration: generationRef.current,
      });
    setBusy(key);
    try {
      const result = await action();
      if (!requestIsCurrent()) return;
      complete(result);
      if (success) toast.success(success);
    } catch (error) {
      if (!requestIsCurrent()) return;
      toast.error("Agent workspace action failed", {
        description: safeAgentWorkspaceError(error),
      });
    } finally {
      if (requestIsCurrent()) setBusy(null);
    }
  }

  const workspaceLabel = agentWorkspaceStatus(workspace);
  const isBusy = (key: string) => busy === key;

  async function refreshInstructions(): Promise<void> {
    await runAction(
      "instructions",
      () =>
        getAgentInstructions(projectId, instructionTarget.trim() || undefined),
      setInstructions,
    );
  }

  async function refreshRepositoryMap(): Promise<void> {
    await runAction(
      "repository-map",
      () =>
        getAgentRepositoryMap(projectId, {
          maxPaths: MAP_PATH_LIMIT,
          maxTotalBytes: MAP_BYTE_LIMIT,
        }),
      setRepositoryMap,
    );
  }

  function invalidatePreparedCommit(): void {
    setPreparedCommit(null);
    setConfirmedCommit(null);
    setCommitConfirmationError(null);
  }

  function applyGitStatus(status: AgentGitStatus): void {
    const available = new Set(status.files.map((file) => file.path));
    setGitStatus(status);
    setSelectedCommitPaths(
      (current) => new Set([...current].filter((path) => available.has(path))),
    );
    invalidatePreparedCommit();
  }

  function toggleCommitPath(path: string, selected: boolean): void {
    invalidatePreparedCommit();
    setSelectedCommitPaths((current) => {
      const next = new Set(current);
      if (selected) next.add(path);
      else next.delete(path);
      return next;
    });
  }

  async function prepareCommitPreview(): Promise<void> {
    const ownedPaths = [...selectedCommitPaths];
    const message = commitMessage.trim();
    if (!ownedPaths.length || !message) {
      toast.error("Select changed paths and enter a commit message");
      return;
    }
    await runAction(
      "commit-prepare",
      () => prepareAgentCommit(projectId, ownedPaths, message),
      (preview) => {
        setPreparedCommit(preview);
        setConfirmedCommit(null);
        setCommitConfirmationError(null);
      },
      "Commit preview prepared",
    );
  }

  async function confirmCommitPreview(): Promise<void> {
    if (busy || !preparedCommit) return;
    let confirmation: ReturnType<typeof preparedCommitConfirmation>;
    try {
      confirmation = preparedCommitConfirmation(preparedCommit);
    } catch (error) {
      setCommitConfirmationError(safeAgentWorkspaceError(error));
      return;
    }
    const generation = generationRef.current;
    const requestProjectId = projectId;
    setPreparedCommit(null);
    setCommitConfirmationError(null);
    setBusy("commit-confirm");
    try {
      const confirmed = await confirmAgentPreparedCommit(
        requestProjectId,
        confirmation.preparationId,
        confirmation.confirmationToken,
      );
      if (
        !agentWorkspaceRequestIsCurrent({
          requestProjectId,
          activeProjectId: activeProjectIdRef.current,
          requestGeneration: generation,
          activeGeneration: generationRef.current,
        })
      ) {
        return;
      }
      setConfirmedCommit(confirmed);
      setSelectedCommitPaths(new Set());
      setCommitMessage("");
      toast.success("Prepared commit ref created");
    } catch (error) {
      if (
        agentWorkspaceRequestIsCurrent({
          requestProjectId,
          activeProjectId: activeProjectIdRef.current,
          requestGeneration: generation,
          activeGeneration: generationRef.current,
        })
      ) {
        setCommitConfirmationError(
          agentWorkspaceMutationOutcomeUnknown(error)
            ? "Prepared ref outcome is unknown. Inspect Git status and Studio-owned refs before preparing another preview."
            : safeAgentWorkspaceError(error),
        );
      }
    } finally {
      if (
        agentWorkspaceRequestIsCurrent({
          requestProjectId,
          activeProjectId: activeProjectIdRef.current,
          requestGeneration: generation,
          activeGeneration: generationRef.current,
        })
      ) {
        setBusy(null);
      }
    }
  }

  function invalidatePullRequestHandoff(): void {
    setPullRequestHandoff(null);
    setPullRequestSubmissionResult(null);
  }

  async function preparePullRequestHandoff(): Promise<void> {
    await runAction(
      "github-preview",
      () =>
        prepareAgentPullRequestHandoff(projectId, {
          serverId: githubConnectorId,
          owner: githubOwner,
          repository: githubRepository,
          base: githubBase,
          head: githubHead,
          draft: githubDraft,
        }),
      (preview) => {
        setPullRequestHandoff(preview);
        setPullRequestSubmission(null);
        setPullRequestSubmissionResult(null);
      },
      "GitHub handoff preview prepared",
    );
  }

  async function submitPullRequestHandoff(): Promise<void> {
    if (
      busy ||
      !pullRequestHandoffCanSubmit(pullRequestHandoff, pullRequestSubmission) ||
      !pullRequestHandoff
    ) {
      return;
    }
    const preview = pullRequestHandoff;
    const confirmation = pullRequestHandoffConfirmation(preview);
    const generation = generationRef.current;
    const requestProjectId = projectId;
    setPullRequestHandoff(null);
    setPullRequestSubmission(
      pullRequestSubmissionDisplay(preview, "submitting"),
    );
    setPullRequestSubmissionResult(null);
    setBusy("github-submit");
    try {
      const result = await confirmAgentPullRequestHandoff(
        requestProjectId,
        confirmation.handoffId,
        {
          serverId: confirmation.serverId,
          confirmationToken: confirmation.confirmationToken,
          expectedRequestDigest: confirmation.expectedRequestDigest,
        },
      );
      if (
        !agentWorkspaceRequestIsCurrent({
          requestProjectId,
          activeProjectId: activeProjectIdRef.current,
          requestGeneration: generation,
          activeGeneration: generationRef.current,
        })
      ) {
        return;
      }
      setPullRequestSubmission(
        pullRequestSubmissionDisplay(preview, "submitted"),
      );
      setPullRequestSubmissionResult(result);
      toast.success("GitHub confirmed the pull request submission");
    } catch (error) {
      if (
        agentWorkspaceRequestIsCurrent({
          requestProjectId,
          activeProjectId: activeProjectIdRef.current,
          requestGeneration: generation,
          activeGeneration: generationRef.current,
        })
      ) {
        const outcomeUnknown = agentWorkspaceMutationOutcomeUnknown(error);
        if (outcomeUnknown) {
          setPullRequestSubmission(
            pullRequestSubmissionDisplay(preview, "unknown"),
          );
          toast.error("GitHub submission outcome is unknown", {
            description: "Check GitHub before creating another handoff.",
          });
        } else {
          setPullRequestSubmission(null);
          toast.error("GitHub handoff was rejected before submission", {
            description: safeAgentWorkspaceError(error),
          });
        }
        setPullRequestSubmissionResult(null);
      }
    } finally {
      if (
        agentWorkspaceRequestIsCurrent({
          requestProjectId,
          activeProjectId: activeProjectIdRef.current,
          requestGeneration: generation,
          activeGeneration: generationRef.current,
        })
      ) {
        setBusy(null);
      }
    }
  }

  function updateVerificationCheck(
    index: number,
    patch: Partial<AgentVerificationCheck>,
  ): void {
    setVerificationChecks((checks) =>
      checks.map((check, checkIndex) =>
        checkIndex === index ? { ...check, ...patch } : check,
      ),
    );
  }

  function applyVerificationConfig(config: AgentVerificationConfig): void {
    setVerificationChecks(config.checks);
    setRequireVerificationForGoalCompletion(config.requireForGoalCompletion);
    setVerificationConfigRevision(config.revision);
  }

  async function saveVerification(): Promise<void> {
    const checks = normalizedVerificationChecks(verificationChecks);
    await runAction(
      "verification-save",
      () =>
        saveAgentVerificationConfig(
          projectId,
          checks,
          requireVerificationForGoalCompletion,
          verificationConfigRevision,
        ),
      applyVerificationConfig,
      "Verification checks saved",
    );
  }

  async function runVerification(background: boolean): Promise<void> {
    const key = background ? "verification-background" : "verification-run";
    const checks = normalizedVerificationChecks(verificationChecks);
    if (checks.length === 0) {
      toast.error("Add and save at least one verification check");
      return;
    }
    if (background) {
      await runAction(
        key,
        async () => {
          const config = await saveAgentVerificationConfig(
            projectId,
            checks,
            requireVerificationForGoalCompletion,
            verificationConfigRevision,
          );
          const task = await queueAgentVerification(projectId, config.revision);
          return { config, task };
        },
        ({ config, task }) => {
          applyVerificationConfig(config);
          setBackgroundTasks((tasks) => [task, ...tasks]);
        },
        "Verification queued",
      );
      return;
    }
    await runAction(
      key,
      async () => {
        const config = await saveAgentVerificationConfig(
          projectId,
          checks,
          requireVerificationForGoalCompletion,
          verificationConfigRevision,
        );
        const run = await runAgentVerification(projectId, config.revision);
        return { config, run };
      },
      ({ config, run }) => {
        applyVerificationConfig(config);
        setVerificationRuns((runs) => [run, ...runs]);
      },
      "Verification finished",
    );
  }

  async function createPlan(): Promise<void> {
    const title = planTitle.trim();
    const tasks = planTasks
      .split("\n")
      .map((task) => task.trim())
      .filter(Boolean)
      .map((task) => ({ title: task }));
    if (!title) {
      toast.error("Plan title is required");
      return;
    }
    await runAction(
      "plan-create",
      () => createAgentPlan(projectId, { title, tasks }),
      (plan) => {
        setPlans((current) => [plan, ...current]);
        setPlanTitle("");
        setPlanTasks("");
      },
      "Plan created",
    );
  }

  function replacePlan(updated: AgentPlan): void {
    setPlans((current) =>
      current.map((plan) => (plan.id === updated.id ? updated : plan)),
    );
  }

  async function setPlanStatus(
    plan: AgentPlan,
    status: AgentPlanStatus,
  ): Promise<void> {
    await runAction(
      `plan-${plan.id}`,
      () => updateAgentPlan(projectId, plan, status),
      replacePlan,
    );
  }

  async function setTaskStatus(
    plan: AgentPlan,
    taskId: string,
    status: AgentPlanTaskStatus,
  ): Promise<void> {
    await runAction(
      `task-${taskId}`,
      () => updateAgentPlanTask(projectId, plan, taskId, { status }),
      replacePlan,
    );
  }

  async function mutateBackgroundTask(
    task: AgentBackgroundTask,
    action: "start" | "cancel" | "retry",
  ): Promise<void> {
    const invoke =
      action === "start"
        ? startAgentBackgroundTask
        : action === "cancel"
          ? cancelAgentBackgroundTask
          : retryAgentBackgroundTask;
    await runAction(
      `background-${task.id}`,
      () => invoke(projectId, task.id),
      (updated) => {
        setBackgroundTasks(
          (current) =>
            reconcileAgentBackgroundMutation({
              tasks: current,
              worktrees: [],
              previousTaskId: task.id,
              action,
              updated,
            }).tasks,
        );
        setWorktrees(
          (current) =>
            reconcileAgentBackgroundMutation({
              tasks: [],
              worktrees: current,
              previousTaskId: task.id,
              action,
              updated,
            }).worktrees,
        );
      },
    );
  }

  async function queueBackgroundAgent(start: boolean): Promise<void> {
    const instruction = agentInstruction.trim();
    if (!instruction) {
      toast.error("Agent task instructions are required");
      return;
    }
    const runtime: AgentBackgroundRuntimeSelection = {
      kind: agentRuntimeKind,
      model: agentRuntimeModel,
      providerId:
        agentRuntimeKind === "provider" ? agentRuntimeProviderId : undefined,
      permissionMode: agentPermissionMode,
      reasoningEffort: agentReasoningEffort || undefined,
      maxOutputTokens: Number(agentMaxOutputTokens),
    };
    await runAction(
      start ? "agent-start" : "agent-queue",
      () =>
        queueAgentTask(projectId, {
          instruction,
          runtime,
          planId: agentPlanId || undefined,
          planTaskId: agentPlanTaskId || undefined,
          worktreeId: agentWorktreeId || undefined,
          cleanupWorktreeOnCancel,
          start,
        }),
      (task) => {
        setBackgroundTasks(
          (current) =>
            reconcileAgentBackgroundMutation({
              tasks: current,
              worktrees: [],
              action: "enqueue",
              updated: task,
            }).tasks,
        );
        setAgentInstruction("");
        if (task.worktreeId) {
          setWorktrees(
            (current) =>
              reconcileAgentBackgroundMutation({
                tasks: [],
                worktrees: current,
                action: "enqueue",
                updated: task,
              }).worktrees,
          );
          setAgentWorktreeId("");
          setCleanupWorktreeOnCancel(false);
        }
      },
      start ? "Agent task started" : "Agent task queued",
    );
  }

  function replaceWorktree(updated: AgentWorktree): void {
    setWorktrees((current) =>
      current.map((worktree) =>
        worktree.id === updated.id ? updated : worktree,
      ),
    );
  }

  async function mergeWorktree(worktree: AgentWorktree): Promise<void> {
    const expectedTargetHead = gitStatus?.head;
    if (!expectedTargetHead) {
      toast.error("Refresh Git status before merging");
      return;
    }
    if (
      !window.confirm(
        `Merge ${worktree.branch} into ${gitStatus?.branch ?? "the primary branch"} at ${expectedTargetHead.slice(0, 12)}?`,
      )
    ) {
      return;
    }
    await runAction(
      `worktree-merge-${worktree.id}`,
      () => mergeAgentWorktree(projectId, worktree.id, expectedTargetHead),
      (updated) => {
        replaceWorktree(updated);
        const merge = updated.merge;
        if (merge?.status === "merged") {
          if (merge.resultHead) {
            setGitStatus((current) =>
              current
                ? { ...current, head: merge.resultHead as string }
                : current,
            );
          }
          toast.success("Worktree merged");
        } else if (merge?.status === "conflict") {
          toast.error("Worktree merge needs conflict resolution");
        }
      },
    );
  }

  async function createWorktree(): Promise<void> {
    const branch = worktreeBranch.trim();
    await runAction(
      "worktree-create",
      () => createAgentWorktree(projectId, { branch: branch || undefined }),
      (worktree) => {
        setWorktrees((current) => [...current, worktree]);
        setWorktreeBranch("");
      },
      "Worktree created",
    );
  }

  async function cleanupWorktree(worktree: AgentWorktree): Promise<void> {
    if (
      !window.confirm(
        `Remove Studio worktree ${worktree.branch}? Dirty worktrees will be preserved.`,
      )
    ) {
      return;
    }
    await runAction(
      `worktree-${worktree.id}`,
      () => cleanupAgentWorktree(projectId, worktree.id),
      replaceWorktree,
      "Worktree removed",
    );
  }

  const activePlanCount = useMemo(
    () => plans.filter((plan) => plan.status === "active").length,
    [plans],
  );
  const selectedAgentPlan = useMemo(
    () => plans.find((plan) => plan.id === agentPlanId) ?? null,
    [agentPlanId, plans],
  );
  const parsedAgentMaxOutputTokens = Number(agentMaxOutputTokens);
  const agentRuntimeReady = Boolean(
    agentRuntimeModel.trim() &&
      (agentRuntimeKind === "local" || agentRuntimeProviderId.trim()) &&
      Number.isInteger(parsedAgentMaxOutputTokens) &&
      parsedAgentMaxOutputTokens >= 1 &&
      parsedAgentMaxOutputTokens <= 32_768,
  );
  const filteredCommitFiles = useMemo(() => {
    const query = commitPathFilter.trim().toLocaleLowerCase();
    const files = gitStatus?.files ?? [];
    return query
      ? files.filter((file) => file.path.toLocaleLowerCase().includes(query))
      : files;
  }, [commitPathFilter, gitStatus]);
  const visibleCommitFiles = filteredCommitFiles.slice(0, 200);
  const githubFieldsComplete = Boolean(
    githubConnectorId.trim() &&
      githubOwner.trim() &&
      githubRepository.trim() &&
      githubBase.trim() &&
      githubHead.trim(),
  );
  const githubHandoffLocked = Boolean(pullRequestSubmission);
  const githubCanSubmit = pullRequestHandoffCanSubmit(
    pullRequestHandoff,
    pullRequestSubmission,
  );

  if (loading && !workspace) {
    return (
      <output
        aria-live="polite"
        className="mt-8 flex items-center justify-center gap-2 rounded-[26px] bg-muted/30 px-6 py-12 text-sm text-muted-foreground"
      >
        <Loader2 className="size-4 animate-spin" /> Loading project workspace
      </output>
    );
  }

  return (
    <div
      className="mt-8 flex flex-col gap-3 pb-10"
      aria-busy={loading || Boolean(busy)}
    >
      <div className="flex flex-wrap items-center gap-2 rounded-[22px] bg-muted/30 px-4 py-3">
        <span
          className={`size-2 rounded-full ${
            workspaceLabel.tone === "success"
              ? "bg-emerald-500"
              : workspaceLabel.tone === "danger"
                ? "bg-destructive"
                : "bg-muted-foreground"
          }`}
          aria-hidden="true"
        />
        <span className="text-ui-14 font-semibold">{workspaceLabel.label}</span>
        {workspace ? (
          <Badge variant="outline">
            {workspace.workspaceKind === "folder" ? "Local folder" : "Managed"}
          </Badge>
        ) : null}
        <span className="min-w-0 flex-1" />
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={() => void loadDashboard()}
          disabled={loading || Boolean(busy)}
        >
          <RefreshCw className={loading ? "animate-spin" : ""} /> Refresh all
        </Button>
      </div>

      {loadError ? (
        <div
          role="alert"
          className="rounded-xl border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive"
        >
          Some workspace data could not be loaded: {loadError}
        </div>
      ) : null}

      {workspace && !workspace.available ? (
        <div className="rounded-[22px] border border-destructive/25 bg-destructive/5 px-5 py-8 text-center">
          <p className="text-sm font-semibold text-foreground">
            This project folder is unavailable.
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Reconnect the repository in Unsloth Desktop, then refresh this
            workspace. No repository files were changed.
          </p>
        </div>
      ) : (
        <>
          {workspace?.capabilities.instructions ? (
            <AgentSection
              icon={<FileText className="size-4" />}
              title="Repository instructions"
              detail={
                instructions
                  ? `${instructions.layers.length} AGENTS.md layer${instructions.layers.length === 1 ? "" : "s"}, ${instructions.precedence}`
                  : "No resolved instructions"
              }
              actions={
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() => void refreshInstructions()}
                  disabled={Boolean(busy)}
                >
                  <RefreshCw
                    className={isBusy("instructions") ? "animate-spin" : ""}
                  />
                  Resolve
                </Button>
              }
            >
              <Input
                value={instructionTarget}
                onChange={(event) => setInstructionTarget(event.target.value)}
                placeholder="Optional file or directory scope"
                aria-label="Instruction target path"
                className="h-8 text-xs"
              />
              {instructions?.layers.length ? (
                <div className="mt-2 space-y-2">
                  {instructions.layers.map((layer) => (
                    <details
                      key={layer.path}
                      className="rounded-xl bg-muted/35 px-3 py-2"
                    >
                      <summary className="cursor-pointer text-xs font-medium text-foreground">
                        {layer.path}
                        {layer.truncated ? " (truncated)" : ""}
                      </summary>
                      <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-muted-foreground">
                        {layer.content}
                      </pre>
                    </details>
                  ))}
                </div>
              ) : (
                <div className="mt-2">
                  <Empty>No AGENTS.md applies to this scope.</Empty>
                </div>
              )}
              {instructions?.issues.length ? (
                <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                  {instructions.issues.length} instruction file issue
                  {instructions.issues.length === 1 ? "" : "s"} excluded.
                </p>
              ) : null}
            </AgentSection>
          ) : null}

          {workspace?.capabilities.repositoryMap ? (
            <AgentSection
              icon={<FileCode2 className="size-4" />}
              title="Repository map"
              detail={agentRepositoryMapSummary(repositoryMap)}
              actions={
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() => void refreshRepositoryMap()}
                  disabled={Boolean(busy)}
                >
                  <RefreshCw
                    className={isBusy("repository-map") ? "animate-spin" : ""}
                  />
                  Refresh
                </Button>
              }
            >
              {repositoryMap?.entries.length ? (
                <div className="grid max-h-56 grid-cols-1 gap-1 overflow-auto sm:grid-cols-2">
                  {repositoryMap.entries.slice(0, 100).map((entry) => (
                    <div
                      key={entry.path}
                      className="truncate rounded-lg bg-muted/30 px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground"
                      title={entry.path}
                    >
                      {entry.path}
                    </div>
                  ))}
                </div>
              ) : (
                <Empty>No eligible text files were discovered.</Empty>
              )}
            </AgentSection>
          ) : null}

          {workspace?.capabilities.git ? (
            <AgentSection
              icon={<GitBranch className="size-4" />}
              title="Git"
              detail={
                gitStatus
                  ? `${gitStatus.branch ?? "Detached HEAD"}, ${gitStatus.clean ? "clean" : "working tree changed"}`
                  : "Git status unavailable"
              }
              actions={
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() =>
                    void runAction(
                      "git-status",
                      () => getAgentGitStatus(projectId),
                      applyGitStatus,
                    )
                  }
                  disabled={Boolean(busy)}
                >
                  <RefreshCw
                    className={isBusy("git-status") ? "animate-spin" : ""}
                  />
                  Status
                </Button>
              }
            >
              {gitStatus ? (
                <>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(gitStatus.counts).map(([label, count]) => (
                      <Badge key={label} variant="outline">
                        {count} {label}
                      </Badge>
                    ))}
                  </div>
                  {gitStatus.files.length ? (
                    <div className="mt-2 max-h-40 overflow-auto rounded-xl bg-muted/30 p-2 font-mono text-[11px] leading-5 text-muted-foreground">
                      {gitStatus.files.slice(0, 100).map((file) => (
                        <div
                          key={`${file.code}:${file.path}`}
                          className="truncate"
                        >
                          <span className="mr-2 text-foreground/70">
                            {file.code}
                          </span>
                          {file.path}
                        </div>
                      ))}
                    </div>
                  ) : null}
                  <div className="mt-2 flex gap-1.5">
                    <Button
                      type="button"
                      size="xs"
                      variant="outline"
                      onClick={() =>
                        void runAction(
                          "git-diff",
                          () =>
                            getAgentGitDiff(projectId, {
                              staged: false,
                              maxBytes: 512_000,
                            }),
                          setGitDiff,
                        )
                      }
                      disabled={Boolean(busy)}
                    >
                      <GitCompare /> Working diff
                    </Button>
                    <Button
                      type="button"
                      size="xs"
                      variant="outline"
                      onClick={() =>
                        void runAction(
                          "git-diff-staged",
                          () =>
                            getAgentGitDiff(projectId, {
                              staged: true,
                              maxBytes: 512_000,
                            }),
                          setGitDiff,
                        )
                      }
                      disabled={Boolean(busy)}
                    >
                      <GitCompare /> Staged diff
                    </Button>
                  </div>
                  {gitDiff ? (
                    <details
                      className="mt-2 rounded-xl bg-muted/35 px-3 py-2"
                      open={true}
                    >
                      <summary className="cursor-pointer text-xs font-medium">
                        {gitDiff.staged ? "Staged" : "Working"} diff
                        {gitDiff.truncated ? " (truncated)" : ""}
                      </summary>
                      <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-muted-foreground">
                        {gitDiff.diff || "No diff."}
                      </pre>
                    </details>
                  ) : null}
                  <div className="mt-3 rounded-xl border border-border/60 bg-background/45 p-3">
                    <div className="flex flex-wrap items-start gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-xs font-semibold text-foreground">
                          Prepare selected commit
                        </p>
                        <p className="mt-0.5 text-[11px] text-muted-foreground">
                          Confirmation creates a Studio-owned prepared ref. It
                          does not move the current branch or change the working
                          tree.
                        </p>
                      </div>
                      <Badge variant="outline">
                        {selectedCommitPaths.size} selected
                      </Badge>
                    </div>
                    <div className="mt-2 flex gap-1.5">
                      <Input
                        value={commitPathFilter}
                        onChange={(event) =>
                          setCommitPathFilter(event.target.value)
                        }
                        placeholder="Filter changed paths"
                        aria-label="Filter changed paths"
                        className="h-8 text-xs"
                      />
                      <Button
                        type="button"
                        size="xs"
                        variant="ghost"
                        onClick={() => {
                          invalidatePreparedCommit();
                          setSelectedCommitPaths((current) => {
                            const next = new Set(current);
                            for (const file of visibleCommitFiles) {
                              if (file.code !== "??") next.add(file.path);
                            }
                            return next;
                          });
                        }}
                        disabled={Boolean(busy) || !visibleCommitFiles.length}
                      >
                        Select shown
                      </Button>
                      <Button
                        type="button"
                        size="xs"
                        variant="ghost"
                        onClick={() => {
                          invalidatePreparedCommit();
                          setSelectedCommitPaths(new Set());
                        }}
                        disabled={Boolean(busy) || !selectedCommitPaths.size}
                      >
                        Clear
                      </Button>
                    </div>
                    <div className="mt-2 max-h-44 overflow-auto rounded-lg bg-muted/30 p-2">
                      {visibleCommitFiles.map((file) => (
                        <label
                          key={`${file.code}:${file.path}`}
                          className="flex min-w-0 items-center gap-2 py-1 font-mono text-[11px] text-muted-foreground"
                        >
                          <input
                            type="checkbox"
                            checked={selectedCommitPaths.has(file.path)}
                            onChange={(event) =>
                              toggleCommitPath(file.path, event.target.checked)
                            }
                            disabled={Boolean(busy) || file.code === "??"}
                            className="size-3.5 accent-primary"
                          />
                          <span className="w-5 shrink-0 text-foreground/70">
                            {file.code}
                          </span>
                          <span className="min-w-0 truncate">{file.path}</span>
                          {file.code === "??" ? (
                            <span className="ml-auto shrink-0 font-sans text-[10px]">
                              stage first
                            </span>
                          ) : null}
                        </label>
                      ))}
                      {!visibleCommitFiles.length ? (
                        <p className="py-2 text-center text-[11px] text-muted-foreground">
                          No matching tracked changes.
                        </p>
                      ) : null}
                    </div>
                    {filteredCommitFiles.length > visibleCommitFiles.length ? (
                      <p className="mt-1 text-[10px] text-muted-foreground">
                        Showing 200 of {filteredCommitFiles.length} matches.
                        Refine the path filter to select another file.
                      </p>
                    ) : null}
                    <div className="mt-2 flex gap-2">
                      <Input
                        value={commitMessage}
                        onChange={(event) => {
                          setCommitMessage(event.target.value);
                          invalidatePreparedCommit();
                        }}
                        placeholder="Commit message"
                        aria-label="Prepared commit message"
                        className="h-8 text-xs"
                      />
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={() => void prepareCommitPreview()}
                        disabled={
                          Boolean(busy) ||
                          !selectedCommitPaths.size ||
                          !commitMessage.trim()
                        }
                      >
                        {isBusy("commit-prepare") ? (
                          <Loader2 className="animate-spin" />
                        ) : (
                          <GitCompare />
                        )}
                        Preview
                      </Button>
                    </div>
                    {preparedCommit ? (
                      <div className="mt-3 rounded-lg border border-border/60 bg-muted/25 p-3">
                        <div className="flex flex-wrap items-center gap-2 text-xs">
                          <Badge variant="outline">awaiting confirmation</Badge>
                          <span className="font-medium">
                            {preparedCommit.message}
                          </span>
                          <span className="ml-auto text-[11px] text-muted-foreground">
                            expires {formatDate(preparedCommit.expiresAt)}
                          </span>
                        </div>
                        <div className="mt-2 max-h-32 overflow-auto rounded-lg bg-background/60 p-2 font-mono text-[11px] text-muted-foreground">
                          {(preparedCommit.files ?? []).map((file) => (
                            <div
                              key={`${file.code}:${file.path}`}
                              className="truncate"
                            >
                              <span className="mr-2 text-foreground/70">
                                {file.code}
                              </span>
                              {file.path}
                            </div>
                          ))}
                        </div>
                        <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-background/60 p-2 font-mono text-[11px] leading-5 text-muted-foreground">
                          {preparedCommit.diff || "No diff."}
                        </pre>
                        {preparedCommit.diffTruncated ? (
                          <p className="mt-1 text-[10px] text-amber-700 dark:text-amber-300">
                            The server truncated this diff preview at its review
                            limit.
                          </p>
                        ) : null}
                        <div className="mt-2 flex justify-end gap-1.5">
                          <Button
                            type="button"
                            size="xs"
                            variant="ghost"
                            onClick={() => setPreparedCommit(null)}
                            disabled={Boolean(busy)}
                          >
                            Discard preview
                          </Button>
                          <Button
                            type="button"
                            size="xs"
                            onClick={() => void confirmCommitPreview()}
                            disabled={Boolean(busy)}
                          >
                            {isBusy("commit-confirm") ? (
                              <Loader2 className="animate-spin" />
                            ) : (
                              <CheckCircle2 />
                            )}
                            Confirm prepared ref
                          </Button>
                        </div>
                      </div>
                    ) : null}
                    {confirmedCommit ? (
                      <div className="mt-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-[11px] text-foreground">
                        Prepared ref created at{" "}
                        <span className="font-mono">
                          {confirmedCommit.commitSha?.slice(0, 12)}
                        </span>
                        . The current branch and working tree were not changed.
                      </div>
                    ) : null}
                    {commitConfirmationError ? (
                      <div
                        role="alert"
                        className="mt-2 rounded-lg bg-destructive/10 px-3 py-2 text-[11px] text-destructive"
                      >
                        {commitConfirmationError}
                      </div>
                    ) : null}
                  </div>
                </>
              ) : (
                <Empty>Refresh to read Git status.</Empty>
              )}
            </AgentSection>
          ) : null}

          {workspace?.capabilities.verification ? (
            <AgentSection
              icon={<ShieldCheck className="size-4" />}
              title="Verification"
              detail={latestVerificationSummary(verificationRuns)}
              actions={
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() =>
                    setVerificationChecks((checks) => [
                      ...checks,
                      {
                        name: `check-${checks.length + 1}`,
                        kind: "custom",
                        command: "",
                        required: true,
                        timeoutSeconds: 300,
                        logLimitBytes: 256 * 1024,
                      },
                    ])
                  }
                  disabled={Boolean(busy)}
                >
                  <Plus /> Add check
                </Button>
              }
            >
              <div className="space-y-2">
                {verificationChecks.map((check, index) => (
                  <div
                    key={`${index}:${check.name}`}
                    className="grid grid-cols-[minmax(0,0.8fr)_minmax(0,1.6fr)_auto] gap-2"
                  >
                    <Input
                      value={check.name}
                      onChange={(event) =>
                        updateVerificationCheck(index, {
                          name: event.target.value,
                        })
                      }
                      placeholder="Name"
                      aria-label={`Verification check ${index + 1} name`}
                      className="h-8 text-xs"
                    />
                    <Input
                      value={check.command}
                      onChange={(event) =>
                        updateVerificationCheck(index, {
                          command: event.target.value,
                        })
                      }
                      placeholder="Command"
                      aria-label={`Verification check ${index + 1} command`}
                      className="h-8 font-mono text-xs"
                    />
                    <Button
                      type="button"
                      size="icon-sm"
                      variant="ghost"
                      aria-label={`Remove ${check.name || "verification check"}`}
                      onClick={() =>
                        setVerificationChecks((checks) =>
                          checks.filter(
                            (_, checkIndex) => checkIndex !== index,
                          ),
                        )
                      }
                    >
                      <Trash2 />
                    </Button>
                  </div>
                ))}
                {verificationChecks.length === 0 ? (
                  <Empty>Add a test, lint, build, or custom command.</Empty>
                ) : null}
              </div>
              <div className="mt-3 flex items-start justify-between gap-4 rounded-xl bg-muted/35 px-3 py-3">
                <div className="min-w-0">
                  <label
                    id={`${goalCompletionPolicyId}-label`}
                    htmlFor={goalCompletionPolicyId}
                    className="text-xs font-medium text-foreground"
                  >
                    Require fresh verification before goal completion
                  </label>
                  <p className="mt-1 text-ui-11 text-muted-foreground">
                    Goal completion will require a passing primary-workspace run
                    with every required check after the latest source or policy
                    change.
                  </p>
                  {verificationConfigRevision > 0 ? (
                    <p className="mt-1 text-ui-11 text-muted-foreground">
                      Saved policy revision {verificationConfigRevision}
                    </p>
                  ) : null}
                </div>
                <Switch
                  id={goalCompletionPolicyId}
                  checked={requireVerificationForGoalCompletion}
                  onCheckedChange={setRequireVerificationForGoalCompletion}
                  disabled={Boolean(busy)}
                  aria-labelledby={`${goalCompletionPolicyId}-label`}
                />
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  onClick={() => void saveVerification()}
                  disabled={Boolean(busy)}
                >
                  {isBusy("verification-save") ? (
                    <Loader2 className="animate-spin" />
                  ) : (
                    <CheckCircle2 />
                  )}
                  Save checks
                </Button>
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  onClick={() => void runVerification(false)}
                  disabled={
                    Boolean(busy) ||
                    normalizedVerificationChecks(verificationChecks).length ===
                      0
                  }
                >
                  {isBusy("verification-run") ? (
                    <Loader2 className="animate-spin" />
                  ) : (
                    <Play />
                  )}
                  Run now
                </Button>
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  onClick={() => void runVerification(true)}
                  disabled={
                    Boolean(busy) ||
                    normalizedVerificationChecks(verificationChecks).length ===
                      0
                  }
                >
                  <Play /> Run in background
                </Button>
              </div>
              {verificationRuns.length ? (
                <div className="mt-3 space-y-2">
                  {verificationRuns.slice(0, 5).map((run) => (
                    <details
                      key={run.id}
                      className="rounded-xl bg-muted/35 px-3 py-2"
                    >
                      <summary className="flex cursor-pointer list-none items-center gap-2 text-xs">
                        <Badge variant={statusBadgeVariant(run.status)}>
                          {agentStatusLabel(run.status)}
                        </Badge>
                        <span>{formatDate(run.startedAt)}</span>
                        {run.stale ? (
                          <Badge variant="outline">stale</Badge>
                        ) : null}
                        <span className="ml-auto text-muted-foreground">
                          {run.results.length} checks
                        </span>
                      </summary>
                      <div className="mt-2 space-y-2">
                        {run.results.map((result) => (
                          <details
                            key={result.name}
                            className="rounded-lg bg-background/60 p-2"
                          >
                            <summary className="cursor-pointer text-xs font-medium">
                              {result.name}: {agentStatusLabel(result.status)}
                            </summary>
                            <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-muted-foreground">
                              {result.output || "No output."}
                            </pre>
                          </details>
                        ))}
                      </div>
                    </details>
                  ))}
                </div>
              ) : null}
            </AgentSection>
          ) : null}

          {workspace?.capabilities.plans ? (
            <AgentSection
              icon={<ListChecks className="size-4" />}
              title="Plans"
              detail={`${activePlanCount} active, ${plans.length} total`}
            >
              <div className="grid gap-2 sm:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)_auto]">
                <Input
                  value={planTitle}
                  onChange={(event) => setPlanTitle(event.target.value)}
                  placeholder="Plan title"
                  aria-label="Plan title"
                  className="h-8 text-xs"
                />
                <Textarea
                  value={planTasks}
                  onChange={(event) => setPlanTasks(event.target.value)}
                  placeholder="One task per line"
                  aria-label="Plan tasks"
                  className="min-h-8 py-2 text-xs"
                />
                <Button
                  type="button"
                  size="sm"
                  onClick={() => void createPlan()}
                  disabled={Boolean(busy)}
                >
                  <Plus /> Create
                </Button>
              </div>
              <div className="mt-3 space-y-2">
                {plans.slice(0, 8).map((plan) => {
                  const progress = agentPlanProgress(plan);
                  return (
                    <details
                      key={plan.id}
                      className="rounded-xl bg-muted/35 px-3 py-2"
                      open={plan.status === "active"}
                    >
                      <summary className="flex cursor-pointer list-none items-center gap-2 text-xs">
                        <Badge variant={statusBadgeVariant(plan.status)}>
                          {agentStatusLabel(plan.status)}
                        </Badge>
                        <span className="min-w-0 flex-1 truncate font-medium">
                          {plan.title}
                        </span>
                        <span className="text-muted-foreground">
                          {progress.completed}/{progress.total}
                        </span>
                      </summary>
                      <div className="mt-2 space-y-1.5">
                        {plan.tasks.map((task) => (
                          <div
                            key={task.id}
                            className="flex items-center gap-2 rounded-lg bg-background/60 px-2.5 py-2"
                          >
                            <span className="min-w-0 flex-1 truncate text-xs">
                              {task.title}
                            </span>
                            <select
                              value={task.status}
                              onChange={(event) =>
                                void setTaskStatus(
                                  plan,
                                  task.id,
                                  event.target.value as AgentPlanTaskStatus,
                                )
                              }
                              disabled={Boolean(busy)}
                              aria-label={`Status for ${task.title}`}
                              className="h-7 rounded-full border border-border bg-background px-2 text-[11px] outline-none"
                            >
                              <option value="pending">Pending</option>
                              <option value="running">Running</option>
                              <option value="blocked">Blocked</option>
                              <option value="completed">Completed</option>
                              <option value="cancelled">Cancelled</option>
                            </select>
                          </div>
                        ))}
                      </div>
                      <div className="mt-2 flex justify-end gap-1.5">
                        {plan.status === "active" ? (
                          <Button
                            type="button"
                            size="xs"
                            variant="ghost"
                            onClick={() =>
                              void setPlanStatus(plan, "completed")
                            }
                            disabled={Boolean(busy)}
                          >
                            <CheckCircle2 /> Complete plan
                          </Button>
                        ) : (
                          <Button
                            type="button"
                            size="xs"
                            variant="ghost"
                            onClick={() => void setPlanStatus(plan, "active")}
                            disabled={Boolean(busy)}
                          >
                            <RotateCcw /> Reopen
                          </Button>
                        )}
                      </div>
                    </details>
                  );
                })}
                {plans.length === 0 ? (
                  <Empty>No durable plans yet.</Empty>
                ) : null}
              </div>
            </AgentSection>
          ) : null}

          {workspace?.capabilities.background ? (
            <AgentSection
              icon={<Play className="size-4" />}
              title="Background tasks"
              detail={`${backgroundTasks.length} recorded`}
              actions={
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() =>
                    void runAction(
                      "background-refresh",
                      () => listAgentBackgroundTasks(projectId, 50),
                      setBackgroundTasks,
                    )
                  }
                  disabled={Boolean(busy)}
                >
                  <RefreshCw
                    className={
                      isBusy("background-refresh") ? "animate-spin" : ""
                    }
                  />
                  Refresh
                </Button>
              }
            >
              <div className="rounded-xl border border-border/60 bg-background/45 p-3">
                <Textarea
                  value={agentInstruction}
                  onChange={(event) => setAgentInstruction(event.target.value)}
                  placeholder="Describe the task for the background coding agent"
                  aria-label="Background agent instructions"
                  className="min-h-20 text-xs"
                />
                <div className="mt-2 grid gap-2 sm:grid-cols-3">
                  <select
                    value={agentRuntimeKind}
                    onChange={(event) => {
                      const kind = event.target
                        .value as AgentBackgroundRuntimeKind;
                      setAgentRuntimeKind(kind);
                      if (kind === "local") setAgentRuntimeProviderId("");
                    }}
                    disabled={Boolean(busy)}
                    aria-label="Background agent runtime kind"
                    className="h-8 rounded-lg border border-border bg-background px-2 text-xs outline-none"
                  >
                    <option value="local">Local runtime</option>
                    <option value="provider">Saved provider</option>
                  </select>
                  <Input
                    value={agentRuntimeModel}
                    onChange={(event) =>
                      setAgentRuntimeModel(event.target.value)
                    }
                    placeholder={
                      agentRuntimeKind === "local"
                        ? "Loaded model ID"
                        : "Enabled provider model"
                    }
                    aria-label="Background agent model"
                    disabled={Boolean(busy)}
                    className="h-8 font-mono text-xs"
                  />
                  <Input
                    value={agentRuntimeProviderId}
                    onChange={(event) =>
                      setAgentRuntimeProviderId(event.target.value)
                    }
                    placeholder="Saved provider connection ID"
                    aria-label="Background agent provider connection ID"
                    disabled={Boolean(busy) || agentRuntimeKind !== "provider"}
                    className="h-8 font-mono text-xs"
                  />
                  <select
                    value={agentPermissionMode}
                    onChange={(event) => {
                      const permissionMode = event.target
                        .value as AgentBackgroundPermissionMode;
                      if (
                        backgroundAgentPermissionNeedsConfirmation(
                          permissionMode,
                        ) &&
                        !window.confirm(BACKGROUND_AGENT_FULL_ACCESS_WARNING)
                      ) {
                        return;
                      }
                      setAgentPermissionMode(permissionMode);
                    }}
                    disabled={Boolean(busy)}
                    aria-label="Background agent permission mode"
                    className="h-8 rounded-lg border border-border bg-background px-2 text-xs outline-none"
                  >
                    <option value="off">Run tools in project sandbox</option>
                    <option value="full">Full access, project-bound</option>
                  </select>
                  <Input
                    value={agentReasoningEffort}
                    onChange={(event) =>
                      setAgentReasoningEffort(event.target.value)
                    }
                    placeholder="Reasoning effort, optional"
                    aria-label="Background agent reasoning effort"
                    disabled={Boolean(busy)}
                    className="h-8 text-xs"
                  />
                  <Input
                    type="number"
                    min={1}
                    max={32_768}
                    step={1}
                    value={agentMaxOutputTokens}
                    onChange={(event) =>
                      setAgentMaxOutputTokens(event.target.value)
                    }
                    aria-label="Background agent maximum output tokens"
                    disabled={Boolean(busy)}
                    className="h-8 font-mono text-xs"
                  />
                </div>
                <p className="mt-2 rounded-lg bg-muted/35 px-2.5 py-2 text-[11px] text-muted-foreground">
                  {BACKGROUND_AGENT_PERMISSION_POLICY}
                </p>
                <div className="mt-2 grid gap-2 sm:grid-cols-3">
                  <select
                    value={agentPlanId}
                    onChange={(event) => {
                      setAgentPlanId(event.target.value);
                      setAgentPlanTaskId("");
                    }}
                    disabled={Boolean(busy)}
                    aria-label="Background agent plan snapshot"
                    className="h-8 rounded-lg border border-border bg-background px-2 text-xs outline-none"
                  >
                    <option value="">No plan snapshot</option>
                    {plans.map((plan) => (
                      <option key={plan.id} value={plan.id}>
                        {plan.title}
                      </option>
                    ))}
                  </select>
                  <select
                    value={agentPlanTaskId}
                    onChange={(event) => setAgentPlanTaskId(event.target.value)}
                    disabled={Boolean(busy) || !selectedAgentPlan}
                    aria-label="Background agent plan task snapshot"
                    className="h-8 rounded-lg border border-border bg-background px-2 text-xs outline-none"
                  >
                    <option value="">Whole plan</option>
                    {selectedAgentPlan?.tasks.map((task) => (
                      <option key={task.id} value={task.id}>
                        {task.title}
                      </option>
                    ))}
                  </select>
                  <select
                    value={agentWorktreeId}
                    onChange={(event) => setAgentWorktreeId(event.target.value)}
                    disabled={Boolean(busy)}
                    aria-label="Background agent worktree"
                    className="h-8 rounded-lg border border-border bg-background px-2 font-mono text-xs outline-none"
                  >
                    <option value="">Primary workspace</option>
                    {worktrees
                      .filter(
                        (worktree) =>
                          worktree.status === "active" &&
                          !worktree.backgroundTaskId,
                      )
                      .map((worktree) => (
                        <option key={worktree.id} value={worktree.id}>
                          {worktree.branch}
                        </option>
                      ))}
                  </select>
                </div>
                {agentWorktreeId ? (
                  <div className="mt-2 flex items-center justify-between gap-4 rounded-lg bg-muted/35 px-2.5 py-2">
                    <label
                      id={`${cleanupWorktreePolicyId}-label`}
                      htmlFor={cleanupWorktreePolicyId}
                      className="text-[11px] text-muted-foreground"
                    >
                      Remove this Studio-owned worktree if cancellation leaves
                      it clean
                    </label>
                    <Switch
                      id={cleanupWorktreePolicyId}
                      checked={cleanupWorktreeOnCancel}
                      onCheckedChange={setCleanupWorktreeOnCancel}
                      disabled={Boolean(busy)}
                      aria-labelledby={`${cleanupWorktreePolicyId}-label`}
                    />
                  </div>
                ) : null}
                <div className="mt-2 flex justify-end gap-1.5">
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    onClick={() => void queueBackgroundAgent(false)}
                    disabled={
                      Boolean(busy) ||
                      !agentInstruction.trim() ||
                      !agentRuntimeReady
                    }
                  >
                    {isBusy("agent-queue") ? (
                      <Loader2 className="animate-spin" />
                    ) : (
                      <Plus />
                    )}
                    Queue agent
                  </Button>
                  <Button
                    type="button"
                    size="xs"
                    onClick={() => void queueBackgroundAgent(true)}
                    disabled={
                      Boolean(busy) ||
                      !agentInstruction.trim() ||
                      !agentRuntimeReady
                    }
                  >
                    {isBusy("agent-start") ? (
                      <Loader2 className="animate-spin" />
                    ) : (
                      <Play />
                    )}
                    Start agent
                  </Button>
                </div>
              </div>
              <div className="space-y-1.5">
                {backgroundTasks.slice(0, 20).map((task) => {
                  const actions = agentBackgroundActions(task);
                  const snapshot = agentBackgroundSnapshot(task);
                  const linkedWorktree = snapshot.worktreeId
                    ? worktrees.find(
                        (worktree) => worktree.id === snapshot.worktreeId,
                      )
                    : null;
                  return (
                    <div
                      key={task.id}
                      className="rounded-xl bg-muted/35 px-3 py-2"
                    >
                      <div className="flex items-center gap-2">
                        <Badge variant={statusBadgeVariant(task.status)}>
                          {agentStatusLabel(task.status)}
                        </Badge>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-xs font-medium">
                            {task.kind === "agent"
                              ? task.payload.instruction || "Agent task"
                              : task.kind}
                          </p>
                          <p className="text-[11px] text-muted-foreground">
                            Attempt {task.attempt} {formatDate(task.updatedAt)}
                          </p>
                        </div>
                        {actions.canStart ? (
                          <Button
                            type="button"
                            size="icon-xs"
                            variant="ghost"
                            aria-label="Start task"
                            onClick={() =>
                              void mutateBackgroundTask(task, "start")
                            }
                            disabled={Boolean(busy)}
                          >
                            <Play />
                          </Button>
                        ) : null}
                        {actions.canCancel ? (
                          <Button
                            type="button"
                            size="icon-xs"
                            variant="ghost"
                            aria-label="Cancel task"
                            onClick={() =>
                              void mutateBackgroundTask(task, "cancel")
                            }
                            disabled={Boolean(busy)}
                          >
                            <Square />
                          </Button>
                        ) : null}
                        {actions.canRetry ? (
                          <Button
                            type="button"
                            size="icon-xs"
                            variant="ghost"
                            aria-label="Retry task"
                            onClick={() =>
                              void mutateBackgroundTask(task, "retry")
                            }
                            disabled={Boolean(busy)}
                          >
                            <RotateCcw />
                          </Button>
                        ) : null}
                      </div>
                      {task.kind === "agent" ? (
                        <details className="mt-2 border-t border-border/50 pt-2 text-[11px] text-muted-foreground">
                          <summary className="cursor-pointer font-medium text-foreground/75">
                            Captured context
                            <Badge variant="outline" className="ml-2">
                              immutable
                            </Badge>
                          </summary>
                          <dl className="mt-2 grid gap-x-3 gap-y-1 sm:grid-cols-[auto_minmax(0,1fr)]">
                            <dt>Goal</dt>
                            <dd className="min-w-0 truncate text-foreground/80">
                              {snapshot.goal || "No goal set"}
                              {snapshot.goalStatus
                                ? ` (${snapshot.goalStatus})`
                                : ""}
                            </dd>
                            <dt>Plan</dt>
                            <dd className="min-w-0 truncate text-foreground/80">
                              {snapshot.planTitle
                                ? `${snapshot.planTitle} at revision ${snapshot.planRevision ?? 0}`
                                : "No plan linked"}
                            </dd>
                            {snapshot.planTaskTitle ? (
                              <>
                                <dt>Plan task</dt>
                                <dd className="min-w-0 truncate text-foreground/80">
                                  {snapshot.planTaskTitle}
                                </dd>
                              </>
                            ) : null}
                            <dt>Workspace</dt>
                            <dd className="min-w-0 truncate font-mono text-foreground/80">
                              {linkedWorktree?.branch || "Primary workspace"}
                            </dd>
                            <dt>Runtime</dt>
                            <dd className="min-w-0 truncate font-mono text-foreground/80">
                              {snapshot.runtime
                                ? `${snapshot.runtime.kind}: ${snapshot.runtime.model}`
                                : "No runtime captured"}
                            </dd>
                            {snapshot.runtime?.providerId ? (
                              <>
                                <dt>Provider</dt>
                                <dd className="min-w-0 truncate font-mono text-foreground/80">
                                  {snapshot.runtime.providerId}
                                </dd>
                              </>
                            ) : null}
                            <dt>Permissions</dt>
                            <dd className="text-foreground/80">
                              {snapshot.runtime?.permissionMode ||
                                "Unavailable"}
                            </dd>
                            <dt>Generation</dt>
                            <dd className="text-foreground/80">
                              {snapshot.runtime
                                ? `${snapshot.runtime.maxOutputTokens.toLocaleString()} max output tokens${snapshot.runtime.reasoningEffort ? `, ${snapshot.runtime.reasoningEffort} reasoning` : ""}`
                                : "Unavailable"}
                            </dd>
                            <dt>App exit</dt>
                            <dd className="text-foreground/80">
                              {snapshot.appExitPolicy === "interrupt"
                                ? "Mark active task interrupted"
                                : snapshot.appExitPolicy}
                            </dd>
                          </dl>
                        </details>
                      ) : null}
                      {task.error ? (
                        <p className="mt-2 text-[11px] text-destructive">
                          {safeAgentWorkspaceError(task.error)}
                        </p>
                      ) : null}
                    </div>
                  );
                })}
                {backgroundTasks.length === 0 ? (
                  <Empty>No background tasks recorded.</Empty>
                ) : null}
              </div>
            </AgentSection>
          ) : null}

          {workspace?.capabilities.worktrees ? (
            <AgentSection
              icon={<GitFork className="size-4" />}
              title="Owned worktrees"
              detail="Only Studio-owned worktrees can be removed here"
            >
              <div className="flex gap-2">
                <Input
                  value={worktreeBranch}
                  onChange={(event) => setWorktreeBranch(event.target.value)}
                  placeholder="Optional unsloth-studio/branch"
                  aria-label="Worktree branch"
                  className="h-8 font-mono text-xs"
                />
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => void createWorktree()}
                  disabled={Boolean(busy)}
                >
                  <Plus /> Create
                </Button>
              </div>
              <div className="mt-2 space-y-1.5">
                {worktrees.map((worktree) => {
                  const linkedTask = worktree.backgroundTaskId
                    ? backgroundTasks.find(
                        (task) => task.id === worktree.backgroundTaskId,
                      )
                    : null;
                  const mergeAction = agentWorktreeMergeAction({
                    worktree,
                    gitStatus,
                    linkedTask: linkedTask ?? null,
                  });
                  return (
                    <div
                      key={worktree.id}
                      className="rounded-xl bg-muted/35 px-3 py-2"
                    >
                      <div className="flex items-center gap-2">
                        <Badge variant={statusBadgeVariant(worktree.status)}>
                          {agentStatusLabel(worktree.status)}
                        </Badge>
                        <span className="min-w-0 flex-1 truncate font-mono text-xs">
                          {worktree.branch}
                        </span>
                        {worktree.status === "active" ? (
                          <>
                            <Button
                              type="button"
                              size="xs"
                              variant="outline"
                              aria-label={`Merge ${worktree.branch}`}
                              title={mergeAction.reason ?? undefined}
                              onClick={() => void mergeWorktree(worktree)}
                              disabled={Boolean(busy) || !mergeAction.canMerge}
                            >
                              <GitCompare /> Merge
                            </Button>
                            <Button
                              type="button"
                              size="icon-xs"
                              variant="destructive"
                              aria-label={`Remove ${worktree.branch}`}
                              onClick={() => void cleanupWorktree(worktree)}
                              disabled={Boolean(busy)}
                            >
                              <Trash2 />
                            </Button>
                          </>
                        ) : null}
                      </div>
                      {worktree.backgroundTaskId ? (
                        <p className="mt-1 truncate text-[11px] text-muted-foreground">
                          Linked task: {worktree.backgroundTaskId}
                        </p>
                      ) : null}
                      {worktree.merge ? (
                        <div className="mt-2 border-t border-border/50 pt-2 text-[11px] text-muted-foreground">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge
                              variant={statusBadgeVariant(
                                worktree.merge.status,
                              )}
                            >
                              merge {agentStatusLabel(worktree.merge.status)}
                            </Badge>
                            <span>
                              {worktree.merge.targetBranch} at{" "}
                              {worktree.merge.expectedTargetHead.slice(0, 12)}
                            </span>
                            {worktree.merge.resultHead ? (
                              <span>
                                result {worktree.merge.resultHead.slice(0, 12)}
                              </span>
                            ) : null}
                          </div>
                          {worktree.merge.conflicts.length ? (
                            <ul className="mt-1 list-disc pl-4 font-mono text-destructive">
                              {worktree.merge.conflicts.map((path) => (
                                <li key={path} className="truncate">
                                  {path}
                                </li>
                              ))}
                            </ul>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  );
                })}
                {worktrees.length === 0 ? (
                  <Empty>No owned worktrees.</Empty>
                ) : null}
              </div>
            </AgentSection>
          ) : null}

          {workspace?.capabilities.review ? (
            <AgentSection
              icon={<Clipboard className="size-4" />}
              title="Review"
              detail="Builds local evidence and draft text without submitting anything"
              actions={
                <>
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    onClick={() =>
                      void runAction(
                        "review",
                        () => getAgentReview(projectId),
                        setReview,
                      )
                    }
                    disabled={Boolean(busy)}
                  >
                    <RefreshCw
                      className={isBusy("review") ? "animate-spin" : ""}
                    />
                    Build review
                  </Button>
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    onClick={() =>
                      void runAction(
                        "review-draft",
                        () => createAgentPullRequestDraft(projectId),
                        setPullRequestDraft,
                      )
                    }
                    disabled={Boolean(busy)}
                  >
                    <FileText /> Draft PR text
                  </Button>
                </>
              }
            >
              {review ? (
                <div className="grid gap-2 sm:grid-cols-3">
                  <div className="rounded-xl bg-muted/35 px-3 py-2">
                    <p className="text-[11px] text-muted-foreground">Goal</p>
                    <p className="mt-1 line-clamp-2 text-xs font-medium">
                      {review.goal || "No goal set"}
                    </p>
                  </div>
                  <div className="rounded-xl bg-muted/35 px-3 py-2">
                    <p className="text-[11px] text-muted-foreground">Plans</p>
                    <p className="mt-1 text-xs font-medium">
                      {review.plans.length} recorded
                    </p>
                  </div>
                  <div className="rounded-xl bg-muted/35 px-3 py-2">
                    <p className="text-[11px] text-muted-foreground">
                      Verification
                    </p>
                    <p className="mt-1 text-xs font-medium">
                      {latestVerificationSummary(review.verification)}
                    </p>
                  </div>
                </div>
              ) : (
                <Empty>Review evidence is generated only when requested.</Empty>
              )}
              {pullRequestDraft ? (
                <details
                  className="mt-2 rounded-xl bg-muted/35 px-3 py-2"
                  open={true}
                >
                  <summary className="cursor-pointer text-xs font-medium">
                    Local pull request draft
                  </summary>
                  <Input
                    readOnly={true}
                    value={pullRequestDraft.title}
                    aria-label="Pull request draft title"
                    className="mt-2 h-8 text-xs"
                  />
                  <Textarea
                    readOnly={true}
                    value={pullRequestDraft.body}
                    aria-label="Pull request draft body"
                    className="mt-2 max-h-80 min-h-40 font-mono text-xs"
                  />
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    className="mt-2"
                    onClick={() => {
                      void navigator.clipboard
                        .writeText(
                          `${pullRequestDraft.title}\n\n${pullRequestDraft.body}`,
                        )
                        .then(() => toast.success("Draft copied"))
                        .catch((error) =>
                          toast.error("Copy failed", {
                            description: safeAgentWorkspaceError(error),
                          }),
                        );
                    }}
                  >
                    <Clipboard /> Copy draft
                  </Button>
                </details>
              ) : null}
              <div className="mt-3 rounded-xl border border-border/60 bg-background/45 p-3">
                <div className="flex flex-wrap items-start gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-semibold text-foreground">
                      Connected GitHub handoff
                    </p>
                    <p className="mt-0.5 text-[11px] text-muted-foreground">
                      Preview the exact request first. Submission requires a
                      separate explicit action and is never retried
                      automatically.
                    </p>
                  </div>
                  <Badge variant="outline">connector</Badge>
                </div>
                <div className="mt-2 grid gap-2 sm:grid-cols-3">
                  <Input
                    value={githubConnectorId}
                    onChange={(event) => {
                      setGithubConnectorId(event.target.value);
                      invalidatePullRequestHandoff();
                    }}
                    placeholder="GitHub connector ID"
                    aria-label="GitHub connector ID"
                    disabled={Boolean(busy) || githubHandoffLocked}
                    className="h-8 text-xs"
                  />
                  <Input
                    value={githubOwner}
                    onChange={(event) => {
                      setGithubOwner(event.target.value);
                      invalidatePullRequestHandoff();
                    }}
                    placeholder="Owner"
                    aria-label="GitHub owner"
                    disabled={Boolean(busy) || githubHandoffLocked}
                    className="h-8 text-xs"
                  />
                  <Input
                    value={githubRepository}
                    onChange={(event) => {
                      setGithubRepository(event.target.value);
                      invalidatePullRequestHandoff();
                    }}
                    placeholder="Repository"
                    aria-label="GitHub repository"
                    disabled={Boolean(busy) || githubHandoffLocked}
                    className="h-8 text-xs"
                  />
                  <Input
                    value={githubBase}
                    onChange={(event) => {
                      setGithubBase(event.target.value);
                      invalidatePullRequestHandoff();
                    }}
                    placeholder="Base branch"
                    aria-label="GitHub base branch"
                    disabled={Boolean(busy) || githubHandoffLocked}
                    className="h-8 font-mono text-xs"
                  />
                  <Input
                    value={githubHead}
                    onChange={(event) => {
                      setGithubHead(event.target.value);
                      invalidatePullRequestHandoff();
                    }}
                    placeholder="Head branch"
                    aria-label="GitHub head branch"
                    disabled={Boolean(busy) || githubHandoffLocked}
                    className="h-8 font-mono text-xs"
                  />
                  <div className="flex h-8 items-center justify-between rounded-lg border border-border bg-background px-2.5">
                    <label
                      id={`${githubDraftId}-label`}
                      htmlFor={githubDraftId}
                      className="text-xs text-foreground"
                    >
                      Draft pull request
                    </label>
                    <Switch
                      id={githubDraftId}
                      checked={githubDraft}
                      onCheckedChange={(checked) => {
                        setGithubDraft(checked);
                        invalidatePullRequestHandoff();
                      }}
                      disabled={Boolean(busy) || githubHandoffLocked}
                      aria-labelledby={`${githubDraftId}-label`}
                    />
                  </div>
                </div>
                <div className="mt-2 flex justify-end">
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    onClick={() => void preparePullRequestHandoff()}
                    disabled={
                      Boolean(busy) ||
                      githubHandoffLocked ||
                      Boolean(pullRequestHandoff) ||
                      !githubFieldsComplete
                    }
                  >
                    {isBusy("github-preview") ? (
                      <Loader2 className="animate-spin" />
                    ) : (
                      <GitCompare />
                    )}
                    Preview handoff
                  </Button>
                </div>
                {pullRequestHandoff ? (
                  <div className="mt-3 rounded-lg border border-border/60 bg-muted/25 p-3">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <Badge variant="outline">preview only</Badge>
                      <span className="font-medium">
                        {pullRequestHandoff.request.owner}/
                        {pullRequestHandoff.request.repo}
                      </span>
                      <span className="font-mono text-[11px] text-muted-foreground">
                        {pullRequestHandoff.request.base} from{" "}
                        {pullRequestHandoff.request.head}
                      </span>
                      {pullRequestHandoff.request.draft ? (
                        <Badge variant="secondary">draft</Badge>
                      ) : null}
                      <span className="ml-auto text-[11px] text-muted-foreground">
                        expires {formatDate(pullRequestHandoff.expiresAt)}
                      </span>
                    </div>
                    <Input
                      readOnly={true}
                      value={pullRequestHandoff.request.title}
                      aria-label="GitHub handoff preview title"
                      className="mt-2 h-8 text-xs"
                    />
                    <Textarea
                      readOnly={true}
                      value={pullRequestHandoff.request.body}
                      aria-label="GitHub handoff preview body"
                      className="mt-2 max-h-80 min-h-40 font-mono text-xs"
                    />
                    <div className="mt-2 flex justify-end gap-1.5">
                      <Button
                        type="button"
                        size="xs"
                        variant="ghost"
                        onClick={() => setPullRequestHandoff(null)}
                        disabled={Boolean(busy)}
                      >
                        Discard preview
                      </Button>
                      <Button
                        type="button"
                        size="xs"
                        onClick={() => void submitPullRequestHandoff()}
                        disabled={Boolean(busy) || !githubCanSubmit}
                      >
                        {isBusy("github-submit") ? (
                          <Loader2 className="animate-spin" />
                        ) : (
                          <CheckCircle2 />
                        )}
                        Submit to GitHub
                      </Button>
                    </div>
                  </div>
                ) : null}
                {pullRequestSubmission ? (
                  <div
                    aria-live="polite"
                    aria-atomic="true"
                    className={`mt-3 rounded-lg px-3 py-2 text-[11px] ${
                      pullRequestSubmission.status === "unknown"
                        ? "bg-amber-500/10 text-amber-700 dark:text-amber-300"
                        : "bg-muted/35 text-foreground"
                    }`}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge
                        variant={statusBadgeVariant(
                          pullRequestSubmission.status,
                        )}
                      >
                        {agentStatusLabel(pullRequestSubmission.status)}
                      </Badge>
                      <span className="font-medium">
                        {pullRequestSubmission.connectorName}
                      </span>
                      <span>{pullRequestSubmission.repository}</span>
                    </div>
                    <p className="mt-1">{pullRequestSubmission.detail}</p>
                    {pullRequestSubmissionResult ? (
                      <p className="mt-1 break-words font-mono">
                        {pullRequestSubmissionResult.connectorResult}
                        {pullRequestSubmissionResult.connectorResultTruncated
                          ? " (truncated)"
                          : ""}
                      </p>
                    ) : null}
                    {pullRequestSubmission.status !== "submitting" ? (
                      <Button
                        type="button"
                        size="xs"
                        variant="ghost"
                        className="mt-1"
                        onClick={() => {
                          setPullRequestSubmission(null);
                          setPullRequestSubmissionResult(null);
                        }}
                        disabled={Boolean(busy)}
                      >
                        {pullRequestSubmission.status === "unknown"
                          ? "I checked GitHub, reset"
                          : "Prepare another handoff"}
                      </Button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </AgentSection>
          ) : null}
        </>
      )}
    </div>
  );
}
