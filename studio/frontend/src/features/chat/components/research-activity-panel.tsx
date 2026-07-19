// SPDX-License-Identifier: AGPL-3.0-only

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { openLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  Brain,
  Check,
  ChevronDown,
  ExternalLink,
  FileText,
  Globe2,
  Pencil,
  Plus,
  RotateCcw,
  Search,
  Square,
  Telescope,
  Trash2,
  X,
} from "lucide-react";
import {
  useCallback,
  type ReactElement,
  memo,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { motion, useReducedMotion } from "motion/react";
import {
  approveResearchRun,
  retryResearchRun,
  updateResearchPlan,
} from "../api/research-api";
import {
  type ResearchActivity,
  ensureResearchRunFollowed,
  ingestResearchUpdate,
  isSettledResearchRun,
  useResearchRunStore,
} from "../stores/research-run-store";
import type { ResearchRunStatus } from "../types/research";

const terminalStatuses = new Set<ResearchRunStatus>([
  "completed",
  "failed",
  "cancelled",
]);
const ACTIVITY_FOLLOW_SETTLE_MS = 450;
const ACTIVITY_BOTTOM_THRESHOLD_PX = 24;

function useResearchActivityScroll(runId: string) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const scrollToLatestRef = useRef<() => void>(() => undefined);
  const [isAtBottom, setIsAtBottom] = useState(true);

  useLayoutEffect(() => {
    const element = viewportRef.current;
    if (!element) return;

    let detached = false;
    let pointerActive = false;
    let touchStartY = 0;
    let lastScrollTop = element.scrollTop;
    let followUntil = performance.now() + ACTIVITY_FOLLOW_SETTLE_MS;
    let animationFrame: number | null = null;

    const distanceFromBottom = () =>
      Math.max(
        0,
        element.scrollHeight - element.scrollTop - element.clientHeight,
      );
    const updateAtBottom = (value: boolean) =>
      setIsAtBottom((current) => (current === value ? current : value));
    const requestTick = () => {
      if (animationFrame === null) animationFrame = requestAnimationFrame(tick);
    };
    const tick = () => {
      animationFrame = null;
      if (!detached && performance.now() < followUntil) {
        if (distanceFromBottom() > 1) element.scrollTop = element.scrollHeight;
        updateAtBottom(true);
        requestTick();
        return;
      }
      updateAtBottom(distanceFromBottom() <= ACTIVITY_BOTTOM_THRESHOLD_PX);
    };
    const followLayout = () => {
      if (detached) return;
      followUntil = performance.now() + ACTIVITY_FOLLOW_SETTLE_MS;
      requestTick();
    };
    const detach = () => {
      detached = true;
      followUntil = 0;
      updateAtBottom(false);
    };
    const innerScrollWillConsumeUpward = (target: EventTarget | null) => {
      let node = target instanceof Element ? target : null;
      while (node && node !== element) {
        if (node.scrollTop > 0) {
          const overflowY = window.getComputedStyle(node).overflowY;
          if (overflowY === "auto" || overflowY === "scroll") return true;
        }
        node = node.parentElement;
      }
      return false;
    };
    const scrollToLatest = () => {
      detached = false;
      followUntil = performance.now() + ACTIVITY_FOLLOW_SETTLE_MS;
      element.scrollTop = element.scrollHeight;
      lastScrollTop = element.scrollTop;
      updateAtBottom(true);
      requestTick();
    };
    scrollToLatestRef.current = scrollToLatest;

    const onScroll = () => {
      const scrollTop = element.scrollTop;
      const movingUp = scrollTop < lastScrollTop;
      if (!detached && pointerActive && movingUp) detach();
      if (
        detached &&
        scrollTop > lastScrollTop &&
        distanceFromBottom() <= ACTIVITY_BOTTOM_THRESHOLD_PX
      ) {
        detached = false;
        followLayout();
      }
      lastScrollTop = scrollTop;
      if (detached) updateAtBottom(false);
    };
    const onWheel = (event: WheelEvent) => {
      if (
        event.deltaY < 0 &&
        element.scrollTop > 0 &&
        !innerScrollWillConsumeUpward(event.target)
      ) {
        detach();
      }
    };
    const onTouchStart = (event: TouchEvent) => {
      touchStartY = event.touches[0]?.clientY ?? 0;
    };
    const onTouchMove = (event: TouchEvent) => {
      const y = event.touches[0]?.clientY ?? 0;
      if (
        y - touchStartY > 4 &&
        element.scrollTop > 0 &&
        !innerScrollWillConsumeUpward(event.target)
      ) {
        detach();
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (["ArrowUp", "PageUp", "Home"].includes(event.key)) detach();
    };
    const onPointerDown = () => {
      pointerActive = true;
    };
    const onPointerUp = () => {
      pointerActive = false;
    };

    const resizeObserver = new ResizeObserver(followLayout);
    const mutationObserver = new MutationObserver(followLayout);
    resizeObserver.observe(element, { box: "border-box" });
    mutationObserver.observe(element, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ["data-state", "hidden", "aria-hidden"],
    });
    element.addEventListener("scroll", onScroll, { passive: true });
    element.addEventListener("wheel", onWheel, { passive: true });
    element.addEventListener("touchstart", onTouchStart, { passive: true });
    element.addEventListener("touchmove", onTouchMove, { passive: true });
    element.addEventListener("keydown", onKeyDown);
    element.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);

    scrollToLatest();

    return () => {
      if (animationFrame !== null) cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
      element.removeEventListener("scroll", onScroll);
      element.removeEventListener("wheel", onWheel);
      element.removeEventListener("touchstart", onTouchStart);
      element.removeEventListener("touchmove", onTouchMove);
      element.removeEventListener("keydown", onKeyDown);
      element.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
      scrollToLatestRef.current = () => undefined;
    };
  }, [runId]);

  const scrollToLatest = useCallback(() => scrollToLatestRef.current(), []);
  return { viewportRef, isAtBottom, scrollToLatest };
}

export function researchStatusLabel(status: ResearchRunStatus): string {
  switch (status) {
    case "planning":
      return "Planning";
    case "awaiting_approval":
      return "Review plan";
    case "queued":
      return "Queued";
    case "running":
      return "Researching";
    case "paused":
      return "Paused";
    case "cancelling":
      return "Stopping";
    case "cancelled":
      return "Cancelled";
    case "completed":
      return "Complete";
    case "failed":
      return "Failed";
  }
}

function formatElapsed(start: number, end = Date.now()): string {
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function ActivityIcon({
  activity,
}: { activity: ResearchActivity }): ReactElement {
  const className = "size-3.5";
  if (activity.state === "running") return <Spinner className={className} />;
  if (activity.state === "failed")
    return <X className={cn(className, "text-destructive")} />;
  if (activity.state === "cancelled")
    return <Square className={cn(className, "text-muted-foreground")} />;
  if (activity.kind === "reasoning") return <Brain className={className} />;
  if (activity.kind === "plan") return <FileText className={className} />;
  if (activity.kind === "report") return <FileText className={className} />;
  if (activity.action === "fetch") return <BookOpen className={className} />;
  if (activity.action === "search") return <Search className={className} />;
  return <Check className={className} />;
}

const ActivityRow = memo(function ActivityRow({
  runId,
  activity,
}: {
  runId: string;
  activity: ResearchActivity;
}): ReactElement {
  const storedOpen = useResearchRunStore(
    (state) => state.activityOpenByRunId[runId]?.[activity.id],
  );
  const setActivityOpen = useResearchRunStore(
    (state) => state.setActivityOpen,
  );
  const open =
    storedOpen ??
    (activity.state === "running" || activity.state === "action");
  const hasDetails = Boolean(
    activity.reasoning ||
      activity.plan ||
      activity.input ||
      activity.sources?.length ||
      activity.evidenceSources?.length ||
      activity.excerpt ||
      activity.detail,
  );
  const content = (
    <div className="space-y-2 pb-3 pl-7 pr-1 text-[12.5px] text-muted-foreground">
      {activity.input ? (
        <p
          className={cn(
            "line-clamp-3 break-words rounded-xl bg-muted/45 px-3 py-2 text-foreground/80",
            activity.kind === "step" &&
              "bg-primary/[0.045] ring-1 ring-primary/10",
          )}
        >
          {activity.input}
        </p>
      ) : null}
      {activity.reasoning ? (
        <div className="max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded-xl bg-muted/35 px-3 py-2 leading-relaxed text-foreground/80">
          {activity.state === "running" && activity.reasoning.length > 8000
            ? `…\n${activity.reasoning.slice(-8000)}`
            : activity.reasoning}
        </div>
      ) : null}
      {activity.plan ? (
        <div className="space-y-2 rounded-xl bg-muted/35 px-3 py-2.5">
          <p className="font-medium text-foreground/85">
            {activity.plan.title}
          </p>
          {activity.plan.steps.slice(0, 3).map((step, index) => (
            <div key={`activity-plan-${index}`} className="flex gap-2">
              <span className="text-[10px] tabular-nums text-primary">
                {index + 1}
              </span>
              <span className="min-w-0">
                <span className="block font-medium text-foreground/80">
                  {step.title}
                </span>
                <span className="line-clamp-2 break-words">{step.query}</span>
              </span>
            </div>
          ))}
          {activity.plan.steps.length > 3 ? (
            <p className="pl-5 text-[11px] text-muted-foreground">
              +{activity.plan.steps.length - 3} more steps
            </p>
          ) : null}
        </div>
      ) : null}
      {activity.detail ? (
        <p
          className={cn(
            activity.kind === "step" &&
              activity.state !== "failed" &&
              "font-medium text-primary/75",
          )}
        >
          {activity.detail}
        </p>
      ) : null}
      {activity.sources?.map((source) => (
        <button
          key={`${activity.id}-${source.id ?? source.url}`}
          type="button"
          onClick={() => openLink(source.url)}
          className="group/source flex w-full items-start gap-2 rounded-xl px-2 py-2 text-left transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <Globe2 className="mt-0.5 size-3.5 shrink-0" />
          <span className="min-w-0 flex-1">
            <span className="block line-clamp-2 break-words font-medium text-foreground/85">
              {source.title || source.url}
            </span>
            <span className="block truncate text-[11px]">{source.url}</span>
            {source.snippet ? (
              <span className="mt-1 block line-clamp-2 leading-relaxed">
                {source.snippet}
              </span>
            ) : null}
          </span>
          <ExternalLink className="mt-0.5 size-3 opacity-0 transition-opacity group-hover/source:opacity-100" />
        </button>
      ))}
      {activity.evidenceSources?.map((source) => (
        <div
          key={`${activity.id}-${source.chunkId}`}
          className="rounded-xl bg-muted/45 px-3 py-2"
        >
          <p className="line-clamp-2 break-words font-medium text-foreground/85">
            {source.filename}
            {source.page ? ` · page ${source.page}` : ""}
          </p>
          {source.snippet ? (
            <p className="mt-1 line-clamp-3 leading-relaxed">
              {source.snippet}
            </p>
          ) : null}
        </div>
      ))}
      {activity.excerpt ? (
        <p className="line-clamp-5 whitespace-pre-wrap break-words rounded-xl bg-muted/45 px-3 py-2 leading-relaxed">
          {activity.excerpt}
        </p>
      ) : null}
    </div>
  );

  return (
    <Collapsible
      open={open}
      onOpenChange={(nextOpen) =>
        setActivityOpen(runId, activity.id, nextOpen)
      }
    >
      <div
        className={cn(
          "relative pl-7 before:absolute before:left-[7px] before:top-6 before:h-[calc(100%-12px)] before:w-px before:bg-border last:before:hidden",
          activity.kind === "step" && "before:bg-primary/20",
        )}
      >
        <CollapsibleTrigger
          disabled={!hasDetails}
          className="group/activity flex min-h-10 w-full items-start gap-2 py-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default"
        >
          <span
            className={cn(
              "absolute left-0 top-3 flex size-[15px] items-center justify-center rounded-full bg-background text-muted-foreground",
              activity.kind === "step" &&
                activity.state !== "failed" &&
                "bg-primary/10 text-primary",
              activity.state === "failed" && "text-destructive",
            )}
          >
            <ActivityIcon activity={activity} />
          </span>
          <span className="min-w-0 flex-1 break-words text-[13.5px] font-medium leading-5 text-foreground/90">
            {activity.title}
          </span>
          <time className="mt-0.5 shrink-0 text-[10.5px] tabular-nums text-muted-foreground">
            {new Date(activity.createdAt).toLocaleTimeString([], {
              hour: "numeric",
              minute: "2-digit",
            })}
          </time>
          {hasDetails ? (
            <ChevronDown className="mt-0.5 size-3.5 shrink-0 text-muted-foreground transition-transform group-data-[state=open]/activity:rotate-180" />
          ) : null}
        </CollapsibleTrigger>
        {hasDetails ? <CollapsibleContent>{content}</CollapsibleContent> : null}
      </div>
    </Collapsible>
  );
});

function PlanReview({ runId }: { runId: string }): ReactElement | null {
  const run = useResearchRunStore((state) => state.sessions[runId]?.run);
  const review = useResearchRunStore(
    (state) => state.planReviewByRunId[runId],
  );
  const setOpen = useResearchRunStore((state) => state.setPlanReviewOpen);
  const setEditing = useResearchRunStore(
    (state) => state.setPlanReviewEditing,
  );
  const setDraft = useResearchRunStore((state) => state.setPlanReviewDraft);
  const [pending, setPending] = useState(false);
  const stepKeyPrefix = useId();
  const [stepKeys, setStepKeys] = useState(() =>
    (review?.draft.steps ?? []).map((_, index) => `${stepKeyPrefix}-${index}`),
  );
  const reduceMotion = useReducedMotion();

  if (!run?.plan || run.status !== "awaiting_approval" || !review) return null;
  const { draft, editing, open } = review;

  const start = async () => {
    setPending(true);
    try {
      let latest = run;
      if (JSON.stringify(draft) !== JSON.stringify(run.plan)) {
        latest = await updateResearchPlan(run.id, draft, run.planRevision);
        ingestResearchUpdate(latest);
      }
      if (!latest.planHash)
        throw new Error("The research plan is missing its approval hash.");
      const approved = await approveResearchRun(
        latest.id,
        latest.planRevision,
        latest.planHash,
      );
      ingestResearchUpdate(approved);
    } catch (error) {
      toast.error("Could not start research", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setPending(false);
    }
  };

  const move = (index: number, direction: -1 | 1) => {
    const target = index + direction;
    if (target < 0 || target >= draft.steps.length) return;
    const steps = [...draft.steps];
    [steps[index], steps[target]] = [steps[target], steps[index]];
    const keys = [...stepKeys];
    [keys[index], keys[target]] = [keys[target], keys[index]];
    setStepKeys(keys);
    setDraft(runId, { ...draft, steps });
  };

  return (
    <>
      <section className="mx-4 mt-3 rounded-2xl border border-primary/20 bg-primary/[0.045] p-3">
        <p className="font-heading text-sm font-medium">Research plan ready</p>
        <p className="mt-1 line-clamp-2 break-words text-xs text-muted-foreground">
          {run.plan.title}
        </p>
        <Button
          className="mt-3 w-full"
          size="sm"
          onClick={() => setOpen(runId, true)}
        >
          Review plan
        </Button>
      </section>
      <Dialog
        open={open}
        onOpenChange={(nextOpen) => setOpen(runId, nextOpen)}
      >
        <DialogContent className="max-h-[min(680px,calc(100dvh-6rem))] grid-rows-[auto_minmax(0,1fr)_auto] gap-0 overflow-hidden p-0 sm:max-w-3xl [&>[data-slot=dialog-close]]:right-6 [&>[data-slot=dialog-close]]:top-6">
          <DialogHeader className="border-b border-border/70 px-7 pb-4 pt-6 pr-16">
            <DialogTitle>Review the research plan</DialogTitle>
            <DialogDescription className="max-w-2xl leading-relaxed">
              Research starts only after your approval. Check the scope and
              search approach before continuing.
            </DialogDescription>
          </DialogHeader>
          <div className="min-h-0 overflow-y-scroll px-7 py-5 [scrollbar-gutter:stable]">
            {editing ? (
              <div className="space-y-3">
                <Textarea
                  aria-label="Plan title"
                  value={draft.title}
                  maxLength={200}
                  className="min-h-10 py-2 font-medium"
                  onChange={(event) =>
                    setDraft(runId, { ...draft, title: event.target.value })
                  }
                />
                {draft.steps.map((step, index) => (
                  <motion.div
                    key={stepKeys[index] ?? `${stepKeyPrefix}-${index}`}
                    layout="position"
                    transition={
                      reduceMotion
                        ? { layout: { duration: 0 } }
                        : {
                            layout: {
                              duration: 0.2,
                              ease: [0.22, 1, 0.36, 1],
                            },
                          }
                    }
                    className="border-b border-border/60 py-3 first:pt-0 last:border-b-0 last:pb-0"
                  >
                    <div className="mb-2 flex items-center gap-1">
                      <span className="mr-auto text-[11px] font-medium text-muted-foreground">
                        Step {index + 1}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        onClick={() => move(index, -1)}
                        disabled={index === 0}
                        aria-label={`Move step ${index + 1} up`}
                      >
                        <ArrowUp />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        onClick={() => move(index, 1)}
                        disabled={index === draft.steps.length - 1}
                        aria-label={`Move step ${index + 1} down`}
                      >
                        <ArrowDown />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        disabled={draft.steps.length === 1}
                        onClick={() => {
                          setStepKeys((keys) => keys.filter(
                            (_, stepIndex) => stepIndex !== index,
                          ));
                          setDraft(runId, {
                            ...draft,
                            steps: draft.steps.filter(
                              (_, stepIndex) => stepIndex !== index,
                            ),
                          });
                        }}
                        aria-label={`Remove step ${index + 1}`}
                      >
                        <Trash2 />
                      </Button>
                    </div>
                    <Textarea
                      aria-label={`Step ${index + 1} title`}
                      value={step.title}
                      maxLength={200}
                      className="mb-2 min-h-9 py-2"
                      onChange={(event) => {
                        const steps = [...draft.steps];
                        steps[index] = { ...step, title: event.target.value };
                        setDraft(runId, { ...draft, steps });
                      }}
                    />
                    <Textarea
                      aria-label={`Step ${index + 1} query`}
                      value={step.query}
                      maxLength={500}
                      className="min-h-9 py-2 text-xs"
                      onChange={(event) => {
                        const steps = [...draft.steps];
                        steps[index] = { ...step, query: event.target.value };
                        setDraft(runId, { ...draft, steps });
                      }}
                    />
                  </motion.div>
                ))}
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={draft.steps.length >= 30}
                  onClick={() => {
                    setStepKeys((keys) => [
                      ...keys,
                      `${stepKeyPrefix}-${keys.length}-${Math.random().toString(36).slice(2)}`,
                    ]);
                    setDraft(runId, {
                      ...draft,
                      steps: [
                        ...draft.steps,
                        { title: "New research step", query: "" },
                      ],
                    });
                  }}
                >
                  <Plus /> Add step
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="mb-4 flex items-start justify-between gap-4">
                  <p className="break-words font-heading text-lg font-medium leading-snug text-foreground/90">
                    {draft.title}
                  </p>
                  <span className="shrink-0 rounded-full bg-muted px-2.5 py-1 text-[11px] font-medium text-muted-foreground">
                    {draft.steps.length} steps
                  </span>
                </div>
                {draft.steps.map((step, index) => (
                  <div
                    key={`${index}-${step.query}`}
                    className="flex gap-3 border-b border-border/60 py-3 first:pt-0 last:border-b-0 last:pb-0"
                  >
                    <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-medium text-primary">
                      {index + 1}
                    </span>
                    <span className="min-w-0">
                      <span className="block break-words text-sm font-medium leading-5 text-foreground/90">
                        {step.title}
                      </span>
                      <span className="mt-1 block break-words text-[13px] leading-relaxed text-muted-foreground/90">
                        {step.query}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <DialogFooter className="shrink-0 flex-col gap-3 border-t border-border/70 bg-background px-7 py-4 sm:flex-row sm:items-center sm:justify-between">
            <Button
              variant="outline"
              onClick={() => setEditing(runId, !editing)}
            >
              <Pencil /> {editing ? "Preview plan" : "Edit plan"}
            </Button>
            <div className="flex flex-col-reverse gap-2 sm:flex-row">
              <Button variant="ghost" onClick={() => setOpen(runId, false)}>
                Review later
              </Button>
              <Button
                disabled={
                  pending ||
                  !draft.title.trim() ||
                  draft.steps.some(
                    (step) => !step.title.trim() || !step.query.trim(),
                  )
                }
                onClick={() => void start()}
              >
                {pending ? <Spinner /> : <Telescope />}
                {editing ? "Save and start" : "Start research"}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function ResearchActions({ runId }: { runId: string }): ReactElement | null {
  const run = useResearchRunStore((state) => state.sessions[runId]?.run);
  const [pending, setPending] = useState(false);
  if (!run) return null;
  const canRetry = run.status === "failed" || run.status === "cancelled";
  if (!canRetry) return null;
  const retry = async () => {
    setPending(true);
    try {
      const retried = await retryResearchRun(run.id);
      ingestResearchUpdate(retried);
      useResearchRunStore.getState().setConnectionError(retried.id, null);
      ensureResearchRunFollowed(retried.id, retried);
    } catch (error) {
      toast.error("Could not retry research", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="border-t border-border/70 bg-background/95 p-3 backdrop-blur">
      <Button
        className="w-full"
        disabled={pending}
        onClick={() => void retry()}
      >
        {pending ? <Spinner /> : <RotateCcw />} Retry research
      </Button>
    </div>
  );
}

export function ResearchActivityPanel({
  runId,
  onClose,
  variant = "panel",
}: {
  runId: string;
  onClose: () => void;
  variant?: "panel" | "sheet";
}): ReactElement {
  const session = useResearchRunStore((state) => state.sessions[runId]);
  const [elapsedNow, setElapsedNow] = useState<number | null>(null);
  const { viewportRef, isAtBottom, scrollToLatest } =
    useResearchActivityScroll(runId);
  const hydrating = Boolean(
    session &&
      session.connection === "connecting" &&
      session.lastAppliedSeq < session.run.lastEventSeq,
  );

  useEffect(() => {
    ensureResearchRunFollowed(runId, session?.run);
  }, [runId, session?.following]);

  useEffect(() => {
    if (!session || terminalStatuses.has(session.run.status)) return;
    const timer = window.setInterval(() => setElapsedNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [session?.run.status]);

  if (!session) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }
  const { run, activities } = session;
  const elapsedEnd = run.completedAt ?? elapsedNow ?? run.updatedAt;
  const allowedDomains = run.config?.websitePolicy?.allowedDomains ?? [];
  const blockedDomains = run.config?.websitePolicy?.blockedDomains ?? [];
  const websiteLimitLabel = allowedDomains.length
    ? allowedDomains.length === 1
      ? `Only ${allowedDomains[0]}`
      : `${allowedDomains.length} allowed domains`
    : blockedDomains.length
      ? `${blockedDomains.length} blocked ${blockedDomains.length === 1 ? "domain" : "domains"}`
      : null;
  const websiteLimitTitle = [
    allowedDomains.length ? `Allowed: ${allowedDomains.join(", ")}` : "",
    blockedDomains.length ? `Blocked: ${blockedDomains.join(", ")}` : "",
  ]
    .filter(Boolean)
    .join("\n");

  return (
    <aside
      aria-label="Research activity"
      className="relative flex min-h-0 flex-col bg-background text-foreground"
      style={
        variant === "panel"
          ? {
              height:
                "calc(100% - var(--studio-content-top-inset, 0px) - var(--studio-chat-header-height, 48px))",
              marginTop:
                "calc(var(--studio-content-top-inset, 0px) + var(--studio-chat-header-height, 48px))",
            }
          : {
              height:
                "calc(100% - var(--studio-custom-titlebar-height, 0px))",
              marginTop: "var(--studio-custom-titlebar-height, 0px)",
            }
      }
    >
      <header className="shrink-0 border-b border-border/70 px-4 py-3.5">
        <div className="flex items-start gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-[13px] bg-primary/10 text-primary">
            <Telescope className="size-[18px]" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h2 className="font-heading text-[15px] font-medium">
                Deep research
              </h2>
              <span
                className={cn(
                  "rounded-full bg-muted px-2 py-0.5 text-[10.5px] font-medium text-muted-foreground",
                  run.status === "awaiting_approval" &&
                    "bg-amber-500/10 text-amber-700 dark:text-amber-300",
                  run.status === "failed" &&
                    "bg-destructive/10 text-destructive",
                )}
              >
                {researchStatusLabel(run.status)}
              </span>
            </div>
            <p className="mt-0.5 line-clamp-2 break-words text-xs text-muted-foreground">
              {run.plan?.title ?? "Investigating your question"}
            </p>
            {websiteLimitLabel ? (
              <p
                className="mt-1 flex items-center gap-1 text-[10.5px] font-medium text-primary/75"
                title={websiteLimitTitle}
              >
                <Globe2 className="size-3" />
                <span className="truncate">{websiteLimitLabel}</span>
              </p>
            ) : null}
            <p className="mt-1 text-[10.5px] tabular-nums text-muted-foreground">
              {formatElapsed(run.createdAt, elapsedEnd)} · {run.sources.length}{" "}
              sources ·{" "}
              {run.steps.filter((step) => step.status === "completed").length}{" "}
              actions
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onClose}
            aria-label="Close research activity"
          >
            <X />
          </Button>
        </div>
        {session.connection === "reconnecting" ? (
          <div
            role="status"
            className="mt-2 flex items-center gap-2 text-[11px] text-amber-700 dark:text-amber-300"
          >
            <Spinner className="size-3" /> Reconnecting to research activity…
          </div>
        ) : session.connection === "disconnected" &&
          !isSettledResearchRun(run, session.lastAppliedSeq) ? (
          <div
            role="status"
            className="mt-2 flex items-center justify-between gap-2 text-[11px] text-destructive"
          >
            <span>Research activity is unavailable.</span>
            <Button
              size="sm"
              variant="ghost"
              className="h-7 px-2 text-[11px]"
              onClick={() => {
                useResearchRunStore
                  .getState()
                  .setConnectionError(runId, null);
                ensureResearchRunFollowed(runId, run);
              }}
            >
              Reconnect
            </Button>
          </div>
        ) : null}
      </header>
      {/* Key on runId only: keying on planRevision remounted PlanReview mid-approve
          (updateResearchPlan bumps the revision), resetting local `pending` and
          re-enabling "Start research" during the in-flight approve. */}
      <PlanReview key={runId} runId={runId} />
      <div
        ref={viewportRef}
        role="log"
        aria-live="off"
        aria-label="Research activity timeline"
        tabIndex={0}
        className="min-h-0 flex-1 overflow-y-auto px-4 py-3 [overflow-anchor:none] focus-visible:outline-none"
      >
        {hydrating ? (
          <div className="flex items-center gap-2 py-3 text-sm text-muted-foreground">
            <Spinner /> Restoring research activity…
          </div>
        ) : activities.length ? (
          activities.map((activity) => (
            <ActivityRow key={activity.id} runId={runId} activity={activity} />
          ))
        ) : (
          <div className="flex items-center gap-2 py-3 text-sm text-muted-foreground">
            <Spinner /> Loading research activity…
          </div>
        )}
      </div>
      {isAtBottom ? null : (
        <Button
          size="sm"
          variant="outline"
          className="absolute bottom-16 left-1/2 z-10 -translate-x-1/2 bg-background"
          onClick={scrollToLatest}
        >
          <ArrowDown /> Latest
        </Button>
      )}
      <ResearchActions runId={runId} />
    </aside>
  );
}

export function ResearchActivitySheet({
  runId,
  open,
  onOpenChange,
}: {
  runId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-screen max-w-none p-0 sm:max-w-none"
        showCloseButton={false}
      >
        <SheetHeader className="sr-only">
          <SheetTitle>Deep research</SheetTitle>
          <SheetDescription>Chronological research activity</SheetDescription>
        </SheetHeader>
        <ResearchActivityPanel
          key={runId}
          runId={runId}
          onClose={() => onOpenChange(false)}
          variant="sheet"
        />
      </SheetContent>
    </Sheet>
  );
}
