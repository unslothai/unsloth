// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { AlertDialog } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Spinner } from "@/components/ui/spinner";
import { getAuthSessionEpoch, useFullAccessAllowed } from "@/features/auth";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  Cancel01Icon,
  Robot01Icon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { ToolApprovalGoneError } from "../api/chat-api";
import { cancelSubagent, resolveSubagentApproval } from "../api/subagents-api";
import {
  useSubagentClock,
  useSubagentDetailPolling,
} from "../hooks/use-subagents";
import { FullAccessConfirmContent } from "../permission-mode-select";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useSubagentsStore } from "../stores/subagents-store";
import type {
  SubagentRecord,
  SubagentStatus,
  SubagentTranscriptItem,
} from "../types/subagents";
import { withChatFullAccess } from "../utils/chat-full-access";
import {
  anySubagentRunning,
  formatSubagentDuration,
  isSubagentRunning,
  shortSubagentModel,
  subagentElapsedMs,
  summarizeSubagentArguments,
} from "../utils/subagents";

const NO_RECORDS: readonly SubagentRecord[] = [];

function StatusGlyph({
  status,
  className,
}: {
  status: SubagentStatus;
  className?: string;
}): ReactElement {
  const t = useT();
  switch (status) {
    case "running":
      return (
        <Spinner
          label={t("subagents.status.running")}
          className={cn("size-4", className)}
        />
      );
    case "completed":
      return (
        <HugeiconsIcon
          icon={Tick02Icon}
          strokeWidth={2}
          className={cn("size-4 shrink-0 text-primary", className)}
        />
      );
    case "failed":
      return (
        <HugeiconsIcon
          icon={Cancel01Icon}
          strokeWidth={2}
          className={cn("size-4 shrink-0 text-destructive", className)}
        />
      );
    default:
      return (
        <HugeiconsIcon
          icon={StopIcon}
          strokeWidth={2}
          className={cn("size-4 shrink-0 text-muted-foreground", className)}
        />
      );
  }
}

function statusLabelKey(status: SubagentStatus) {
  return `subagents.status.${status}` as const;
}

function toolCountLabel(t: ReturnType<typeof useT>, count: number): string {
  return count === 1
    ? t("subagents.toolCallsOne")
    : t("subagents.toolCalls", { count });
}

/**
 * Allow / Deny for one pending tool call, answered through the shared tool-confirm endpoint. Like
 * the main chat's approval card it also offers Full Access when the account may use it and the
 * chat is not already on it; a grant covers the whole running chat, subagents included.
 */
function ApprovalButtons({
  sessionId,
  approvalId,
  threadId,
  subagentId,
  onResolved,
}: {
  sessionId: string | null;
  approvalId: string;
  /** The chat that spawned the subagent: the one a Full Access grant belongs to. */
  threadId: string | null;
  subagentId: string;
  onResolved?: () => void;
}): ReactElement | null {
  const t = useT();
  const fullAccessAllowed = useFullAccessAllowed();
  const permissionMode = useChatRuntimeStore(
    (state) =>
      withChatFullAccess(state, state.chatFullAccessGrants, threadId)
        .permissionMode,
  );
  const grantChatFullAccess = useChatRuntimeStore(
    (state) => state.grantChatFullAccess,
  );
  const [pending, setPending] = useState<"allow" | "deny" | "full" | null>(
    null,
  );
  const [fullAccessOpen, setFullAccessOpen] = useState(false);
  if (!sessionId) return null;
  const answer = async (decision: "allow" | "deny", fullAccess = false) => {
    const epoch = getAuthSessionEpoch();
    const choiceVersion =
      useChatRuntimeStore.getState().chatPermissionChoiceVersion;
    setPending(fullAccess ? "full" : decision);
    try {
      const ok = await resolveSubagentApproval(
        sessionId,
        approvalId,
        decision,
        fullAccess,
      );
      if (!ok || getAuthSessionEpoch() !== epoch) {
        toast.error(t("subagents.approvalFailed"));
        return;
      }
      if (
        fullAccess &&
        threadId &&
        useChatRuntimeStore.getState().chatPermissionChoiceVersion ===
          choiceVersion
      ) {
        // The run id only keys the grant; the thread is what the chat's permission overlay reads.
        grantChatFullAccess({
          runId: `subagent:${subagentId}`,
          sessionId,
          threadId,
        });
      }
      onResolved?.();
    } catch (error) {
      if (error instanceof ToolApprovalGoneError) {
        toast.info(t("subagents.approvalGone"));
        onResolved?.();
      } else {
        toast.error(t("subagents.approvalFailed"), {
          description: error instanceof Error ? error.message : undefined,
        });
      }
    } finally {
      setPending(null);
    }
  };
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Button
        size="sm"
        className="h-7 px-2.5 text-ui-11"
        disabled={pending !== null}
        onClick={() => void answer("allow")}
      >
        {pending === "allow" ? <Spinner className="size-3" /> : null}
        {t("subagents.allow")}
      </Button>
      <Button
        size="sm"
        variant="outline"
        className="h-7 px-2.5 text-ui-11"
        disabled={pending !== null}
        onClick={() => void answer("deny")}
      >
        {t("subagents.deny")}
      </Button>
      {fullAccessAllowed && permissionMode !== "full" && threadId ? (
        <>
          <Button
            size="sm"
            variant="destructive"
            className="h-7 px-2.5 text-ui-11"
            disabled={pending !== null}
            onClick={() => setFullAccessOpen(true)}
          >
            {pending === "full" ? <Spinner className="size-3" /> : null}
            {t("subagents.fullAccess")}
          </Button>
          <AlertDialog open={fullAccessOpen} onOpenChange={setFullAccessOpen}>
            <FullAccessConfirmContent
              onConfirm={() => {
                setFullAccessOpen(false);
                void answer("allow", true);
              }}
              onClose={() => setFullAccessOpen(false)}
            />
          </AlertDialog>
        </>
      ) : null}
    </div>
  );
}

function StopButton({
  subagentId,
  onStopped,
}: {
  subagentId: string;
  onStopped?: () => void;
}): ReactElement {
  const t = useT();
  const [pending, setPending] = useState(false);
  return (
    <Button
      size="sm"
      variant="outline"
      className="h-7 shrink-0 gap-1 px-2.5 text-ui-11"
      disabled={pending}
      onClick={async () => {
        setPending(true);
        try {
          await cancelSubagent(subagentId);
          onStopped?.();
        } catch (error) {
          toast.error(t("subagents.stopFailed"), {
            description: error instanceof Error ? error.message : undefined,
          });
        } finally {
          setPending(false);
        }
      }}
    >
      {pending ? (
        <Spinner className="size-3" />
      ) : (
        <HugeiconsIcon icon={StopIcon} className="size-3" strokeWidth={2} />
      )}
      {t("subagents.stop")}
    </Button>
  );
}

function PendingApprovalNotice({
  record,
}: {
  record: SubagentRecord;
}): ReactElement | null {
  const t = useT();
  const approval = record.pendingApproval;
  if (!approval) return null;
  const summary = summarizeSubagentArguments(approval.arguments, 160);
  return (
    <div className="flex flex-col gap-2 rounded-md bg-amber-500/10 px-2.5 py-2 text-amber-800 dark:text-amber-200">
      <div className="min-w-0">
        <p className="text-ui-11 font-medium">
          {t("subagents.approvalTitle", { tool: approval.toolName })}
        </p>
        {summary ? (
          <p className="mt-0.5 line-clamp-2 break-all font-mono text-ui-10p5 opacity-80">
            {summary}
          </p>
        ) : null}
      </div>
      <ApprovalButtons
        sessionId={record.sessionId}
        approvalId={approval.approvalId}
        threadId={record.parentThreadId}
        subagentId={record.id}
      />
    </div>
  );
}

function SubagentRow({
  record,
  now,
  onOpen,
}: {
  record: SubagentRecord;
  now: number;
  onOpen: () => void;
}): ReactElement {
  const t = useT();
  const running = isSubagentRunning(record);
  const model = shortSubagentModel(record.model);
  const line =
    record.status === "failed" || record.status === "cancelled"
      ? (record.error ?? record.resultPreview)
      : record.resultPreview;
  return (
    <li
      className={cn(
        "rounded-lg border border-border/70 bg-background transition-colors hover:bg-muted/40",
        record.pendingApproval && "border-amber-500/40",
      )}
    >
      <div className="flex items-start gap-1 pr-2">
        <button
          type="button"
          onClick={onOpen}
          className="flex min-w-0 flex-1 items-start gap-2.5 rounded-lg px-3 py-2.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <span className="mt-0.5 flex size-4 shrink-0 items-center justify-center">
            <StatusGlyph status={record.status} />
          </span>
          <span className="min-w-0 flex-1">
            <span className="flex items-baseline gap-2">
              <span className="min-w-0 flex-1 truncate text-ui-13p5 font-medium text-foreground/90">
                {record.description}
              </span>
              <span className="shrink-0 text-ui-10p5 tabular-nums text-muted-foreground">
                {formatSubagentDuration(subagentElapsedMs(record, now))}
              </span>
            </span>
            <span className="mt-0.5 flex flex-wrap items-center gap-x-2 text-ui-11 text-muted-foreground">
              {model ? (
                <span
                  className="max-w-[calc(160px*var(--ui-space-scale,1))] truncate"
                  title={record.model ?? undefined}
                >
                  {model}
                </span>
              ) : null}
              <span>{toolCountLabel(t, record.toolCallCount)}</span>
            </span>
            {line ? (
              <span
                className={cn(
                  "mt-1 line-clamp-1 block break-words text-ui-11p5 leading-snug text-muted-foreground",
                  record.status === "failed" && "text-destructive",
                )}
              >
                {line}
              </span>
            ) : null}
          </span>
        </button>
        {running ? (
          <div className="pt-2">
            <StopButton subagentId={record.id} />
          </div>
        ) : null}
      </div>
      {record.pendingApproval ? (
        <div className="px-3 pb-2.5">
          <PendingApprovalNotice record={record} />
        </div>
      ) : null}
    </li>
  );
}

function ToolItem({
  item,
  record,
}: {
  item: Extract<SubagentTranscriptItem, { kind: "tool" }>;
  record: SubagentRecord;
}): ReactElement {
  const t = useT();
  const [open, setOpen] = useState(false);
  const summary = summarizeSubagentArguments(item.arguments);
  const awaiting = item.status === "awaiting_approval";
  const statusText =
    item.status === "running"
      ? t("subagents.toolStatus.running")
      : awaiting
        ? t("subagents.toolStatus.awaitingApproval")
        : t("subagents.toolStatus.done");
  const hasResult = Boolean(item.result);
  return (
    <div
      className={cn(
        "rounded-lg border border-border/70 bg-muted/30",
        awaiting && "border-amber-500/40",
      )}
    >
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger
          disabled={!hasResult}
          className="group/tool flex w-full items-center gap-2 px-2.5 py-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default"
        >
          {item.status === "running" ? (
            <Spinner className="size-3.5" />
          ) : (
            <HugeiconsIcon
              icon={awaiting ? StopIcon : Tick02Icon}
              strokeWidth={2}
              className={cn(
                "size-3.5 shrink-0",
                awaiting
                  ? "text-amber-600 dark:text-amber-300"
                  : "text-muted-foreground",
              )}
            />
          )}
          <span className="min-w-0 flex-1">
            <span className="block truncate font-mono text-ui-11p5 font-medium text-foreground/90">
              {item.toolName}
            </span>
            {summary ? (
              <span className="block truncate text-ui-11 text-muted-foreground">
                {summary}
              </span>
            ) : null}
          </span>
          <span className="shrink-0 text-ui-10p5 text-muted-foreground">
            {statusText}
          </span>
          {hasResult ? (
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              className="size-3.5 shrink-0 text-muted-foreground transition-transform group-data-[state=open]/tool:rotate-180"
            />
          ) : null}
        </CollapsibleTrigger>
        {hasResult ? (
          <CollapsibleContent>
            <pre className="mx-2.5 mb-2.5 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md bg-background/70 p-2 text-ui-11 leading-relaxed text-foreground/85">
              {item.result}
            </pre>
          </CollapsibleContent>
        ) : null}
      </Collapsible>
      {awaiting && item.approvalId ? (
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-amber-500/30 px-2.5 py-2">
          <span className="text-ui-11 text-amber-800 dark:text-amber-200">
            {t("subagents.approvalTitle", { tool: item.toolName })}
          </span>
          <ApprovalButtons
            sessionId={record.sessionId}
            approvalId={item.approvalId}
            threadId={record.parentThreadId}
            subagentId={record.id}
          />
        </div>
      ) : null}
    </div>
  );
}

function ReasoningItem({ text }: { text: string }): ReactElement {
  const t = useT();
  return (
    <Collapsible>
      <CollapsibleTrigger className="group/reasoning flex items-center gap-1.5 text-ui-11p5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <span>{t("subagents.thinking")}</span>
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className="size-3 transition-transform group-data-[state=open]/reasoning:rotate-180"
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <p className="mt-1 whitespace-pre-wrap break-words border-l-2 border-border pl-3 text-ui-12 leading-relaxed text-muted-foreground">
          {text}
        </p>
      </CollapsibleContent>
    </Collapsible>
  );
}

function TranscriptItemView({
  item,
  record,
}: {
  item: SubagentTranscriptItem;
  record: SubagentRecord;
}): ReactElement | null {
  switch (item.kind) {
    case "text":
      return item.text.trim() ? (
        <MarkdownPreview
          markdown={item.text}
          className="max-h-none overflow-visible border-0 bg-transparent p-0 text-ui-13p5"
          defer={true}
        />
      ) : null;
    case "reasoning":
      return item.text.trim() ? <ReasoningItem text={item.text} /> : null;
    case "tool":
      return <ToolItem item={item} record={record} />;
    case "status":
      return (
        <p className="text-ui-11 italic text-muted-foreground">{item.text}</p>
      );
    case "error":
      return (
        <div
          role="alert"
          className="whitespace-pre-wrap break-words rounded-md bg-destructive/10 px-3 py-2 text-ui-12 text-destructive"
        >
          {item.text}
        </div>
      );
    default:
      return null;
  }
}

function SubagentDetail({
  summary,
  now,
  onBack,
}: {
  summary: SubagentRecord;
  now: number;
  onBack: () => void;
}): ReactElement {
  const t = useT();
  const navigate = useNavigate();
  const running = isSubagentRunning(summary);
  const detail = useSubagentDetailPolling(summary.id, running);
  // The list poll can see a run end before the detail poll does; never show the older "running".
  const record: SubagentRecord =
    detail && !(isSubagentRunning(detail) && !running)
      ? detail
      : { ...summary, transcript: detail?.transcript };
  const transcript = detail?.transcript;
  const model = shortSubagentModel(record.model);

  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const stickRef = useRef(true);
  const onScroll = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }, []);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the transcript is the trigger
  useLayoutEffect(() => {
    const el = scrollerRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [transcript]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="shrink-0 border-b border-border/70 px-3 py-2.5">
        <div className="flex items-start gap-2">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onBack}
            aria-label={t("subagents.back")}
          >
            <HugeiconsIcon icon={ArrowLeft01Icon} />
          </Button>
          <div className="min-w-0 flex-1 pt-0.5">
            <div className="flex items-center gap-2">
              <StatusGlyph status={record.status} />
              <h3 className="min-w-0 flex-1 truncate text-ui-13p5 font-medium">
                {record.description}
              </h3>
            </div>
            <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-ui-11 text-muted-foreground">
              <span>{t(statusLabelKey(record.status))}</span>
              {model ? (
                <span
                  className="max-w-[calc(192px*var(--ui-space-scale,1))] truncate"
                  title={record.model ?? undefined}
                >
                  {model}
                </span>
              ) : null}
              <span>{toolCountLabel(t, record.toolCallCount)}</span>
              <span className="tabular-nums">
                {formatSubagentDuration(subagentElapsedMs(record, now))}
              </span>
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1.5 pt-0.5">
            {running ? <StopButton subagentId={record.id} /> : null}
            {!running && record.childThreadId ? (
              <Button
                size="sm"
                variant="outline"
                className="h-7 px-2.5 text-ui-11"
                onClick={() =>
                  void navigate({
                    to: "/chat",
                    search: { thread: record.childThreadId },
                  })
                }
              >
                {t("subagents.openChat")}
              </Button>
            ) : null}
          </div>
        </div>
      </div>
      <div
        ref={scrollerRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3 [overflow-anchor:none]"
      >
        {record.pendingApproval ? (
          <PendingApprovalNotice record={record} />
        ) : null}
        {record.prompt ? (
          <Collapsible>
            <CollapsibleTrigger className="group/prompt flex w-full items-center gap-1.5 text-ui-11p5 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              <span>{t("subagents.prompt")}</span>
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                className="size-3 transition-transform group-data-[state=open]/prompt:rotate-180"
              />
            </CollapsibleTrigger>
            <CollapsibleContent>
              <p className="mt-1.5 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/45 px-3 py-2 text-ui-12 leading-relaxed text-foreground/85">
                {record.prompt}
              </p>
            </CollapsibleContent>
          </Collapsible>
        ) : null}
        {transcript === undefined ? (
          <div className="flex items-center gap-2 py-2 text-sm text-muted-foreground">
            <Spinner /> {t("subagents.loadingTranscript")}
          </div>
        ) : transcript.length === 0 ? (
          <p className="py-2 text-sm text-muted-foreground">
            {running ? t("subagents.working") : t("subagents.emptyTranscript")}
          </p>
        ) : (
          transcript.map((item, index) => (
            <TranscriptItemView
              // The transcript only grows, so a position is a stable identity.
              key={
                item.kind === "tool"
                  ? `tool-${item.toolCallId}`
                  : `${item.kind}-${index}`
              }
              item={item}
              record={record}
            />
          ))
        )}
        {record.status === "failed" &&
        record.error &&
        !transcript?.some((item) => item.kind === "error") ? (
          <div
            role="alert"
            className="whitespace-pre-wrap break-words rounded-md bg-destructive/10 px-3 py-2 text-ui-12 text-destructive"
          >
            {record.error}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function SubagentsPanel({
  threadId,
  onClose,
  variant = "panel",
}: {
  threadId: string;
  onClose: () => void;
  variant?: "panel" | "sheet";
}): ReactElement {
  const t = useT();
  const records =
    useSubagentsStore((state) => state.byThread[threadId]) ?? NO_RECORDS;
  const selectedId = useSubagentsStore((state) => state.selectedId);
  const select = useSubagentsStore((state) => state.select);
  const anyRunning = anySubagentRunning(records);
  const now = useSubagentClock(anyRunning);
  const selected = selectedId
    ? records.find((record) => record.id === selectedId)
    : undefined;
  const runningCount = records.filter(isSubagentRunning).length;

  // A selection from another chat, or a row the server no longer lists, falls back to the list.
  useEffect(() => {
    if (selectedId && !selected && records.length > 0) select(null);
  }, [selectedId, selected, records.length, select]);

  return (
    <aside
      aria-label={t("subagents.title")}
      className={cn(
        "relative flex min-h-0 flex-col bg-background text-foreground",
        variant === "sheet" && "h-full",
        variant === "panel" && "side-pane-lock-cover",
      )}
      style={
        variant === "panel"
          ? {
              // Keep the panel's controls below the absolute chat header.
              height:
                "calc(100% - var(--studio-content-top-inset, 0px) - var(--studio-chat-header-height, 48px))",
              marginTop:
                "calc(var(--studio-content-top-inset, 0px) + var(--studio-chat-header-height, 48px))",
            }
          : undefined
      }
    >
      <header className="shrink-0 border-b border-border/70 px-4 py-3.5">
        <div className="flex items-start gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
            <HugeiconsIcon
              icon={Robot01Icon}
              className="size-[calc(18px*var(--ui-space-scale,1))]"
            />
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="font-heading text-ui-15 font-medium">
              {t("subagents.title")}
            </h2>
            <p className="mt-0.5 text-ui-11 tabular-nums text-muted-foreground">
              {runningCount > 0
                ? t("subagents.summaryRunning", {
                    running: runningCount,
                    total: records.length,
                  })
                : t("subagents.summaryTotal", { total: records.length })}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onClose}
            aria-label={t("subagents.close")}
          >
            <HugeiconsIcon icon={Cancel01Icon} />
          </Button>
        </div>
      </header>
      {selected ? (
        <SubagentDetail
          key={selected.id}
          summary={selected}
          now={now}
          onBack={() => select(null)}
        />
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
          {records.length === 0 ? (
            <div className="px-2 py-6 text-center">
              <p className="text-sm font-medium text-foreground/85">
                {t("subagents.empty")}
              </p>
              <p className="mt-1 text-ui-12 text-muted-foreground">
                {t("subagents.emptyDescription")}
              </p>
            </div>
          ) : (
            <ul
              className="flex flex-col gap-2"
              aria-label={t("subagents.title")}
            >
              {records.map((record) => (
                <SubagentRow
                  key={record.id}
                  record={record}
                  now={now}
                  onOpen={() => select(record.id)}
                />
              ))}
            </ul>
          )}
        </div>
      )}
    </aside>
  );
}

export function SubagentsSheet({
  threadId,
  open,
  onOpenChange,
}: {
  threadId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  const t = useT();
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-screen max-w-none p-0 sm:max-w-none"
        showCloseButton={false}
      >
        <SheetHeader className="sr-only">
          <SheetTitle>{t("subagents.title")}</SheetTitle>
          <SheetDescription>{t("subagents.description")}</SheetDescription>
        </SheetHeader>
        <SubagentsPanel
          threadId={threadId}
          onClose={() => onOpenChange(false)}
          variant="sheet"
        />
      </SheetContent>
    </Sheet>
  );
}
