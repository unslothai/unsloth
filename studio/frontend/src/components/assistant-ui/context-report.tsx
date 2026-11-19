// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Progress } from "@/components/ui/progress";
import {
  type ContextView,
  countContext,
  contextReportBranch,
  deriveContextView,
  formatTokenCountFull,
  getSeverityColor,
  shownContextReport,
  useChatRuntimeStore,
  useContextReportStore,
} from "@/features/chat";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { useAui, useAuiState } from "@assistant-ui/react";
import { ChevronRightIcon, XIcon } from "lucide-react";
import { type FC, useEffect, useMemo, useRef, useState } from "react";

// An error without a message: the model or branch changed while counting.
type Count =
  | { state: "loading" }
  | { state: "done"; view: ContextView }
  | { state: "error"; message?: string };

export const ContextReport: FC = () => {
  const threadKey = useAuiState(({ threadListItem }) => threadListItem.id);
  const messages = useAuiState(({ thread }) => thread.messages);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const request = useContextReportStore((s) => s.report);
  const branch = useMemo(() => contextReportBranch(messages), [messages]);
  const report = shownContextReport(request, threadKey, branch, checkpoint);
  const dismissContext = useContextReportStore((s) => s.dismissContext);
  useEffect(() => {
    if (
      request?.threadKey === threadKey &&
      !report &&
      useContextReportStore.getState().report === request
    ) {
      dismissContext();
    }
  }, [request, threadKey, report, dismissContext]);
  return report ? (
    <ContextCard key={report.nonce} onDismiss={dismissContext} />
  ) : null;
};

const ContextCard: FC<{ onDismiss: () => void }> = ({ onDismiss }) => {
  const t = useT();
  const aui = useAui();
  const ref = useRef<HTMLElement>(null);
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  // The report describes the branch it was asked under.
  const branch = useMemo(() => aui.thread().getState().messages, [aui]);
  const [count, setCount] = useState<Count>({ state: "loading" });

  useEffect(() => {
    ref.current?.scrollIntoView({ block: "nearest" });
  }, []);
  useEffect(() => {
    let cancelled = false;
    countContext(
      branch,
      remoteId ?? undefined,
      () => aui.thread().getState().messages,
    ).then(
      (counted) => {
        if (cancelled) return;
        const view =
          counted &&
          deriveContextView(
            counted,
            useChatRuntimeStore.getState().loadedContextLength,
          );
        setCount(view ? { state: "done", view } : { state: "error" });
      },
      (error: unknown) => {
        if (cancelled) return;
        setCount({
          state: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [aui, branch, remoteId]);

  const summary =
    count.state === "done"
      ? t("context.summary", {
          used: formatTokenCountFull(count.view.used),
          window: formatTokenCountFull(count.view.window),
          percent: count.view.percent,
        })
      : count.state === "loading"
        ? t("context.loading")
        : count.message
          ? t("context.failed", { error: count.message })
          : t("context.modelChanged");
  return (
    <section
      ref={ref}
      aria-label={t("context.title")}
      className="mx-auto mb-4 grid w-full max-w-(--thread-content-max-width) gap-3 rounded-[16px] border border-border/70 bg-muted/20 px-4 py-3 text-sm"
    >
      <header className="flex items-center justify-between gap-3">
        <p aria-live="polite">
          <span className="font-medium">{t("context.title")}</span>
          <span className="text-muted-foreground">
            {" · "}
            {summary}
          </span>
        </p>
        <button
          type="button"
          onClick={onDismiss}
          aria-label={t("common.close")}
          className="rounded-full p-1 text-muted-foreground outline-none hover:bg-accent hover:text-foreground focus-visible:bg-accent"
        >
          <XIcon className="size-3.5" />
        </button>
      </header>
      {count.state === "done" ? <Breakdown view={count.view} /> : null}
    </section>
  );
};

const Breakdown: FC<{ view: ContextView }> = ({ view }) => {
  const t = useT();
  const label = {
    system: t("context.system"),
    tools: t("context.tools", { count: view.toolNames.length }),
    messages: t("context.messages"),
  };
  // Unrounded, as the header ring reads it, so the two change color together.
  const percent = Math.min(100, (view.used / view.window) * 100);
  return (
    <>
      <div className="relative" aria-hidden={true}>
        <Progress
          value={percent}
          className="h-1"
          indicatorClassName={getSeverityColor(percent).bar}
        />
        {view.compactAt ? (
          <div
            className="absolute -inset-y-0.5 w-px bg-[color-mix(in_oklab,var(--foreground)_calc(60%*var(--contrast-wash-gain,1)),transparent)]"
            style={{ left: `${view.compactAt.share * 100}%` }}
          />
        ) : null}
      </div>
      <dl className="grid grid-cols-[1fr_auto] gap-x-6 gap-y-1">
        {view.segments.map(({ key, tokens }) => (
          <Row
            key={key}
            label={label[key]}
            tokens={tokens}
            detail={key === "tools" ? view.toolNames.join(", ") : undefined}
          />
        ))}
        <Row label={t("context.free")} tokens={view.free} />
      </dl>
      {view.compactAt ? (
        <p className="text-muted-foreground text-xs">
          {t("context.compactsAt", {
            tokens: formatTokenCountFull(view.compactAt.tokens),
          })}
        </p>
      ) : null}
      {view.prompt ? <PromptToggle prompt={view.prompt} /> : null}
    </>
  );
};

const Row: FC<{ label: string; tokens: number; detail?: string }> = ({
  label,
  tokens,
  detail,
}) => {
  const [open, setOpen] = useState(false);
  return (
    <>
      <dt className="min-w-0">
        {detail ? (
          <button
            type="button"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            className="inline-flex items-center gap-1 outline-none focus-visible:underline"
          >
            {label}
            <ChevronRightIcon
              className={cn(
                "size-3 text-muted-foreground transition-transform",
                open && "rotate-90",
              )}
            />
          </button>
        ) : (
          label
        )}
        {open ? (
          <span className="block text-muted-foreground text-xs">{detail}</span>
        ) : null}
      </dt>
      <dd className="text-right font-mono tabular-nums">
        {formatTokenCountFull(tokens)}
      </dd>
    </>
  );
};

const TEXT_BUTTON =
  "text-muted-foreground outline-none hover:text-foreground focus-visible:text-foreground";

const PromptToggle: FC<{ prompt: string }> = ({ prompt }) => {
  const t = useT();
  const [open, setOpen] = useState(false);
  return (
    <div className="grid gap-2 text-xs">
      <div className="flex gap-3">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className={TEXT_BUTTON}
        >
          {open ? t("context.hidePrompt") : t("context.showPrompt")}
        </button>
        {open ? (
          <button
            type="button"
            onClick={() => void copyToClipboard(prompt)}
            className={TEXT_BUTTON}
          >
            {t("chatMenu.copy")}
          </button>
        ) : null}
      </div>
      {open ? (
        <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-[12px] bg-muted px-3 py-2 font-mono">
          {prompt}
        </pre>
      ) : null}
    </div>
  );
};
