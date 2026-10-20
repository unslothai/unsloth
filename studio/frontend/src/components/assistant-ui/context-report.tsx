// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Progress } from "@/components/ui/progress";
import {
  type ContextView,
  countContext,
  deriveContextView,
  externalModelContextLength,
  formatTokenCountFull,
  isExternalModelId,
  shownContextReport,
  usagePercent,
  useChatRuntimeStore,
  useContextReportStore,
} from "@/features/chat";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { useAui, useAuiState } from "@assistant-ui/react";
import { ChevronRightIcon, XIcon } from "lucide-react";
import { type FC, useEffect, useMemo, useRef, useState } from "react";

// No message on an error: the loaded model changed while counting.
type Count =
  | { state: "loading" }
  | { state: "done"; view: ContextView }
  | { state: "error"; message?: string };

/** Mounted once per Thread, under the last message. The report goes once a turn lands after it. */
export const ContextReport: FC = () => {
  const threadKey = useAuiState(({ threadListItem }) => threadListItem.id);
  const lastMessageId = useAuiState(
    ({ thread }) => thread.messages.at(-1)?.id ?? null,
  );
  const report = useContextReportStore((s) =>
    shownContextReport(s.report, threadKey, lastMessageId),
  );
  const dismissContext = useContextReportStore((s) => s.dismissContext);
  return report ? (
    <ContextCard key={report.nonce} onDismiss={dismissContext} />
  ) : null;
};

const ContextCard: FC<{ onDismiss: () => void }> = ({ onDismiss }) => {
  const t = useT();
  // Read once: the report describes the model it was asked under.
  const [checkpoint] = useState(
    () => useChatRuntimeStore.getState().params.checkpoint,
  );
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ block: "nearest" });
  }, []);
  return (
    <section
      ref={ref}
      aria-label={t("context.title")}
      className="mx-auto mb-4 grid w-full max-w-(--thread-content-max-width) gap-3 text-sm"
    >
      {isExternalModelId(checkpoint) ? (
        <ApiTotal checkpoint={checkpoint} onDismiss={onDismiss} />
      ) : (
        <LocalCount onDismiss={onDismiss} />
      )}
    </section>
  );
};

const Header: FC<{ summary: string; onDismiss: () => void }> = ({
  summary,
  onDismiss,
}) => {
  const t = useT();
  return (
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
  );
};

const UsageBar: FC<{ percent: number; markAt?: number }> = ({
  percent,
  markAt,
}) => (
  <div className="relative" aria-hidden={true}>
    <Progress
      value={percent}
      className="h-1"
      indicatorClassName="bg-foreground/45"
    />
    {markAt != null ? (
      <div
        className="absolute -inset-y-0.5 w-px bg-foreground/60"
        style={{ left: `${markAt * 100}%` }}
      />
    ) : null}
  </div>
);

/** A local model: the backend splits the prompt it would send. */
const LocalCount: FC<{ onDismiss: () => void }> = ({ onDismiss }) => {
  const t = useT();
  const aui = useAui();
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  // Read once: the report describes the branch it was asked under.
  const branch = useMemo(() => aui.thread().getState().messages, [aui]);
  const [count, setCount] = useState<Count>({ state: "loading" });

  useEffect(() => {
    let cancelled = false;
    countContext(branch, remoteId ?? undefined).then(
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
  }, [branch, remoteId]);

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
    <>
      <Header summary={summary} onDismiss={onDismiss} />
      {count.state === "done" ? <Breakdown view={count.view} /> : null}
    </>
  );
};

/** An API model: no local tokenizer, so the last reply's total against the catalogued window. */
const ApiTotal: FC<{ checkpoint: string; onDismiss: () => void }> = ({
  checkpoint,
  onDismiss,
}) => {
  const t = useT();
  const [{ used, window }] = useState(() => {
    const usage = useChatRuntimeStore.getState().contextUsage;
    return {
      used: usage?.contextTokens ?? usage?.totalTokens ?? null,
      window: externalModelContextLength(checkpoint),
    };
  });
  const percent = used != null && window ? usagePercent(used, window) : null;
  return (
    <>
      <Header
        summary={
          used == null
            ? t("context.noUsageYet")
            : window && percent != null
              ? t("context.summary", {
                  used: formatTokenCountFull(used),
                  window: formatTokenCountFull(window),
                  percent,
                })
              : t("context.usedOnly", { used: formatTokenCountFull(used) })
        }
        onDismiss={onDismiss}
      />
      {percent != null ? <UsageBar percent={percent} /> : null}
      <p className="text-muted-foreground text-xs">
        {t("context.apiNoBreakdown")}
      </p>
    </>
  );
};

const Breakdown: FC<{ view: ContextView }> = ({ view }) => {
  const t = useT();
  const label = {
    system: t("context.system"),
    tools: t("context.tools", { count: view.toolNames.length }),
    messages: t("context.messages"),
  };
  return (
    <>
      <UsageBar percent={view.percent} markAt={view.compactAt?.share} />
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

/** A `detail` (the tool names) stays folded under its label until clicked. */
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

/** The exact text the count priced. */
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
            {t("context.copyPrompt")}
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
