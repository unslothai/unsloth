// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { useT } from "@/i18n";
import { askHide, askResize, pillServerPort } from "@/lib/pill-native";
import {
  fetchInferenceStatus,
  fetchPillSettings,
  getCachedSettings,
} from "../pill/api";
import {
  classifyFetchError,
  ensureModelLoaded,
  PillRunError,
} from "../pill/run-action";
import { streamCompletion } from "../pill/stream";

type AskPhase = "input" | "loading" | "streaming" | "done" | "error";

type Turn = { question: string; answer: string };

function shortModelName(model: string): string {
  return model.split("/").pop() ?? model;
}

function SparkIcon(): ReactElement {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="size-[18px] shrink-0 text-muted-foreground"
    >
      <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" />
      <circle cx="12" cy="12" r="3.2" />
    </svg>
  );
}

function Key({ children }: { children: ReactNode }): ReactElement {
  return (
    <kbd className="rounded-[4px] border border-border/70 bg-muted/60 px-[5px] py-px font-sans text-[10px] leading-[14px] text-muted-foreground">
      {children}
    </kbd>
  );
}

export function AskApp(): ReactElement {
  const t = useT();
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<AskPhase>("input");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [modelLabel, setModelLabel] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  const [errorKey, setErrorKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [context, setContext] = useState<string | null>(null);
  const [showNonce, setShowNonce] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const answerRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const lastSizeRef = useRef({ width: 0, height: 0 });

  useEffect(() => {
    let disposed = false;
    const cleanups: Array<() => void> = [];

    void (async () => {
      const { isTauri } = await import("@/lib/api-base");
      if (!isTauri) return;
      const { listen } = await import("@tauri-apps/api/event");
      const unlistenShow = await listen<string | null>("ask://show", (event) => {
        // Every summon is a fresh conversation with fresh settings.
        abortRef.current?.abort();
        setContext(event.payload ?? null);
        setTurns([]);
        setQuery("");
        setErrorKey(null);
        setPhase("input");
        setShowNonce((nonce) => nonce + 1);
        void fetchPillSettings()
          .then((settings) =>
            setModelLabel(
              settings.defaultModel
                ? shortModelName(settings.defaultModel)
                : null,
            ),
          )
          .catch(() => undefined);
      });
      const unlistenHide = await listen("ask://hide", () => {
        abortRef.current?.abort();
      });
      const unlistenPort = await listen<number>("server-port", (event) => {
        void import("@/lib/api-base").then(({ setApiBase }) =>
          setApiBase(event.payload),
        );
      });
      if (disposed) {
        unlistenShow();
        unlistenHide();
        unlistenPort();
        return;
      }
      cleanups.push(unlistenShow, unlistenHide, unlistenPort);

      // The server-port broadcast may predate this listener; pull the current
      // port, falling back to the value the main window persisted.
      let port = await pillServerPort().catch(() => null);
      if (port == null) {
        const stored = window.localStorage.getItem("unsloth_backend_port");
        port = stored ? Number(stored) || null : null;
      }
      if (port != null) {
        const { setApiBase } = await import("@/lib/api-base");
        setApiBase(port);
      }
      const settings = await fetchPillSettings().catch(() => null);
      if (!disposed && settings?.defaultModel) {
        setModelLabel(shortModelName(settings.defaultModel));
      }
    })();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        abortRef.current?.abort();
        void askHide();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    cleanups.push(() => window.removeEventListener("keydown", onKeyDown));

    return () => {
      disposed = true;
      for (const cleanup of cleanups) cleanup();
    };
  }, []);

  // Refocus (and preselect the previous question) on every summon.
  useEffect(() => {
    if (showNonce === 0) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [showNonce]);

  useLayoutEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const rect = node.getBoundingClientRect();
    const width = Math.ceil(rect.width);
    const height = Math.ceil(rect.height);
    // Streaming updates land per token; a native resize per token saturates
    // the main thread and freezes the app. Only resize on real change.
    if (
      width === lastSizeRef.current.width &&
      height === lastSizeRef.current.height
    ) {
      return;
    }
    lastSizeRef.current = { width, height };
    void askResize(width, height).catch(() => undefined);
  }, [phase, turns, errorKey, loadingModel]);

  // Follow the stream.
  useEffect(() => {
    const node = answerRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [turns]);

  const submit = async (): Promise<void> => {
    const question = query.trim();
    if (!question || phase === "streaming" || phase === "loading") return;
    abortRef.current?.abort();
    const abort = new AbortController();
    abortRef.current = abort;
    const history = turns;
    setTurns([...history, { question, answer: "" }]);
    setQuery("");
    setErrorKey(null);
    setPhase("streaming");

    try {
      const status = await fetchInferenceStatus().catch((error: unknown) => {
        throw new PillRunError(classifyFetchError(error));
      });
      // Fresh-first: the default model may have changed in the main window.
      const settings =
        (await fetchPillSettings().catch(() => null)) ?? getCachedSettings();
      const model = settings?.defaultModel ?? null;

      if (model) {
        await ensureModelLoaded(
          model,
          settings?.defaultGgufVariant ?? null,
          abort.signal,
          (loading) => {
            setLoadingModel(shortModelName(loading));
            setPhase("loading");
          },
        );
        setLoadingModel(null);
        setPhase("streaming");
      } else if (!status.active_model) {
        throw new PillRunError("noModel");
      }

      const used = model ?? status.active_model ?? "default";
      setModelLabel(shortModelName(used));

      const withContext = (text: string, first: boolean): string =>
        first && context ? `${text}\n\nText:\n"""\n${context}\n"""` : text;
      const messages = history.flatMap((turn, index) => [
        { role: "user" as const, content: withContext(turn.question, index === 0) },
        { role: "assistant" as const, content: turn.answer },
      ]);
      messages.push({
        role: "user",
        content: withContext(question, history.length === 0),
      });
      let sawToken = false;
      for await (const delta of streamCompletion(
        { model: used, messages, stream: true },
        abort.signal,
      )) {
        sawToken = true;
        setTurns((current) => {
          const next = current.slice();
          const last = next[next.length - 1];
          next[next.length - 1] = { ...last, answer: last.answer + delta };
          return next;
        });
      }
      if (!sawToken) throw new PillRunError("failed");
      setPhase("done");
    } catch (error) {
      if (abort.signal.aborted) {
        setPhase("done");
      } else if (error instanceof PillRunError) {
        setErrorKey(error.errorKey === "noModel" ? "noModel" : "failed");
        setPhase("error");
      } else {
        setErrorKey("failed");
        setPhase("error");
      }
      setLoadingModel(null);
    } finally {
      if (abortRef.current === abort) abortRef.current = null;
    }
  };

  const lastAnswer = turns.length > 0 ? turns[turns.length - 1].answer : "";

  const copyAnswer = (): void => {
    void navigator.clipboard.writeText(lastAnswer).then(() => setCopied(true));
  };

  const clearThread = (): void => {
    abortRef.current?.abort();
    setTurns([]);
    setContext(null);
    setErrorKey(null);
    setPhase("input");
    inputRef.current?.focus();
  };
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  // Cmd+C with nothing selected copies the finished answer.
  useEffect(() => {
    if (phase !== "done" || !lastAnswer) return;
    const onCopyKey = (event: KeyboardEvent) => {
      if (
        event.metaKey &&
        event.key === "c" &&
        !window.getSelection()?.toString()
      ) {
        copyAnswer();
      }
    };
    window.addEventListener("keydown", onCopyKey);
    return () => window.removeEventListener("keydown", onCopyKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, lastAnswer]);

  const busy = phase === "loading" || phase === "streaming";

  return (
    <div
      key={showNonce}
      ref={containerRef}
      className="ask-pop w-[640px] overflow-hidden rounded-2xl border border-border/60 bg-popover/70 text-popover-foreground shadow-2xl"
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
        className="flex items-center gap-3 px-5 py-4"
      >
        <SparkIcon />
        <input
          ref={inputRef}
          autoFocus
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={
            turns.length > 0
              ? t("systemPill.ask.followUp")
              : t("systemPill.ask.placeholder")
          }
          spellCheck={false}
          className="w-full bg-transparent text-[17px] text-foreground outline-none placeholder:text-muted-foreground/80"
        />
        {busy && (
          <span className="size-4 shrink-0 animate-spin rounded-full border-2 border-muted-foreground/70 border-t-transparent" />
        )}
      </form>

      {context && (
        <div className="flex items-center gap-2 px-5 pb-3 -mt-1">
          <span className="flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/50 py-0.5 pl-2.5 pr-1 text-[11px] text-muted-foreground">
            {t("systemPill.ask.context", { chars: String(context.length) })}
            <button
              type="button"
              onClick={() => setContext(null)}
              className="rounded-full px-1 hover:bg-accent hover:text-accent-foreground"
            >
              ×
            </button>
          </span>
        </div>
      )}

      {(turns.length > 0 || phase === "error") && (
        <div
          ref={answerRef}
          className="max-h-80 overflow-y-auto border-t border-border/50 px-5 py-3.5 text-[13.5px] leading-6"
        >
          {turns.map((turn, index) => (
            <div key={index} className={index > 0 ? "mt-3" : undefined}>
              {(index > 0 || turns.length > 1) && (
                <div className="mb-1 text-[11.5px] font-medium text-muted-foreground">
                  {turn.question}
                </div>
              )}
              <div className="whitespace-pre-wrap">
                {turn.answer}
                {phase === "streaming" && index === turns.length - 1 && (
                  <span className="ask-caret" />
                )}
              </div>
            </div>
          ))}
          {phase === "error" && (
            <span className="text-muted-foreground">
              {t(
                errorKey === "noModel"
                  ? "systemPill.ask.noModel"
                  : "systemPill.ask.failed",
              )}
            </span>
          )}
        </div>
      )}

      <div className="flex h-9 items-center justify-between border-t border-border/50 bg-muted/30 px-4">
        <span className="flex min-w-0 items-center gap-1.5 text-[11px] text-muted-foreground">
          {phase === "loading" && loadingModel ? (
            t("systemPill.ask.loading", { model: loadingModel })
          ) : (
            <>
              <span className="size-[6px] shrink-0 rounded-full bg-emerald-500/80" />
              <span className="truncate">
                {modelLabel ?? t("systemPill.ask.autoModel")}
              </span>
            </>
          )}
        </span>
        <span className="flex shrink-0 items-center gap-3 text-[11px] text-muted-foreground">
          {turns.length > 0 && (
            <button
              type="button"
              onClick={clearThread}
              className="rounded-md px-1.5 py-0.5 hover:bg-accent hover:text-accent-foreground"
            >
              {t("systemPill.ask.clear")}
            </button>
          )}
          {phase === "done" && lastAnswer && (
            <button
              type="button"
              onClick={copyAnswer}
              className="rounded-md px-1.5 py-0.5 hover:bg-accent hover:text-accent-foreground"
            >
              {copied ? t("systemPill.ask.copied") : t("systemPill.ask.copy")}
            </button>
          )}
          <span className="flex items-center gap-1">
            <Key>⏎</Key> {t("systemPill.ask.enterHint")}
          </span>
          <span className="flex items-center gap-1">
            <Key>esc</Key> {t("systemPill.ask.escHint")}
          </span>
        </span>
      </div>
    </div>
  );
}
