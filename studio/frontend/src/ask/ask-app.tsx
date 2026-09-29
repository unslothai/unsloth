// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  adoptBackendPort,
  AskError,
  resolveModel,
  streamAnswer,
  type ChatMessage,
} from "./chat";

type Phase = "input" | "loading" | "streaming" | "done" | "error";
// `complete`: the stream reached a clean stop, so the turn can go into follow-up history.
type Turn = { question: string; answer: string; complete: boolean };

const shortName = (model: string): string => model.split("/").pop() ?? model;

function Key({ children }: { children: ReactNode }): ReactElement {
  return (
    <kbd className="rounded-[4px] border border-border/70 bg-muted/60 px-1.25 py-px font-sans text-ui-10 leading-ui-14 text-muted-foreground">
      {children}
    </kbd>
  );
}

export function AskApp(): ReactElement {
  const t = useT();
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<Phase>("input");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [model, setModel] = useState<string | null>(null);
  const [error, setError] = useState<AskError["kind"] | null>(null);
  const [copied, setCopied] = useState(false);
  const [session, setSession] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const answerRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const sizeRef = useRef({ width: 0, height: 0 });

  const cancel = (): void => {
    abortRef.current?.abort();
    abortRef.current = null;
  };

  useEffect(() => {
    // Every summon starts a fresh conversation.
    const show = listen("ask://show", () => {
      cancel();
      setTurns([]);
      setQuery("");
      setError(null);
      setPhase("input");
      setSession((value) => value + 1);
    });
    const hide = listen("ask://hide", cancel);
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        cancel();
        void invoke("ask_hide");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      void show.then((unlisten) => unlisten());
      void hide.then((unlisten) => unlisten());
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  useEffect(() => {
    inputRef.current?.focus();
  }, [session]);

  useLayoutEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const width = Math.ceil(node.offsetWidth);
    const height = Math.ceil(node.offsetHeight);
    // A native resize per streamed token stalls the app: only resize on a real change.
    if (width === sizeRef.current.width && height === sizeRef.current.height) return;
    sizeRef.current = { width, height };
    void invoke("ask_resize", { width, height }).catch(() => undefined);
  });

  useEffect(() => {
    const node = answerRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [turns]);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  const submit = async (): Promise<void> => {
    const question = query.trim();
    if (!question || phase === "loading" || phase === "streaming") return;
    cancel();
    const abort = new AbortController();
    abortRef.current = abort;
    const history = turns.filter((turn) => turn.complete);
    setTurns([...turns, { question, answer: "", complete: false }]);
    setQuery("");
    setError(null);
    setPhase("streaming");
    // A new summon replaces the controller; stop touching state once that happens.
    const current = (): boolean => abortRef.current === abort;
    const update = (change: (turn: Turn) => Turn): void =>
      setTurns((all) => [...all.slice(0, -1), change(all[all.length - 1])]);

    try {
      if (!adoptBackendPort()) throw new AskError("failed");
      const used = await resolveModel(abort.signal, (loading) => {
        if (!current()) return;
        setModel(shortName(loading));
        setPhase("loading");
      });
      if (!current()) return;
      setModel(shortName(used));
      setPhase("streaming");
      const messages: ChatMessage[] = history.flatMap((turn) => [
        { role: "user" as const, content: turn.question },
        { role: "assistant" as const, content: turn.answer },
      ]);
      messages.push({ role: "user", content: question });
      for await (const delta of streamAnswer(used, messages, abort.signal)) {
        if (!current()) return;
        update((turn) => ({ ...turn, answer: turn.answer + delta }));
      }
      if (!current()) return;
      update((turn) => ({ ...turn, complete: true }));
      setPhase("done");
    } catch (cause) {
      if (!current()) return;
      if (abort.signal.aborted) {
        setPhase("done");
        return;
      }
      setError(cause instanceof AskError ? cause.kind : "failed");
      setPhase("error");
    } finally {
      if (current()) abortRef.current = null;
    }
  };

  const lastAnswer = turns.at(-1)?.answer ?? "";
  const busy = phase === "loading" || phase === "streaming";
  const linkClass =
    "rounded-md px-1.5 py-0.5 hover:bg-accent hover:text-accent-foreground";

  return (
    <div
      key={session}
      ref={containerRef}
      className="ask-pop w-160 overflow-hidden rounded-2xl border border-border/60 bg-popover/70 text-popover-foreground shadow-2xl"
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
        className="flex items-center gap-3 px-5 py-4"
      >
        <input
          ref={inputRef}
          autoFocus
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={t(turns.length > 0 ? "askBar.followUp" : "askBar.placeholder")}
          spellCheck={false}
          className="w-full bg-transparent text-ui-17 text-foreground outline-none placeholder:text-muted-foreground/80"
        />
        {busy && (
          <span className="size-4 shrink-0 animate-spin rounded-full border-2 border-muted-foreground/70 border-t-transparent" />
        )}
      </form>

      {(turns.length > 0 || phase === "error") && (
        <div
          ref={answerRef}
          className="max-h-80 overflow-y-auto border-t border-border/50 px-5 py-3.5 text-ui-13p5 leading-6"
        >
          {turns.map((turn, index) => (
            <div key={index} className={index > 0 ? "mt-3" : undefined}>
              {turns.length > 1 && (
                <div className="mb-1 text-ui-11p5 font-medium text-muted-foreground">
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
              {t(error === "noModel" ? "askBar.noModel" : "askBar.failed")}
            </span>
          )}
        </div>
      )}

      <div className="flex h-9 items-center justify-between border-t border-border/50 bg-muted/30 px-4 text-ui-11 text-muted-foreground">
        <span className="truncate">
          {phase === "loading" && model
            ? t("askBar.loading", { model })
            : (model ?? t("askBar.autoModel"))}
        </span>
        <span className="flex shrink-0 items-center gap-3">
          {turns.length > 0 && (
            <button
              type="button"
              className={linkClass}
              onClick={() => {
                cancel();
                setTurns([]);
                setError(null);
                setPhase("input");
                inputRef.current?.focus();
              }}
            >
              {t("askBar.clear")}
            </button>
          )}
          {phase === "done" && lastAnswer && (
            <button
              type="button"
              className={linkClass}
              onClick={() =>
                void copyToClipboard(lastAnswer).then((ok) => setCopied(ok))
              }
            >
              {t(copied ? "askBar.copied" : "askBar.copy")}
            </button>
          )}
          <span className="flex items-center gap-1">
            <Key>⏎</Key> {t("askBar.enterHint")}
          </span>
          <span className="flex items-center gap-1">
            <Key>esc</Key> {t("askBar.escHint")}
          </span>
        </span>
      </div>
    </div>
  );
}
