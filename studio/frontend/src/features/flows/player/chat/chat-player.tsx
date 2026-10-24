// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowLeftIcon, RotateCcwIcon, SendIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import type { PlayerApi } from "../player-api";
import {
  type ChatEntry,
  type ChatTurn,
  restoreChat,
  serializeChat,
  turnParams,
} from "./chat-state";

const SLOT = "chat";

export function ChatPlayer({
  flowId,
  api,
  entry,
  title,
  onExit,
}: {
  flowId: string;
  api: PlayerApi;
  entry: ChatEntry;
  title: string;
  onExit: () => void;
}) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState("");
  const [thinking, setThinking] = useState(false);
  // Sending before the saved conversation arrives would drop it from the model's history and overwrite the save.
  const [loaded, setLoaded] = useState(false);
  const generation = useRef(0);
  const inflight = useRef<AbortController | null>(null);
  const saves = useRef<Promise<unknown>>(Promise.resolve());
  const saveWarned = useRef(false);
  const bottom = useRef<HTMLDivElement>(null);

  const [attempt, setAttempt] = useState(0);

  // biome-ignore lint/correctness/useExhaustiveDependencies: attempt re-runs the load after a failure
  useEffect(() => {
    let live = true;
    api
      .loadSession(flowId, SLOT)
      .then((saved) => {
        if (!live) return;
        if (saved) setTurns(restoreChat(saved.state));
        setLoaded(true);
      })
      .catch(() => {
        // A failed read stays blocked: replying on an empty history would save over the real conversation.
        if (live)
          toast("Could not load this conversation", {
            action: { label: "Retry", onClick: () => setAttempt((n) => n + 1) },
          });
      });
    return () => {
      live = false;
      inflight.current?.abort();
    };
  }, [api, flowId, attempt]);

  // One save at a time, in order, so an earlier reply's save can never land after New chat cleared the slot.
  const persist = (next: ChatTurn[]) => {
    saves.current = saves.current
      .catch(() => undefined)
      .then(() => api.saveSession(flowId, SLOT, serializeChat(next)))
      .catch(() => {
        if (saveWarned.current) return;
        saveWarned.current = true;
        toast("This conversation could not be saved");
      });
  };

  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll to the newest turn whenever the conversation changes
  useEffect(() => {
    bottom.current?.scrollIntoView?.({ block: "end" });
  }, [turns, thinking]);

  const send = async () => {
    const message = draft.trim();
    if (!message || thinking || !loaded) return;
    const before = turns;
    const asked: ChatTurn[] = [...before, { role: "user", content: message }];
    setTurns(asked);
    setDraft("");
    setThinking(true);
    const mine = ++generation.current;
    const controller = new AbortController();
    inflight.current = controller;
    try {
      const res = await api.invoke(
        flowId,
        entry.output,
        {},
        { params: turnParams(before, message), signal: controller.signal },
      );
      if (mine !== generation.current) return;
      const reply = res.outputs.reply;
      if (typeof reply !== "string" || !reply.trim())
        throw new Error("The flow gave no reply");
      const next: ChatTurn[] = [
        ...asked,
        { role: "assistant", content: reply },
      ];
      setTurns(next);
      persist(next);
    } catch (e) {
      if (mine !== generation.current) return;
      setTurns(before);
      setDraft(message);
      toast(e instanceof Error ? e.message : String(e));
    } finally {
      if (inflight.current === controller) inflight.current = null;
      if (mine === generation.current) setThinking(false);
    }
  };

  const reset = () => {
    generation.current += 1;
    // Stop the abandoned reply so the next message does not queue behind it on the GPU.
    inflight.current?.abort();
    inflight.current = null;
    setTurns([]);
    setThinking(false);
    persist([]);
  };

  return (
    <div
      className="flex h-full flex-col bg-zinc-950 text-white"
      data-testid="chat-player"
    >
      <div className="flex items-center gap-3 border-white/10 border-b px-4 py-3">
        <button
          type="button"
          onClick={onExit}
          aria-label="Back to the editor"
          title="Back to the editor"
          className="rounded-full p-1.5 text-white/60 hover:bg-white/10 hover:text-white"
        >
          <ArrowLeftIcon className="size-4" />
        </button>
        <p className="flex-1 truncate font-medium">{title}</p>
        <button
          type="button"
          onClick={reset}
          className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-white/60 hover:bg-white/10 hover:text-white"
          data-testid="chat-reset"
        >
          <RotateCcwIcon className="size-3.5" /> New chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex max-w-2xl flex-col gap-3">
          <div className="self-start max-w-[85%] whitespace-pre-wrap rounded-2xl bg-white/8 px-4 py-2.5 text-white/85">
            {entry.welcome}
          </div>
          {turns.map((t, i) => (
            <div
              // biome-ignore lint/suspicious/noArrayIndexKey: turns only ever append or clear
              key={i}
              data-testid={t.role === "user" ? "chat-user" : "chat-reply"}
              className={
                t.role === "user"
                  ? "self-end max-w-[85%] whitespace-pre-wrap rounded-2xl bg-teal-600/80 px-4 py-2.5"
                  : "self-start max-w-[85%] whitespace-pre-wrap rounded-2xl bg-white/8 px-4 py-2.5 text-white/90"
              }
            >
              {t.content}
            </div>
          ))}
          {thinking ? (
            <div
              className="self-start rounded-2xl bg-white/8 px-4 py-2.5 text-white/50"
              data-testid="chat-thinking"
            >
              ...
            </div>
          ) : null}
          <div ref={bottom} />
        </div>
      </div>
      <form
        // The right gutter keeps Send clear of the player's fullscreen button on narrow screens.
        className="border-white/10 border-t py-3 pr-14 pl-4"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <div className="mx-auto flex max-w-2xl items-end gap-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              // Enter also confirms an IME composition (Chinese, Japanese, Korean input): that is not a send.
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                void send();
              }
            }}
            rows={1}
            placeholder={
              loaded ? "Type a message" : "Loading the conversation..."
            }
            className="max-h-40 min-h-10 flex-1 resize-none rounded-xl border border-white/15 bg-white/5 px-3 py-2 outline-none focus:border-teal-400/60"
            data-testid="chat-say"
          />
          <button
            type="submit"
            disabled={thinking || !loaded || !draft.trim()}
            aria-label="Send"
            title="Send"
            className="rounded-xl bg-teal-500 p-2.5 text-black disabled:opacity-40"
          >
            <SendIcon className="size-4" />
          </button>
        </div>
      </form>
    </div>
  );
}
