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
  const generation = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    api
      .loadSession(flowId, SLOT)
      .then((saved) => {
        if (live && saved) setTurns(restoreChat(saved.state));
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [api, flowId]);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ block: "end" });
  }, [turns, thinking]);

  const send = async () => {
    const message = draft.trim();
    if (!message || thinking) return;
    const before = turns;
    const asked: ChatTurn[] = [...before, { role: "user", content: message }];
    setTurns(asked);
    setDraft("");
    setThinking(true);
    const mine = ++generation.current;
    try {
      const res = await api.invoke(
        flowId,
        entry.output,
        {},
        { params: turnParams(before, message) },
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
      void api
        .saveSession(flowId, SLOT, serializeChat(next))
        .catch(() => undefined);
    } catch (e) {
      if (mine !== generation.current) return;
      setTurns(before);
      setDraft(message);
      toast(e instanceof Error ? e.message : String(e));
    } finally {
      if (mine === generation.current) setThinking(false);
    }
  };

  const reset = () => {
    generation.current += 1;
    setTurns([]);
    setThinking(false);
    void api
      .saveSession(flowId, SLOT, serializeChat([]))
      .catch(() => undefined);
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
        className="border-white/10 border-t px-4 py-3"
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
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            rows={1}
            placeholder="Type a message"
            className="max-h-40 min-h-10 flex-1 resize-none rounded-xl border border-white/15 bg-white/5 px-3 py-2 outline-none focus:border-teal-400/60"
            data-testid="chat-say"
          />
          <button
            type="submit"
            disabled={thinking || !draft.trim()}
            aria-label="Send"
            className="rounded-xl bg-teal-500 p-2.5 text-black disabled:opacity-40"
          >
            <SendIcon className="size-4" />
          </button>
        </div>
      </form>
    </div>
  );
}
