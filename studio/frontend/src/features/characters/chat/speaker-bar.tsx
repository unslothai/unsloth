// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { errorMessage } from "@/lib/error-message";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { isExternalModelId } from "@/features/chat/external-providers";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { PencilEdit02Icon, PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useAui, useAuiState } from "@assistant-ui/react";
import { useEffect, useRef, useState } from "react";
import { Portrait } from "../components/portrait";
import { lastSpeakerId, roleplayMetaOf } from "../meta";
import { shortName } from "../prompt/speakers";
import { writeMyReply } from "./write-my-reply";
import { useAskToReply } from "./ask-bridge";
import { useCharacterChat } from "./character-chat-context";

const CHIP =
  "flex h-8 shrink-0 cursor-pointer items-center gap-1.5 rounded-full bg-muted py-0.5 text-ui-13 font-medium text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:opacity-60 dark:bg-card";

/** Above the composer in a character chat: who replies next, and a draft of your own reply. */
export function CharacterSpeakerBar() {
  const t = useT();
  const aui = useAui();
  const thread = useCharacterChat();
  const threadId = thread?.threadId ?? null;
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  // Every composer mounts this bar; in an ordinary chat its selectors do no work per token.
  const active = thread !== null;
  const running = useAuiState(({ thread: state }) => active && state.isRunning);
  const lastReplySpeakerId = useAuiState(({ thread: state }) => (active ? lastSpeakerId(state.messages) : null));
  const runningSpeakerId = useAuiState(({ thread: state }) => {
    if (!active) return null;
    const last = state.messages[state.messages.length - 1];
    if (!state.isRunning || last?.role !== "assistant") return null;
    const meta = roleplayMetaOf(last);
    return meta?.kind === "reply" ? meta.speakerId : null;
  });
  const ask = useAskToReply();
  const [writing, setWriting] = useState(false);
  const abort = useRef<AbortController | null>(null);
  // The bar stays mounted across chat switches, so a draft still running belongs to the chat it started in.
  useEffect(() => () => abort.current?.abort(), [threadId]);
  if (!thread || !threadId) return null;
  const muted = new Set(thread.scene.cast.filter((m) => m.muted).map((m) => m.characterId));
  const canWrite = Boolean(checkpoint) && !isExternalModelId(checkpoint);

  async function write() {
    // The ref, not state: a second click before the re-render must stop the first draft, not start another.
    if (abort.current) {
      abort.current.abort();
      return;
    }
    const controller = new AbortController();
    abort.current = controller;
    setWriting(true);
    try {
      const origin = threadId!;
      const typed = aui.composer().getState().text;
      const text = await writeMyReply(origin, aui.thread().getState().messages, controller.signal);
      // Only into the same chat, and never over what the user typed while it was being written.
      const untouched = aui.composer().getState().text === typed;
      if (text && untouched && aui.threads().getState().mainThreadId === origin) aui.composer().setText(text);
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        toast.error(t("characters.chat.writeFailed"), { description: errorMessage(error) });
      }
    } finally {
      if (abort.current === controller) abort.current = null;
      setWriting(false);
    }
  }

  const solo = !thread.isGroup ? thread.cast[0] : null;
  return (
    <div
      className="character-speaker-bar relative mb-2 flex items-center gap-1.5 px-1 pb-0.5"
      aria-label={t("characters.chat.nextReply")}
      data-testid="speaker-bar"
    >
      {solo ? (
        <button
          type="button"
          disabled={running}
          onClick={() => ask(solo.id, solo.name)}
          className={cn(CHIP, "pl-1 pr-3")}
          title={t("characters.chat.continueScene", { name: shortName(solo.name) })}
        >
          <Portrait subject={solo} className="size-7 rounded-full text-[0.7rem]" />
          <HugeiconsIcon icon={PlayIcon} strokeWidth={1.8} className="size-3.5 text-muted-foreground" />
          <span className="truncate">{t("characters.chat.letContinue", { name: shortName(solo.name) })}</span>
        </button>
      ) : (
        <>
          <span className="mr-1 shrink-0 text-ui-12 font-medium text-muted-foreground max-sm:hidden">{t("characters.chat.nextReply")}</span>
          {/* Only the chips scroll, so Write my reply stays in reach however long the names are. */}
          <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto [mask-image:linear-gradient(to_right,#000_calc(100%-1.5rem),transparent)] [scrollbar-width:none]">
          {thread.cast.map((character) => {
            const speaking = runningSpeakerId === character.id;
            const isMuted = muted.has(character.id);
            return (
              <Tooltip key={character.id}>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    disabled={running}
                    onClick={() => ask(character.id, character.name)}
                    aria-label={t("characters.chat.replyNow", { name: character.name })}
                    className={cn(
                      CHIP,
                      "pl-0.5 pr-3",
                      speaking && "bg-primary/15 ring-1 ring-primary/40 disabled:opacity-100",
                      isMuted && "opacity-55",
                      lastReplySpeakerId === character.id && !speaking && "ring-1 ring-border",
                    )}
                  >
                    <Portrait
                      subject={character}
                      className={cn("size-7 rounded-full text-[0.7rem]", speaking && "animate-pulse")}
                    />
                    <span className="max-w-40 truncate">{character.name}</span>
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top" className="tooltip-compact">
                  {speaking ? t("characters.chat.speaking", { name: character.name }) : t("characters.chat.replyNow", { name: character.name })}
                </TooltipContent>
              </Tooltip>
            );
          })}
          </div>
        </>
      )}
      {canWrite ? (
        <button
          type="button"
          disabled={running && !writing}
          onClick={() => void write()}
          className={cn(CHIP, "ml-auto shrink-0 px-3", writing && "bg-primary/12 text-foreground")}
          title={t("characters.chat.writeHint", { name: thread.persona.name })}
          data-testid="write-my-reply"
        >
          <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.8} className={cn("size-3.5", writing && "animate-pulse")} />
          {writing ? t("characters.chat.writing") : t("characters.chat.writeMyReply")}
        </button>
      ) : null}
    </div>
  );
}

export function useCharacterComposerPlaceholder(): string | null {
  const t = useT();
  const thread = useCharacterChat();
  if (!thread) return null;
  return thread.isGroup
    ? t("characters.chat.placeholderGroup")
    : t("characters.chat.placeholder", { name: thread.cast[0].name });
}
