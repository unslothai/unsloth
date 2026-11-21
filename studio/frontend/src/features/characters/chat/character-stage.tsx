// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useAuiState } from "@assistant-ui/react";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { Portrait } from "../components/portrait";
import { useAvatarUrl, useIsCutout, viewAvatar } from "../images";
import { chatViewOf } from "../chat-view";
import { useStageFocus } from "../stage-focus";
import type { Character } from "../types";
import { NovelNameplate, NovelScene, SpeakingDots } from "./novel-scene";
import { useAskToReply } from "./ask-bridge";
import { GroupRoundController } from "./group-round-controller";
import { CharacterChatContext } from "./character-chat-context";
import { useThreadScene } from "./use-thread-scene";

/** A character chat's layout: plain, stage (speaker beside the chat) or novel (cast over the scene). Keeps the thread mounted across views. */
export function CharacterStage({ threadId, children }: { threadId: string | null | undefined; children: ReactNode }) {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const id = threadId ?? activeThreadId ?? null;
  // Worked out once here, so messages and the composer read context instead of each subscribing.
  const thread = useThreadScene(id);
  const view = thread ? chatViewOf(thread.scene.settings) : "chat";
  const stage = view === "stage";
  const novel = view === "novel";
  return (
    <CharacterChatContext.Provider value={thread}>
    <div
      className={cn(
        "relative flex min-h-0 min-w-0 flex-1 basis-0",
        stage ? "character-stage-layout flex-col [container-type:inline-size] md:flex-row" : "flex-col",
        novel && "character-novel isolate",
      )}
    >
      {thread ? <GroupRoundController /> : null}
      {stage && thread ? <StagePanel threadId={id} cast={thread.cast} /> : null}
      {novel && thread ? <NovelScene threadId={id} cast={thread.cast} /> : null}
      <div
        className={cn(
          "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden",
          novel &&
            "character-novel-dialogue relative z-10 mx-auto mt-auto w-[calc(100%-2rem)] max-w-[calc(920px*var(--ui-space-scale,1))] flex-none basis-auto h-[var(--character-novel-panel-h)] min-h-[var(--character-novel-panel-min)] overflow-visible rounded-t-[30px] bg-background/75 shadow-[0_-30px_80px_-40px_rgb(0_0_0/0.6)] ring-1 ring-white/25 backdrop-blur-2xl dark:bg-background/70 dark:ring-white/10 max-md:h-[58%] max-md:w-full [--studio-chat-header-height:calc(1.75rem*var(--ui-space-scale,1))] [--studio-chat-scroller-top:0px] [--studio-content-top-inset:0px]",
        )}
      >
        {children}
        {novel && thread ? <NovelNameplate threadId={id} cast={thread.cast} /> : null}
      </div>
    </div>
    </CharacterChatContext.Provider>
  );
}

function StagePanel({ threadId, cast }: { threadId: string | null; cast: Character[] }) {
  const t = useT();
  const { focus, speaking } = useStageFocus(threadId, cast);
  const ask = useAskToReply();
  const running = useAuiState(({ thread }) => thread.isRunning);
  const focusUrl = useAvatarUrl("character", focus?.id, focus?.avatarRev ?? 0, Boolean(focus?.hasAvatar));
  const cutout = useIsCutout(focusUrl, true);
  if (!focus) return null;
  const others = cast.filter((c) => c.id !== focus.id);
  return (
    <aside
      className="character-stage relative flex shrink-0 gap-3 border-border/60 p-3 max-md:mt-12 max-md:h-37 max-md:items-end max-md:border-b md:w-[clamp(240px,30%,440px)] md:flex-col md:border-r md:p-4 md:pt-16"
      data-testid="character-stage"
      aria-label={t("characters.chat.stage")}
    >
      <figure
        key={focus.id}
        // Marked on the figure itself: a :has() rule above it would restyle the whole thread on every change.
        data-cutout={cutout || undefined}
        className={cn(
          "character-stage-figure relative min-h-0 animate-in fade-in-0 zoom-in-[0.985] duration-500 max-md:aspect-[2/3] max-md:h-full md:flex-1",
          // A cut-out stands beside the conversation instead of filling a tile.
          cutout ? "overflow-visible" : "overflow-hidden rounded-[26px] bg-muted shadow-[0_24px_60px_-30px_rgb(0_0_0/0.6)]",
        )}
      >
        <button
          type="button"
          className="block size-full cursor-zoom-in focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring disabled:cursor-default"
          onClick={() => viewAvatar("character", focus.id, focus.name, focus.avatarRev)}
          disabled={!focus.hasAvatar}
          aria-label={t("characters.actions.viewPicture")}
        >
          <Portrait
            subject={focus}
            variant="card"
            cutout={true}
            className="size-full"
          />
        </button>
        <figcaption
          className={cn(
            "pointer-events-none absolute inset-x-0 bottom-0 flex flex-col gap-0.5 px-4 pb-3.5 pt-16 max-md:hidden",
            // Over a cut-out the name can fall on the figure, so it gets a halo in the page color instead of the scrim.
            cutout
              ? "text-foreground [text-shadow:0_0_3px_var(--background),0_0_8px_var(--background),0_0_14px_var(--background)]"
              : "bg-gradient-to-t from-black/70 via-black/25 to-transparent text-white",
          )}
        >
          <span className={cn("font-heading text-ui-20 font-semibold leading-tight", !cutout && "[text-shadow:0_1px_2px_rgb(0_0_0/0.4)]")}>
            {focus.name}
          </span>
          {speaking ? <SpeakingDots name={focus.name} onPage={cutout} /> : null}
        </figcaption>
      </figure>
      {others.length ? (
        <div className="flex shrink-0 gap-2 max-md:flex-col max-md:justify-end md:flex-wrap" aria-label={t("characters.chat.castLabel")}>
          {others.map((character) => (
            <button
              key={character.id}
              type="button"
              onClick={() => ask(character.id, character.name)}
              disabled={running}
              title={t("characters.chat.replyNow", { name: character.name })}
              aria-label={t("characters.chat.replyNow", { name: character.name })}
              className="group/cast flex cursor-pointer items-center gap-2 rounded-full bg-muted/70 py-1 pl-1 pr-3 text-ui-13 font-medium text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:opacity-60 dark:bg-card"
            >
              <Portrait
                subject={character}
                className="size-8 rounded-full text-[0.7rem] opacity-80 grayscale-[30%] transition-[opacity,filter] group-hover/cast:opacity-100 group-hover/cast:grayscale-0"
              />
              <span className="max-w-36 truncate max-md:hidden">{character.name}</span>
            </button>
          ))}
        </div>
      ) : null}
    </aside>
  );
}
