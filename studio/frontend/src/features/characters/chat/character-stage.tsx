// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { Portrait } from "../components/portrait";
import { useImageUrl, useIsCutout, viewPicture } from "../images";
import { chatViewOf, useStageFocus } from "../stage-focus";
import type { Character } from "../types";
import { NovelNameplate, NovelScene, SpeakingDots } from "./novel-scene";
import { useAskToReply } from "./speaker-bar";
import { RoleplayController } from "./roleplay-controller";
import { useThreadScene } from "./use-thread-scene";

/**
 * The chat area of a character chat. Its controller keeps group rounds going. Stage view stands the
 * speaking character beside the conversation; visual novel view puts the cast over the scene with the
 * messages in a panel below. The thread keeps one place in the tree, so switching never remounts it.
 */
export function CharacterStage({ threadId, children }: { threadId: string | null | undefined; children: ReactNode }) {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const id = threadId ?? activeThreadId ?? null;
  const thread = useThreadScene(id);
  const view = thread ? chatViewOf(thread.scene.settings) : "chat";
  const stage = view === "stage";
  const novel = view === "novel";
  return (
    <div
      className={cn(
        "relative flex min-h-0 min-w-0 flex-1 basis-0",
        stage ? "character-stage-layout flex-col [container-type:inline-size] md:flex-row" : "flex-col",
        novel && "character-novel isolate",
      )}
    >
      {thread ? <RoleplayController /> : null}
      {stage && thread ? <StagePanel threadId={id} cast={thread.cast} /> : null}
      {novel && thread ? <NovelScene threadId={id} cast={thread.cast} /> : null}
      <div
        className={cn(
          "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden",
          novel &&
            "novel-dialogue relative z-10 mx-auto mt-auto w-[calc(100%-2rem)] max-w-[calc(920px*var(--ui-space-scale,1))] flex-none basis-auto h-[52%] min-h-[calc(440px*var(--ui-space-scale,1))] overflow-visible rounded-t-[30px] bg-background/75 shadow-[0_-30px_80px_-40px_rgb(0_0_0/0.6)] ring-1 ring-white/25 backdrop-blur-2xl dark:bg-background/70 dark:ring-white/10 max-md:h-[58%] max-md:w-full [--studio-chat-header-height:calc(1.75rem*var(--ui-space-scale,1))] [--studio-chat-scroller-top:0px] [--studio-content-top-inset:0px]",
        )}
      >
        {children}
        {novel && thread ? <NovelNameplate threadId={id} cast={thread.cast} /> : null}
      </div>
    </div>
  );
}

function StagePanel({ threadId, cast }: { threadId: string | null; cast: Character[] }) {
  const t = useT();
  const { focus, speaking } = useStageFocus(threadId, cast);
  const ask = useAskToReply();
  // Marked on the figure itself: a :has() rule above it would restyle the whole thread on every change.
  const focusUrl = useImageUrl("character", focus?.id, focus?.avatarRev ?? 0, Boolean(focus?.hasAvatar));
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
        data-cutout={cutout || undefined}
        className="stage-figure relative min-h-0 overflow-hidden rounded-[26px] bg-muted shadow-[0_24px_60px_-30px_rgb(0_0_0/0.6)] animate-in fade-in-0 zoom-in-[0.985] duration-500 max-md:aspect-[2/3] max-md:h-full md:flex-1"
      >
        <button
          type="button"
          className="block size-full cursor-zoom-in"
          onClick={() => viewPicture("character", focus.id, focus.name, focus.avatarRev)}
          disabled={!focus.hasAvatar}
          aria-label={t("characters.actions.viewPicture")}
        >
          <Portrait
            id={focus.id}
            name={focus.name}
            hasImage={focus.hasAvatar}
            rev={focus.avatarRev}
            variant="card"
            cutout={true}
            className="size-full"
          />
        </button>
        <figcaption className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col gap-0.5 bg-gradient-to-t from-black/70 via-black/25 to-transparent px-4 pb-3.5 pt-16 text-white max-md:hidden">
          <span className="font-heading text-ui-20 font-semibold leading-tight [text-shadow:0_1px_2px_rgb(0_0_0/0.4)]">
            {focus.name}
          </span>
          {speaking ? <SpeakingDots name={focus.name} /> : null}
        </figcaption>
      </figure>
      {others.length ? (
        <div className="flex shrink-0 gap-2 max-md:flex-col max-md:justify-end md:flex-wrap" aria-label={t("characters.chat.castLabel")}>
          {others.map((character) => (
            <button
              key={character.id}
              type="button"
              onClick={() => ask(character.id, character.name)}
              title={t("characters.chat.replyNow", { name: character.name })}
              aria-label={t("characters.chat.replyNow", { name: character.name })}
              className="group/cast flex cursor-pointer items-center gap-2 rounded-full bg-muted/70 py-1 pl-1 pr-3 text-ui-13 font-medium text-foreground transition-colors hover:bg-accent dark:bg-card"
            >
              <Portrait
                id={character.id}
                name={character.name}
                hasImage={character.hasAvatar}
                rev={character.avatarRev}
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
