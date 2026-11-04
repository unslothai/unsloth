// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useAuiState } from "@assistant-ui/react";
import type { ReactNode } from "react";
import { Portrait } from "../components/portrait";
import { openCharacterLightbox } from "../lightbox-store";
import { roleplayMetaOf } from "../meta";
import type { Character } from "../types";
import { useAskToReply } from "./speaker-bar";
import { RoleplayController } from "./roleplay-controller";
import { useThreadScene } from "./use-thread-scene";

function useStageFocus(cast: Character[]): { focus: Character | null; speaking: boolean } {
  const state = useAuiState(({ thread }) => {
    let lastId: string | null = null;
    for (let i = thread.messages.length - 1; i >= 0; i -= 1) {
      const meta = roleplayMetaOf(thread.messages[i]);
      if (meta?.kind === "reply" || meta?.kind === "turn") {
        lastId = meta.speakerId;
        break;
      }
    }
    return `${thread.isRunning ? 1 : 0}|${lastId ?? ""}`;
  });
  const [running, id] = state.split("|");
  const focus = cast.find((c) => c.id === id) ?? cast[0] ?? null;
  return { focus, speaking: running === "1" };
}

/**
 * The chat area of a character chat. Its controller keeps group rounds going; with stage view on,
 * the speaking character stands beside the conversation in a tall portrait, the rest of the cast
 * waiting below them.
 */
export function CharacterStage({ threadId, children }: { threadId: string | null | undefined; children: ReactNode }) {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const thread = useThreadScene(threadId ?? activeThreadId);
  const stage = Boolean(thread?.scene.settings.stage);
  return (
    <div className={cn("flex min-h-0 min-w-0 flex-1 basis-0", stage ? "flex-col md:flex-row" : "flex-col")}>
      {thread ? <RoleplayController /> : null}
      {stage && thread ? <StagePanel cast={thread.cast} /> : null}
      <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden">{children}</div>
    </div>
  );
}

function StagePanel({ cast }: { cast: Character[] }) {
  const t = useT();
  const { focus, speaking } = useStageFocus(cast);
  const ask = useAskToReply();
  if (!focus) return null;
  const others = cast.filter((c) => c.id !== focus.id);
  const open = (character: Character) =>
    openCharacterLightbox({
      kind: "character",
      id: character.id,
      name: character.name,
      hasImage: character.hasAvatar,
      rev: character.avatarRev,
    });
  return (
    <aside
      className="character-stage relative flex shrink-0 gap-3 border-border/60 p-3 max-md:mt-12 max-md:h-37 max-md:items-end max-md:border-b md:w-[clamp(240px,30%,440px)] md:flex-col md:border-r md:p-4 md:pt-16"
      data-testid="character-stage"
      aria-label={t("characters.chat.stage")}
    >
      <figure
        key={focus.id}
        className="relative min-h-0 overflow-hidden rounded-[26px] bg-muted shadow-[0_24px_60px_-30px_rgb(0_0_0/0.6)] animate-in fade-in-0 zoom-in-[0.985] duration-500 max-md:aspect-[2/3] max-md:h-full md:flex-1"
      >
        <button
          type="button"
          className="block size-full cursor-zoom-in"
          onClick={() => open(focus)}
          aria-label={t("characters.actions.viewPicture")}
        >
          <Portrait
            id={focus.id}
            name={focus.name}
            hasImage={focus.hasAvatar}
            rev={focus.avatarRev}
            variant="card"
            className="size-full"
          />
        </button>
        <figcaption className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col gap-0.5 bg-gradient-to-t from-black/70 via-black/25 to-transparent px-4 pb-3.5 pt-16 text-white max-md:hidden">
          <span className="font-heading text-ui-20 font-semibold leading-tight [text-shadow:0_1px_2px_rgb(0_0_0/0.4)]">
            {focus.name}
          </span>
          {speaking ? (
            <span className="flex items-center gap-1.5 text-ui-12 text-white/80">
              <span className="flex gap-0.5">
                <span className="size-1 animate-bounce rounded-full bg-white/90 [animation-delay:-0.3s]" />
                <span className="size-1 animate-bounce rounded-full bg-white/90 [animation-delay:-0.15s]" />
                <span className="size-1 animate-bounce rounded-full bg-white/90" />
              </span>
              {t("characters.chat.speaking", { name: focus.name.split(/\s+/)[0] })}
            </span>
          ) : null}
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
