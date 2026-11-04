// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBackgroundStore } from "@/features/chat-background";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { CSSProperties } from "react";
import { Portrait } from "../components/portrait";
import { nameHue, useImageUrl, viewPicture } from "../images";
import { useStageFocus } from "../stage-focus";
import type { Character } from "../types";
import { useAskToReply } from "./speaker-bar";

// Enough figures to read as a scene; a bigger cast shows the speaker and the members nearest them.
const MAX_FIGURES = 4;

function shownCast(cast: Character[], focus: Character): Character[] {
  if (cast.length <= MAX_FIGURES) return cast;
  const at = cast.findIndex((c) => c.id === focus.id);
  const start = Math.min(Math.max(0, at - 1), cast.length - MAX_FIGURES);
  return cast.slice(start, start + MAX_FIGURES);
}

/**
 * Visual novel view: the cast stands over the scene, the speaker in front, with the messages in a
 * panel below them. The scene is the user's chat background when one is set, otherwise a soft blur of
 * the speaker's picture.
 */
export function NovelScene({ threadId, cast }: { threadId: string | null; cast: Character[] }) {
  const t = useT();
  const { focus, speaking } = useStageFocus(threadId, cast);
  const hasWallpaper = useBackgroundStore((s) => Boolean(s.value));
  const ask = useAskToReply();
  if (!focus) return null;
  const figures = shownCast(cast, focus);
  return (
    <>
      {hasWallpaper ? null : <NovelBackdrop character={focus} />}
      <section
        className="relative z-0 -mb-8 flex min-h-0 flex-1 items-end justify-center gap-[2%] px-[4%] pt-16 max-md:-mb-6 max-md:pt-14"
        aria-label={t("characters.chat.viewNovel")}
        data-testid="character-novel"
      >
        {figures.map((character) => {
          const active = character.id === focus.id;
          const onClick = active || cast.length === 1
            ? () => character.hasAvatar && viewPicture("character", character.id, character.name)
            : () => ask(character.id, character.name);
          return (
            <button
              key={character.id}
              type="button"
              onClick={onClick}
              aria-label={
                active || cast.length === 1
                  ? t("characters.actions.viewPicture")
                  : t("characters.chat.replyNow", { name: character.name })
              }
              title={active ? character.name : t("characters.chat.replyNow", { name: character.name })}
              className={cn(
                "group/figure relative aspect-[2/3] min-w-0 max-w-[34%] shrink cursor-pointer overflow-hidden rounded-t-[30px] shadow-[0_30px_80px_-30px_rgb(0_0_0/0.7)] transition-[height,opacity,filter,transform] duration-500 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                active ? "z-[1] h-full" : "h-[80%] opacity-60 brightness-90 saturate-[0.7] hover:opacity-90 hover:saturate-100",
                active && character.hasAvatar && "cursor-zoom-in",
              )}
              data-active={active || undefined}
            >
              <Portrait
                id={character.id}
                name={character.name}
                hasImage={character.hasAvatar}
                rev={character.avatarRev}
                variant="card"
                className="size-full"
              />
              {active ? (
                <span className="pointer-events-none absolute inset-x-0 top-0 flex flex-col gap-0.5 bg-gradient-to-b from-black/55 via-black/15 to-transparent px-4 pb-10 pt-3.5 text-left text-white">
                  <span className="font-heading text-ui-18 font-semibold leading-tight [text-shadow:0_1px_2px_rgb(0_0_0/0.4)]">
                    {character.name}
                  </span>
                  {speaking ? <SpeakingDots name={character.name} /> : null}
                </span>
              ) : null}
            </button>
          );
        })}
      </section>
    </>
  );
}

/** "Seraphina is replying" with three bouncing dots. */
export function SpeakingDots({ name }: { name: string }) {
  const t = useT();
  return (
    <span className="flex items-center gap-1.5 text-ui-12 text-white/80">
      <span className="flex gap-0.5">
        <span className="size-1 animate-bounce rounded-full bg-white/90 [animation-delay:-0.3s]" />
        <span className="size-1 animate-bounce rounded-full bg-white/90 [animation-delay:-0.15s]" />
        <span className="size-1 animate-bounce rounded-full bg-white/90" />
      </span>
      {t("characters.chat.speaking", { name: name.split(/\s+/)[0] })}
    </span>
  );
}

function NovelBackdrop({ character }: { character: Character }) {
  const url = useImageUrl("character", character.id, character.avatarRev, character.hasAvatar);
  const hue = nameHue(character.name);
  const tint: CSSProperties = {
    background: `radial-gradient(120% 90% at 50% 10%, hsl(${hue} 40% 55% / 0.35), transparent 70%)`,
  };
  return (
    <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden" aria-hidden={true}>
      {url ? (
        <img
          key={url}
          src={url}
          alt=""
          className="absolute inset-0 size-full scale-125 object-cover opacity-45 blur-3xl saturate-150 animate-in fade-in-0 duration-700 dark:opacity-35"
        />
      ) : null}
      <div className="absolute inset-0" style={tint} />
    </div>
  );
}
