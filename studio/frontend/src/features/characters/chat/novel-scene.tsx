// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBackgroundStore } from "@/features/chat-background";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useEffect, useState, type CSSProperties } from "react";
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

/** The user's chat background as its own picture, at full strength rather than behind the chat's scrim. */
function useWallpaperUrl(): string | null {
  const blob = useBackgroundStore((s) => s.value?.blob ?? null);
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!blob) {
      setUrl(null);
      return;
    }
    const next = URL.createObjectURL(blob);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [blob]);
  return url;
}

/**
 * Visual novel view: the cast stands in the scene, the speaker in front and the others a step behind,
 * their lower halves behind the dialogue panel. The scene is the user's chat background when one is set,
 * otherwise a wash of the speaker's colors.
 */
export function NovelScene({ threadId, cast }: { threadId: string | null; cast: Character[] }) {
  const t = useT();
  const { focus, speaking } = useStageFocus(threadId, cast);
  const wallpaper = useWallpaperUrl();
  const ask = useAskToReply();
  if (!focus) return null;
  const figures = shownCast(cast, focus);
  return (
    <>
      <NovelBackdrop character={focus} wallpaper={wallpaper} />
      <section
        className="novel-figures pointer-events-none absolute inset-x-0 bottom-[22%] top-14 z-0 flex items-end justify-center px-[3%] max-md:bottom-[44%] max-md:top-12"
        aria-label={t("characters.chat.viewNovel")}
        data-testid="character-novel"
      >
        {figures.map((character) => {
          const active = character.id === focus.id;
          const solo = cast.length === 1;
          const onClick = active || solo
            ? () => character.hasAvatar && viewPicture("character", character.id, character.name)
            : () => ask(character.id, character.name);
          return (
            <button
              key={character.id}
              type="button"
              onClick={onClick}
              aria-label={
                active || solo ? t("characters.actions.viewPicture") : t("characters.chat.replyNow", { name: character.name })
              }
              title={active ? character.name : t("characters.chat.replyNow", { name: character.name })}
              className={cn(
                "novel-figure pointer-events-auto relative -mx-[1.5%] aspect-[2/3] h-full min-w-0 max-w-[36%] shrink origin-bottom cursor-pointer focus-visible:outline-none",
                active ? "z-[2] max-sm:max-w-[78%]" : "novel-figure-waiting z-[1] scale-[0.84] max-sm:hidden",
                active && speaking && "novel-figure-speaking",
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
                className="novel-figure-picture size-full rounded-t-[40px]"
              />
            </button>
          );
        })}
      </section>
    </>
  );
}

/** The speaker's name on the dialogue panel, as a visual novel shows it. */
export function NovelNameplate({ threadId, cast }: { threadId: string | null; cast: Character[] }) {
  const { focus, speaking } = useStageFocus(threadId, cast);
  if (!focus) return null;
  const hue = nameHue(focus.name);
  const style: CSSProperties = {
    background: `linear-gradient(135deg, hsl(${hue} 52% 46%), hsl(${(hue + 28) % 360} 56% 36%))`,
  };
  return (
    <div
      key={focus.id}
      className="pointer-events-none absolute -top-4 left-[clamp(1rem,6%,3.5rem)] z-20 flex items-center gap-2.5 rounded-full py-1.5 pl-4 pr-4 text-white shadow-[0_10px_24px_-10px_rgb(0_0_0/0.6)] ring-1 ring-white/20 animate-in fade-in-0 slide-in-from-bottom-1 duration-300"
      style={style}
      data-testid="novel-nameplate"
    >
      <span className="font-heading text-ui-15 font-semibold tracking-[-0.01em]">{focus.name}</span>
      {speaking ? (
        <span className="flex gap-0.5" aria-hidden={true}>
          <span className="size-1 animate-bounce rounded-full bg-white/90 [animation-delay:-0.3s]" />
          <span className="size-1 animate-bounce rounded-full bg-white/90 [animation-delay:-0.15s]" />
          <span className="size-1 animate-bounce rounded-full bg-white/90" />
        </span>
      ) : null}
    </div>
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

function NovelBackdrop({ character, wallpaper }: { character: Character; wallpaper: string | null }) {
  const portrait = useImageUrl("character", character.id, character.avatarRev, character.hasAvatar && !wallpaper);
  const hue = nameHue(character.name);
  const wash: CSSProperties = {
    background:
      `radial-gradient(90% 70% at 50% 35%, hsl(${hue} 55% 62% / 0.55), transparent 70%), ` +
      `linear-gradient(180deg, hsl(${(hue + 200) % 360} 35% 72% / 0.5), hsl(${hue} 30% 40% / 0.35))`,
  };
  return (
    <div className="novel-backdrop pointer-events-none absolute inset-0 -z-10 overflow-hidden bg-background" aria-hidden={true}>
      {wallpaper ? (
        <img src={wallpaper} alt="" className="absolute inset-0 size-full object-cover animate-in fade-in-0 duration-500" />
      ) : (
        <>
          {portrait ? (
            <img
              key={portrait}
              src={portrait}
              alt=""
              className="absolute inset-0 size-full scale-125 object-cover opacity-70 blur-[70px] saturate-[1.6] animate-in fade-in-0 duration-700 dark:opacity-55"
            />
          ) : null}
          <div className="absolute inset-0 transition-[background] duration-700" style={wash} />
        </>
      )}
      {/* Seats the dialogue panel and keeps the header readable over any picture. */}
      <div className="absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-background/70 to-transparent" />
      <div className="absolute inset-x-0 bottom-0 h-[45%] bg-gradient-to-t from-background/60 to-transparent" />
    </div>
  );
}
