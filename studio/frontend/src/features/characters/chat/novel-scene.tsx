// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBackgroundStore } from "@/features/chat-background";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Portrait } from "../components/portrait";
import { nameHue, viewPicture } from "../images";
import { shortName } from "../prompt/speakers";
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
 * otherwise a plain stage in the theme's colors.
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
      <NovelBackdrop wallpaper={wallpaper} />
      <section
        className="novel-figures pointer-events-none absolute inset-x-0 bottom-[34%] top-14 z-0 flex items-end justify-center px-[3%] max-md:bottom-[44%] max-md:top-12"
        aria-label={t("characters.chat.viewNovel")}
        data-testid="character-novel"
      >
        {figures.map((character) => (
          <NovelFigure
            key={character.id}
            character={character}
            active={character.id === focus.id}
            speaking={speaking && character.id === focus.id}
            solo={cast.length === 1}
            onAsk={() => ask(character.id, character.name)}
          />
        ))}
      </section>
    </>
  );
}

/**
 * Breathing that ends where it began: once the reply is done, the cycle in progress finishes before the
 * animation is removed, so the figure settles instead of snapping back.
 */
function useSettlingBreath(on: boolean) {
  const [breathing, setBreathing] = useState(on);
  const stopping = useRef(false);
  useEffect(() => {
    if (on) {
      stopping.current = false;
      setBreathing(true);
    } else {
      stopping.current = true;
    }
  }, [on]);
  const onAnimationIteration = () => {
    if (stopping.current) {
      stopping.current = false;
      setBreathing(false);
    }
  };
  return { breathing, onAnimationIteration };
}

function NovelFigure({
  character,
  active,
  speaking,
  solo,
  onAsk,
}: {
  character: Character;
  active: boolean;
  speaking: boolean;
  solo: boolean;
  onAsk: () => void;
}) {
  const t = useT();
  const { breathing, onAnimationIteration } = useSettlingBreath(speaking);
  const onClick = active || solo ? () => character.hasAvatar && viewPicture("character", character.id, character.name, character.avatarRev) : onAsk;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={active || solo ? t("characters.actions.viewPicture") : t("characters.chat.replyNow", { name: character.name })}
      title={active ? character.name : t("characters.chat.replyNow", { name: character.name })}
      className={cn(
        "novel-figure pointer-events-auto relative -mx-[1.5%] aspect-[2/3] h-full min-w-0 max-w-[36%] shrink origin-bottom cursor-pointer focus-visible:outline-none",
        active ? "z-[2] max-sm:max-w-[78%]" : "novel-figure-waiting z-[1] scale-[0.84] max-sm:hidden",
        active && character.hasAvatar && "cursor-zoom-in",
      )}
      data-active={active || undefined}
    >
      <span
        className={cn("novel-figure-body block size-full origin-bottom", breathing && "novel-figure-speaking")}
        onAnimationIteration={onAnimationIteration}
      >
        <Portrait
          id={character.id}
          name={character.name}
          hasImage={character.hasAvatar}
          rev={character.avatarRev}
          variant="card"
          cutout={true}
          className="novel-figure-picture size-full rounded-t-[40px]"
        />
      </span>
    </button>
  );
}

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
      <span className="font-heading text-ui-15 font-semibold tracking-[-0.01em]">{shortName(focus.name)}</span>
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

export function SpeakingDots({ name }: { name: string }) {
  const t = useT();
  return (
    <span className="flex items-center gap-1.5 text-ui-12 text-white/80">
      <span className="flex gap-0.5">
        <span className="size-1 animate-bounce rounded-full bg-white/90 [animation-delay:-0.3s]" />
        <span className="size-1 animate-bounce rounded-full bg-white/90 [animation-delay:-0.15s]" />
        <span className="size-1 animate-bounce rounded-full bg-white/90" />
      </span>
      {t("characters.chat.speaking", { name: shortName(name) })}
    </span>
  );
}

function NovelBackdrop({ wallpaper }: { wallpaper: string | null }) {
  return (
    <div className="novel-backdrop pointer-events-none absolute inset-0 -z-10 overflow-hidden bg-background" aria-hidden={true}>
      {wallpaper ? (
        <img src={wallpaper} alt="" className="absolute inset-0 size-full object-cover animate-in fade-in-0 duration-500" />
      ) : (
        // No wallpaper: the chat's own background, with a soft shadow under the cast.
        <div className="absolute inset-0 bg-[radial-gradient(70%_40%_at_50%_70%,rgb(0_0_0/0.06),transparent)] dark:bg-[radial-gradient(70%_40%_at_50%_70%,rgb(0_0_0/0.25),transparent)]" />
      )}
      {/* Seats the dialogue panel and keeps the header readable over any picture. */}
      <div className="absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-background/70 to-transparent" />
      <div className="absolute inset-x-0 bottom-0 h-[45%] bg-gradient-to-t from-background/60 to-transparent" />
    </div>
  );
}
