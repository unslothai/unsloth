// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBackgroundStore } from "@/features/chat-background";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useAuiState } from "@assistant-ui/react";
import { useObjectUrl } from "@/hooks/use-object-url";
import { useEffect, useRef, useState } from "react";
import { Portrait } from "../components/portrait";
import { viewAvatar } from "../images";
import { shortName } from "../prompt/speakers";
import { useStageFocus } from "../stage-focus";
import type { Character } from "../types";
import { useAskToReply } from "./ask-bridge";

// Enough figures to read as a scene; a bigger cast shows the speaker and the members nearest them.
const MAX_FIGURES = 4;

function shownCast(cast: Character[], focus: Character): Character[] {
  if (cast.length <= MAX_FIGURES) return cast;
  const at = cast.findIndex((c) => c.id === focus.id);
  const start = Math.min(Math.max(0, at - 1), cast.length - MAX_FIGURES);
  return cast.slice(start, start + MAX_FIGURES);
}

/** Visual novel view: speaker in front, the rest a step behind, over the chat wallpaper or a plain stage. */
export function NovelScene({ threadId, cast }: { threadId: string | null; cast: Character[] }) {
  const t = useT();
  const { focus, speaking } = useStageFocus(threadId, cast);
  // The chat background at full strength, not behind the chat's scrim.
  const wallpaper = useObjectUrl(useBackgroundStore((s) => s.value?.blob ?? null));
  const ask = useAskToReply();
  const running = useAuiState(({ thread }) => thread.isRunning);
  if (!focus) return null;
  const figures = shownCast(cast, focus);
  return (
    <>
      <NovelBackdrop wallpaper={wallpaper} />
      <section
        className={cn(
          "character-novel-figures pointer-events-none absolute inset-x-0 bottom-[34%] top-[var(--character-novel-strip-top)] z-0 flex items-end justify-center px-[3%] max-md:bottom-[44%] max-md:top-12",
          cast.length === 1 && "character-novel-figures-solo",
        )}
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
            named={cast.length > 1}
            onAsk={running ? undefined : () => ask(character.id, character.name)}
          />
        ))}
        {cast.length > 1 ? (
          <span role="status" className="sr-only">
            {speaking ? t("characters.chat.speaking", { name: focus.name }) : ""}
          </span>
        ) : null}
      </section>
    </>
  );
}

/** Lets the current breath finish before stopping, so the figure settles instead of snapping. */
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
  named,
  onAsk,
}: {
  character: Character;
  active: boolean;
  speaking: boolean;
  solo: boolean;
  named: boolean;
  onAsk?: () => void;
}) {
  const t = useT();
  const { breathing, onAnimationIteration } = useSettlingBreath(speaking);
  const viewing = active || solo;
  const onClick = viewing
    ? character.hasAvatar
      ? () => viewAvatar("character", character.id, character.name, character.avatarRev)
      : undefined
    : onAsk;
  return (
    <button
      type="button"
      onClick={onClick}
      // Nothing to do (no picture to view, or a reply running): not a control.
      disabled={!onClick}
      aria-label={viewing ? t("characters.actions.viewPicture") : t("characters.chat.replyNow", { name: character.name })}
      title={active ? character.name : t("characters.chat.replyNow", { name: character.name })}
      className={cn(
        "character-novel-figure pointer-events-auto relative -mx-[1.5%] aspect-[2/3] h-full min-w-0 max-w-[36%] shrink origin-bottom cursor-pointer rounded-t-[40px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default",
        active ? "z-[2] max-sm:max-w-[78%]" : "character-novel-figure-waiting z-[1] scale-[0.84] max-sm:hidden",
        active && character.hasAvatar && "cursor-zoom-in",
      )}
      data-active={active || undefined}
    >
      <span
        className={cn("character-novel-figure-body block size-full origin-bottom", breathing && "character-novel-figure-speaking")}
        onAnimationIteration={onAnimationIteration}
      >
        <Portrait
          subject={character}
          variant="card"
          cutout={true}
          className="character-novel-figure-picture size-full rounded-t-[40px]"
        />
      </span>
      {named && active ? (
        <span
          aria-hidden={true}
          className="character-novel-figure-name pointer-events-none absolute left-1/2 z-[3] flex -translate-x-1/2 items-center gap-2 whitespace-nowrap rounded-full bg-background/85 px-3.5 py-1 text-foreground shadow-[0_8px_20px_-10px_rgb(0_0_0/0.5)] ring-1 ring-border/60 backdrop-blur-md animate-in fade-in-0 duration-300"
          data-testid="character-novel-figure-name"
        >
          <span className="font-heading text-ui-14 font-semibold">{character.name}</span>
          {speaking ? <Dots className="bg-muted-foreground" /> : null}
        </span>
      ) : null}
    </button>
  );
}

function Dots({ className = "bg-white/90" }: { className?: string }) {
  return (
    <span className="flex gap-0.5" aria-hidden={true}>
      <span className={cn("size-1 animate-bounce rounded-full [animation-delay:-0.3s]", className)} />
      <span className={cn("size-1 animate-bounce rounded-full [animation-delay:-0.15s]", className)} />
      <span className={cn("size-1 animate-bounce rounded-full", className)} />
    </span>
  );
}

/** onPage: drawn over the page rather than a dark scrim. */
export function SpeakingDots({ name, onPage = false }: { name: string; onPage?: boolean }) {
  const t = useT();
  return (
    <span className={cn("flex items-center gap-1.5 text-ui-12", !onPage && "text-white/80")}>
      <Dots className={onPage ? "bg-muted-foreground" : undefined} />
      {t("characters.chat.speaking", { name: shortName(name) })}
    </span>
  );
}

function NovelBackdrop({ wallpaper }: { wallpaper: string | null }) {
  return (
    <div className="character-novel-backdrop pointer-events-none absolute inset-0 -z-10 overflow-hidden bg-background" aria-hidden={true}>
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
