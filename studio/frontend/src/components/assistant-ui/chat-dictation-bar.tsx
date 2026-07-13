// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cancelActiveStudioDictation } from "@/features/chat/adapters/studio-dictation-adapter";
import { subscribeDictationLevel } from "@/features/chat/adapters/studio-model-dictation-adapter";
import { useAui, useAuiState } from "@assistant-ui/react";
import { CheckIcon, XIcon } from "lucide-react";
import { type FC, useEffect, useRef, useState } from "react";
import { Spinner } from "@/components/ui/spinner";
import { TooltipIconButton } from "./tooltip-icon-button";

// Dense row of dots that rise into thin centered bars, ChatGPT-style. Many,
// thin dots read as a smooth waveform across any composer width.
const BAR_COUNT = 96;
// Peak multiple of a dot's height for the loudest audio (dot is ~3px tall).
const MAX_SCALE = 9;
// If no real mic level arrives for this long (e.g. the browser speech engine
// gives us no stream), fall back to a gentle idle shimmer so the bar is alive.
const IDLE_AFTER_MS = 350;

function formatElapsed(ms: number): string {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * ChatGPT-style recording UI, rendered in place of the composer input while
 * dictating: a live waveform in the middle with a discard (X) and a confirm
 * (tick) on the right, matching ChatGPT's layout. The tick transcribes the
 * recording; X throws it away and keeps any text already in the composer. The
 * left composer tools (the "+") stay visible alongside it.
 */
export const ChatDictationBar: FC = () => {
  const aui = useAui();
  const isDictating = useAuiState((s) => s.composer.dictation != null);
  const [transcribing, setTranscribing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const barsRef = useRef<number[]>(new Array(BAR_COUNT).fill(0));
  const rowRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isDictating) return;

    const startedAt = Date.now();
    let lastLevelAt = 0;
    const barEls = rowRef.current
      ? Array.from(rowRef.current.children).filter(
          (el): el is HTMLElement => el instanceof HTMLElement,
        )
      : [];
    // Paint the bars imperatively (not via React state) so the ~60fps waveform
    // does not thrash renders. The spans carry no style prop, so React never
    // overwrites these transforms when the elapsed timer re-renders. At rest
    // each span is a small round dot (scaleY 1); louder audio scales it up into
    // a thin centered bar.
    for (const el of barEls) {
      el.style.transform = "scaleY(1)";
      el.style.opacity = "0.3";
    }

    // Push a new level onto the rolling waveform and paint it. Painting the DOM
    // directly (not React state) keeps the ~60fps animation cheap.
    const push = (level: number) => {
      const bars = barsRef.current;
      bars.push(level);
      if (bars.length > BAR_COUNT) bars.shift();
      for (let i = 0; i < barEls.length; i++) {
        const v = bars[i] ?? 0;
        barEls[i].style.transform = `scaleY(${1 + v * (MAX_SCALE - 1)})`;
        barEls[i].style.opacity = `${0.3 + v * 0.7}`;
      }
    };

    const unsub = subscribeDictationLevel((level) => {
      lastLevelAt = Date.now();
      push(level);
    });

    // Drives the elapsed timer and the idle shimmer when no levels arrive.
    const interval = window.setInterval(() => {
      setElapsed(Date.now() - startedAt);
      if (Date.now() - lastLevelAt > IDLE_AFTER_MS) {
        const t = Date.now() / 260;
        push(0.12 + 0.1 * Math.abs(Math.sin(t)));
      }
    }, 90);

    // Reset in cleanup (runs when dictation ends or on unmount) so the next
    // session starts fresh, without a synchronous setState in the effect body.
    return () => {
      unsub();
      window.clearInterval(interval);
      setTranscribing(false);
      setElapsed(0);
      barsRef.current = new Array(BAR_COUNT).fill(0);
    };
  }, [isDictating]);

  if (!isDictating) return null;

  const discard = () => {
    cancelActiveStudioDictation();
  };

  const confirm = () => {
    setTranscribing(true);
    aui.composer().stopDictation();
  };

  return (
    <div
      // order-2 places the bar in the input's slot, after the left "+" tools
      // (order 1) and matching ChatGPT's [+] [waveform] [X] [tick] layout.
      className="unsloth-dictation-bar order-2 flex min-w-0 flex-1 items-center gap-2"
      role="group"
      aria-label="Voice recording"
    >
      <div
        ref={rowRef}
        aria-hidden="true"
        className="unsloth-dictation-wave flex h-8 min-w-0 flex-1 items-center justify-between overflow-hidden"
      >
        {Array.from({ length: BAR_COUNT }).map((_, i) => (
          <span
            key={`wave-bar-${i}`}
            className="h-[3px] w-[3px] shrink-0 origin-center rounded-full bg-foreground opacity-30 transition-transform duration-75"
          />
        ))}
      </div>
      <span className="shrink-0 tabular-nums text-sm text-muted-foreground">
        {formatElapsed(elapsed)}
      </span>
      <div className="flex shrink-0 items-center gap-1">
        <TooltipIconButton
          tooltip="Discard recording"
          aria-label="Discard recording"
          variant="ghost"
          onClick={discard}
          disabled={transcribing}
          className="size-8 rounded-full text-muted-foreground hover:text-foreground"
        >
          <XIcon className="size-5" />
        </TooltipIconButton>
        <TooltipIconButton
          tooltip={transcribing ? "Transcribing…" : "Stop and transcribe"}
          aria-label="Stop and transcribe"
          variant="default"
          onClick={confirm}
          disabled={transcribing}
          className="size-8 rounded-full"
        >
          {transcribing ? (
            <Spinner className="size-4" />
          ) : (
            <CheckIcon className="size-5" />
          )}
        </TooltipIconButton>
      </div>
    </div>
  );
};
