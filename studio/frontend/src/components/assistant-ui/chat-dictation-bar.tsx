// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { cancelActiveStudioDictation } from "@/features/chat/adapters/studio-dictation-adapter";
import { subscribeDictationLevel } from "@/features/chat/adapters/studio-model-dictation-adapter";
import { useAui, useAuiState } from "@assistant-ui/react";
import { CheckIcon, XIcon } from "lucide-react";
import { type FC, useEffect, useRef, useState } from "react";
import { TooltipIconButton } from "./tooltip-icon-button";

// ChatGPT keeps a dotted center line across the composer and only turns the
// most recent samples at the right edge into waveform bars.
const BAR_COUNT = 32;
const SAMPLE_INTERVAL_MS = 55;
const WAVE_BARS = Array.from(
  { length: BAR_COUNT },
  (_, index) => `wave-bar-${index}`,
);
// If no real mic level arrives for this long (e.g. the browser speech engine
// gives us no stream), fall back to a gentle idle shimmer so the bar is alive.
const IDLE_AFTER_MS = 350;

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
  const transcribingRef = useRef(false);
  const barsRef = useRef<number[]>(new Array(BAR_COUNT).fill(0));
  const rowRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    transcribingRef.current = transcribing;
  }, [transcribing]);

  useEffect(() => {
    if (!isDictating) {
      return;
    }

    let lastLevelAt = 0;
    let peak = 0;
    let idlePhase = 0;
    const barEls = rowRef.current
      ? Array.from(rowRef.current.children).filter(
          (el): el is HTMLElement => el instanceof HTMLElement,
        )
      : [];
    // Paint the bars imperatively (not via React state) so the live waveform
    // does not thrash renders. The spans carry no style prop, so React never
    // overwrites these transforms.
    for (const el of barEls) {
      el.style.transform = "scaleY(0.075)";
      el.style.opacity = "0.35";
    }

    // Push a new level onto the rolling waveform and paint it. Painting the DOM
    // directly (not React state) keeps the animation cheap.
    const push = (level: number) => {
      const bars = barsRef.current;
      bars.push(level);
      if (bars.length > BAR_COUNT) {
        bars.shift();
      }
      for (let i = 0; i < barEls.length; i++) {
        const v = bars[i] ?? 0;
        const scale = Math.max(0.075, Math.min(1, v ** 0.68));
        barEls[i].style.transform = `scaleY(${scale})`;
        barEls[i].style.opacity = `${0.35 + v * 0.65}`;
      }
    };

    const unsub = subscribeDictationLevel((level) => {
      // The analyser reports zero continuously after its MediaStream stops.
      // Ignore those frames so the transcription shimmer can take over.
      if (level > 0.01) {
        lastLevelAt = performance.now();
      }
      peak = Math.max(peak, level);
    });

    // Sample the analyser at a calmer rate than requestAnimationFrame. This
    // preserves enough history to form a readable waveform instead of racing
    // all speech off the left edge in half a second.
    const interval = window.setInterval(() => {
      const now = performance.now();
      let level = peak;
      peak = 0;
      if (now - lastLevelAt > IDLE_AFTER_MS) {
        idlePhase += transcribingRef.current ? 0.55 : 0.3;
        const amplitude = transcribingRef.current ? 0.16 : 0.08;
        level = 0.06 + amplitude * Math.abs(Math.sin(idlePhase));
      }
      push(level);
    }, SAMPLE_INTERVAL_MS);

    // Reset in cleanup (runs when dictation ends or on unmount) so the next
    // session starts fresh, without a synchronous setState in the effect body.
    return () => {
      unsub();
      window.clearInterval(interval);
      setTranscribing(false);
      transcribingRef.current = false;
      barsRef.current = new Array(BAR_COUNT).fill(0);
    };
  }, [isDictating]);

  if (!isDictating) {
    return null;
  }

  const discard = () => {
    cancelActiveStudioDictation();
  };

  const confirm = () => {
    setTranscribing(true);
    aui.composer().stopDictation();
  };

  return (
    <fieldset
      // order-2 places the bar in the input's slot, after the left "+" tools
      // (order 1) and matching ChatGPT's [+] [waveform] [X] [tick] layout.
      className="unsloth-dictation-bar order-2 flex min-w-0 flex-1 items-center gap-1.5"
      aria-label="Voice recording"
    >
      <div
        aria-hidden="true"
        className="unsloth-dictation-wave relative h-10 min-w-0 flex-1 overflow-hidden text-foreground"
      >
        <div className="unsloth-dictation-baseline absolute inset-0" />
        <div
          ref={rowRef}
          className="absolute inset-y-0 right-0 flex w-[clamp(10rem,28%,18rem)] items-center justify-end gap-[5px]"
        >
          {WAVE_BARS.map((barId) => (
            <span
              key={barId}
              className="h-9 w-[3px] shrink-0 origin-center scale-y-[0.075] rounded-full bg-foreground/80 opacity-35 transition-transform duration-75"
            />
          ))}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-0.5">
        <TooltipIconButton
          tooltip="Discard recording"
          aria-label="Discard recording"
          variant="ghost"
          onClick={discard}
          disabled={transcribing}
          className="size-9 rounded-full text-foreground/85 hover:text-foreground"
        >
          <XIcon className="size-[22px] stroke-[1.75px]" />
        </TooltipIconButton>
        <TooltipIconButton
          tooltip={transcribing ? "Transcribing…" : "Stop and transcribe"}
          aria-label="Stop and transcribe"
          variant="ghost"
          onClick={confirm}
          disabled={transcribing}
          className="size-9 rounded-full text-foreground/90 hover:text-foreground"
        >
          {transcribing ? (
            <Spinner className="size-4" />
          ) : (
            <CheckIcon className="size-[23px] stroke-[1.75px]" />
          )}
        </TooltipIconButton>
      </div>
    </fieldset>
  );
};
