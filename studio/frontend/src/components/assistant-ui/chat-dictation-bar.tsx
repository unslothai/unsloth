// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  cancelActiveStudioDictation,
  subscribeDictationLevel,
} from "@/features/chat";
import { useAui, useAuiState } from "@assistant-ui/react";
import { CheckIcon, XIcon } from "lucide-react";
import { type FC, useEffect, useRef, useState } from "react";
import { TooltipIconButton } from "./tooltip-icon-button";

// Dense row of dots that rise into centered recording bars.
const BAR_COUNT = 84;
// Peak height multiple for the loudest audio (dot is 4px). Under the 40px pill
// so bars don't touch the edges.
const MAX_SCALE = 8;
// Time for a sample to slide one bar-width left (drift speed). Bars interpolate
// between samples each frame, so a slower interval stays smooth.
const PUSH_INTERVAL_MS = 165;
// If no real mic level arrives for this long (e.g. Web Audio unavailable), fall
// back to a gentle idle shimmer so the bar stays alive.
const IDLE_AFTER_MS = 450;
const WAVE_BAR_IDS = Array.from(
  { length: BAR_COUNT },
  (_, index) => `wave-bar-${index}`,
);

function formatElapsed(ms: number): string {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Recording UI shown in place of the composer input: a live waveform with
 * discard and confirm on the right. Confirm transcribes; discard keeps the
 * existing composer text.
 */
export const ChatDictationBar: FC = () => {
  const aui = useAui();
  const isDictating = useAuiState((s) => s.composer.dictation != null);
  const [transcribing, setTranscribing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const transcribingRef = useRef(false);
  // Extra slot: newest sample lands here; the last visible bar slides toward it.
  const barsRef = useRef<number[]>(new Array(BAR_COUNT + 1).fill(0));
  const rowRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isDictating) {
      return;
    }

    const startedAt = Date.now();
    let peak = 0; // loudest level seen since the last waveform advance
    let smoothed = 0;
    let lastLevelAt = 0;
    const barEls = rowRef.current
      ? Array.from(rowRef.current.children).filter(
          (el): el is HTMLElement => el instanceof HTMLElement,
        )
      : [];
    // Paint the bars imperatively (not via React state) so the waveform does not
    // thrash renders. The spans carry no style prop, so React never overwrites
    // these transforms on an elapsed-timer re-render. At rest each is a round dot
    // (scaleY 1); louder audio scales it into a thin centered bar.
    for (const el of barEls) {
      el.style.transform = "scaleY(1)";
      el.style.opacity = "0.62";
    }

    // Push one new sample, dropping the oldest.
    const commitSample = () => {
      const bars = barsRef.current;
      let level = peak;
      peak = 0;
      if (Date.now() - lastLevelAt > IDLE_AFTER_MS) {
        level = 0.075 + 0.055 * (1 + Math.sin(Date.now() / 360));
      }
      // Quiet speech needs a perceptual lift. Fast attack, slower release:
      // removes twitching while preserving clear peaks.
      const visual = Math.min(1, Math.max(0, level) ** 0.62 * 1.45);
      smoothed =
        visual >= smoothed
          ? smoothed * 0.2 + visual * 0.8
          : smoothed * 0.78 + visual * 0.22;
      bars.push(smoothed);
      while (bars.length > BAR_COUNT + 1) {
        bars.shift();
      }
    };

    // Keep the loudest mic level between advances so downsampling to
    // PUSH_INTERVAL_MS doesn't swallow peaks in quiet gaps.
    const unsub = subscribeDictationLevel((level) => {
      if (level > peak) {
        peak = level;
      }
      lastLevelAt = Date.now();
    });

    // Repaint every frame; bars interpolate between samples so the wave glides.
    // Timer updates at most once per second.
    let lastPushAt = performance.now();
    let shownSecond = -1;
    let raf = 0;

    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (transcribingRef.current) {
        return;
      }

      const now = performance.now();
      const bars = barsRef.current;
      let steps = 0;
      while (now - lastPushAt >= PUSH_INTERVAL_MS && steps < BAR_COUNT + 1) {
        commitSample();
        lastPushAt += PUSH_INTERVAL_MS;
        steps++;
      }
      // Drop stale backlog if rAF was paused (tab backgrounded).
      if (now - lastPushAt >= PUSH_INTERVAL_MS) {
        lastPushAt = now;
      }

      const phase = Math.min(1, (now - lastPushAt) / PUSH_INTERVAL_MS);
      for (let i = 0; i < barEls.length; i++) {
        // Bar i drifts toward its right neighbour as phase goes 0 to 1.
        const a = bars[i] ?? 0;
        const b = bars[i + 1] ?? a;
        const v = a + (b - a) * phase;
        barEls[i].style.transform = `scaleY(${1 + v * (MAX_SCALE - 1)})`;
        barEls[i].style.opacity = `${0.62 + v * 0.38}`;
      }

      const elapsedMs = Date.now() - startedAt;
      const second = Math.floor(elapsedMs / 1000);
      if (second !== shownSecond) {
        shownSecond = second;
        setElapsed(elapsedMs);
      }
    };
    raf = requestAnimationFrame(frame);

    // Reset in cleanup (dictation end or unmount) so the next session starts
    // fresh, without a synchronous setState in the effect body.
    return () => {
      unsub();
      cancelAnimationFrame(raf);
      transcribingRef.current = false;
      setTranscribing(false);
      setElapsed(0);
      barsRef.current = new Array(BAR_COUNT + 1).fill(0);
    };
  }, [isDictating]);

  if (!isDictating) {
    return null;
  }

  const discard = () => {
    cancelActiveStudioDictation();
  };

  const confirm = () => {
    // Freeze the timer + waveform before the transcription round trip completes
    // and the session ends.
    transcribingRef.current = true;
    setTranscribing(true);
    aui.composer().stopDictation();
  };

  return (
    <fieldset
      // order-2 places the bar in the input's slot after the left "+" tools.
      className="unsloth-dictation-bar order-2 m-0 flex min-w-0 flex-1 items-center gap-2 border-0 p-0"
      aria-label="Voice recording"
    >
      <div
        ref={rowRef}
        aria-hidden="true"
        className="unsloth-dictation-wave grid h-10 min-w-0 flex-1 items-center overflow-hidden px-2"
        style={{
          gridTemplateColumns: `repeat(${BAR_COUNT}, minmax(1px, 3px))`,
          justifyContent: "space-between",
        }}
      >
        {WAVE_BAR_IDS.map((barId) => (
          <span
            key={barId}
            className="h-1 w-full origin-center rounded-full bg-foreground opacity-[0.62] will-change-transform"
          />
        ))}
      </div>
      <span className="shrink-0 tabular-nums text-sm text-muted-foreground">
        {formatElapsed(elapsed)}
      </span>
      <div className="flex shrink-0 items-center gap-1">
        <TooltipIconButton
          type="button"
          tooltip="Discard recording"
          aria-label="Discard recording"
          variant="ghost"
          onClick={discard}
          className="size-8 rounded-full text-muted-foreground hover:text-foreground"
        >
          <XIcon className="size-5" />
        </TooltipIconButton>
        <TooltipIconButton
          type="button"
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
    </fieldset>
  );
};
