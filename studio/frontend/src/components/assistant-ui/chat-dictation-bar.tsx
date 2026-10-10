// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  cancelActiveStudioDictation,
  subscribeDictationLevel,
} from "@/features/chat";
import { useAui, useAuiState } from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { type FC, useEffect, useRef, useState } from "react";
import { TooltipIconButton } from "./tooltip-icon-button";

const BAR_COUNT = 84;
// Peak height multiple (dot is 4px); stays under the 40px pill.
const MAX_SCALE = 8;
const PUSH_INTERVAL_MS = 165;
// No mic level for this long (e.g. no Web Audio): fall back to an idle shimmer.
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

/** Recording UI in place of the composer input. Escape, or stop pressed again while transcribing, discards. */
export const ChatDictationBar: FC<{
  onSend?: () => void;
  sendDisabled?: boolean;
}> = ({ onSend, sendDisabled }) => {
  const aui = useAui();
  const isDictating = useAuiState((s) => s.composer.dictation != null);
  const [transcribing, setTranscribing] = useState<"stop" | "send" | null>(
    null,
  );
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
    let peak = 0;
    let smoothed = 0;
    let lastLevelAt = 0;
    const barEls = rowRef.current
      ? Array.from(rowRef.current.children).filter(
          (el): el is HTMLElement => el instanceof HTMLElement,
        )
      : [];
    // Painted imperatively so React never overwrites these transforms on a timer re-render.
    for (const el of barEls) {
      el.style.transform = "scaleY(1)";
      el.style.opacity = "0.62";
    }

    const commitSample = () => {
      const bars = barsRef.current;
      let level = peak;
      peak = 0;
      if (Date.now() - lastLevelAt > IDLE_AFTER_MS) {
        level = 0.075 + 0.055 * (1 + Math.sin(Date.now() / 360));
      }
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

    // Keep the loudest level between advances so downsampling does not swallow peaks.
    const unsub = subscribeDictationLevel((level) => {
      if (level > peak) {
        peak = level;
      }
      lastLevelAt = Date.now();
    });

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

    // Reset in cleanup, not with a synchronous setState in the effect body.
    return () => {
      unsub();
      cancelAnimationFrame(raf);
      transcribingRef.current = false;
      setTranscribing(null);
      setElapsed(0);
      barsRef.current = new Array(BAR_COUNT + 1).fill(0);
    };
  }, [isDictating]);

  // No discard button, so Escape drops a recording; while transcribing it aborts the request.
  useEffect(() => {
    if (!isDictating) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      // defaultPrevented: an open dialog or menu already claimed this Escape.
      if (event.key !== "Escape" || event.defaultPrevented) {
        return;
      }
      cancelActiveStudioDictation();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isDictating]);

  if (!isDictating) {
    return null;
  }

  const freeze = (source: "stop" | "send") => {
    transcribingRef.current = true;
    setTranscribing(source);
  };

  // A second press while transcribing discards: touch has no Escape key.
  const stop = () => {
    if (transcribing !== null) {
      cancelActiveStudioDictation();
      return;
    }
    freeze("stop");
    aui.composer().stopDictation();
  };

  const send = () => {
    if (sendDisabled) return;
    if (!onSend) {
      stop();
      return;
    }
    freeze("send");
    onSend();
  };

  return (
    <fieldset
      // order-2 places the bar in the input's slot after the left "+" tools.
      className="unsloth-dictation-bar order-2 m-0 flex min-w-0 flex-1 items-center gap-0 border-0 p-0"
      aria-label="Voice recording"
    >
      <div
        ref={rowRef}
        aria-hidden="true"
        className="unsloth-dictation-wave grid h-10 min-w-0 flex-1 items-center overflow-hidden pl-4 pr-2"
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
      {/* Classed so narrow panes can tighten these: both are shrink-0 and set the bar's floor. */}
      <span className="unsloth-dictation-timer mr-4 shrink-0 tabular-nums text-sm text-muted-foreground">
        {formatElapsed(elapsed)}
      </span>
      <div className="unsloth-dictation-actions flex shrink-0 items-center gap-2.5">
        <TooltipIconButton
          type="button"
          tooltip={
            transcribing !== null ? "Cancel transcription" : "Stop recording"
          }
          aria-label={
            transcribing !== null ? "Cancel transcription" : "Stop recording"
          }
          variant="ghost"
          onClick={stop}
          // Neutral grey: --secondary is brand green on light and too close to --card on dark.
          className="size-9 rounded-full bg-accent text-foreground hover:bg-accent/70 dark:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.16*var(--contrast-wash-gain,1)))]"
        >
          {transcribing === "stop" ? (
            <Spinner className="size-3.5" />
          ) : (
            <SquareIcon className="size-3 fill-current" />
          )}
        </TooltipIconButton>
        <TooltipIconButton
          type="button"
          tooltip={transcribing === "send" ? "Transcribing…" : "Send message"}
          aria-label="Send message"
          variant="default"
          onClick={send}
          disabled={transcribing !== null || sendDisabled}
          className="aui-composer-send size-9 rounded-full"
        >
          {transcribing === "send" ? (
            <Spinner className="size-[calc(18px*var(--ui-space-scale,1))]" />
          ) : (
            <ArrowUpIcon className="unsloth-send-icon aui-composer-send-icon size-[calc(21px*var(--ui-space-scale,1))] stroke-2" />
          )}
        </TooltipIconButton>
      </div>
    </fieldset>
  );
};
