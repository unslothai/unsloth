// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PlayIcon, SquareIcon } from "lucide-react";
import { useEffect, useRef, useState, type FC, type KeyboardEvent, type PointerEvent } from "react";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  HOLD_TO_STOP_MS,
  createHoldToStop,
  type HoldPhase,
  type HoldToStop,
} from "@/features/chat/voice/hold-to-stop";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  requestVoicePauseToggle,
  requestVoiceStopSpeaking,
} from "@/features/chat/voice/voice-loop-bridge";
import { voiceTrace } from "@/features/chat/voice/voice-trace";
import { cn } from "@/lib/utils";

// The ring that fills while the button is held, drawn just inside the button's edge.
const SIZE = 36;
const RING_R = 15.5;
const RING_C = 2 * Math.PI * RING_R;

// Two slim bars: lucide's filled pause reads as a block at this size.
const PauseBars: FC = () => (
  <svg aria-hidden={true} viewBox="0 0 16 16" className="size-4" fill="currentColor">
    <rect x="3.5" y="2.5" width="2.6" height="11" rx="1.3" />
    <rect x="9.9" y="2.5" width="2.6" height="11" rx="1.3" />
  </svg>
);

/**
 * Voice mode's one composer control. Tap: pause or resume. Hold: a ring fills
 * over HOLD_TO_STOP_MS while the glyph turns into a square; release once it is
 * full to stop everything (reply, run, held speech) and listen again. Releasing
 * before it fills is a tap or nothing, never a stop.
 */
export const VoicePauseStopButton: FC = () => {
  const paused = useChatRuntimeStore((s) => s.voicePaused);
  const [phase, setPhase] = useState<HoldPhase>("idle");
  const holdRef = useRef<HoldToStop | null>(null);
  const pressedAtRef = useRef(0);
  if (!holdRef.current) holdRef.current = createHoldToStop(setPhase);
  useEffect(() => () => holdRef.current?.dispose(), []);

  const press = () => {
    pressedAtRef.current = performance.now();
    holdRef.current?.press();
  };
  const release = () => {
    const outcome = holdRef.current?.release() ?? "none";
    const heldMs = Math.round(performance.now() - pressedAtRef.current);
    voiceTrace("ui.holdStop", { outcome, heldMs });
    if (outcome === "tap") requestVoicePauseToggle();
    else if (outcome === "stop") requestVoiceStopSpeaking();
  };
  const cancel = () => holdRef.current?.cancel();

  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    press();
  };
  const onPointerUp = (e: PointerEvent<HTMLButtonElement>) => {
    if (e.button !== 0) return;
    release();
  };
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key !== " " && e.key !== "Enter") return;
    // The button's own click would fire on keyup and tap a second time.
    e.preventDefault();
    if (e.repeat) return;
    press();
  };
  const onKeyUp = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key !== " " && e.key !== "Enter") return;
    e.preventDefault();
    release();
  };

  const holding = phase !== "idle";
  const armed = phase === "armed";
  const label = paused ? "Resume" : "Pause";
  return (
    <Tooltip disableHoverableContent={true}>
      <TooltipTrigger asChild={true}>
        <Button
          type="button"
          variant="default"
          size="icon"
          aria-label={`${label}, hold to stop`}
          aria-pressed={paused}
          className={cn(
            "aui-composer-cancel relative ml-1.5 size-9 select-none overflow-hidden rounded-full transition-colors duration-150",
            armed && "bg-destructive text-destructive-foreground hover:bg-destructive",
          )}
          style={{ touchAction: "none", WebkitTouchCallout: "none" } as never}
          onPointerDown={onPointerDown}
          onPointerUp={onPointerUp}
          onPointerCancel={cancel}
          onLostPointerCapture={cancel}
          onKeyDown={onKeyDown}
          onKeyUp={onKeyUp}
          onBlur={cancel}
          onClick={(e) => e.preventDefault()}
        >
          {/* Both glyphs share the cell; the square fades in as the ring fills. */}
          <span className="relative grid size-full place-items-center">
            <span
              className="col-start-1 row-start-1 grid place-items-center transition-opacity"
              style={{
                opacity: holding ? 0 : 1,
                transitionDuration: `${holding ? HOLD_TO_STOP_MS : 150}ms`,
              }}
            >
              {paused ? (
                <PlayIcon className="size-4 fill-current" />
              ) : (
                <PauseBars />
              )}
            </span>
            <span
              className="col-start-1 row-start-1 grid place-items-center transition-opacity"
              style={{
                opacity: holding ? 1 : 0,
                transitionDuration: `${holding ? HOLD_TO_STOP_MS : 150}ms`,
              }}
            >
              <SquareIcon className="size-3 fill-current" />
            </span>
          </span>
          <svg
            aria-hidden={true}
            className="pointer-events-none absolute inset-0"
            viewBox={`0 0 ${SIZE} ${SIZE}`}
            width={SIZE}
            height={SIZE}
          >
            <circle
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={RING_R}
              fill="none"
              stroke="currentColor"
              strokeWidth={2.5}
              strokeLinecap="round"
              strokeDasharray={RING_C}
              strokeDashoffset={holding ? 0 : RING_C}
              transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}
              style={{
                opacity: holding ? 0.9 : 0,
                transition: holding
                  ? `stroke-dashoffset ${HOLD_TO_STOP_MS}ms linear, opacity 80ms`
                  : "stroke-dashoffset 200ms ease-out, opacity 200ms",
              }}
            />
          </svg>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {label} · hold to stop
      </TooltipContent>
    </Tooltip>
  );
};
