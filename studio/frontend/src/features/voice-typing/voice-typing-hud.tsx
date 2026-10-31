// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The overlay shown while dictating from another app. It renders in its own tiny
// transparent window (voice-hud.html). Fixed size: logo (hover to stop) on the
// left, a live voice animation in the centre and an MM:SS timer on the right.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { type CSSProperties, useEffect, useRef, useState } from "react";

// Per-bar periods (ms) taken from the reference animation, so bars never move in lockstep.
const BAR_PERIODS = [474, 433, 407, 458, 400, 427, 441, 419, 487, 442] as const;
const BAR_IDS = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"] as const;
const BAR_WEIGHTS = [0.55, 0.7, 0.85, 0.95, 1, 1, 0.92, 0.82, 0.68, 0.52] as const;
const MIN_H = 4;
const MAX_H = 24;

interface HudPayload {
  state?: string;
  level?: number;
  sublabel?: string | null;
}

const ACCENTS: Record<string, string> = {
  preparing: "#93c5fd",
  recording: "#f87171",
  transcribing: "#fcd34d",
  fixing: "#a78bfa",
  refining: "#67e8f9",
  blocked: "#ef4444",
};

function formatElapsed(milliseconds: number): string {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  return `${mm}:${String(seconds % 60).padStart(2, "0")}`;
}

export function VoiceTypingHud() {
  const [state, setState] = useState("");
  const [message, setMessage] = useState("");
  const [elapsedMs, setElapsedMs] = useState(0);
  const stateRef = useRef("");
  const levelRef = useRef(0);
  const targetLevelRef = useRef(0);
  const barRefs = useRef<(HTMLSpanElement | null)[]>([]);
  const startedAt = useRef<number | null>(null);
  const frozenMs = useRef(0);

  useEffect(() => {
    const subscribing = listen<HudPayload>("voice-typing:hud", (event) => {
      const payload = event.payload;
      if (payload.state) {
        stateRef.current = payload.state;
        setState(payload.state);
        setMessage(payload.sublabel ?? "");
        if (payload.state !== "recording") targetLevelRef.current = 0;
      }
      if (typeof payload.level === "number") {
        targetLevelRef.current = payload.level;
      }
    });
    return () => {
      void subscribing.then((unlisten) => unlisten());
    };
  }, []);

  // Real-time animation: bars follow the live microphone level while recording,
  // and travel as a wave while the model loads or the transcript is produced.
  useEffect(() => {
    if (!state) return;
    let frame = 0;
    const draw = (now: number) => {
      const current = stateRef.current;
      const target = targetLevelRef.current;
      // Fast attack, slower release: the meter reads as a natural voice.
      levelRef.current +=
        (target - levelRef.current) * (target > levelRef.current ? 0.45 : 0.12);
      const level = Math.pow(Math.min(1, levelRef.current * 1.4), 0.7);
      for (let i = 0; i < BAR_IDS.length; i += 1) {
        const bar = barRefs.current[i];
        if (!bar) continue;
        const phase = (now / BAR_PERIODS[i]) * Math.PI;
        let height: number;
        let opacity: number;
        if (current === "recording") {
          const wobble = 0.55 + 0.45 * Math.sin(phase);
          height = MIN_H + (MAX_H - MIN_H) * level * BAR_WEIGHTS[i] * wobble;
          opacity = 0.45 + 0.55 * Math.min(1, level * 1.5 + 0.1);
        } else {
          const speed = current === "transcribing" ? 260 : 380;
          const wave = 0.5 + 0.5 * Math.sin(now / speed - i * 0.62);
          height = MIN_H + (MAX_H * 0.6 - MIN_H) * wave * BAR_WEIGHTS[i];
          opacity = 0.4 + 0.6 * wave;
        }
        bar.style.height = `${height.toFixed(1)}px`;
        bar.style.opacity = opacity.toFixed(2);
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [state]);

  useEffect(() => {
    if (state === "recording") {
      startedAt.current ??= performance.now();
      const tick = () => {
        if (startedAt.current !== null) {
          setElapsedMs(performance.now() - startedAt.current);
        }
      };
      tick();
      const timer = window.setInterval(tick, 250);
      return () => window.clearInterval(timer);
    }
    if (
      state === "transcribing" ||
      state === "fixing" ||
      state === "refining"
    ) {
      if (startedAt.current !== null) {
        frozenMs.current = performance.now() - startedAt.current;
        setElapsedMs(frozenMs.current);
      }
      return;
    }
    if (state === "preparing") {
      startedAt.current = null;
      frozenMs.current = 0;
      setElapsedMs(0);
      return;
    }
    startedAt.current = null;
    frozenMs.current = 0;
    setElapsedMs(0);
  }, [state]);

  if (!ACCENTS[state]) {
    // Nothing active: keep the (transparent) window empty.
    return null;
  }

  const handleCancel = () => {
    stateRef.current = "";
    setState("");
    void invoke("voice_typing_cancel").catch(() => {});
  };

  return (
    <div className="vt-stage">
      <output
        className={`vt-pill vt-${state}`}
        style={{ "--vt-accent": ACCENTS[state] } as CSSProperties}
        aria-label={state}
      >
        <button
          type="button"
          className="vt-logo"
          onClick={handleCancel}
          title="Stop (Esc)"
          aria-label="Stop dictation"
        >
          <img src="/circle-logo-small.png" alt="" draggable={false} />
          <span className="vt-stop" aria-hidden="true">
            <svg viewBox="0 0 10 10" fill="currentColor">
              <rect x="1" y="1" width="8" height="8" rx="1.6" />
            </svg>
          </span>
        </button>
        {state === "blocked" ? (
          <span className="vt-blocked" role="alert">
            {message || "Abusive Language is Discouraged!"}
          </span>
        ) : (
          <>
            <span className="vt-wave" aria-hidden="true">
              {BAR_IDS.map((id, index) => (
                <span
                  key={id}
                  className="vt-bar"
                  ref={(node) => {
                    barRefs.current[index] = node;
                  }}
                />
              ))}
            </span>
            <span className="vt-time" aria-hidden="true">
              {formatElapsed(elapsedMs)}
            </span>
          </>
        )}
      </output>
    </div>
  );
}
