// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The overlay that sits at the bottom of the screen while dictating from another
// app. It renders in its own tiny window (voice-hud.html) and is styled inline so
// it stays independent of the main app's theme. The pill is deliberately compact:
// state dot, waveform, elapsed time and the Esc hint, with no transcript text.

import { useT } from "@/i18n";
import { listen } from "@tauri-apps/api/event";
import { type CSSProperties, useEffect, useRef, useState } from "react";

const BAR_COUNT = 7;
const BAR_IDS = ["b1", "b2", "b3", "b4", "b5", "b6", "b7"] as const;

interface HudPayload {
  state?: string;
  level?: number;
}

const ACCENTS: Record<string, string> = {
  preparing: "#93c5fd",
  recording: "#f87171",
  transcribing: "#fcd34d",
  done: "#6ee7b7",
};

/** Center-weighted bars so the meter reads like a small waveform. */
function barHeights(level: number, state: string): number[] {
  if (state === "done") return Array.from({ length: BAR_COUNT }, () => 5);
  if (state !== "recording") {
    return Array.from({ length: BAR_COUNT }, (_, index) =>
      index % 2 === 0 ? 14 : 8,
    );
  }
  const clamped = Math.max(0.03, Math.min(1, level));
  return Array.from({ length: BAR_COUNT }, (_, index) => {
    const center =
      1 - Math.abs(index - (BAR_COUNT - 1) / 2) / ((BAR_COUNT - 1) / 2);
    return 4 + clamped * (0.45 + 0.55 * center) * 20;
  });
}

function formatElapsed(milliseconds: number): string {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export function VoiceTypingHud() {
  const t = useT();
  const [state, setState] = useState("");
  const [level, setLevel] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const startedAt = useRef<number | null>(null);
  const frozenMs = useRef(0);

  useEffect(() => {
    const subscribing = listen<HudPayload>("voice-typing:hud", (event) => {
      const payload = event.payload;
      if (payload.state) {
        setState(payload.state);
        if (payload.state !== "recording") setLevel(0);
      }
      if (typeof payload.level === "number") setLevel(payload.level);
    });
    return () => {
      void subscribing.then((unlisten) => unlisten());
    };
  }, []);

  useEffect(() => {
    if (state === "recording") {
      startedAt.current ??= performance.now();
      const tick = () => {
        if (startedAt.current !== null) {
          setElapsedMs(performance.now() - startedAt.current);
        }
      };
      tick();
      const timer = window.setInterval(tick, 200);
      return () => window.clearInterval(timer);
    }
    if (state === "transcribing") {
      if (startedAt.current !== null) {
        frozenMs.current = performance.now() - startedAt.current;
        setElapsedMs(frozenMs.current);
      }
      return;
    }
    if (state === "done") {
      setElapsedMs(frozenMs.current);
      return;
    }
    startedAt.current = null;
    frozenMs.current = 0;
    setElapsedMs(0);
  }, [state]);

  const label =
    state === "preparing"
      ? t("settings.voice.voiceTyping.hudPreparing")
      : state === "recording"
        ? t("settings.voice.voiceTyping.hudListening")
        : state === "transcribing"
          ? t("settings.voice.voiceTyping.hudTranscribing")
          : state === "done"
            ? t("settings.voice.voiceTyping.hudInserted")
            : "";

  if (!label) {
    // Nothing recorded yet: keep the (transparent) window empty.
    return null;
  }

  const accent = ACCENTS[state] ?? "#8b93a1";
  const heights = barHeights(level, state);
  const active = state === "preparing" || state === "recording";

  return (
    <div className="vt-stage">
      <output
        className={`vt-pill vt-${state}`}
        style={{ "--vt-accent": accent } as CSSProperties}
        aria-label={label}
      >
        <span className="vt-dot" aria-hidden="true" />
        <span className="vt-wave" aria-hidden="true">
          {BAR_IDS.map((id, index) => (
            <span
              key={id}
              className="vt-bar"
              style={{
                height: `${heights[index]}px`,
                animationDelay: `${index * 70}ms`,
              }}
            />
          ))}
        </span>
        <span className="vt-time" aria-hidden="true">
          {state === "done" ? (
            <svg
              viewBox="0 0 16 16"
              className="vt-check"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M3.5 8.5 6.5 11.5 12.5 4.5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            formatElapsed(elapsedMs)
          )}
        </span>
        {active ? (
          <span className="vt-esc" aria-hidden="true">
            <kbd>Esc</kbd>
          </span>
        ) : null}
      </output>
    </div>
  );
}
