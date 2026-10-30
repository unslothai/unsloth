// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The overlay that sits at the bottom of the screen while dictating from another
// app. It renders in its own tiny window (voice-hud.html) and is styled inline so
// it stays independent of the main app's theme. The pill is deliberately compact:
// state dot, waveform, elapsed time and the Esc hint, with no transcript text.

import { listen } from "@tauri-apps/api/event";
import { type CSSProperties, useEffect, useRef, useState } from "react";

const BAR_COUNT = 9;
const BAR_IDS = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"] as const;
const BAR_WEIGHTS = [0.42, 0.62, 0.82, 0.95, 1.0, 0.92, 0.78, 0.58, 0.4] as const;

interface HudPayload {
  state?: string;
  level?: number;
  label?: string;
  sublabel?: string;
}

const ACCENTS: Record<string, string> = {
  preparing: "#93c5fd",
  recording: "#f87171",
  transcribing: "#fcd34d",
  done: "#6ee7b7",
};

const FALLBACK_LABELS: Record<string, string> = {
  preparing: "Getting ready…",
  recording: "Listening",
  transcribing: "Transcribing…",
  done: "Inserted",
};

/** Center-weighted bars so the meter reads like a natural voice waveform. */
function barHeights(level: number, state: string): number[] {
  if (state === "done") return Array.from({ length: BAR_COUNT }, () => 4);
  if (state !== "recording") {
    return Array.from({ length: BAR_COUNT }, (_, index) =>
      index % 2 === 0 ? 13 : 7,
    );
  }
  const clamped = Math.max(0.03, Math.min(1, level));
  return BAR_WEIGHTS.map((weight) => 4 + Math.pow(clamped, 0.75) * weight * 20);
}

function formatElapsed(milliseconds: number): string {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

/** Keep the tail of long live transcripts visible inside the compact pill. */
function tailPreview(text: string, maxChars = 34): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= maxChars) return clean;
  return `…${clean.slice(clean.length - maxChars + 1).trimStart()}`;
}

export function VoiceTypingHud() {
  const [state, setState] = useState("");
  const [level, setLevel] = useState(0);
  const [customLabel, setCustomLabel] = useState("");
  const [sublabel, setSublabel] = useState("");
  const [elapsedMs, setElapsedMs] = useState(0);
  const startedAt = useRef<number | null>(null);
  const frozenMs = useRef(0);

  useEffect(() => {
    const subscribing = listen<HudPayload>("voice-typing:hud", (event) => {
      const payload = event.payload;
      if (payload.state) {
        setState(payload.state);
        if (payload.state !== "recording") setLevel(0);
        if (payload.state === "preparing" || payload.state === "idle") {
          setSublabel("");
        }
      }
      if (typeof payload.label === "string") {
        setCustomLabel(payload.label);
      }
      if (payload.sublabel !== undefined) {
        setSublabel(payload.sublabel ?? "");
      }
      if (typeof payload.level === "number") {
        setLevel((prev) =>
          payload.level! > prev
            ? prev * 0.25 + payload.level! * 0.75
            : prev * 0.6 + payload.level! * 0.4,
        );
      }
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

  const fallback = FALLBACK_LABELS[state] ?? "";
  const label = customLabel || fallback;

  if (!fallback) {
    // Nothing recorded yet: keep the (transparent) window empty.
    return null;
  }

  const accent = ACCENTS[state] ?? "#8b93a1";
  const heights = barHeights(level, state);
  const active = state === "preparing" || state === "recording";
  const displayCopy = sublabel ? tailPreview(sublabel) : label;

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
                animationDelay: `${index * 60}ms`,
              }}
            />
          ))}
        </span>
        <span
          className={`vt-copy ${sublabel ? "vt-copy-live" : ""}`}
          aria-hidden="true"
        >
          {displayCopy}
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
