// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The overlay that sits at the bottom of the screen while dictating from another
// app. It renders in its own tiny window (voice-hud.html) and is styled inline so
// it stays independent of the main app's theme.

import { listen } from "@tauri-apps/api/event";
import { useEffect, useState } from "react";
import { useT } from "@/i18n";

const BAR_COUNT = 9;

interface HudPayload {
  state?: string;
  level?: number;
}

const DOT_COLORS: Record<string, string> = {
  preparing: "#60a5fa",
  recording: "#f87171",
  transcribing: "#fbbf24",
  done: "#34d399",
};

const BAR_COLORS: Record<string, string> = {
  preparing: "#93c5fd",
  recording: "#fca5a5",
  transcribing: "#fcd34d",
  done: "#6ee7b7",
};

/** Center-weighted bars so the meter reads like a small waveform. */
function barHeights(level: number): number[] {
  const clamped = Math.max(0.03, Math.min(1, level));
  return Array.from({ length: BAR_COUNT }, (_, index) => {
    const center =
      1 - Math.abs(index - (BAR_COUNT - 1) / 2) / ((BAR_COUNT - 1) / 2);
    return 4 + clamped * (0.45 + 0.55 * center) * 22;
  });
}

export function VoiceTypingHud() {
  const t = useT();
  const [state, setState] = useState("");
  const [level, setLevel] = useState(0);

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

  const heights = barHeights(state === "recording" ? level : 0);
  const barColor = BAR_COLORS[state] ?? "#8b93a1";

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily:
          "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 16px",
          borderRadius: 999,
          background: "rgba(15, 17, 21, 0.88)",
          boxShadow:
            "0 10px 30px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255, 255, 255, 0.08)",
          color: "#e8eaee",
          fontSize: 12.5,
          fontWeight: 500,
          letterSpacing: 0.1,
        }}
      >
        <span
          className={state === "recording" ? "vt-pulse" : undefined}
          style={{
            width: 8,
            height: 8,
            borderRadius: 999,
            background: DOT_COLORS[state] ?? "#8b93a1",
            flexShrink: 0,
          }}
        />
        <span>{label}</span>
        <span
          style={{
            display: "flex",
            alignItems: "flex-end",
            gap: 3,
            height: 24,
            color: barColor,
          }}
        >
          {heights.map((height, index) => (
            <span
              key={index}
              className="vt-bar"
              style={{ width: 3, height: `${height}px` }}
            />
          ))}
        </span>
      </div>
    </div>
  );
}
