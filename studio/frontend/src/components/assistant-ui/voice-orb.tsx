// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { requestVoiceToggle } from "@/components/assistant-ui/thread";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { cn } from "@/lib/utils";
import { XIcon } from "lucide-react";
import { useEffect, type FC } from "react";

export const orbConfig = {
  listening: {
    gradient: "radial-gradient(circle at 40% 35%, #34d399, #059669)",
    animation: "voice-orb-breathe",
    duration: "3s",
    shadow: "0 0 32px 8px rgba(52,211,153,0.35)",
  },
  thinking: {
    gradient: "radial-gradient(circle at 40% 35%, #fbbf24, #d97706)",
    animation: "voice-orb-pulse",
    duration: "1s",
    shadow: "0 0 32px 8px rgba(251,191,36,0.35)",
  },
  synthesizing: {
    gradient: "radial-gradient(circle at 40% 35%, #c4b5fd, #8b5cf6)",
    animation: "voice-orb-pulse",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(167,139,250,0.35)",
  },
  speaking: {
    gradient: "radial-gradient(circle at 40% 35%, #38bdf8, #0284c7)",
    animation: "voice-orb-speak",
    duration: "0.8s",
    shadow: "0 0 32px 8px rgba(56,189,248,0.35)",
  },
} as const;

// Grey gradient for the mini orb when the voice loop isn't running yet.
export const ORB_IDLE_GRADIENT =
  "radial-gradient(circle at 40% 35%, #9ca3af, #4b5563)";

export const VoiceOrb: FC = () => {
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);
  // Minimized: the loop keeps running (orbState stays set, Esc still exits) but
  // the full-screen overlay is hidden so the chat is visible underneath.
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const showOverlay = Boolean(orbState) && !collapsed;

  const cfg = orbState ? orbConfig[orbState] : null;

  // Esc disables voice mode, but only while the orb is active — the listener is
  // attached only when orbState is set, so it never fires globally.
  useEffect(() => {
    if (!orbState) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        requestVoiceToggle();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [orbState]);

  return (
    <>
      <style>{`
        @keyframes voice-orb-breathe {
          0%, 100% { transform: scale(1);    }
          50%       { transform: scale(1.05); }
        }
        @keyframes voice-orb-pulse {
          0%, 100% { transform: scale(1);    }
          50%       { transform: scale(1.06); }
        }
        @keyframes voice-orb-speak {
          0%, 100% { transform: scale(1);    }
          50%       { transform: scale(1.1);  }
        }
      `}</style>

      <div
        className={cn(
          // When active, the backdrop must intercept pointer events so hovers
          // don't fall through to hidden chat history underneath (tooltips,
          // hover menus, side panels). The composer dock sits at z-40 above this
          // and keeps its own pointer-events-auto, so it stays interactive. When
          // inactive the backdrop is still mounted (opacity-0), so it must let
          // clicks pass through.
          "absolute inset-0 z-30 flex items-center justify-center bg-background",
          "transition-opacity duration-500",
          showOverlay
            ? "pointer-events-auto opacity-100"
            : "pointer-events-none opacity-0",
        )}
      >
        {showOverlay && (
          <button
            type="button"
            onClick={() => requestVoiceToggle()}
            aria-label="Exit voice mode"
            className={cn(
              "pointer-events-auto absolute right-4 top-4 flex size-9 items-center justify-center",
              "rounded-full text-foreground/90 transition-colors",
              "hover:bg-accent hover:text-foreground",
            )}
          >
            <XIcon className="size-5" strokeWidth={2.5} />
          </button>
        )}
        <div
          className="transition-all duration-500"
          aria-hidden
          style={{
            width: 140,
            height: 140,
            borderRadius: "50%",
            background: cfg?.gradient ?? orbConfig.listening.gradient,
            boxShadow: cfg?.shadow ?? "none",
            animation: cfg
              ? `${cfg.animation} ${cfg.duration} ease-in-out infinite`
              : undefined,
          }}
        />
      </div>
    </>
  );
};
