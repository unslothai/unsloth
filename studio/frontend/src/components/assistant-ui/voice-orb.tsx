// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { cn } from "@/lib/utils";
import { XIcon } from "lucide-react";
import { useEffect, useRef, type FC } from "react";

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
  // Minimized: the loop keeps running (orbState stays set) but the full-screen
  // overlay is hidden so the chat is visible underneath.
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const setVoiceOrbCollapsed = useChatRuntimeStore((s) => s.setVoiceOrbCollapsed);
  const showOverlay = Boolean(orbState) && !collapsed;
  const overlayRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  const cfg = orbState ? orbConfig[orbState] : null;

  // Esc only minimizes the orb back to chat -- it does NOT stop voice mode. The
  // single place to turn voice off is the + menu's Voice toggle, so exiting the
  // orb view is never confused with ending the session. Listener attached only
  // while the overlay is showing.
  useEffect(() => {
    if (!showOverlay) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setVoiceOrbCollapsed(true);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [showOverlay, setVoiceOrbCollapsed]);

  // Focus containment: the overlay hides the chat/sidebar visually but they stay
  // in the DOM and keyboard-focusable, so Tab lands on them and pops their
  // tooltips over the orb. Start focus on the close button and bounce any focus
  // that escapes to hidden UI back to it; the composer dock (z-40) stays
  // reachable so you can still type.
  useEffect(() => {
    if (!showOverlay) return;
    closeRef.current?.focus();
    const onFocusIn = (e: FocusEvent) => {
      const t = e.target;
      if (!(t instanceof HTMLElement)) return;
      if (overlayRef.current?.contains(t)) return;
      if (t.closest(".aui-thread-composer-dock")) return;
      closeRef.current?.focus();
    };
    document.addEventListener("focusin", onFocusIn);
    return () => document.removeEventListener("focusin", onFocusIn);
  }, [showOverlay]);

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
        ref={overlayRef}
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
            ref={closeRef}
            type="button"
            onClick={() => setVoiceOrbCollapsed(true)}
            aria-label="Back to chat"
            title="Back to chat (voice keeps running)"
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
