// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { cn } from "@/lib/utils";
import {
  AudioLinesIcon,
  LoaderCircleIcon,
  MicIcon,
  SparklesIcon,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, type FC } from "react";

export const orbConfig = {
  listening: {
    gradient: "radial-gradient(circle at 40% 35%, #34d399, #059669)",
    animation: "voice-orb-breathe",
    duration: "3s",
    shadow: "0 0 32px 8px rgba(52,211,153,0.35)",
  },
  // Salmon pink: the mic is actively hearing your voice (distinct from the idle
  // green "listening"). A quicker breathe so it reads as responsive/live.
  hearing: {
    gradient: "radial-gradient(circle at 40% 35%, #ffb0a0, #fb7185)",
    animation: "voice-orb-breathe",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(251,113,133,0.4)",
  },
  // Grey-blue slow pulse: a model is loading / warming (voice slot or Whisper),
  // so it clearly reads as "not ready yet" -- distinct from the idle grey and the
  // ready green. Separate from lilac, which now means only TTS generating speech.
  loading: {
    gradient: "radial-gradient(circle at 40% 35%, #94a3b8, #475569)",
    animation: "voice-orb-pulse",
    duration: "2s",
    shadow: "0 0 26px 6px rgba(148,163,184,0.3)",
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

type OrbStateName = keyof typeof orbConfig;

// Grey gradient for the mini orb when the voice loop isn't running yet.
export const ORB_IDLE_GRADIENT =
  "radial-gradient(circle at 40% 35%, #9ca3af, #4b5563)";

// The full-overlay orb is a dark 3D glass sphere. Each state carves an icon into
// it as a recessed "black hole" (near-black fill + a light bottom rim so the
// shape reads as pressed into the surface) and shows a one-word caption below.
const orbMeta: Record<OrbStateName, { label: string; icon: IconKind }> = {
  listening: { label: "Listening", icon: "mic" },
  hearing: { label: "Hearing you", icon: "wave" },
  loading: { label: "Warming up", icon: "spinner" },
  thinking: { label: "Thinking", icon: "dots" },
  synthesizing: { label: "Generating", icon: "sparkle" },
  speaking: { label: "Speaking", icon: "wave" },
};

type IconKind = "mic" | "wave" | "dots" | "spinner" | "sparkle";

// Deboss: near-black fill with a faint highlight below and a dark cut above, so
// the glyph looks carved into the sphere rather than sitting on top of it.
const HOLE = "#08080b";
const DEBOSS =
  "drop-shadow(0 1px 0.5px rgba(255,255,255,0.22)) drop-shadow(0 -1px 1.5px rgba(0,0,0,0.6))";

// Three dots that jump left-to-right in sequence (thinking). This is a vertical
// translate, not a scale, so the dots never grow or shrink independently -- their
// size stays locked to the ball's breathe.
const Dots: FC = () => (
  <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
    {[0, 1, 2].map((i) => (
      <motion.span
        key={i}
        style={{ width: 11, height: 11, borderRadius: "50%", background: HOLE, filter: DEBOSS }}
        animate={{ y: [0, -9, 0] }}
        transition={{ duration: 0.7, repeat: Infinity, ease: "easeInOut", delay: i * 0.15 }}
      />
    ))}
  </div>
);

const OrbIcon: FC<{ icon: IconKind }> = ({ icon }) => {
  // Soundwave (hearing / speaking): a static glyph -- the ball's own pulse
  // (faster/harder for speaking) supplies the motion, and the icon rides it, so
  // it grows and shrinks exactly with the sphere instead of on its own scaleY.
  if (icon === "wave")
    return (
      <AudioLinesIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
    );
  if (icon === "dots") return <Dots />;
  if (icon === "spinner")
    return (
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
        style={{ display: "grid", placeItems: "center" }}
      >
        <LoaderCircleIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
      </motion.div>
    );
  if (icon === "sparkle")
    return (
      // Rotate-only twinkle: no scale, so it never grows/shrinks on its own.
      <motion.div
        animate={{ rotate: [-8, 8, -8] }}
        transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
        style={{ display: "grid", placeItems: "center" }}
      >
        <SparklesIcon size={44} strokeWidth={2.3} color={HOLE} style={{ filter: DEBOSS }} />
      </motion.div>
    );
  // mic (listening): a resting state, so the icon gets NO independent animation.
  // It rides the ball's breathe scale only, so it grows and shrinks in perfect
  // sync with the sphere -- reading as carved into the surface, not floating.
  return <MicIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />;
};

export const VoiceOrb: FC = () => {
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);
  // Minimized: the loop keeps running (orbState stays set) but the full-screen
  // overlay is hidden so the chat is visible underneath.
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const setVoiceOrbCollapsed = useChatRuntimeStore((s) => s.setVoiceOrbCollapsed);
  const showOverlay = Boolean(orbState) && !collapsed;
  const overlayRef = useRef<HTMLDivElement>(null);

  const cfg = orbState ? orbConfig[orbState] : null;
  const meta = orbState ? orbMeta[orbState] : null;

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
          "absolute inset-0 z-30 flex flex-col items-center justify-center gap-8 bg-background",
          "transition-opacity duration-500",
          showOverlay
            ? "pointer-events-auto opacity-100"
            : "pointer-events-none opacity-0",
        )}
      >
        {/* Original sphere: a single radial-gradient ball with a soft glow that
            breathes/pulses via the CSS keyframes. The state icon is a near-black
            deboss ("door hole") carved into the surface. It lives INSIDE the ball,
            so it grows and shrinks with the sphere -- reading as part of the
            surface rather than floating on top. */}
        <div
          className="grid place-items-center transition-all duration-500"
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
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={meta?.icon ?? "none"}
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.85 }}
              transition={{ duration: 0.18 }}
              style={{ display: "grid", placeItems: "center" }}
            >
              {meta && <OrbIcon icon={meta.icon} />}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Live caption: names the current state so the color never needs a
            legend. Studio heading font (Hellix / Space Grotesk) to match chrome. */}
        <div className="h-5">
          <AnimatePresence mode="wait">
            {meta && (
              <motion.span
                key={meta.label}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.22 }}
                className="text-[13px] font-medium uppercase tracking-[0.28em] text-foreground/55"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {meta.label}
              </motion.span>
            )}
          </AnimatePresence>
        </div>
      </div>
    </>
  );
};
