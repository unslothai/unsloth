// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { voiceOutputLevel } from "@/features/chat/hooks/use-tts-player";
import { cn } from "@/lib/utils";
import {
  AudioLinesIcon,
  AudioWaveformIcon,
  BrainIcon,
  LoaderCircleIcon,
  MicIcon,
  MicOffIcon,
  PencilLineIcon,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, type FC } from "react";

export const orbConfig = {
  waiting: {
    gradient:
      "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))",
    animation: "voice-orb-breathe",
    duration: "4.5s",
    shadow: "0 0 26px 6px var(--voice-orb-idle-glow)",
  },
  listening: {
    gradient: "radial-gradient(circle at 40% 35%, #34d399, #059669)",
    animation: "voice-orb-breathe",
    duration: "3s",
    shadow: "0 0 32px 8px rgba(52,211,153,0.35)",
  },
  hearing: {
    gradient: "radial-gradient(circle at 40% 35%, #ffb0a0, #fb7185)",
    animation: "voice-orb-breathe",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(251,113,133,0.4)",
  },
  loading: {
    gradient: "radial-gradient(circle at 40% 35%, #94a3b8, #475569)",
    animation: "voice-orb-pulse",
    duration: "2s",
    shadow: "0 0 26px 6px rgba(148,163,184,0.3)",
  },
  transcribing: {
    gradient: "radial-gradient(circle at 40% 35%, #e879f9, #c026d3)",
    animation: "voice-orb-pulse",
    duration: "1.1s",
    shadow: "0 0 30px 7px rgba(217,70,239,0.4)",
  },
  thinking: {
    gradient: "radial-gradient(circle at 40% 35%, #a5b4fc, #4f46e5)",
    animation: "voice-orb-breathe",
    duration: "2.4s",
    shadow: "0 0 32px 8px rgba(129,140,248,0.35)",
  },
  generating: {
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

export const ORB_IDLE_GRADIENT =
  "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))";
export const ORB_IDLE_SHADOW = "0 0 26px 6px var(--voice-orb-idle-glow)";

const orbMeta: Record<OrbStateName, { label: string; icon: IconKind }> = {
  waiting: { label: "Select a model", icon: "mic-off" },
  listening: { label: "Listening", icon: "mic" },
  hearing: { label: "Hearing you", icon: "wave" },
  loading: { label: "Warming up", icon: "spinner" },
  transcribing: { label: "Transcribing", icon: "pencil" },
  thinking: { label: "Thinking", icon: "brain" },
  generating: { label: "Writing the answer", icon: "dots" },
  synthesizing: { label: "Generating voice", icon: "waveform" },
  speaking: { label: "Speaking", icon: "bars-live" },
};

type IconKind =
  | "mic"
  | "mic-off"
  | "wave"
  | "bars-live"
  | "dots"
  | "brain"
  | "spinner"
  | "pencil"
  | "waveform";

const HOLE = "#08080b";
const DEBOSS =
  "drop-shadow(0 1px 0.5px rgba(255,255,255,0.22)) drop-shadow(0 -1px 1.5px rgba(0,0,0,0.6))";
// box-shadow, not drop-shadow: WebKit leaves ghost bands when a filtered element's transform changes per frame.
const DEBOSS_BOX =
  "0 1px 0.5px rgba(255,255,255,0.22), 0 -1px 1.5px rgba(0,0,0,0.6)";

const Dots: FC = () => (
  <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
    {[0, 1, 2].map((i) => (
      <motion.span
        key={i}
        style={{ width: 11, height: 11, borderRadius: "50%", background: HOLE, boxShadow: DEBOSS_BOX }}
        animate={{ y: [0, -9, 0] }}
        transition={{ duration: 0.7, repeat: Infinity, ease: "easeInOut", delay: i * 0.15 }}
      />
    ))}
  </div>
);

const SPEAK_GAIN = 5.5;
const SpeakingBars: FC = () => {
  const refs = useRef<Array<HTMLSpanElement | null>>([]);
  useEffect(() => {
    const weights = [0.55, 0.85, 1, 0.82, 0.5];
    const smooth = [0, 0, 0, 0, 0];
    let raf = 0;
    const tick = () => {
      const lvl = Math.min(1, voiceOutputLevel.current * SPEAK_GAIN);
      const t = performance.now() / 1000;
      for (let i = 0; i < refs.current.length; i++) {
        const shimmer = 0.78 + 0.22 * Math.sin(t * 9 + i * 1.7);
        const target = Math.min(1, 0.16 + lvl * weights[i] * shimmer);
        smooth[i] += (target - smooth[i]) * 0.4;
        const el = refs.current[i];
        if (el) el.style.transform = `scaleY(${Math.max(0.14, smooth[i])})`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, height: 40 }}>
      {[0, 1, 2, 3, 4].map((i) => (
        <span
          key={i}
          ref={(el) => {
            refs.current[i] = el;
          }}
          style={{
            width: 6,
            height: 40,
            borderRadius: 3,
            background: HOLE,
            transformOrigin: "center",
            boxShadow: DEBOSS_BOX,
            willChange: "transform",
          }}
        />
      ))}
    </div>
  );
};

const OrbIcon: FC<{ icon: IconKind }> = ({ icon }) => {
  if (icon === "bars-live") return <SpeakingBars />;
  if (icon === "wave")
    return (
      <AudioLinesIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
    );
  if (icon === "dots") return <Dots />;
  if (icon === "brain")
    return <BrainIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />;
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
  if (icon === "pencil")
    return (
      <PencilLineIcon size={44} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
    );
  if (icon === "waveform")
    return (
      <AudioWaveformIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
    );
  if (icon === "mic-off")
    return (
      <MicOffIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
    );
  return <MicIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />;
};

export const VoiceOrb: FC = () => {
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const setVoiceOrbCollapsed = useChatRuntimeStore((s) => s.setVoiceOrbCollapsed);
  const showOverlay = Boolean(orbState) && !collapsed;

  const cfg = orbState ? orbConfig[orbState] : null;
  const meta = orbState ? orbMeta[orbState] : null;

  useEffect(() => {
    if (!showOverlay) return;
    // Window capture, before the composer's cancelOnEscape: Escape hides the orb, it never stops the reply.
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !e.isComposing) {
        e.preventDefault();
        e.stopPropagation();
        setVoiceOrbCollapsed(true);
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
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
        className={cn(
          "absolute inset-0 z-30 flex flex-col items-center justify-center gap-8 bg-background",
          "transition-opacity duration-200 ease-out will-change-[opacity]",
          showOverlay
            ? "pointer-events-auto opacity-100"
            : "pointer-events-none opacity-0",
        )}
      >
        <div
          // Not `all`: transitioning transform fights the breathe keyframe.
          className="grid place-items-center transition-[background,box-shadow] duration-300"
          aria-hidden
          style={{
            width: 140,
            height: 140,
            borderRadius: "50%",
            background: cfg?.gradient ?? ORB_IDLE_GRADIENT,
            boxShadow: cfg?.shadow ?? "none",
            // Only while on screen: a minimized orb kept repainting and made the swap back to chat stutter.
            animation:
              cfg && showOverlay
                ? `${cfg.animation} ${cfg.duration} ease-in-out infinite`
                : undefined,
          }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={showOverlay ? (meta?.icon ?? "none") : "hidden"}
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

        <div className="h-5">
          <AnimatePresence mode="wait">
            {meta && (
              <motion.span
                key={meta.label}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.22 }}
                className="text-ui-13 font-medium uppercase tracking-[0.28em] text-foreground/55"
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
