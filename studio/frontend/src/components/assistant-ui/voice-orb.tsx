// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { cn } from "@/lib/utils";
import type { FC } from "react";

const orbConfig = {
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
  speaking: {
    gradient: "radial-gradient(circle at 40% 35%, #38bdf8, #0284c7)",
    animation: "voice-orb-speak",
    duration: "0.8s",
    shadow: "0 0 32px 8px rgba(56,189,248,0.35)",
  },
} as const;

export const VoiceOrb: FC = () => {
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);

  const cfg = orbState ? orbConfig[orbState] : null;

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
          orbState
            ? "pointer-events-auto opacity-100"
            : "pointer-events-none opacity-0",
        )}
        aria-hidden
      >
        <div
          className="transition-all duration-500"
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
