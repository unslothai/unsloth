// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { motion } from "motion/react";
import type { Rect } from "../types";

type SpotlightOverlayProps = {
  rect: Rect | null;
  vw: number;
  vh: number;
  maskId: string;
};

export function SpotlightOverlay({ rect, vw, vh, maskId }: SpotlightOverlayProps) {
  const hole = rect ?? { x: vw / 2 - 140, y: vh / 2 - 90, w: 280, h: 180 };
  const r = 22;

  return (
    <svg
      className="absolute inset-0 size-full"
      viewBox={`0 0 ${vw} ${vh}`}
      preserveAspectRatio="none"
      aria-hidden={true}
    >
      <defs>
        <radialGradient id={`${maskId}-v`} cx="50%" cy="45%" r="80%">
          <stop offset="0%" stopColor="rgba(6, 9, 15, 0.35)" />
          <stop offset="55%" stopColor="rgba(6, 9, 15, 0.65)" />
          <stop offset="100%" stopColor="rgba(6, 9, 15, 0.88)" />
        </radialGradient>
        <mask id={maskId}>
          <rect x="0" y="0" width={vw} height={vh} fill="white" />
          <motion.rect
            x={hole.x}
            y={hole.y}
            width={hole.w}
            height={hole.h}
            rx={r}
            fill="black"
            transition={{ type: "spring", stiffness: 260, damping: 30 }}
          />
        </mask>
      </defs>
      <rect
        x="0"
        y="0"
        width={vw}
        height={vh}
        fill={`url(#${maskId}-v)`}
        mask={`url(#${maskId})`}
      />
    </svg>
  );
}

