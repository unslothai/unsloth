// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Quick opacity + small slide on reveal. Deliberately avoids animating
// `height: auto`, whose per-frame remeasure caused the janky reflow / flashing
// the export panels used to show. Layout settles instantly; only opacity moves.
export const collapseAnim = {
  initial: { opacity: 0, y: -4 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0 },
  transition: { duration: 0.15, ease: "easeOut" as const },
};
