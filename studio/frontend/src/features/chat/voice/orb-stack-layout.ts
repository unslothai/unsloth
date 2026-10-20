// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Where the full orb's stack sits under the ball: the status line, the decision chip and
 * the approval card, each a fixed gap below the one above it. The ball itself never moves;
 * with the decision arcs off their ring takes no room, so the stack starts under the ball's
 * own reach, and a hidden chip hands its slot to the card. Offsets are from the ball's
 * centre to each item's top, in CSS pixels.
 */
export const ORB_STACK_GAP = 32;
/** The decision ring's outer edge (voice-orb.tsx: radius 110, stroke 6.5). */
export const ORB_RING_EXTENT = 116.5;
/** The ball's widest wave: radius 70, the level swell 3 and a full band 22.4. */
export const ORB_BALL_EXTENT = 96;
export const ORB_STATUS_HEIGHT = 20;
export const ORB_CHIP_HEIGHT = 36;

export interface OrbStackLayout {
  status: number;
  chip: number | null;
  card: number | null;
}

export function orbStackLayout({
  arcs,
  chip,
  card,
}: { arcs: boolean; chip: boolean; card: boolean }): OrbStackLayout {
  const status = (arcs ? ORB_RING_EXTENT : ORB_BALL_EXTENT) + ORB_STACK_GAP;
  const chipTop = status + ORB_STATUS_HEIGHT + ORB_STACK_GAP;
  const cardTop = chip ? chipTop + ORB_CHIP_HEIGHT + ORB_STACK_GAP : chipTop;
  return { status, chip: chip ? chipTop : null, card: card ? cardTop : null };
}
