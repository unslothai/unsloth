// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** MiniMax-H3's combined budget across picture, video and standalone-audio references. */
export const MAX_H3_REFERENCES = 12;

export function hasReferenceCapacity(
  images: number,
  videos: number,
  audios: number,
): boolean {
  return images + videos + audios < MAX_H3_REFERENCES;
}
