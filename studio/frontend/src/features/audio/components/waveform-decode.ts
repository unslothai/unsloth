// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { computePeaks } from "./waveform-peaks";

const DECODE_MAX_BYTES = 60 * 1024 * 1024;

/** Decodes the Blob itself: fetching a blob: URL is blocked by the page's connect-src. */
export async function decodePeaks(
  blob: Blob,
): Promise<{ peaks: number[] | null; durationS: number | null }> {
  if (blob.size > DECODE_MAX_BYTES) return { peaks: null, durationS: null };
  const Offline =
    window.OfflineAudioContext ||
    (
      window as unknown as {
        webkitOfflineAudioContext?: typeof OfflineAudioContext;
      }
    ).webkitOfflineAudioContext;
  if (!Offline) return { peaks: null, durationS: null };
  try {
    const context = new Offline(1, 1, 22050);
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const channels = Array.from(
      { length: buffer.numberOfChannels },
      (_, index) => buffer.getChannelData(index),
    );
    return { peaks: computePeaks(channels), durationS: buffer.duration };
  } catch {
    return { peaks: null, durationS: null };
  }
}
