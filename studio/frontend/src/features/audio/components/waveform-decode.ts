// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { computePeaks } from "./waveform-peaks";

/** Above this, decoding a waveform costs more memory than the bars are worth; the card stays flat. */
const DECODE_MAX_BYTES = 60 * 1024 * 1024;

/**
 * Bar heights and length for a clip, from its bytes. Decodes the Blob itself: fetching a blob: URL is
 * blocked by the page's connect-src. Returns null peaks when the browser cannot decode it in Web Audio.
 */
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
    // decodeAudioData decodes at the context's rate; the rate only affects the sample count.
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
