// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { BlobUrlCache } from "@/lib/blob-url-cache";
import { useEffect, useState } from "react";
import { fetchAudioBlob } from "../api";
import { computePeaks } from "../components/waveform-peaks";

// Bar heights by clip id. Peaks are tiny, so they outlive the group that made them: going back
// to an earlier separation redraws at once. Insertion order is the LRU order.
const PEAKS_CAP = 64;
const peaksByClip = new Map<string, readonly number[]>();

function rememberPeaks(clipId: string, peaks: readonly number[]): void {
  peaksByClip.delete(clipId);
  peaksByClip.set(clipId, peaks);
  while (peaksByClip.size > PEAKS_CAP) {
    const oldest = peaksByClip.keys().next().value;
    if (oldest === undefined) break;
    peaksByClip.delete(oldest);
  }
}

function cachedPeaks(clipId: string): readonly number[] | undefined {
  const peaks = peaksByClip.get(clipId);
  if (peaks) rememberPeaks(clipId, peaks);
  return peaks;
}

// Peaks only need the shape, so decode at a low rate: a 3-minute stereo stem is about 32 MB
// as float at 22.05 kHz instead of about 64 MB at 44.1 kHz, and the buffer is dropped at once.
const PEAKS_DECODE_RATE = 22_050;

// From the Blob, not its object URL: the page's CSP (connect-src) refuses fetch() on blob: URLs.
async function decodePeaks(blob: Blob): Promise<number[]> {
  const bytes = await blob.arrayBuffer();
  const context = new OfflineAudioContext(1, 1, PEAKS_DECODE_RATE);
  const buffer = await context.decodeAudioData(bytes);
  const channels: Float32Array[] = [];
  for (let index = 0; index < buffer.numberOfChannels; index += 1)
    channels.push(buffer.getChannelData(index));
  return computePeaks(channels);
}

export interface StemSourceInput {
  clipId: string;
  /** The gallery file route. */
  url: string;
}

export interface StemSources {
  /** Object URLs by clip id, once fetched. Revoked when the group changes or on unmount. */
  srcById: Readonly<Record<string, string>>;
  /** Bar heights by clip id, once decoded. */
  peaksById: Readonly<Record<string, readonly number[]>>;
  /** Clips whose audio could not be fetched. */
  failedIds: readonly string[];
}

const EMPTY: StemSources = { srcById: {}, peaksById: {}, failedIds: [] };

function groupKey(stems: readonly StemSourceInput[]): string {
  return stems.map((stem) => `${stem.clipId}\u0000${stem.url}`).join("\u0001");
}

/** The audio and bar heights of one separation's stems. The gallery's object-URL cache is a
 *  budgeted LRU that would evict stems mid-playback (one 3-minute stem is about 32 MB), so the
 *  group gets its own cache with every stem pinned until the group changes. Peaks are decoded
 *  one stem at a time so only one decoded buffer is alive at once. */
export function useStemSources(stems: readonly StemSourceInput[]): StemSources {
  const key = groupKey(stems);
  const [state, setState] = useState<{ key: string } & StemSources>({
    key: "",
    ...EMPTY,
  });

  useEffect(() => {
    if (!key) return;
    const group: StemSourceInput[] = key.split("\u0001").map((entry) => {
      const [clipId, url] = entry.split("\u0000");
      return { clipId, url };
    });
    let cancelled = false;
    // Sized to the group: nothing is ever evicted while it is shown, so prune is never called.
    const cache = new BlobUrlCache(Number.POSITIVE_INFINITY);
    const peaksById: Record<string, readonly number[]> = {};
    const failedIds: string[] = [];
    // Each stem's bytes until its peaks are drawn; the object URL alone keeps them after that.
    const blobs = new Map<string, Blob>();
    for (const { clipId } of group) {
      const peaks = cachedPeaks(clipId);
      if (peaks) peaksById[clipId] = peaks;
    }
    const publish = () => {
      if (cancelled) return;
      setState({
        key,
        srcById: cache.toRecord(),
        peaksById: { ...peaksById },
        failedIds: [...failedIds],
      });
    };
    publish();

    const loadSrc = async ({ clipId, url }: StemSourceInput) => {
      try {
        const blob = await fetchAudioBlob(url);
        if (cancelled) return;
        cache.set(clipId, URL.createObjectURL(blob), blob.size);
        blobs.set(clipId, blob);
      } catch {
        if (cancelled) return;
        failedIds.push(clipId);
      }
      publish();
    };

    void (async () => {
      const loads = new Map(
        group.map((stem) => [stem.clipId, loadSrc(stem)] as const),
      );
      // In display order, one at a time.
      for (const { clipId } of group) {
        await loads.get(clipId);
        if (cancelled) return;
        const blob = blobs.get(clipId);
        blobs.delete(clipId);
        if (peaksById[clipId] || !blob) continue;
        try {
          const peaks = await decodePeaks(blob);
          rememberPeaks(clipId, peaks);
          if (cancelled) return;
          peaksById[clipId] = peaks;
          publish();
        } catch {
          // Undecodable here: the row keeps its flat placeholder and still plays.
        }
      }
    })();

    return () => {
      cancelled = true;
      cache.clear();
    };
  }, [key]);

  return state.key === key ? state : EMPTY;
}
