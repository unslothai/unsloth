// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type LevelListener = (level: number) => void;
type FrameListener = (rawRms: number, now: number) => void;

const levelListeners = new Set<LevelListener>();

/** Subscribe to the live microphone level (0..1) during dictation. */
export function subscribeDictationLevel(listener: LevelListener): () => void {
  levelListeners.add(listener);
  return () => {
    levelListeners.delete(listener);
  };
}

function publishLevel(level: number): void {
  for (const listener of levelListeners) {
    listener(level);
  }
}

/**
 * Start a lightweight Web Audio meter for the shared recording waveform and
 * optional voice-activity detection. Returns an idempotent cleanup function.
 */
export function startDictationLevelMeter(
  source: MediaStream,
  onFrame?: FrameListener,
): () => void {
  let audioContext: AudioContext | null = null;
  let levelRaf = 0;
  let stopped = false;

  const stop = () => {
    if (stopped) {
      return;
    }
    stopped = true;
    if (levelRaf) {
      cancelAnimationFrame(levelRaf);
    }
    levelRaf = 0;
    audioContext?.close().catch(() => {
      // A closing or already-closed context is harmless.
    });
    audioContext = null;
    publishLevel(0);
  };

  try {
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctx) {
      return stop;
    }
    audioContext = new Ctx();
    audioContext.resume().catch(() => {
      // Some browsers resume automatically after microphone permission.
    });
    const node = audioContext.createMediaStreamSource(source);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    node.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      if (stopped) {
        return;
      }
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (const sample of data) {
        const value = (sample - 128) / 128;
        sum += value * value;
      }
      const rms = Math.sqrt(sum / data.length);
      onFrame?.(rms, performance.now());
      // Perceptual boost so normal speech remains clearly visible.
      publishLevel(Math.min(1, rms * 3.2));
      levelRaf = requestAnimationFrame(tick);
    };
    levelRaf = requestAnimationFrame(tick);
  } catch {
    // The waveform is cosmetic; recording can continue without Web Audio.
    stop();
  }

  return stop;
}
