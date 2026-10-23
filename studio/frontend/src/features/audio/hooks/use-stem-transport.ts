// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

const DRIFT_TOLERANCE_S = 0.04;
const DRIFT_CHECK_MS = 250;
const CANPLAY_TIMEOUT_MS = 10_000;

export interface StemTransportSource {
  id: string;
  src: string | null;
}

export interface StemTransport {
  ready: boolean;
  playing: boolean;
  position: number;
  duration: number;
  play(): void;
  pause(): void;
  toggle(): void;
  seek(seconds: number): void;
}

interface StemVoice {
  element: HTMLAudioElement;
  source: MediaElementAudioSourceNode | null;
  gain: GainNode | null;
}

function waitForCanPlay(element: HTMLAudioElement): Promise<void> {
  const canPlay = () =>
    !element.seeking && element.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA;
  if (canPlay()) return Promise.resolve();
  const events = ["canplay", "canplaythrough", "seeked", "error"] as const;
  return new Promise((resolve) => {
    const finish = () => {
      window.clearTimeout(timer);
      for (const name of events) element.removeEventListener(name, check);
      resolve();
    };
    const check = () => {
      if (canPlay() || element.error) finish();
    };
    const timer = window.setTimeout(finish, CANPLAY_TIMEOUT_MS);
    for (const name of events) element.addEventListener(name, check);
  });
}

function setGain(voice: StemVoice, value: number, context: AudioContext) {
  voice.gain?.gain.setTargetAtTime(value, context.currentTime, 0.015);
}

/** Media elements through gain nodes, not decoded buffers, so memory stays near the blob size. */
export function useStemTransport({
  sources,
  gains,
  active,
  durationHint = 0,
}: {
  sources: readonly StemTransportSource[];
  gains: readonly number[];
  active: boolean;
  durationHint?: number;
}): StemTransport {
  const contextRef = useRef<AudioContext | null>(null);
  const voicesRef = useRef<StemVoice[]>([]);
  const gainsRef = useRef(gains);
  gainsRef.current = gains;
  const positionRef = useRef(0);
  const playingRef = useRef(false);
  // Lets a start still waiting on `canplay` know it is stale.
  const startToken = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [mediaDuration, setMediaDuration] = useState(0);

  const ready = sources.length > 0 && sources.every((source) => source.src);
  const srcKey = ready
    ? sources.map((source) => source.src ?? "").join("\u0000")
    : "";
  const duration =
    mediaDuration > 0 ? mediaDuration : Math.max(0, durationHint);
  const durationRef = useRef(duration);
  durationRef.current = duration;

  const updatePosition = useCallback((seconds: number) => {
    positionRef.current = seconds;
    setPosition(seconds);
  }, []);

  const stopAll = useCallback(() => {
    startToken.current += 1;
    playingRef.current = false;
    setPlaying(false);
    for (const voice of voicesRef.current) voice.element.pause();
  }, []);

  useEffect(() => {
    if (!srcKey) return;
    const voices: StemVoice[] = srcKey.split("\u0000").map((src) => {
      const element = new Audio();
      element.preload = "auto";
      element.src = src;
      return { element, source: null, gain: null };
    });
    voicesRef.current = voices;
    updatePosition(0);
    setMediaDuration(0);
    const first = voices[0]?.element;
    const onMetadata = () => {
      const lengths = voices
        .map((voice) => voice.element.duration)
        .filter((value) => Number.isFinite(value) && value > 0);
      if (lengths.length > 0) setMediaDuration(Math.max(...lengths));
    };
    const onEnded = () => {
      startToken.current += 1;
      playingRef.current = false;
      setPlaying(false);
      for (const voice of voices) voice.element.pause();
      updatePosition(0);
      for (const voice of voices) voice.element.currentTime = 0;
    };
    for (const voice of voices)
      voice.element.addEventListener("loadedmetadata", onMetadata);
    first?.addEventListener("ended", onEnded);
    return () => {
      startToken.current += 1;
      playingRef.current = false;
      setPlaying(false);
      first?.removeEventListener("ended", onEnded);
      for (const voice of voices) {
        voice.element.removeEventListener("loadedmetadata", onMetadata);
        voice.element.pause();
        voice.source?.disconnect();
        voice.gain?.disconnect();
        voice.element.removeAttribute("src");
        voice.element.load();
      }
      if (voicesRef.current === voices) voicesRef.current = [];
    };
  }, [srcKey, updatePosition]);

  useEffect(
    () => () => {
      const context = contextRef.current;
      contextRef.current = null;
      void context?.close().catch(() => {});
    },
    [],
  );

  // Created on the first Play, a user gesture, so the browser lets the context run.
  const ensureGraph = useCallback((): AudioContext => {
    let context = contextRef.current;
    if (!context) {
      context = new AudioContext();
      contextRef.current = context;
    }
    for (const [index, voice] of voicesRef.current.entries()) {
      if (voice.source) continue;
      voice.source = context.createMediaElementSource(voice.element);
      voice.gain = context.createGain();
      voice.gain.gain.value = gainsRef.current[index] ?? 1;
      voice.source.connect(voice.gain).connect(context.destination);
    }
    return context;
  }, []);

  const gainKey = gains.join(",");
  useEffect(() => {
    const context = contextRef.current;
    if (!context) return;
    const values = gainKey.split(",").map(Number);
    for (const [index, voice] of voicesRef.current.entries())
      setGain(voice, values[index] ?? 1, context);
  }, [gainKey]);

  const start = useCallback(
    async (from: number) => {
      const voices = voicesRef.current;
      if (voices.length === 0) return;
      const token = ++startToken.current;
      const context = ensureGraph();
      for (const voice of voices) voice.element.pause();
      for (const voice of voices) voice.element.currentTime = from;
      updatePosition(from);
      playingRef.current = true;
      setPlaying(true);
      try {
        await context.resume();
        await Promise.all(voices.map((voice) => waitForCanPlay(voice.element)));
        if (token !== startToken.current) return;
        await Promise.all(voices.map((voice) => voice.element.play()));
      } catch {
        if (token === startToken.current) stopAll();
      }
    },
    [ensureGraph, stopAll, updatePosition],
  );

  const play = useCallback(() => {
    if (playingRef.current) return;
    const from =
      positionRef.current >= durationRef.current - 0.05
        ? 0
        : positionRef.current;
    void start(from);
  }, [start]);

  const pause = useCallback(() => {
    if (!playingRef.current) return;
    const current = voicesRef.current[0]?.element.currentTime;
    stopAll();
    if (current !== undefined && Number.isFinite(current))
      updatePosition(current);
  }, [stopAll, updatePosition]);

  const toggle = useCallback(() => {
    if (playingRef.current) pause();
    else play();
  }, [pause, play]);

  const seek = useCallback(
    (seconds: number) => {
      const end = durationRef.current;
      const next = Math.min(end > 0 ? end : 0, Math.max(0, seconds));
      if (playingRef.current) {
        void start(next);
        return;
      }
      startToken.current += 1;
      updatePosition(next);
      for (const voice of voicesRef.current) voice.element.currentTime = next;
    },
    [start, updatePosition],
  );

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      const voices = voicesRef.current;
      const lead = voices[0]?.element;
      if (!lead || lead.paused || lead.seeking) return;
      const now = lead.currentTime;
      updatePosition(now);
      for (const voice of voices.slice(1)) {
        const element = voice.element;
        if (element.seeking || element.ended) continue;
        if (Math.abs(element.currentTime - now) > DRIFT_TOLERANCE_S)
          element.currentTime = now;
      }
    }, DRIFT_CHECK_MS);
    return () => window.clearInterval(timer);
  }, [playing, updatePosition]);

  useEffect(() => {
    if (!active) pause();
  }, [active, pause]);

  return {
    ready: Boolean(srcKey),
    playing,
    position,
    duration,
    play,
    pause,
    toggle,
    seek,
  };
}
