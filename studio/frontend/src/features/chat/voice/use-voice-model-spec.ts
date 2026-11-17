// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type AudioOptionSpec,
  audioVoiceLabel,
  parseAudioOptions,
} from "@/features/audio/audio-options";
import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";
import { CHAT_MODEL_VOICE_SPEAKERS, isChatModelVoice } from "./voice-models";

// The family the chat model's own voice reports for its two speakers.
export const CHAT_MODEL_FAMILY = "chat_model";

export interface VoiceModelSpec {
  /** "unavailable": neither the loaded model nor its spec could be read (offline and not downloaded). */
  state: "loading" | "ready" | "unavailable";
  family: string | null;
  /** The model's built-in speakers, from its voice option. */
  speakers: string[];
  /** Every Speak option it declares, the voice included. */
  specs: AudioOptionSpec[];
}

interface SpecBody {
  audio_family?: string | null;
  audio_options?: unknown;
}

const LOADING: VoiceModelSpec = { state: "loading", family: null, speakers: [], specs: [] };
const UNAVAILABLE: VoiceModelSpec = { state: "unavailable", family: null, speakers: [], specs: [] };
// A spec only changes with a new download, so one read serves every panel open for a while.
const SPEC_TTL_MS = 5 * 60_000;
const specCache = new Map<string, { at: number; body: Promise<SpecBody | null> }>();

async function readJson(url: string): Promise<SpecBody | null> {
  const res = await authFetch(url);
  return res.ok ? ((await res.json()) as SpecBody) : null;
}

async function readSpec(modelId: string, loaded: boolean): Promise<SpecBody | null> {
  if (loaded) {
    // The loaded quant's own list: a quant can embed a different spec than the default one.
    const status = await readJson(`/api/inference/status?model=${encodeURIComponent(modelId)}`).catch(() => null);
    if (Array.isArray(status?.audio_options)) return status;
  }
  const key = modelId.toLowerCase();
  const hit = specCache.get(key);
  if (hit && Date.now() - hit.at < SPEC_TTL_MS) return hit.body;
  const body = readJson(
    `/api/inference/audio/model-options?model=${encodeURIComponent(modelId)}`,
  ).catch(() => null);
  specCache.set(key, { at: Date.now(), body });
  // A failure is not kept: the next open asks again (the model may have finished downloading).
  void body.then((value) => {
    if (value === null && specCache.get(key)?.body === body) specCache.delete(key);
  });
  return body;
}

function toSpec(body: SpecBody): VoiceModelSpec {
  const specs = parseAudioOptions(body.audio_options);
  const voice = specs.find((spec) => spec.name === "voice");
  return {
    state: "ready",
    family: body.audio_family ?? null,
    speakers: voice?.type === "enum" ? [...(voice.values ?? [])] : [],
    specs,
  };
}

const CHAT_MODEL_SPEC: VoiceModelSpec = {
  state: "ready",
  family: CHAT_MODEL_FAMILY,
  speakers: [...CHAT_MODEL_VOICE_SPEAKERS],
  specs: [],
};

/** A speech model's speakers and options, loaded or not, so voice mode can be set up before Start. */
export function useVoiceModelSpec(modelId: string, loaded: boolean): VoiceModelSpec {
  const key = `${loaded ? 1 : 0}:${modelId}`;
  // Keyed, so a switch of model shows "loading" at once instead of the last model's options.
  const [read, setRead] = useState<{ key: string; spec: VoiceModelSpec } | null>(null);
  const local = !modelId || isChatModelVoice(modelId);
  useEffect(() => {
    if (local) return;
    let cancelled = false;
    void readSpec(modelId, loaded).then((body) => {
      if (!cancelled) setRead({ key, spec: body ? toSpec(body) : UNAVAILABLE });
    });
    return () => {
      cancelled = true;
    };
  }, [key, local, modelId, loaded]);
  if (!modelId) return UNAVAILABLE;
  if (isChatModelVoice(modelId)) return CHAT_MODEL_SPEC;
  return read?.key === key ? read.spec : LOADING;
}

/** A speaker as people read it: the chat model's two by gender, an audio model's by its catalog name. */
export function voiceSpeakerLabel(
  speaker: string,
  family: string | null,
  t: (key: "voicePanel.speakerMale" | "voicePanel.speakerFemale") => string,
): string {
  if (family !== CHAT_MODEL_FAMILY) return audioVoiceLabel(speaker, family);
  return t(speaker === "male" ? "voicePanel.speakerMale" : "voicePanel.speakerFemale");
}
