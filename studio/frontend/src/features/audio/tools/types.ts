// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AudioConvertCaps } from "@/features/chat/types/api";
import type { ComponentType } from "react";
import type { AudioOptionSpec, AudioOptionValues } from "../audio-options";
import type {
  AudioSourceRef,
  ConvertMode,
  ConvertStyle,
} from "../audio-run-request";
import type { AudioWorkflowId } from "../workflows";

export type AudioReferenceTextMode = "required" | "optional" | "unused";

export interface AudioModelContext {
  audioType: string | null;
  audioFamily: string | null;
  musicGeneration: boolean;
  cudaMusicGeneration: boolean;
  /** MiniMax Music 3 and YuE2 need a description beside the lyrics. */
  musicNeedsDescription: boolean;
  audioWorkflows?: readonly string[];
  requiredInputs?: readonly string[];
  referenceTextMode?: AudioReferenceTextMode | null;
  convert?: AudioConvertCaps | null;
  convertMode?: ConvertMode;
  audioMusic?: boolean;
  /** Ids of the saved voices, once the list has loaded. */
  savedVoiceIds?: readonly string[] | null;
}

export interface AudioRunPatch {
  instructions?: string;
  language?: string;
  options?: AudioOptionValues;
  inputs?: { reference?: AudioSourceRef; emotion?: AudioSourceRef };
  route?: string;
  text?: string;
  speed?: number;
  referenceTextMode?: "required" | "optional" | "hidden";
  /** `route` mirrors options.route so the page can tell a run reloads the model. */
  convert?: { style?: ConvertStyle; route?: string };
}

export interface CoreInputs {
  text: string;
  referenceText?: string;
  hasReference?: boolean;
  edit?: EditCoreInputs;
}

export interface EditCoreInputs {
  transcript: string;
  edited: string;
  mode: "words" | "delivery";
  speed: number;
  pitchSteps: number;
}

export interface AudioToolPanelProps<V> {
  value: V;
  onChange: (value: V) => void;
  specs: AudioOptionSpec[];
  disabled: boolean;
  ctx: AudioModelContext;
  core?: CoreInputs;
}

export interface AudioToolPanel<V> {
  id: string;
  /** Runtime families the panel serves. Ignored when `appliesTo` is given. */
  families: readonly string[];
  workflows: readonly AudioWorkflowId[];
  title: string;
  claims: readonly string[];
  /** Matches on more than the family, e.g. on audio_type for native runtimes. */
  appliesTo?: (ctx: AudioModelContext) => boolean;
  initial: (specs: AudioOptionSpec[]) => V;
  Component: ComponentType<AudioToolPanelProps<V>>;
  toRequest: (value: V, ctx?: AudioModelContext) => AudioRunPatch;
  validate?: (
    value: V,
    core: CoreInputs,
    ctx: AudioModelContext,
  ) => string | null;
}

// biome-ignore lint/suspicious/noExplicitAny: each panel owns its value shape; the host only passes it through.
export type AnyAudioToolPanel = AudioToolPanel<any>;
