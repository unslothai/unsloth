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

/** Whether a clone model needs the reference clip's transcript, as the status reports it. */
export type AudioReferenceTextMode = "required" | "optional" | "unused";

/** What a tool panel knows about the loaded model when deciding whether it applies. */
export interface AudioModelContext {
  audioType: string | null;
  audioFamily: string | null;
  musicGeneration: boolean;
  cudaMusicGeneration: boolean;
  /** MiniMax Music 3 and YuE2 need a description beside the lyrics. */
  musicNeedsDescription: boolean;
  /** The pages the loaded model runs on (status `audio_workflows`). Absent on older servers. */
  audioWorkflows?: readonly string[];
  /** Request inputs the model's spec marks required (status `audio_required_inputs`). */
  requiredInputs?: readonly string[];
  /** Status `audio_reference_text`: null when the model does not clone. */
  referenceTextMode?: AudioReferenceTextMode | null;
  convert?: AudioConvertCaps | null;
  convertMode?: ConvertMode;
}

/** The part of a generation request a panel contributes. */
export interface AudioRunPatch {
  instructions?: string;
  language?: string;
  options?: AudioOptionValues;
  inputs?: { reference?: AudioSourceRef; emotion?: AudioSourceRef };
  route?: string;
  text?: string;
  /** Top-level speech speed (F5). */
  speed?: number;
  /** A panel that changes whether the transcript is used, over what the model reports. */
  referenceTextMode?: "required" | "optional" | "hidden";
  /** `route` mirrors options.route so the page can tell a run reloads the model. */
  convert?: { style?: ConvertStyle; route?: string };
}

export interface CoreInputs {
  text: string;
  /** Clone: the transcript typed for the reference clip. */
  referenceText?: string;
  /** Clone: whether a reference clip is picked. */
  hasReference?: boolean;
}

export interface AudioToolPanelProps<V> {
  value: V;
  onChange: (value: V) => void;
  specs: AudioOptionSpec[];
  disabled: boolean;
  ctx: AudioModelContext;
  /** The page's own inputs, for panels that preview what they do to them. */
  core?: CoreInputs;
}

/** Model-specific controls shown between a page's own inputs and Advanced, only for models that have them. */
export interface AudioToolPanel<V> {
  id: string;
  /** Runtime families the panel serves. Ignored when `appliesTo` is given. */
  families: readonly string[];
  workflows: readonly AudioWorkflowId[];
  title: string;
  /** Spec option names the panel renders itself, so Advanced leaves them out. */
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

/** A panel of any value type, as the registry lists them. */
// biome-ignore lint/suspicious/noExplicitAny: each panel owns its value shape; the host only passes it through.
export type AnyAudioToolPanel = AudioToolPanel<any>;
