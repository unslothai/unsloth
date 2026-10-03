// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentType } from "react";
import type { AudioOptionSpec, AudioOptionValues } from "../audio-options";
import type { AudioWorkflowId } from "../workflows";

/** What a tool panel knows about the loaded model when deciding whether it applies. */
export interface AudioModelContext {
  audioType: string | null;
  audioFamily: string | null;
  musicGeneration: boolean;
  cudaMusicGeneration: boolean;
  /** MiniMax Music 3 and YuE2 need a description beside the lyrics. */
  musicNeedsDescription: boolean;
}

/** The part of a generation request a panel contributes. */
export interface AudioRunPatch {
  instructions?: string;
  language?: string;
  options?: AudioOptionValues;
  inputs?: Record<string, string>;
  route?: string;
  text?: string;
}

export interface CoreInputs {
  text: string;
}

export interface AudioToolPanelProps<V> {
  value: V;
  onChange: (value: V) => void;
  specs: AudioOptionSpec[];
  disabled: boolean;
  ctx: AudioModelContext;
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
  toRequest: (value: V) => AudioRunPatch;
  validate?: (value: V, core: CoreInputs, ctx: AudioModelContext) => string | null;
}
