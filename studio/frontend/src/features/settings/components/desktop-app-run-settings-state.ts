// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The logic of Settings → Agents → Desktop apps → Run settings: three controls that are views
// of settings Studio already has, never copies of them.
//   Context window  → the model's remembered Context Length (Run settings, Remember).
//   MTP             → the model's remembered speculative decoding mode (Run settings).
//   Keep loaded     → "Unload after idle" (Settings → Model auto-switch), vetoed by
//                     Model memory → Keep resident.

import type { PerModelConfig } from "@/features/model-picker";
import type { OpenAIAutoSwitchSettings, OpenAIAutoSwitchUpdate } from "../api/openai-auto-switch";

/** What GET /api/settings/claude-desktop/model-settings answers (routes/desktop_app_models.py). */
export type DesktopAppModelSettings = {
  model: string;
  config_id: string | null;
  alias_id?: string | null;
  gguf_variant: string | null;
  is_gguf: boolean;
  custom_context_length: number | null;
  speculative_type: string | null;
  trained_context: number | null;
  mtp_available: boolean | null;
  guard_budget_bytes: number | null;
  guard_context: number | null;
  loaded: boolean;
  loaded_context: number | null;
};

/** Mirrors CLAUDE_MIN_CONTEXT / CODEX_MIN_CONTEXT in utils/claude_desktop.py: what one turn of
 *  each app needs, with room to answer. */
export const CLAUDE_MIN_CONTEXT = 65_536;
export const CODEX_MIN_CONTEXT = 32_768;
/** Mirrors AGENT_CONTEXT_MIN: the slider's bottom, and the loader's own fit floor. */
export const AGENT_CONTEXT_MIN = 8_192;
export const AGENT_CONTEXT_STEP = 1_024;

/** The two settings this panel reads from the per-model record. */
export type SharedModelFields = Pick<PerModelConfig, "customContextLength" | "speculativeType">;

/** The model's shared values: this browser's remembered record when there is one (what Run
 *  settings shows), else what the server would apply to a gateway load. */
export function sharedModelFields(
  local: SharedModelFields | null,
  info: DesktopAppModelSettings | null,
): SharedModelFields {
  if (local) {
    return {
      customContextLength: local.customContextLength ?? null,
      speculativeType: local.speculativeType ?? null,
    };
  }
  return {
    customContextLength: info?.custom_context_length ?? null,
    speculativeType: info?.speculative_type ?? null,
  };
}

export type ContextView = {
  /** Slider bounds. */
  min: number;
  max: number;
  /** Where the thumb sits: the pin, or the full trained context for Auto. */
  value: number;
  /** No pin: a gateway load asks for the full trained context (max_seq_length 0). */
  auto: boolean;
  /** What a load will actually get: the pin, or the full context capped by the memory guard. */
  effective: number;
  /** The memory guard caps the full context on this machine now. */
  guardCapped: boolean;
  /** About how much the guard lets through, when it caps. */
  guardContext: number | null;
  belowClaude: boolean;
  belowCodex: boolean;
};

export function contextView(
  fields: SharedModelFields,
  info: DesktopAppModelSettings | null,
): ContextView | null {
  const trained = info?.trained_context ?? null;
  if (!trained || trained <= 0) {
    return null;
  }
  const min = Math.min(AGENT_CONTEXT_MIN, trained);
  const pin = fields.customContextLength;
  const auto = pin == null || pin <= 0;
  const value = auto ? trained : Math.max(min, Math.min(trained, pin));
  const guardContext = info?.guard_context ?? null;
  const guardCapped = guardContext != null && guardContext < value;
  const effective = guardCapped && auto ? guardContext : value;
  return {
    min,
    max: trained,
    value,
    auto,
    effective,
    guardCapped,
    guardContext,
    belowClaude: effective < CLAUDE_MIN_CONTEXT,
    belowCodex: effective < CODEX_MIN_CONTEXT,
  };
}

/** The per-model patch for a slider position. The top is Auto, not a pin of the trained
 *  length: Auto is "full context, as much as the memory guard allows", which is what a pin of
 *  the maximum means to a user and what keeps working when memory frees up. */
export function contextPatch(value: number | null, trained: number): Partial<PerModelConfig> {
  if (value == null || value >= trained) {
    return { customContextLength: null };
  }
  const snapped = Math.max(
    Math.min(AGENT_CONTEXT_MIN, trained),
    Math.round(value / AGENT_CONTEXT_STEP) * AGENT_CONTEXT_STEP,
  );
  return { customContextLength: Math.min(snapped, trained) };
}

const MTP_MODES = new Set(["mtp", "mtp+ngram"]);

export type MtpView = {
  /** The model has embedded MTP heads (null: unknown). */
  available: boolean | null;
  on: boolean;
  /** Still the Auto mode: Studio decides, and runs MTP when the heads are there. */
  auto: boolean;
};

export function mtpView(fields: SharedModelFields, info: DesktopAppModelSettings | null): MtpView {
  const mode = fields.speculativeType;
  const available = info?.mtp_available ?? null;
  const auto = mode == null || mode === "auto";
  return {
    available,
    auto,
    on: auto ? available === true : MTP_MODES.has(mode),
  };
}

export function mtpPatch(on: boolean): Partial<PerModelConfig> {
  return { speculativeType: on ? "mtp" : "off" };
}

/** "Keep loaded" choices in seconds; 0 keeps the model loaded until something else replaces it. */
export const KEEP_ALIVE_CHOICES = [0, 300, 900, 1800, 3600] as const;

export type KeepAliveView = {
  /** Idle seconds before an unload, 0 = never. */
  seconds: number;
  /** Model memory → Keep resident is on: nothing unloads, whatever is set here. */
  vetoed: boolean;
};

export function keepAliveView(
  settings: OpenAIAutoSwitchSettings | null,
  keepResident: boolean,
): KeepAliveView {
  // Unload after idle runs only while auto-switch is on (get_auto_unload_idle_seconds).
  const seconds = settings?.enabled ? settings.autoUnloadIdleSeconds : 0;
  return { seconds: seconds > 0 ? seconds : 0, vetoed: keepResident };
}

/** The same write Settings → Model auto-switch makes: a timeout turns auto-switch on with it
 *  (handleSaveIdle), "never" clears the timeout and leaves auto-switch as it is. */
export function keepAliveUpdate(
  seconds: number,
  settings: OpenAIAutoSwitchSettings | null,
): OpenAIAutoSwitchUpdate {
  if (seconds <= 0) {
    return { enabled: settings?.enabled ?? false, autoUnloadIdleSeconds: 0 };
  }
  return { enabled: true, autoUnloadIdleSeconds: Math.max(60, Math.round(seconds)) };
}

/** A choice list that always contains the current value (one typed in Settings, say 7 min). */
export function keepAliveChoices(current: number): number[] {
  const choices: number[] = [...KEEP_ALIVE_CHOICES];
  if (current > 0 && !choices.includes(current)) {
    choices.push(current);
    choices.sort((a, b) => a - b);
  }
  return choices;
}

/** The model the controls start on: the loaded one, else the first a Desktop app is set to. */
export function defaultRunSettingsModel(
  models: { id: string; loaded: boolean }[],
  preferred: string[],
): string | null {
  const loaded = models.find((model) => model.loaded);
  if (loaded) {
    return loaded.id;
  }
  for (const id of preferred) {
    if (models.some((model) => model.id === id)) {
      return id;
    }
  }
  return models[0]?.id ?? null;
}
