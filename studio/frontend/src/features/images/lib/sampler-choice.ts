// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Backend ids: core/inference/diffusion_sampler.py. Labels follow AUTOMATIC1111. */
export const DEFAULT_SAMPLER = "default";
export const AUTOMATIC_SCHEDULE = "automatic";

const SAMPLER_LABELS: Record<string, string> = {
  default: "Model default",
  euler: "Euler",
  euler_a: "Euler a",
  dpmpp_2m: "DPM++ 2M",
  dpmpp_2m_sde: "DPM++ 2M SDE",
};

export const SCHEDULE_LABELS: Record<string, string> = {
  automatic: "Automatic",
  uniform: "Uniform",
  karras: "Karras",
  exponential: "Exponential",
};

export interface SamplerOption {
  id: string;
  label: string;
  schedule_types: string[];
}

export interface SamplerChoice {
  sampler: string;
  scheduleType: string;
}

export const DEFAULT_SAMPLER_CHOICE: SamplerChoice = {
  sampler: DEFAULT_SAMPLER,
  scheduleType: AUTOMATIC_SCHEDULE,
};

/** `choice` if the loaded model offers it, else the nearest that it does (shipped scheduler last). */
export function normalizeSamplerChoice(
  options: SamplerOption[],
  choice: Partial<SamplerChoice> | null | undefined,
): SamplerChoice {
  const option = options.find((o) => o.id === choice?.sampler);
  if (!option) return DEFAULT_SAMPLER_CHOICE;
  const scheduleType =
    choice?.scheduleType && option.schedule_types.includes(choice.scheduleType)
      ? choice.scheduleType
      : AUTOMATIC_SCHEDULE;
  return { sampler: option.id, scheduleType };
}

/** Request fields: omitted for the shipped scheduler, so the request is what it was before the control. */
export function samplerRequestFields(
  options: SamplerOption[],
  choice: SamplerChoice,
): { sampler?: string; schedule_type?: string } {
  const valid = normalizeSamplerChoice(options, choice);
  return {
    ...(valid.sampler !== DEFAULT_SAMPLER ? { sampler: valid.sampler } : {}),
    ...(valid.scheduleType !== AUTOMATIC_SCHEDULE ? { schedule_type: valid.scheduleType } : {}),
  };
}

/** "DPM++ 2M Karras" for the recipe panel; null when the image ran on the shipped scheduler. */
export function samplerRecipeLabel(
  sampler: string | null | undefined,
  scheduleType: string | null | undefined,
): string | null {
  if (!sampler && !scheduleType) return null;
  const name = SAMPLER_LABELS[sampler ?? DEFAULT_SAMPLER] ?? sampler ?? "";
  const schedule = scheduleType ? (SCHEDULE_LABELS[scheduleType] ?? scheduleType) : "";
  return [name, schedule].filter(Boolean).join(" ");
}

const KEY = "unsloth:images:sampler-by-model";

/** Choices remembered per model key; a model without an entry runs its shipped scheduler. */
export function readSamplerChoices(): Record<string, SamplerChoice> {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

/** ``choices`` with ``modelKey`` set to ``choice`` (dropped when it is the default), persisted. */
export function withSamplerChoice(
  choices: Record<string, SamplerChoice>,
  modelKey: string,
  choice: SamplerChoice,
): Record<string, SamplerChoice> {
  if (!modelKey) return choices;
  const next = { ...choices };
  if (choice.sampler === DEFAULT_SAMPLER && choice.scheduleType === AUTOMATIC_SCHEDULE) {
    delete next[modelKey];
  } else {
    next[modelKey] = choice;
  }
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // storage can be unavailable in private browsing
  }
  return next;
}
