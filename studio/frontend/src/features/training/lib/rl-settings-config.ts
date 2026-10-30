// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept apart from yaml-config so the training store can restore a run without loading js-yaml.
import {
  GRPO_VARIANTS,
  type GrpoRewardSelection,
  type GrpoPythonRewardIsolation,
  type GrpoVariant,
  type TrainingObjective,
} from "@/types/training";

import {
  GRPO_FORMAT_WARMUP_MIN_CONTEXT,
  GRPO_FORMAT_WARMUP_STEPS,
} from "@/config/training";

const RL_OBJECTIVES = ["dpo", "orpo", "grpo"] as const;

export interface YamlRlSettings {
  trainingObjective: TrainingObjective;
  rlBeta?: number | null;
  rlMaxPromptLength?: number | null;
  grpoVariant?: GrpoVariant;
  grpoNumGenerations?: number;
  grpoMaxCompletionLength?: number | null;
  grpoTemperature?: number;
  grpoEpsilonHigh?: number | null;
  grpoMaskTruncatedCompletions?: boolean;
  grpoReasoningFormat?: boolean;
  grpoFormatWarmupSteps?: number;
  grpoPythonRewardIsolation?: GrpoPythonRewardIsolation;
  grpoEnableThinking?: boolean;
  grpoSystemPrompt?: string;
  grpoRewards?: GrpoRewardSelection[];
}

const isNum = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);
const numOrNull = (v: unknown): number | null | undefined =>
  v === null ? null : isNum(v) ? v : undefined;

/** A saved run's objective and RL settings. The stored config keeps them as `objective`,
 * `rl_settings` (the same keys as the YAML rl section) and `reward_specs`. */
export function parseRunConfigRlSettings(
  config: Record<string, unknown>,
): YamlRlSettings {
  const settings = config.rl_settings;
  return parseRlSection(
    {
      ...(settings !== null && typeof settings === "object" ? settings : {}),
      objective: config.objective,
      rewards: config.reward_specs,
    },
    config.max_seq_length,
  );
}

export function parseRlSection(rl: unknown, context?: unknown): YamlRlSettings {
  if (rl == null || typeof rl !== "object" || Array.isArray(rl)) {
    return { trainingObjective: "sft" };
  }
  const r = rl as Record<string, unknown>;
  const objective = RL_OBJECTIVES.find((o) => o === r.objective);
  if (!objective) {
    return { trainingObjective: "sft" };
  }
  const out: YamlRlSettings = { trainingObjective: objective };
  const set = <K extends keyof YamlRlSettings>(
    key: K,
    value: YamlRlSettings[K] | undefined,
  ) => {
    if (value !== undefined) {
      out[key] = value;
    }
  };
  // Out-of-range values are dropped, matching the backend's TrainingStartRequest bounds.
  const inRange = (v: unknown, lo: number, hi: number, loOpen = false) => {
    const n = numOrNull(v);
    return n == null || (n <= hi && (loOpen ? n > lo : n >= lo))
      ? n
      : undefined;
  };
  set("rlBeta", inRange(r.beta, 0, 10));
  const intInRange = (v: unknown, lo: number, hi: number) => {
    const n = inRange(v, lo, hi);
    return n == null || Number.isInteger(n) ? n : undefined;
  };
  set("rlMaxPromptLength", intInRange(r.max_prompt_length, 16, Infinity));
  if (objective !== "grpo") {
    return out;
  }
  set(
    "grpoVariant",
    GRPO_VARIANTS.find((v) => v === r.variant),
  );
  set("grpoNumGenerations", intInRange(r.num_generations, 2, 16) ?? undefined);
  set(
    "grpoMaxCompletionLength",
    intInRange(r.max_completion_length, 16, Infinity),
  );
  set(
    "grpoTemperature",
    isNum(r.temperature) && r.temperature > 0 && r.temperature <= 2
      ? r.temperature
      : undefined,
  );
  set("grpoEpsilonHigh", inRange(r.epsilon_high, 0, 1, true));
  if (typeof r.reasoning_format === "boolean") {
    out.grpoReasoningFormat = r.reasoning_format;
  }
  set(
    "grpoFormatWarmupSteps",
    // The backend's range: anything else would only fail at Start.
    intInRange(r.format_warmup_steps, 0, 1000) ?? undefined,
  );
  // An omitted count gets the backend's default, not the store's 0; a written 0 stays.
  if (
    out.grpoReasoningFormat === true &&
    out.grpoFormatWarmupSteps === undefined &&
    !(isNum(context) && context < GRPO_FORMAT_WARMUP_MIN_CONTEXT)
  ) {
    out.grpoFormatWarmupSteps = GRPO_FORMAT_WARMUP_STEPS;
  }
  if (typeof r.mask_truncated_completions === "boolean") {
    out.grpoMaskTruncatedCompletions = r.mask_truncated_completions;
  }
  if (
    r.python_reward_isolation === "required" ||
    r.python_reward_isolation === "auto"
  ) {
    out.grpoPythonRewardIsolation = r.python_reward_isolation;
  }
  if (typeof r.enable_thinking === "boolean") {
    out.grpoEnableThinking = r.enable_thinking;
  }
  if (typeof r.system_prompt === "string") {
    out.grpoSystemPrompt = r.system_prompt;
  } else if (r.system_prompt === null) {
    out.grpoSystemPrompt = "";
  }
  if (Array.isArray(r.rewards)) {
    out.grpoRewards = r.rewards.flatMap((item) => {
      const x = item as Record<string, unknown> | null;
      return x && typeof x.name === "string" && x.name
        ? [
            {
              name: x.name,
              weight:
                isNum(x.weight) && x.weight >= -10 && x.weight <= 10
                  ? x.weight
                  : 1,
            },
          ]
        : [];
    });
  }
  return out;
}
