// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  EXTRA_ARGS_MAX_BYTES,
  EXTRA_ARGS_MAX_TOKENS,
} from "../model-picker/model-config/llama-extra-args";
import { KV_CACHE_DTYPES } from "../model-picker/model-config/per-model-config";

type ValueRule = (value: string) => boolean;
const encoder = new TextEncoder();
const decimal = /^-?(?:0|[1-9]\d*)(?:\.\d+)?$/;
const number =
  (min: number, max: number, integer = false): ValueRule =>
  (value) => {
    if (value.length > 24 || value !== value.trim() || !decimal.test(value)) {
      return false;
    }
    const parsed = Number(value);
    return (
      Number.isFinite(parsed) &&
      parsed >= min &&
      parsed <= max &&
      (!integer || (!value.includes(".") && Number.isSafeInteger(parsed)))
    );
  };
const choice =
  (values: readonly string[]): ValueRule =>
  (value) =>
    values.includes(value);
const threads = number(-1, 1024, true);
const context: ValueRule = (value) =>
  value === "0" || number(128, 2_147_483_647, true)(value);
const batch = number(1, 65_536, true);
const layers = number(-1, 2_147_483_647, true);
const cache: ValueRule = (value) =>
  value === "f16" || KV_CACHE_DTYPES.some((dtype) => dtype === value);
const ratio = number(0, 1);
const tensorRatio = number(0, 1_000_000);

const rules: Readonly<Record<string, ValueRule | null>> = {
  "--threads": threads,
  "--threads-batch": threads,
  "--ctx-size": context,
  "--batch-size": batch,
  "--ubatch-size": batch,
  "--gpu-layers": layers,
  "--n-cpu-moe": number(0, 2_147_483_647, true),
  "--main-gpu": number(0, 255, true),
  "--flash-attn": choice(["on", "off", "auto"]),
  "--cache-type-k": cache,
  "--cache-type-v": cache,
  "--split-mode": choice(["none", "layer", "row", "tensor"]),
  "--tensor-split": (value) => {
    const ratios = value.split(",");
    return (
      ratios.length <= 256 &&
      ratios.every(tensorRatio) &&
      ratios.some((part) => Number(part) > 0)
    );
  },
  "--rope-scaling": choice(["none", "linear", "yarn"]),
  "--rope-scale": number(0.001, 1_000_000),
  "--rope-freq-base": number(0.001, 1_000_000_000),
  "--rope-freq-scale": number(0.001, 1_000_000),
  "--yarn-orig-ctx": context,
  "--yarn-ext-factor": number(-1, 1_000_000),
  "--yarn-attn-factor": number(0, 1_000_000),
  "--yarn-beta-fast": number(0, 1_000_000),
  "--yarn-beta-slow": number(0, 1_000_000),
  "--seed": number(-1, 4_294_967_295, true),
  "--temp": number(0, 100),
  "--top-k": number(0, 1_000_000, true),
  "--top-p": ratio,
  "--min-p": ratio,
  "--repeat-last-n": number(-1, 1_000_000, true),
  "--repeat-penalty": number(0, 100),
  "--presence-penalty": number(-2, 2),
  "--frequency-penalty": number(-2, 2),
  "--no-warmup": null,
  "--no-context-shift": null,
};

const aliases: Readonly<Record<string, string>> = {
  "-t": "--threads",
  "-tb": "--threads-batch",
  "-c": "--ctx-size",
  "-b": "--batch-size",
  "-ub": "--ubatch-size",
  "--n-gpu-layers": "--gpu-layers",
  "-ngl": "--gpu-layers",
  "-ncmoe": "--n-cpu-moe",
  "-mg": "--main-gpu",
  "-fa": "--flash-attn",
  "-ctk": "--cache-type-k",
  "-ctv": "--cache-type-v",
  "-sm": "--split-mode",
};

function boundedTokens(value: unknown): value is string[] {
  if (!Array.isArray(value) || value.length > EXTRA_ARGS_MAX_TOKENS) {
    return false;
  }
  let bytes = 0;
  for (const token of value) {
    if (typeof token !== "string" || token.length > EXTRA_ARGS_MAX_BYTES) {
      return false;
    }
    bytes += encoder.encode(token).byteLength;
    if (bytes > EXTRA_ARGS_MAX_BYTES) {
      return false;
    }
  }
  return true;
}

export function validSharedExtraArgs(value: unknown): value is string[] {
  if (!boundedTokens(value)) {
    return false;
  }
  const seen = new Set<string>();
  const tokens = value[Symbol.iterator]();
  for (const token of tokens) {
    const flag = Object.hasOwn(aliases, token) ? aliases[token] : token;
    if (!Object.hasOwn(rules, flag) || seen.has(flag)) {
      return false;
    }
    seen.add(flag);
    const rule = rules[flag];
    if (rule !== null) {
      const argument = tokens.next().value;
      if (typeof argument !== "string" || !rule(argument)) {
        return false;
      }
    }
  }
  return true;
}
