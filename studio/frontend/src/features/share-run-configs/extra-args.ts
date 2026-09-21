// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  EXTRA_ARGS_MAX_BYTES,
  EXTRA_ARGS_MAX_TOKENS,
  diagnoseExtraArgs,
  extraArgFlagName,
  formatExtraArgs,
} from "../model-picker/model-config/llama-extra-args";
import {
  CONTEXT_LENGTH_MIN,
  KV_CACHE_DTYPES,
  N_BATCH_MAX,
  N_BATCH_MIN,
} from "../model-picker/model-config/per-model-config";

type ValueRule = (value: string) => boolean;
const decimal = /^-?(?:0|[1-9]\d*)(?:\.\d+)?$/;
const flagName = /^--?[A-Za-z][A-Za-z0-9_-]{0,63}$/;
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
  value === "0" || number(CONTEXT_LENGTH_MIN, 2_147_483_647)(value);
const batch = number(N_BATCH_MIN, N_BATCH_MAX);
const layers = number(-1, 2_147_483_647);
const cache: ValueRule = (value) =>
  value === "f16" || KV_CACHE_DTYPES.some((dtype) => dtype === value);
const ratio = number(0, 1);
const tensorRatio = number(0, 1_000_000);

const sharePolicy: readonly [string, ValueRule | null][] = [
  ["--threads -t", threads],
  ["--threads-batch -tb", threads],
  ["--ctx-size -c", context],
  ["--batch-size -b", batch],
  ["--ubatch-size -ub", batch],
  ["--gpu-layers --n-gpu-layers -ngl", layers],
  ["--n-cpu-moe -ncmoe", number(0, 2_147_483_647)],
  ["--main-gpu -mg", number(0, 255, true)],
  ["--flash-attn -fa", choice(["on", "off", "auto"])],
  ["--cache-type-k -ctk", cache],
  ["--cache-type-v -ctv", cache],
  ["--split-mode -sm", choice(["none", "layer", "row", "tensor"])],
  [
    "--tensor-split -ts",
    (value) => {
      const ratios = value.split(",");
      return (
        ratios.length <= 256 &&
        ratios.every(tensorRatio) &&
        ratios.some((part) => Number(part) > 0)
      );
    },
  ],
  ["--rope-scaling", choice(["none", "linear", "yarn"])],
  ["--rope-scale", number(0.001, 1_000_000)],
  ["--rope-freq-base", number(0.001, 1_000_000_000)],
  ["--rope-freq-scale", number(0.001, 1_000_000)],
  ["--yarn-orig-ctx", (value) => context(value) && !value.includes(".")],
  ["--yarn-ext-factor", number(-1, 1_000_000)],
  ["--yarn-attn-factor", number(0, 1_000_000)],
  ["--yarn-beta-fast", number(0, 1_000_000)],
  ["--yarn-beta-slow", number(0, 1_000_000)],
  ["--seed -s", number(-1, 4_294_967_295, true)],
  ["--temp --temperature", number(0, 100)],
  ["--top-k", number(0, 1_000_000, true)],
  ["--top-p", ratio],
  ["--min-p", ratio],
  ["--repeat-last-n", number(-1, 1_000_000, true)],
  ["--repeat-penalty", number(0, 100)],
  ["--presence-penalty", number(-2, 2)],
  ["--frequency-penalty", number(-2, 2)],
  ["--no-warmup", null],
  ["--no-context-shift", null],
];

const sharedFlags = new Map(
  sharePolicy.flatMap(([spellings, permits]) => {
    const names = spellings.split(" ");
    return names.map(
      (name) => [name, { canonical: names[0], permits }] as const,
    );
  }),
);

function boundedTokens(value: unknown): value is string[] {
  if (!Array.isArray(value) || value.length > EXTRA_ARGS_MAX_TOKENS) {
    return false;
  }
  for (const token of value) {
    if (typeof token !== "string" || token.length > EXTRA_ARGS_MAX_BYTES) {
      return false;
    }
  }
  return true;
}

function upstreamError(value: string[]): string | null {
  return (
    diagnoseExtraArgs(formatExtraArgs(value), null).find(
      (diagnostic) => diagnostic.level === "error",
    )?.message ?? null
  );
}

function sharedValueError(
  flag: string,
  permits: ValueRule | null,
  token: string,
  tokens: Iterator<string, undefined>,
): string | null {
  const equals = token.indexOf("=");
  if (permits === null) {
    return equals === -1
      ? upstreamError([token])
      : `${flag} does not take a value.`;
  }
  const argument =
    equals === -1 ? tokens.next().value : token.slice(equals + 1);
  if (argument === undefined) {
    return `${flag} requires a value.`;
  }
  return permits(argument)
    ? upstreamError(equals === -1 ? [token, argument] : [token])
    : `${flag} has an invalid or unsupported value.`;
}

function sharePolicyError(value: string[]): string | null {
  const seen = new Set<string>();
  const tokens = value[Symbol.iterator]();
  for (const token of tokens) {
    const flag = extraArgFlagName(token);
    const policy = flag === null ? undefined : sharedFlags.get(flag);
    if (!policy) {
      return flag !== null && flag === flag.trim() && flagName.test(flag)
        ? `${flag} is not supported in shared links.`
        : "Extra arguments contain an unexpected token.";
    }
    if (seen.has(policy.canonical)) {
      return `${policy.canonical} is specified more than once.`;
    }
    seen.add(policy.canonical);
    const issue = sharedValueError(
      policy.canonical,
      policy.permits,
      token,
      tokens,
    );
    if (issue !== null) {
      return issue;
    }
  }
  return null;
}

export function sharedExtraArgsError(value: unknown): string | null {
  if (!boundedTokens(value)) {
    return `Extra arguments must be a list of at most ${EXTRA_ARGS_MAX_TOKENS} text tokens totaling at most ${EXTRA_ARGS_MAX_BYTES} bytes.`;
  }
  return sharePolicyError(value) ?? upstreamError(value);
}

export function validSharedExtraArgs(value: unknown): value is string[] {
  return sharedExtraArgsError(value) === null;
}
