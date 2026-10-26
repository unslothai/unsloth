// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Mirrors llama_server_args.py override parsing; malformed input returns null instead of throwing. */

const SPLIT_MODE_FLAGS = new Set(["-sm", "--split-mode"]);
const GPU_LAYER_FLAGS = new Set(["-ngl", "--gpu-layers", "--n-gpu-layers"]);
/** `_LAYER_OFFLOAD_FLAGS | _MOE_OFFLOAD_FLAGS`, which manual mode owns and strips. */
const OFFLOAD_SHADOWING_FLAGS = new Set([
  ...GPU_LAYER_FLAGS,
  "-fit",
  "--fit",
  "-ncmoe",
  "--n-cpu-moe",
  "-cmoe",
  "--cpu-moe",
]);

/** Shorts start with a letter, so `-1` is a value. Underscores fold as in llama.cpp. */
function flagName(token: string): string | null {
  const trimmed = token.trim();
  if (!trimmed.startsWith("-") || trimmed === "-" || trimmed === "--") {
    return null;
  }
  const second = trimmed[1];
  if (second !== undefined && (/[0-9]/.test(second) || second === ".")) {
    return null;
  }
  const name = trimmed.split("=", 1)[0];
  return name.startsWith("--") ? name.replaceAll("_", "-") : name;
}

function lastFlagValue(
  args: readonly string[] | null | undefined,
  flags: ReadonlySet<string>,
): string | null {
  let value: string | null = null;
  for (let i = 0; i < (args?.length ?? 0); i += 1) {
    const token = String(args?.[i]);
    const flag = flagName(token);
    if (flag === null || !flags.has(flag)) {
      continue;
    }
    if (token.includes("=")) {
      value = token.slice(token.indexOf("=") + 1);
      continue;
    }
    const next = args?.[i + 1];
    if (next === undefined || flagName(String(next)) !== null) {
      return null;
    }
    value = String(next);
    i += 1;
  }
  return value;
}

export function resolveTensorParallel(
  extraArgs: readonly string[] | null | undefined,
  tensorParallel: boolean,
): boolean {
  const override = lastFlagValue(extraArgs, SPLIT_MODE_FLAGS);
  return override === null
    ? tensorParallel
    : override.trim().toLowerCase() === "tensor";
}

/** Malformed must stay distinct from no override, or the saved setting is lost silently. */
export type GpuLayersOverride =
  | { kind: "absent" }
  | { kind: "value"; layers: number }
  | { kind: "invalid" };

export function parseGpuLayersOverride(
  extraArgs: readonly string[] | null | undefined,
): GpuLayersOverride {
  const raw = lastFlagValue(extraArgs, GPU_LAYER_FLAGS);
  if (raw === null) {
    return { kind: "absent" };
  }
  const value = Number.parseInt(raw, 10);
  return Number.isInteger(value) && value >= -1 && String(value) === raw.trim()
    ? { kind: "value", layers: value }
    : { kind: "invalid" };
}

/** Only under manual: in auto an inherited `-ngl` is respected. */
export function stripManagedOffloadFlags(
  extraArgs: readonly string[] | null | undefined,
): string[] | null | undefined {
  if (extraArgs == null) {
    return extraArgs;
  }
  const kept: string[] = [];
  for (let i = 0; i < extraArgs.length; i += 1) {
    const token = String(extraArgs[i]);
    const flag = flagName(token);
    if (flag === null || !OFFLOAD_SHADOWING_FLAGS.has(flag)) {
      kept.push(token);
      continue;
    }
    if (!token.includes("=")) {
      const next = extraArgs[i + 1];
      if (next !== undefined && flagName(String(next)) === null) {
        i += 1;
      }
    }
  }
  return kept;
}
