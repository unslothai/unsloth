// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The pass-through arguments `/load` resolves before its already-loaded comparator runs. Two of
 *  them change what the request means rather than adding to it, so a comparison against the
 *  resident runtime has to resolve them the same way or it judges a request the server never
 *  received. Mirrors `parse_split_mode_override`, `resolve_tensor_parallel` and
 *  `parse_gpu_layers_override` in llama_server_args.py, plus the manual offload strip the route
 *  applies before comparing. A malformed list answers null rather than throwing, unlike the
 *  backend's parser: the load itself is where a bad argument belongs. */

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

/** `_flag_name`: the flag a token names, or null when it is a value. Shorts always start with a
 *  letter, so `-1` and `-0.5` are values, which is what lets `--n-gpu-layers -1` parse at all.
 *  Long options fold underscores the way llama.cpp does. The backend's attached `-np8` folding
 *  is left out: nothing here reads `-np`. */
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

/** `_last_flag_value`: the last-wins value among *flags*, in either form. */
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
    // A flag with no value is malformed; the load rejects it, and until then it says nothing about what was requested.
    if (next === undefined || flagName(String(next)) !== null) {
      return null;
    }
    value = String(next);
    i += 1;
  }
  return value;
}

/** `resolve_tensor_parallel`: an explicit `--split-mode` in extras last-wins over the toggle, and
 *  tensor parallelism is on only when that mode is `tensor`. */
export function resolveTensorParallel(
  extraArgs: readonly string[] | null | undefined,
  tensorParallel: boolean,
): boolean {
  const override = lastFlagValue(extraArgs, SPLIT_MODE_FLAGS);
  return override === null
    ? tensorParallel
    : override.trim().toLowerCase() === "tensor";
}

/** `parse_gpu_layers_override`, in the three states its caller has to tell apart. The backend
 *  raises on a malformed value, so the load fails and says so. Folding that into "no override"
 *  would strip the offending token, adopt the rest, and lose the user's saved setting silently. */
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

/** The offload flags manual mode owns, dropped as the route drops them before comparing. Only
 *  under manual: in auto an inherited `-ngl` is respected and reaches the child. */
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
    // Its value goes with it, unless the flag carried one itself.
    if (!token.includes("=")) {
      const next = extraArgs[i + 1];
      if (next !== undefined && flagName(String(next)) === null) {
        i += 1;
      }
    }
  }
  return kept;
}
