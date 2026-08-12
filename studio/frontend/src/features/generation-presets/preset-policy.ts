// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ResolvedControl,
  isResolvedHonored,
  resolvedSelectValue,
} from "../../lib/resolved-precision.ts";
import type {
  ImageGenerationPresetLoadConfig,
  VideoGenerationPresetLoadConfig,
} from "./types";

export const DEFAULT_PRESET_NAME = "Default";

export type ResidentMediaLoadTarget = {
  repoId: string;
  kind: "gguf" | "single_file" | "pipeline";
  filename?: string;
};

type ResidentMediaStatus = {
  loaded?: boolean;
  repo_id?: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
};

type ResidentLoadStatus = {
  resolved?: Record<string, ResolvedControl> | null;
};

export function reapplyTargetFromStatus(
  status: ResidentMediaStatus | null | undefined,
): ResidentMediaLoadTarget | null {
  if (!(status?.loaded && status.repo_id)) {
    return null;
  }
  const kind = status.model_kind;
  if (kind !== "gguf" && kind !== "single_file" && kind !== "pipeline") {
    return null;
  }
  if (kind === "pipeline") {
    return { repoId: status.repo_id, kind };
  }
  if (!status.gguf_filename) {
    return null;
  }
  return {
    repoId: status.repo_id,
    kind,
    filename: status.gguf_filename,
  };
}

export type DynamicDefaultRollback = () => boolean;

export function chainDynamicDefaultRollback(
  previous: DynamicDefaultRollback | undefined,
  next: DynamicDefaultRollback,
): DynamicDefaultRollback {
  if (!previous) {
    return next;
  }
  return () => next() && previous();
}

function resolvedOption<T extends string>(
  control: ResolvedControl | undefined,
  options: readonly T[],
  aliases: Record<string, T> = {},
): T | null {
  return resolvedSelectValue(
    control,
    (value) =>
      options.find((option) => option === value) ?? aliases[value] ?? null,
  );
}

function commonLoadConfigFromStatus(
  status: ResidentLoadStatus | null | undefined,
): VideoGenerationPresetLoadConfig | null {
  const record = status?.resolved;
  if (!record) {
    return null;
  }
  const speedMode = resolvedOption(record.speed_mode, [
    "auto",
    "off",
    "eager",
    "default",
    "max",
  ]);
  const transformerQuant = resolvedOption(
    record.transformer_quant,
    ["auto", "none", "fp8", "int8", "nvfp4", "mxfp8"],
    { off: "none" },
  );
  const attentionBackend = resolvedOption(
    record.attention_backend,
    ["auto", "native", "cudnn", "flash3", "sage"],
    {
      _native_cudnn: "cudnn",
      _native_flash3: "flash3",
      _native_sage: "sage",
    },
  );
  const memoryMode = resolvedOption(record.memory_mode, [
    "auto",
    "fast",
    "balanced",
    "low_vram",
  ]);
  const transformerCache = resolvedOption(record.transformer_cache, [
    "auto",
    "off",
    "fbcache",
  ]);
  if (
    !speedMode ||
    !transformerQuant ||
    !attentionBackend ||
    !memoryMode ||
    !transformerCache
  ) {
    return null;
  }
  return {
    speedMode,
    transformerQuant,
    attentionBackend,
    memoryMode,
    transformerCache,
  };
}

export function imageLoadConfigFromStatus(
  status: ResidentLoadStatus | null | undefined,
): ImageGenerationPresetLoadConfig | null {
  const common = commonLoadConfigFromStatus(status);
  const control = status?.resolved?.cpu_offload;
  if (!common || !control) {
    return null;
  }
  const cpuRequest = control.requested;
  const cpuOffload =
    control.source === "explicit" &&
    (isResolvedHonored(control) && cpuRequest !== undefined
      ? cpuRequest === true
      : control.value === true);
  return {
    ...common,
    cpuOffload,
  };
}

export function videoLoadConfigFromStatus(
  status: ResidentLoadStatus | null | undefined,
): VideoGenerationPresetLoadConfig | null {
  return commonLoadConfigFromStatus(status);
}

/**
 * Whether a Reapply of the resident build knows what load options it would submit. Either the
 * status names them, or the build reports no resolved record at all -- the native sd.cpp engine
 * takes none of these options, so there is nothing there for a Reapply to silently replace.
 */
export function residentLoadConfigIsKnown(
  status: ResidentLoadStatus | null | undefined,
  config: unknown,
): boolean {
  return Boolean(config) || !status?.resolved;
}

export function configKey(value: unknown): string {
  return JSON.stringify(value ?? null);
}

export function mergeUntouchedParams<Params extends object>(
  baseline: Params,
  current: Params,
  next: Params,
): Params {
  const merged = { ...next };
  for (const key of Object.keys(merged) as Array<keyof Params>) {
    if (configKey(current[key]) !== configKey(baseline[key])) {
      merged[key] = current[key];
    }
  }
  return merged;
}

export function getBuiltinVariantName(usedNames: Set<string>): string {
  let suffix = 1;
  let candidate = `${DEFAULT_PRESET_NAME} ${suffix}`;
  while (usedNames.has(candidate)) {
    suffix += 1;
    candidate = `${DEFAULT_PRESET_NAME} ${suffix}`;
  }
  return candidate;
}

export function closestResolutionIndex(
  presets: [number, number][],
  targetWidth: number,
  targetHeight: number,
): number {
  if (presets.length === 0) {
    return 0;
  }
  const targetAspect = targetWidth / targetHeight;
  const targetArea = targetWidth * targetHeight;
  let bestIndex = 0;
  let bestScore = Number.POSITIVE_INFINITY;
  presets.forEach(([width, height], index) => {
    const aspectDistance = Math.abs(Math.log(width / height / targetAspect));
    const areaDistance = Math.abs(Math.log((width * height) / targetArea));
    const score = aspectDistance * 4 + areaDistance;
    if (score < bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });
  return bestIndex;
}

export function closestDurationIndex(
  durations: Array<{ seconds: number }>,
  targetSeconds: number,
): number {
  if (durations.length === 0) {
    return 0;
  }
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  durations.forEach(({ seconds }, index) => {
    const distance = Math.abs(seconds - targetSeconds);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}
