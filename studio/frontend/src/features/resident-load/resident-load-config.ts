// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What the resident build was loaded with, read back from its status.
 *
 * These are not generation settings and they do not belong to a saved recipe: they describe how
 * the model currently sits in VRAM, they only change on a load, and the running process is their
 * owner. The page mirrors them into the Advanced selects so those stop being pure request state.
 */

import {
  type ResolvedControl,
  isResolvedHonored,
  resolvedSelectValue,
} from "../../lib/resolved-precision.ts";

export interface MediaLoadConfig {
  speedMode: "auto" | "off" | "eager" | "default" | "max";
  transformerQuant: "auto" | "none" | "fp8" | "int8" | "nvfp4" | "mxfp8";
  attentionBackend: "auto" | "native" | "cudnn" | "flash3" | "sage";
  memoryMode: "auto" | "fast" | "balanced" | "low_vram";
  transformerCache: "auto" | "off" | "fbcache";
}

export interface ImageLoadConfig extends MediaLoadConfig {
  cpuOffload: boolean;
}

export type VideoLoadConfig = MediaLoadConfig;

type ResidentMediaStatus = {
  loaded?: boolean;
  repo_id?: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
};

export type ResidentLoadStatus = {
  resolved?: Record<string, ResolvedControl> | null;
};

export type ResidentMediaLoadTarget = {
  repoId: string;
  kind: "gguf" | "single_file" | "pipeline";
  filename?: string;
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
): VideoLoadConfig | null {
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
): ImageLoadConfig | null {
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
): VideoLoadConfig | null {
  return commonLoadConfigFromStatus(status);
}
