// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images Advanced "Engine" choice for GGUF picks: Auto (the host's own choice), diffusers, or the native
// sd.cpp engine, remembered per checkpoint. Only a load that landed records its choice, so a refused pick
// (no native binary, an unsupported family) is never replayed on the next load of that model.

export type EngineChoice = "auto" | "diffusers" | "sd_cpp";

export const ENGINE_OPTIONS: Array<[EngineChoice, string]> = [
  ["auto", "Auto"],
  ["diffusers", "Diffusers"],
  ["sd_cpp", "Native (sd.cpp)"],
];

export const ENGINE_HINT =
  "Which engine runs this GGUF. Auto keeps the host's choice: diffusers on a GPU, native sd.cpp on CPU. Native applies LoRAs to GGUF models and usually needs less VRAM, but has no torch.compile or quantised fast paths and is often slower on CUDA. Diffusers runs a GGUF without LoRA. Applies on the next load (Reapply).";

const KEY = "unsloth:images:engine-by-model";

export function engineModelKey(repoId: string | null | undefined, filename: string | null | undefined): string {
  return repoId && filename ? `${repoId}|${filename}` : "";
}

function isPick(value: unknown): value is Exclude<EngineChoice, "auto"> {
  return value === "diffusers" || value === "sd_cpp";
}

export function readEngineChoices(): Record<string, Exclude<EngineChoice, "auto">> {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const out: Record<string, Exclude<EngineChoice, "auto">> = {};
    for (const [k, v] of Object.entries(value)) if (isPick(v)) out[k] = v;
    return out;
  } catch {
    return {};
  }
}

/** Store the choice for one checkpoint; Auto drops the entry. */
export function withEngineChoice(
  choices: Record<string, Exclude<EngineChoice, "auto">>,
  modelKey: string,
  choice: EngineChoice,
): Record<string, Exclude<EngineChoice, "auto">> {
  if (!modelKey) return choices;
  const next = { ...choices };
  if (isPick(choice)) next[modelKey] = choice;
  else delete next[modelKey];
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // storage can be unavailable in private browsing
  }
  return next;
}

/** Whether a load of this kind may carry an engine pick: the native engine loads GGUF only. */
export function sendsEngine(kind: string | null | undefined, filename: string | null | undefined): boolean {
  if (kind) return kind === "gguf";
  return (filename ?? "").toLowerCase().endsWith(".gguf");
}

/** The request field for a load, omitted for Auto so an untouched load stays byte-identical. */
export function engineRequestField(choice: EngineChoice | undefined): Exclude<EngineChoice, "auto"> | undefined {
  return isPick(choice) ? choice : undefined;
}

interface EngineStatus {
  loaded: boolean;
  engine?: string | null;
  engine_forced?: string | null;
  native_engine_eligible?: boolean;
  lora_via_native?: boolean;
  supports_lora?: boolean;
  resolved?: Record<string, { source: string; requested?: string | boolean | null }> | null;
}

/** The select shows only for a loaded GGUF the native engine can run. */
export function engineSelectVisible(status: EngineStatus | null | undefined): boolean {
  return Boolean(status?.loaded && status.native_engine_eligible);
}

/** The select value for the loaded build: what its load asked for, Auto when it asked for nothing. */
export function loadedEngineChoice(status: EngineStatus | null | undefined): EngineChoice {
  const record = status?.loaded ? status.resolved?.engine : undefined;
  if (record?.source === "explicit" && isPick(record.requested)) return record.requested;
  return "auto";
}

/** The note under the LoRA slot: LoRA is off on this engine, and a native reload would turn it on. A server pin
 *  (UNSLOTH_DIFFUSION_ENGINE) would ignore the reload's pick, so it offers nothing then. */
export function offersNativeLoraReload(status: EngineStatus | null | undefined): boolean {
  return Boolean(
    status?.loaded && status.lora_via_native && !status.supports_lora && !status.engine_forced,
  );
}

interface ReloadStatus {
  loaded: boolean;
  repo_id?: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
  display_repo_id?: string | null;
}

/** The GGUF the native-reload note names: the page's own last load only while it is still the resident model (another
 *  client may have replaced it), else the checkpoint status reports. */
export function nativeReloadTarget<T extends { repoId: string; kind?: string | null; filename?: string | null }>(
  last: T | null | undefined,
  status: ReloadStatus | null | undefined,
): T | { repoId: string; kind: "gguf"; filename: string; displayRepoId?: string } | null {
  if (!status?.loaded || status.model_kind !== "gguf" || !status.repo_id || !status.gguf_filename) return null;
  if (last && last.repoId === status.repo_id && last.kind === "gguf" && last.filename === status.gguf_filename) {
    return last;
  }
  return {
    repoId: status.repo_id,
    kind: "gguf",
    filename: status.gguf_filename,
    displayRepoId: status.display_repo_id ?? undefined,
  };
}
