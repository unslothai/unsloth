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
  lora_after_reload?: boolean;
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

/** The note under the LoRA slot after Speed=Auto compiled the model on its 3rd image: a reload defers again. */
export function offersCompiledLoraReload(status: EngineStatus | null | undefined): boolean {
  return Boolean(status?.loaded && status.lora_after_reload && !status.supports_lora);
}

interface ReloadStatus {
  loaded: boolean;
  repo_id?: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
  display_repo_id?: string | null;
  base_repo?: string | null;
}

/** The resident GGUF's effective base repo, so a reload from status rebuilds the same companions (an API load may have
 *  named a non-default base). Omitted when status names none. */
function baseRepoOf(status: ReloadStatus): { baseRepo?: string } {
  return status.base_repo ? { baseRepo: status.base_repo } : {};
}

/** The page's own GGUF target with the resident's base repo when it names none (one seeded from a status read that
 *  left it out), so the reload keeps the companions it runs with now. ``last`` itself otherwise. */
function withResidentBase<T extends { kind?: string | null }>(last: T, status: ReloadStatus): T {
  if (last.kind !== "gguf" || (last as { baseRepo?: string }).baseRepo || !status.base_repo) return last;
  return { ...last, ...baseRepoOf(status) };
}

/** The GGUF the native-reload note names: the page's own last load only while it is still the resident model (another
 *  client may have replaced it), else the checkpoint status reports. */
export function nativeReloadTarget<T extends { repoId: string; kind?: string | null; filename?: string | null }>(
  last: T | null | undefined,
  status: ReloadStatus | null | undefined,
): T | { repoId: string; kind: "gguf"; filename: string; displayRepoId?: string; baseRepo?: string } | null {
  if (!status?.loaded || status.model_kind !== "gguf" || !status.repo_id || !status.gguf_filename) return null;
  if (last && last.repoId === status.repo_id && last.kind === "gguf" && last.filename === status.gguf_filename) {
    return withResidentBase(last, status);
  }
  return {
    repoId: status.repo_id,
    kind: "gguf",
    filename: status.gguf_filename,
    displayRepoId: status.display_repo_id ?? undefined,
    ...baseRepoOf(status),
  };
}

interface EngineAdvanced {
  transformer_quant?: string;
  text_encoder_quant?: string;
  text_encoder_file?: string | string[];
  vae_file?: string;
}

/** The Advanced values an explicit native load can carry. The load and plan routes refuse an explicit precision
 *  (``_assert_native_precision_unset``, 409) and separate text-encoder / VAE files (``_refuse_native_component_files``,
 *  400) on sd.cpp, so a native pick drops them rather than fail. Only an explicit pick, which the router serves natively
 *  or refuses: under a server pin (UNSLOTH_DIFFUSION_ENGINE) the pick is ignored, and a pin to sd_cpp still falls back
 *  to diffusers when native cannot run the pick, which must then build with the requested options. */
export function nativeLoadAdvanced<T extends EngineAdvanced>(
  advanced: T,
  engine: EngineChoice | undefined,
  forced?: string | null,
  // Whether the pick is a GGUF, the only kind the native engine loads.
  gguf = true,
): T {
  if (!gguf || forced || engine !== "sd_cpp") return advanced;
  return { ...advanced, transformer_quant: undefined, text_encoder_quant: undefined, text_encoder_file: undefined, vae_file: undefined };
}

interface LoadedEngineStatus extends EngineStatus {
  model_kind?: string | null;
  repo_id?: string | null;
  gguf_filename?: string | null;
}

/** Record the choice a resident GGUF loaded with, from any accepted status (a load that finished while the page was
 *  closed is never polled to ready). The latest load wins: Auto drops a stored explicit entry, as the ready branch does.
 *  Returns ``choices`` itself when there is nothing new, so an unchanged status re-renders and writes nothing. */
export function rememberLoadedEngine(
  choices: Record<string, Exclude<EngineChoice, "auto">>,
  status: LoadedEngineStatus | null | undefined,
): Record<string, Exclude<EngineChoice, "auto">> {
  if (!status?.loaded || status.model_kind !== "gguf") return choices;
  const key = engineModelKey(status.repo_id, status.gguf_filename);
  const choice = loadedEngineChoice(status);
  if (!key || (isPick(choice) ? choices[key] === choice : !(key in choices))) return choices;
  return withEngineChoice(choices, key, choice);
}

interface ResidentStatus extends ReloadStatus {
  component_files?: Record<string, string> | null;
}

/** A Reapply target for a resident model this page did not load: a pipeline by repo id, a GGUF by repo id + file (so
 *  an Engine change on it can take effect). None for a GGUF with separate component files, whose paths status names
 *  only by basename, so a reload from status would silently drop them. */
export function residentReloadTarget(
  status: ResidentStatus | null | undefined,
):
  | { repoId: string; kind: "pipeline"; filename?: undefined; displayRepoId?: string; baseRepo?: undefined }
  | { repoId: string; kind: "gguf"; filename: string; displayRepoId?: string; baseRepo?: string }
  | null {
  if (!status?.loaded || !status.repo_id) return null;
  const displayRepoId = status.display_repo_id ?? undefined;
  if (status.model_kind === "pipeline") return { repoId: status.repo_id, kind: "pipeline", displayRepoId };
  if (
    status.model_kind === "gguf" &&
    status.gguf_filename &&
    Object.keys(status.component_files ?? {}).length === 0
  ) {
    return { repoId: status.repo_id, kind: "gguf", filename: status.gguf_filename, displayRepoId, ...baseRepoOf(status) };
  }
  return null;
}

/** What Reapply reloads: the page's own last load while it is still the resident model, else the checkpoint status
 *  reports (another client may have replaced it), never a stale pick the page is no longer showing. */
export function reapplyTarget<T extends { repoId: string; kind?: string | null; filename?: string | null }>(
  last: T | null | undefined,
  status: ResidentStatus | null | undefined,
): T | ReturnType<typeof residentReloadTarget> {
  if (!status?.loaded) return last ?? null;
  if (
    last &&
    last.repoId === status.repo_id &&
    last.kind === status.model_kind &&
    (last.kind === "pipeline" || last.filename === status.gguf_filename)
  ) {
    return withResidentBase(last, status);
  }
  return residentReloadTarget(status);
}

/** Whether a Reapply of the resident GGUF would move it to another engine. The two engines read different companion
 *  files, so such a reload goes through the download plan first. A server pin ignores the pick, so it moves nothing. */
export function reapplyChangesEngine(choice: EngineChoice, status: EngineStatus & { model_kind?: string | null } | null | undefined): boolean {
  if (!status?.loaded || status.model_kind !== "gguf" || status.engine_forced) return false;
  return choice !== loadedEngineChoice(status);
}

/** The picker source a reload of ``repoId`` takes: a local path loads directly, a Hub repo id is planned and staged. */
export function reloadSource(repoId: string): "hub" | "local" {
  return /^(\/|~|[A-Za-z]:[\\/]|\\\\)/.test(repoId) ? "local" : "hub";
}
