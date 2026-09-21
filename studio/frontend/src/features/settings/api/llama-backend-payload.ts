// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Import-free payload types and parsers for the llama.cpp backend picker. */

/** Backends a user can ask for. "auto" means hardware detection. */
export const LLAMA_BACKENDS = [
  "auto",
  "cpu",
  "cuda",
  "rocm",
  "vulkan",
] as const;
export type LlamaBackend = (typeof LLAMA_BACKENDS)[number];

/** Backends an install can report running on. Metal is macOS-only and not selectable. */
export type LlamaEffectiveBackend = Exclude<LlamaBackend, "auto"> | "metal";

export function isLlamaBackend(value: unknown): value is LlamaBackend {
  return (
    typeof value === "string" &&
    (LLAMA_BACKENDS as readonly string[]).includes(value)
  );
}

export interface LlamaBackendOption {
  backend: LlamaBackend;
  available: boolean;
  /** For "auto", the backend detection picks right now, so it can be labelled. */
  resolvedBackend: LlamaEffectiveBackend | null;
  releaseTag: string | null;
  downloadSizeBytes: number | null;
}

export interface LlamaBackendJob {
  state: "idle" | "running" | "success" | "error";
  operation: "update" | "switch" | null;
  requestedBackend: LlamaBackend | null;
  message: string;
  error: string | null;
  progress: number | null;
  reloadRequired: boolean | null;
  startedAt: string | null;
  finishedAt: string | null;
}

export interface LlamaBackendStatus {
  /** Whether this install's backend can be switched from here at all. */
  supported: boolean;
  /** Why switching is unavailable. */
  reason: string | null;
  /** Backend pinned by the environment, which overrides anything chosen here. */
  envBackend: LlamaBackend | null;
  /** What the install runs on now. */
  backend: LlamaEffectiveBackend | null;
  /** The recorded choice; null when this client does not recognize it. */
  backendRequest: LlamaBackend | null;
  /** Whether that choice resolves to the installed bundle and paired sidecars. */
  selectionApplied: boolean;
  installedTag: string | null;
  options: LlamaBackendOption[];
  job: LlamaBackendJob;
}

export interface LlamaBackendSwitchStarted {
  started: boolean;
  /** already_running | already_selected | not_prebuilt | unknown_backend | ... */
  reason: string | null;
  message: string | null;
  job: LlamaBackendJob;
}

type ApiObject = Record<string, unknown>;

function asString(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

export function parseLlamaBackendJob(value: unknown): LlamaBackendJob {
  const job = (value ?? {}) as ApiObject;
  const state = job.state;
  return {
    state:
      state === "running" || state === "success" || state === "error"
        ? state
        : "idle",
    operation:
      job.operation === "update" || job.operation === "switch"
        ? job.operation
        : null,
    requestedBackend: isLlamaBackend(job.requested_backend)
      ? job.requested_backend
      : null,
    message: typeof job.message === "string" ? job.message : "",
    error: asString(job.error),
    progress: typeof job.progress === "number" ? job.progress : null,
    reloadRequired:
      typeof job.reload_required === "boolean" ? job.reload_required : null,
    startedAt: asString(job.started_at),
    finishedAt: asString(job.finished_at),
  };
}

function parseOption(value: unknown): LlamaBackendOption | null {
  const option = (value ?? {}) as ApiObject;
  if (!isLlamaBackend(option.backend)) {
    // Older clients cannot label or submit unknown backends.
    return null;
  }
  return {
    backend: option.backend,
    available: option.available === true,
    resolvedBackend:
      (asString(option.resolved_backend) as LlamaEffectiveBackend | null) ??
      null,
    releaseTag: asString(option.release_tag),
    downloadSizeBytes:
      typeof option.download_size_bytes === "number"
        ? option.download_size_bytes
        : null,
  };
}

export function parseLlamaBackendStatus(value: unknown): LlamaBackendStatus {
  const payload = (value ?? {}) as ApiObject;
  const options = Array.isArray(payload.options) ? payload.options : [];
  return {
    supported: payload.supported === true,
    reason: asString(payload.reason),
    envBackend: isLlamaBackend(payload.env_backend)
      ? payload.env_backend
      : null,
    backend:
      (asString(payload.backend) as LlamaEffectiveBackend | null) ?? null,
    backendRequest: isLlamaBackend(payload.backend_request)
      ? payload.backend_request
      : null,
    selectionApplied: payload.selection_applied !== false,
    installedTag: asString(payload.installed_tag),
    options: options
      .map(parseOption)
      .filter((option): option is LlamaBackendOption => option !== null),
    job: parseLlamaBackendJob(payload.job),
  };
}

export function llamaBackendSelectionNeedsApply(
  status: LlamaBackendStatus | null,
  selected: LlamaBackend | null,
): boolean {
  // A null backendRequest is a choice written by a newer Unsloth. Untouched it is
  // not dirty, but picking a backend over it is an explicit replacement.
  const requested = selected ?? status?.backendRequest ?? null;
  if (!status || requested === null) {
    return false;
  }
  return requested !== status.backendRequest || !status.selectionApplied;
}

export function parseLlamaBackendSwitchStarted(
  value: unknown,
): LlamaBackendSwitchStarted {
  const payload = (value ?? {}) as ApiObject;
  return {
    started: payload.started === true,
    reason: asString(payload.reason),
    message: asString(payload.message),
    job: parseLlamaBackendJob(payload.job),
  };
}

/** Show installable options and preserve the selected value if it became unavailable. */
export function visibleLlamaBackendOptions(
  status: LlamaBackendStatus | null,
  selected: LlamaBackend | null,
): LlamaBackendOption[] {
  if (!status) {
    return [];
  }
  return status.options.filter(
    (option) => option.available || option.backend === selected,
  );
}
