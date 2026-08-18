// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "@/features/model-picker";

import {
  parseGpuLayersOverride,
  resolveTensorParallel,
  stripManagedOffloadFlags,
} from "./llama-extra-args-normalize";
import type { GpuIndexKind } from "@/hooks/gpu-selection";

import type { InferenceStatusResponse } from "../types/api";

/** The resident load's own invocation, as `/api/inference/status` echoes it. */
type ResidentRuntime = Pick<
  InferenceStatusResponse,
  | "requested_context_length"
  | "cache_type_kv"
  | "mlx_kv_bits_requested"
  | "speculative_type"
  | "spec_draft_n_max"
  | "requested_parallel_slots"
  | "requested_n_batch"
  | "requested_n_ubatch"
  | "tensor_parallel"
  | "chat_template_override"
  | "requested_llama_extra_args"
  | "gpu_memory_mode"
  | "gpu_layers"
  | "n_cpu_moe"
  | "requested_gpu_ids"
  | "gpu_ids"
  | "is_gguf"
  | "is_diffusion"
  | "diffusion_requested_ngl"
  | "diffusion_split_supported"
  | "tensor_parallel_dropped_by_arch_gate"
  | "gpu_placement_paravirtual"
  | "tensor_split"
  | "cpu_fallback_reason"
>;

function sameList(
  left: readonly (string | number)[] | null | undefined,
  right: readonly (string | number)[] | null | undefined,
): boolean {
  const a = left ?? [];
  const b = right ?? [];
  return a.length === b.length && a.every((item, index) => item === b[index]);
}

/** Placement is a set, not an order: the backend narrows and reorders it at fit time. */
function sameGpuSet(
  left: readonly number[] | null | undefined,
  right: readonly number[] | null | undefined,
): boolean {
  const sort = (ids: readonly number[]) => [...ids].sort((a, b) => a - b);
  return sameList(sort(left ?? []), sort(right ?? []));
}

/**
 * What a config field resolves to when left unset.
 *
 * These four are not per-model, so omitting them is not silence: the applier fills them
 * from a standing preference or a constant (`applyPerModelConfigToRuntime`), and the load
 * sends that. The caller supplies them rather than this module reading the store, so the
 * comparison stays a pure function of its inputs. Required, not optional: a resolver a
 * caller can forget is one that silently keeps a runtime the user did not ask for.
 */
export type StandingConfigDefaults = {
  /** `readPersistedSpeculativeType()`, already normalized. */
  speculativeType: string | null;
  /** `readPersistedGpuMemoryMode()`. */
  gpuMemoryMode: "auto" | "manual";
  /** `GPU_LAYERS_AUTO`. */
  gpuLayers: number;
  /** The applier's own constant, 0. */
  nCpuMoe: number;
  /**
   * `reconcilePersistedGpuIds`, with the device cache already warm and the resident model's
   * diffusion flag bound. `performLoad` sends the reconciled pick, not the saved one: a
   * selection saved in another index namespace, or naming GPUs that are gone, becomes
   * Automatic before `/load`. Comparing the raw ids would let a saved physical `[1]` adopt a
   * server pinned to Vulkan device 1.
   */
  reconcileGpuIds: (
    ids: number[] | null,
    savedIndexKind: GpuIndexKind | null | undefined,
  ) => number[] | null;
  /**
   * `resolveLoadMaxSeqLength` bound to the inputs `performLoad` gives it, so the comparison
   * is against the n_ctx the load would send. An unset length is not simply 0: for a GGUF
   * re-pick it resolves to the resident context, and only otherwise to 0, which is Auto.
   */
  resolveContextLength: (customContextLength: number | null) => number;
  /**
   * The slot count the server resolves an unset `--parallel` to, or null when it could not
   * be read. `_resolve_parallel_slots` fills an omitted request from the server-wide
   * default and stores THAT as `requested_parallel_slots`, so an unset ask is never null on
   * the status side and comparing the two directly reloaded every default pick.
   */
  parallelSlots: number | null;
  /**
   * `splitRatio` as the store holds it now, which is what the load sends. Never a config
   * field: `applyPerModelConfigToRuntime` clears it, so applying any remembered config
   * asks for the default distribution rather than the resident custom one.
   */
  splitRatio: number[] | null;
  /**
   * `normalizeSpeculativeType`, passed rather than imported. It lives on the chat runtime
   * store, which reaches React, and this module is deliberately a leaf so the node suite
   * can drive it; copying its mapping here (comma-chained legacy echoes and all) would be
   * a second copy free to drift from the one the load actually uses.
   */
  normalizeSpeculative: (value: string | null | undefined) => string | null;
};

/**
 * One setting the resident load can disagree about. `pinned` is whether the config has an
 * opinion at all, `agrees` whether the running server satisfies it. The pair survives for
 * `llamaExtraArgs`, the only field the applier leaves unresolved.
 */
type SettingCheck = {
  /** Placement, which the backend rewrites wholesale on a preserved CPU fallback. */
  placement?: true;
  /**
   * One of the two fields `_mlx_runtime_settings_match` compares. The non-GGUF branch of
   * /load checks identity and those, then answers already_loaded, so nothing else here
   * may decide against a safetensors or MLX resident.
   */
  mlxComparable?: true;
  /**
   * Placement the diffusion branch of `_runtime_matches_intent` replaces wholesale with
   * one `_diffusion_manual_ngl` comparison, rather than comparing field by field.
   */
  ggufPlacement?: true;
  /** The diffusion branch's own comparison, which has no meaning off it. */
  diffusionOnly?: true;
  /**
   * A chat-only invocation setting. `_runtime_matches_intent` guards these on
   * `not self._is_diffusion`, and the status nulls the ones it publishes at all, so
   * comparing them against a diffusion runtime rejects a load that would deduplicate.
   */
  chatOnly?: true;
  pinned: (config: PerModelConfig) => boolean;
  agrees: (
    config: PerModelConfig,
    status: ResidentRuntime,
    standing: StandingConfigDefaults,
  ) => boolean;
};

/**
 * Fallback reasons the backend retries on an IDENTICAL next load, taken from the arms of
 * `LlamaCppBackend._runtime_matches_intent` that return False to force the repair: a drafter
 * fetch that can be attempted again, and the stand-down the UI asks the user to fix by
 * updating llama.cpp. The rest are not here on purpose. "drafter_no_vram" and
 * "mla_mtp_disabled" are Auto-mode policy, and "runtime_error" only reopens when the draft
 * count changes, which the settings comparison already sees; treating them as repairable
 * would prompt to stop running chats on every re-pick and repair nothing.
 */
const RETRYABLE_SPEC_FALLBACKS = new Set([
  "drafter_not_found",
  "binary_no_mtp",
  "binary_outdated",
]);

/** The two of those the user is told to fix by updating llama.cpp. */
const BINARY_SPEC_FALLBACKS = new Set(["binary_no_mtp", "binary_outdated"]);

/** The modes the backend's retry arms are guarded on; anything else asked for no drafter. */
const SPECULATIVE_MODES = new Set([
  "auto",
  "mtp",
  "mtp+ngram",
  "dspark",
  "dflash",
]);

/**
 * Whether the resident load's speculative decoding is degraded in a way the next identical
 * `/load` would repair.
 *
 * This is the one place where sending a request the runtime already satisfies is not a
 * no-op, so it is the one reason to decline the shortcut on settings that match. The
 * decision stays coarser than the backend's: the status echoes the fallback reason but not
 * `_dflash_retry_needed` or an inconclusive capability probe, so a load that would dedupe
 * on the far side can still be sent. That costs one round trip through `already_loaded`,
 * which returns before any teardown.
 */
export function residentSpeculativeNeedsRepair(
  status: Pick<
    InferenceStatusResponse,
    | "spec_fallback_reason"
    | "spec_fallback_binary_changed"
    | "spec_probe_retry_pending"
    | "spec_dflash_retry_pending"
    | "spec_dspark_sidecar_absent"
    | "spec_drafter_kind"
  >,
  resolvedSpeculativeType: string | null,
  /**
   * Whether the load carries a `gguf_path`, which the route sets from the identifier
   * alone: `source.gguf_path if model_identifier.lower().endswith(".gguf") else None`.
   */
  sendsGgufPath = false,
): boolean {
  const mode = resolvedSpeculativeType ?? "auto";
  // Two arms that record no fallback reason, so the reason check below cannot see them.
  // The probe arm is not gated on a mode; the DFlash one is, as the backend gates it.
  if (status.spec_probe_retry_pending === true) {
    return true;
  }
  if (
    status.spec_dflash_retry_pending === true &&
    (mode === "auto" || mode === "dflash")
  ) {
    return true;
  }
  if (
    !RETRYABLE_SPEC_FALLBACKS.has(status.spec_fallback_reason ?? "") ||
    !SPECULATIVE_MODES.has(mode)
  ) {
    return false;
  }
  // The drafter_not_found arm reloads so the next Apply retries the fetch, but excludes
  // the two kinds whose absence is not transient: DFlash asks through its retry flag
  // above, and an absent DSpark sidecar is the permanent state of every repo but one, so
  // retrying either would relaunch an identical server forever.
  if (status.spec_fallback_reason === "drafter_not_found") {
    // The arm is guarded on `intent.gguf_path is None`, so a standalone file never
    // reaches it and an identical load dedupes rather than retrying the fetch.
    if (sendsGgufPath) {
      return false;
    }
    if (status.spec_drafter_kind === "dflash") {
      return false;
    }
    if (
      status.spec_drafter_kind === "dspark" &&
      status.spec_dspark_sidecar_absent === true
    ) {
      return false;
    }
  }
  // A binary stand-down repairs only once a different llama-server is installed, which is
  // the necessary condition in spec_binary_fallback_can_retry. Without that the reload
  // dedupes and the prompt was for nothing. Only an explicit false settles it: a backend
  // too old to report it keeps the coarser answer rather than suppress a real repair, and
  // the field says nothing about a stand-down that was never about the binary.
  return !(
    BINARY_SPEC_FALLBACKS.has(status.spec_fallback_reason ?? "") &&
    status.spec_fallback_binary_changed === false
  );
}

/** The mode the load sends, which is what the backend gates its placement arms on. */
const requestedGpuMemoryMode = (
  config: PerModelConfig,
  standing: StandingConfigDefaults,
): "auto" | "manual" => config.gpuMemoryMode ?? standing.gpuMemoryMode;

const cleanTemplate = (value: string | null | undefined): string | null =>
  value?.trim() ? value : null;

/**
 * Mirrors the fields `_runtime_matches_intent` reloads for, plus the MLX pair
 * `_mlx_runtime_settings_match` compares.
 *
 * Nearly every check is unconditionally pinned. A config reaches here only after
 * `applyModelLoadConfigToRuntime` wrote it over the runtime store (`chat-page.tsx:3242`,
 * `hub-page.tsx:1329`), and that applier resolves each of these with `?? null`, so the
 * snapshot `performLoad` takes reads null rather than inheriting the resident value: an
 * unset field asks for the default, not for whatever is running. Only `llamaExtraArgs` is
 * genuinely optional, being the one field with no `?? null` fallback.
 */
const SETTING_CHECKS: SettingCheck[] = [
  {
    // Resolved, not compared raw: an unset length is Auto, which the load sends as 0 for a
    // cross-model GGUF pick and as the resident context when re-picking the same one.
    // Reading null as "no opinion" against a status echoing either number was a reload.
    pinned: () => true,
    // A GGUF invocation field: a safetensors or MLX status never sets it, and the general
    // non-GGUF rule below is what keeps this from reading that absence as 0.
    agrees: (c, s, standing) =>
      standing.resolveContextLength(c.customContextLength ?? null) ===
      (s.requested_context_length ?? 0),
  },
  {
    pinned: () => true,
    agrees: (c, s) => (c.kvCacheDtype ?? null) === (s.cache_type_kv ?? null),
  },
  {
    mlxComparable: true,
    pinned: () => true,
    agrees: (c, s) =>
      (c.mlxKvBits ?? null) === (s.mlx_kv_bits_requested ?? null),
  },
  {
    // Always pinned: an unset mode resolves to the standing preference, and the load sends
    // it. Reading it as silence let a pick asking for "off" adopt a resident MTP runtime.
    pinned: () => true,
    agrees: (c, s, standing) =>
      (standing.normalizeSpeculative(c.speculativeType) ??
        standing.speculativeType) ===
      (standing.normalizeSpeculative(s.speculative_type) ??
        standing.speculativeType),
  },
  {
    // Only when the pick names one, as _runtime_matches_intent is guarded on
    // `intent.spec_draft_n_max is not None`: an unset limit asks for no change, so
    // comparing null against the count the resident load was launched with was a reload.
    pinned: (c) => c.specDraftNMax != null,
    agrees: (c, s) => c.specDraftNMax === (s.spec_draft_n_max ?? null),
  },
  {
    chatOnly: true,
    // Unknown default: null against the status's resolved count is a reload, the safe
    // direction. defaultParallelSlots is the EFFECTIVE count, so a build that clamps to one
    // slot reloads rather than adopts, which is the same bias.
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.nParallel ?? standing.parallelSlots) ===
      (s.requested_parallel_slots ?? standing.parallelSlots),
  },
  {
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) => (c.nBatch ?? null) === (s.requested_n_batch ?? null),
  },
  {
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) => (c.nUbatch ?? null) === (s.requested_n_ubatch ?? null),
  },
  {
    // Not nullable, so it always has an opinion; a status omitting it ran without.
    pinned: () => true,
    agrees: (c, s) =>
      // The toggle after a pass-through --split-mode has last-won over it, which is what
      // resolve_tensor_parallel hands the comparator.
      resolveTensorParallel(c.llamaExtraArgs, c.tensorParallel) ===
        (s.tensor_parallel ?? false) ||
      // Rewritten away with the rest of the placement on a virtualised Metal device.
      s.gpu_placement_paravirtual === true ||
      // A split the architecture gate normalized away: that layer-mode runtime IS this
      // request as the gate rewrote it, and the backend accepts it back unchanged. The
      // resolved mode again, since a pass-through --split-mode is what was gated.
      (resolveTensorParallel(c.llamaExtraArgs, c.tensorParallel) &&
        s.tensor_parallel !== true &&
        s.tensor_parallel_dropped_by_arch_gate === true),
  },
  {
    // Blank-trimmed on both ends: the applier and the load both send "" as null.
    mlxComparable: true,
    pinned: () => true,
    agrees: (c, s) =>
      cleanTemplate(c.chatTemplateOverride) ===
      cleanTemplate(s.chat_template_override),
  },
  {
    // undefined: never read the stored value. null: the user cleared the box, which agrees
    // only with a load invoked with no pass-through args.
    chatOnly: true,
    pinned: (c) => c.llamaExtraArgs !== undefined,
    agrees: (c, s, standing) =>
      sameList(
        requestedGpuMemoryMode(c, standing) === "manual"
          ? stripManagedOffloadFlags(c.llamaExtraArgs)
          : c.llamaExtraArgs,
        s.requested_llama_extra_args,
      ),
  },
  {
    // Standing preference, like the speculative mode above.
    placement: true,
    ggufPlacement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.gpuMemoryMode ?? standing.gpuMemoryMode) ===
      (s.gpu_memory_mode ?? standing.gpuMemoryMode),
  },
  {
    // Resolves to GPU_LAYERS_AUTO rather than to a preference, but the load still sends it.
    // Only under Manual, as _runtime_matches_intent compares it: under Auto the fitter
    // chooses the offload, so the layer count the load carries decides nothing.
    placement: true,
    ggufPlacement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      requestedGpuMemoryMode(c, standing) !== "manual" ||
      requestedGpuLayers(c, standing) === (s.gpu_layers ?? standing.gpuLayers),
  },
  {
    // Manual with a non-negative pin, the same guard the backend uses. A config keeps a
    // hidden nCpuMoe after the layer slider goes back to Auto, and llama.cpp records 0,
    // so comparing it there rejected an otherwise identical runtime.
    placement: true,
    ggufPlacement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      requestedGpuMemoryMode(c, standing) !== "manual" ||
      requestedGpuLayers(c, standing) < 0 ||
      (c.nCpuMoe ?? standing.nCpuMoe) === (s.n_cpu_moe ?? standing.nCpuMoe),
  },
  {
    // Absent resolves to a null selection in the applier, and a null selection is sent as
    // Automatic, so this is pinned like the rest: an unset pick does not adopt a server
    // that was placed on a chosen GPU. Reconciled first, since that is what /load sends.
    placement: true,
    pinned: () => true,
    agrees: (c, s, standing) => {
      const reconciled = standing.reconcileGpuIds(
        c.selectedGpuIds ?? null,
        c.selectedGpuIndexKind,
      );
      // The diffusion runner drives one device, so matches_gpu_ids reduces the request to
      // its lowest id and the status reports only that. Comparing the configured set
      // rejected a runtime the backend would have called identical.
      const pick =
        s.is_diffusion === true && reconciled?.length
          ? [Math.min(...reconciled)]
          : reconciled;
      if (sameGpuSet(pick, s.requested_gpu_ids)) {
        return true;
      }
      // Either pool, as matches_gpu_ids accepts either: fitting may narrow the request to
      // the smallest subset that holds the model, and asking for that subset does not
      // reload. Comparing only the raw request prompted for a load that dedupes. Guarded on
      // a non-empty echo, since an absent one is no placement rather than Automatic, and
      // reading it as Automatic would make an unpinned pick match every pinned server.
      return Boolean(s.gpu_ids?.length) && sameGpuSet(pick, s.gpu_ids);
    },
  },
  {
    // The split is placement the config cannot carry: the applier clears splitRatio, so a
    // remembered config asks for the default distribution while a resident manual load may
    // be running a custom one. Omitting it kept that custom split with nothing saying so.
    placement: true,
    ggufPlacement: true,
    pinned: () => true,
    agrees: (_c, s, standing) => sameList(standing.splitRatio, s.tensor_split),
  },
  {
    // A managed override the backend would reject outright. It raises there, so the load
    // fails and says so; folding it into "no override" here would strip the token, find
    // the rest agreeable and adopt, losing the saved setting in silence.
    pinned: (c) => c.llamaExtraArgs !== undefined,
    agrees: (c, _s, standing) =>
      requestedGpuMemoryMode(c, standing) !== "manual" ||
      parseGpuLayersOverride(c.llamaExtraArgs).kind !== "invalid",
  },
  {
    // What the diffusion branch compares in place of the placement fields above:
    // _diffusion_manual_ngl, the layer count only under Manual with a non-negative pin and
    // the runner's own default otherwise. An older shim that dropped a manual NGL leaves
    // the status reporting Auto while keeping the request here, so comparing the mode raw
    // rejected a load that deduplicates.
    diffusionOnly: true,
    pinned: () => true,
    agrees: (c, s, standing) => {
      const ngl = diffusionManualNgl(c, standing);
      if (ngl !== (s.diffusion_requested_ngl ?? null)) {
        return false;
      }
      // The request is retained even when an older shim dropped the split, so once the
      // installed shim gains --ngl support the same request has to go through and finally
      // apply it. The backend rejects it for exactly that window.
      return !(
        ngl !== null &&
        s.gpu_layers !== ngl &&
        s.diffusion_split_supported === true
      );
    },
  },
];

/**
 * The layer count `/load` ends up with. Under manual a pass-through `-ngl` is copied into
 * the first-class field before the comparator runs, so the raw toggle is not what was
 * requested.
 */
function requestedGpuLayers(
  config: PerModelConfig,
  standing: StandingConfigDefaults,
): number {
  const override =
    requestedGpuMemoryMode(config, standing) === "manual"
      ? parseGpuLayersOverride(config.llamaExtraArgs)
      : { kind: "absent" as const };
  return override.kind === "value"
    ? override.layers
    : (config.gpuLayers ?? standing.gpuLayers);
}

/** `_diffusion_manual_ngl`: only an explicit manual count reaches the child. */
function diffusionManualNgl(
  config: PerModelConfig,
  standing: StandingConfigDefaults,
): number | null {
  const layers = requestedGpuLayers(config, standing);
  return requestedGpuMemoryMode(config, standing) === "manual" && layers >= 0
    ? layers
    : null;
}

/**
 * Whether the backend would rewrite this request into the resident CPU fallback.
 *
 * `adopt_load_intent_if_matched` runs `_preserve_cpu_fallback_intent` first, so after a
 * Vulkan startup crash an Auto request becomes the resident manual/zero-layer intent and
 * dedupes. Comparing placement literally rejected it and raised the prompt this PR removes.
 *
 * Mirrors `_cpu_fallback_request_eligible`, minus its environment terms: the resident
 * server already fell back under this same env, and a request carrying its own placement
 * args is excluded here rather than guessed at.
 */
function cpuFallbackPlacementPreserved(
  config: PerModelConfig,
  status: ResidentRuntime,
  standing: StandingConfigDefaults,
): boolean {
  if (status.cpu_fallback_reason !== "vulkan_startup_crash") {
    return false;
  }
  const mode = config.gpuMemoryMode ?? standing.gpuMemoryMode;
  const layers = config.gpuLayers ?? standing.gpuLayers;
  return (
    (mode === "auto" || (mode === "manual" && layers === 0)) &&
    !standing.reconcileGpuIds(
      config.selectedGpuIds ?? null,
      config.selectedGpuIndexKind,
    )?.length &&
    !config.tensorParallel &&
    !standing.splitRatio?.length &&
    (config.nCpuMoe ?? standing.nCpuMoe) === 0 &&
    !config.llamaExtraArgs?.length
  );
}

/**
 * Whether the resident load already runs the settings this pick would ask for.
 *
 * Identity is not the whole of a load: `LlamaCppBackend` reuses a running server only when
 * the request also agrees on context, KV dtype, slots, batch sizes, placement, speculative
 * mode, chat template and pass-through args. An unset field is still an opinion, since the
 * applier resolves it before the load reads it: unset asks for the default, not for whatever
 * is running. What keeps the common case working is the other door: a model the user never
 * configured arrives with no config at all, and a model they did configure carries what
 * `currentRuntimePerModelConfig` wrote from the load that made it resident.
 *
 * The bias is one-sided on purpose. "Differs" costs one reload, which is what happened
 * before any of this existed; a wrong "matches" leaves the user on settings they did not
 * ask for, with the panel rolled back to the resident model so nothing says so.
 *
 * `maxSeqLength` is not compared: a client-side generation cap no status echoes, and it
 * never reaches llama-server's invocation.
 */
export function residentRuntimeMatchesConfig(
  status: ResidentRuntime,
  config: PerModelConfig | null | undefined,
  standing: StandingConfigDefaults,
): boolean {
  // No config at all is not the same as a config that pins nothing: with none, the load
  // path reads the live runtime, which was hydrated from the resident model, so there is
  // nothing that could differ.
  if (!config) {
    return true;
  }
  const placementPreserved =
    // A virtualised Metal device pins every GGUF request to the CPU before either
    // comparator runs, so placement cannot tell two requests apart there at all.
    status.gpu_placement_paravirtual === true ||
    cpuFallbackPlacementPreserved(config, status, standing);
  // The diffusion runner takes no --parallel, no batch sizes and no pass-through args, so
  // the backend does not compare them and the status nulls what it publishes at all.
  const diffusion = status.is_diffusion === true;
  return SETTING_CHECKS.every(
    (check) =>
      // The non-GGUF branch of /load checks identity and the MLX pair, then answers
      // already_loaded, so no llama.cpp invocation field may decide against one.
      (status.is_gguf === false && !check.mlxComparable) ||
      (diffusion && check.chatOnly) ||
      (diffusion && check.ggufPlacement) ||
      (!diffusion && check.diffusionOnly) ||
      (check.placement && placementPreserved) ||
      !check.pinned(config) ||
      check.agrees(config, status, standing),
  );
}
