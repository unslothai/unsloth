// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * VRAM estimate for one downloaded model, for the picker's memory bar.
 *
 * Per row, because the rows are spread across a dozen render sites. The work is
 * shared though: `/kv-cache-estimate` reads GGUF metadata off disk, so answers
 * are cached by the inputs that change them and duplicate requests fold into
 * one. Ten rows of the same model cost one read.
 */

// Deep paths, not the `@/features/chat` barrel, and that is load-bearing. The barrel
// reaches this file back: chat -> apply-inference-status-to-store -> model-picker ->
// model-selector -> pickers -> here. Importing the barrel closed that ring, and under
// dev's unbundled ESM the app died on a blank page with "Cannot access
// 'CHAT_GPU_MEMORY_MODE_KEY' before initialization". Production builds hid it, because
// the bundler hoists the declarations into one module and the ordering stops mattering.
import { estimateKvCache } from "@/features/chat/api/chat-api";
import {
  CHAT_GPU_MEMORY_MODE_KEY,
  CHAT_SPECULATIVE_TYPE_KEY,
} from "@/features/chat/stores/chat-runtime-keys";
import {
  readPersistedGpuMemoryMode,
  readPersistedSpeculativeType,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "@/features/hub/lib/model-identity";
import {
  PER_MODEL_CONFIG_STORAGE_KEY,
  PER_MODEL_CONFIG_UPDATED_EVENT,
  type PerModelConfig,
  listPerModelConfigs,
} from "@/features/model-picker/model-config/per-model-config";
import {
  loadVramBudgetSettings,
  subscribeVramBudgetSettings,
} from "@/features/settings/api/vram-budget";
import {
  type ModelMemorySegments,
  computeModelMemory,
  estimateCacheKey,
  estimateIsUnsized,
  extraArgsAddResidentFiles,
  extraArgsOwnPlacement,
  extraArgsShapeKvCache,
} from "@/lib/model-memory";
import { useInferenceGpuInfo } from "./use-gpu-info";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";

/** The on-disk model a row would load. */
export interface ModelMemorySource {
  repoId: string;
  quant: string;
  /** Size from the listing, so the bar draws before the estimate arrives. */
  sizeBytes?: number | null;
  /**
   * The concrete thing this row loads, when the listing resolved one.
   *
   * Selecting the row loads this, while the estimate resolved by repo id alone,
   * so on a duplicate or recovered HF cache the two could pick different copies
   * of the same quant and the bar would describe a model the click does not
   * open. Falls back to the repo id upstream, so sending it changes nothing for
   * an ordinary row.
   */
  loadId?: string | null;
}

interface Estimate {
  kvBytes: number | null;
  weightsBytes: number | null;
  specBytes: number | null;
  /** Context the KV figure was computed at, for the per-token rate. */
  nCtx: number;
  /** Vision projector footprint, resident alongside the weights. */
  projectorBytes: number | null;
  /** The host-heap share of kvBytes (SWA checkpoint snapshots), never on the card. */
  kvCheckpointBytes: number | null;
  /** The share of specBytes no shorter context can reduce (the drafter's weights). */
  specFixedBytes: number | null;
  /** llama.cpp's compute buffers, which every launch reserves. */
  computeBytes: number | null;
  /** False only when the loader is free to shrink the context to fit. */
  contextIsPinned: boolean | null;
  /** An inherited LLAMA_ARG_DEVICE confines the launch to fewer cards than the
   *  aggregate budget credits. */
  inheritedDevicePin: boolean | null;
  /** The planner's own GPU-resident total, which supersedes the segment sum. */
  gpuTotalBytes: number | null;
  /** What the planner still reserves at the shortest context. */
  gpuFloorBytes: number | null;
  /** The configured drafter could not be priced, so the total is a floor. */
  specUnpriced: boolean;
}

const CACHE = new Map<string, Estimate>();
const IN_FLIGHT = new Map<string, Promise<Estimate>>();

/**
 * Cap on remembered answers. A session can browse a lot of models, and each
 * entry is keyed by settings as well as identity, so the key space is larger
 * than the model count. Oldest-first eviction; a re-fetch costs one disk read.
 */
const CACHE_LIMIT = 256;

/**
 * How long a failure is remembered. Failures are cached so an unsizable model
 * doesn't re-request on every reopen, but the common cause is the backend being
 * briefly unavailable -- without expiry those rows would stay blank for the rest
 * of the session.
 */
const FAILURE_TTL_MS = 30_000;

const failedAt = new Map<string, number>();

function remember(cacheKey: string, estimate: Estimate, failed: boolean): void {
  // Only a new key grows the map, so re-pricing an existing one must not evict
  // an unrelated row to make room it does not need.
  if (CACHE.size >= CACHE_LIMIT && !CACHE.has(cacheKey)) {
    const oldest = CACHE.keys().next().value;
    if (oldest !== undefined) {
      CACHE.delete(oldest);
      failedAt.delete(oldest);
    }
  }
  CACHE.set(cacheKey, estimate);
  if (failed) failedAt.set(cacheKey, Date.now());
  else failedAt.delete(cacheKey);
}

function cached(cacheKey: string): Estimate | undefined {
  const hit = CACHE.get(cacheKey);
  if (!hit) return undefined;
  const failedTime = failedAt.get(cacheKey);
  if (failedTime !== undefined && Date.now() - failedTime > FAILURE_TTL_MS) {
    CACHE.delete(cacheKey);
    failedAt.delete(cacheKey);
    return undefined;
  }
  return hit;
}

const MISS: Estimate = {
  kvBytes: null,
  weightsBytes: null,
  specBytes: null,
  nCtx: 0,
  projectorBytes: null,
  kvCheckpointBytes: null,
  specFixedBytes: null,
  computeBytes: null,
  contextIsPinned: null,
  inheritedDevicePin: null,
  gpuTotalBytes: null,
  gpuFloorBytes: null,
  specUnpriced: false,
};

/**
 * Re-read saved settings whenever any of them are written.
 *
 * Two sources, because a model with no override follows the standing
 * preference: the per-model configs, which announce themselves, and the
 * standing GPU-memory mode and speculative type, which live in localStorage and
 * so raise no same-tab event of their own. The runtime store changes in lockstep
 * with those writes in-session, so it stands in as their notification.
 */
function subscribeToConfigChanges(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const onConfigWrite = () => {
    configsDirty = true;
    onChange();
  };
  window.addEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, onConfigWrite);
  // The custom event above is same-tab only, by construction: the native
  // storage event fires in every OTHER document sharing the origin and never in
  // the one that made the write. So settings edited in a second Studio tab
  // arrive here as a storage event and nowhere else, and without this a mounted
  // bar kept its old fit or OOM verdict indefinitely.
  const onStorage = (event: Event) => {
    const key = (event as StorageEvent).key;
    // A null key means the whole store was cleared, which counts.
    if (key == null || WATCHED_STORAGE_KEYS.includes(key)) onConfigWrite();
  };
  window.addEventListener("storage", onStorage);
  const unsubscribeStore = useChatRuntimeStore.subscribe(onChange);
  return () => {
    window.removeEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, onConfigWrite);
    window.removeEventListener("storage", onStorage);
    unsubscribeStore();
  };
}

/** localStorage keys whose cross-tab writes change what the bar should show. */
const WATCHED_STORAGE_KEYS = [
  PER_MODEL_CONFIG_STORAGE_KEY,
  CHAT_GPU_MEMORY_MODE_KEY,
  CHAT_SPECULATIVE_TYPE_KEY,
];

/** Subscriber for a disabled bar: nothing to watch, nothing to unwatch. */
const subscribeNothing = () => () => {};

/** Snapshot for a disabled bar. Constant, so it can never force a re-render. */
const readZeroEpoch = () => 0;

let configEpoch = 0;
let lastConfigSignature = "";
let lastPrefSignature = "";
// Serialising every saved config is the expensive half, and only a config write
// can change it. The store ticks on every streamed token, so doing that work on
// each tick would put an O(all configs) stringify in the render path.
let configsDirty = true;

/**
 * A value that changes whenever any saved config does, so the estimate can be
 * re-keyed. The settings sheet sits directly beside these rows, and without
 * this a context change leaves the bar showing the old KV segment.
 */
function readConfigEpoch(): number {
  // The standing GPU-memory mode and speculative type matter as much as the
  // saved configs: a model with no override follows them. Without them,
  // switching the global mode to Manual left every mounted bar drawn against a
  // budget the load would no longer use, and turning MTP on globally left the
  // draft reserve missing, until the row happened to remount. Both are two
  // short strings, so they are cheap to check on every read.
  // The session GPU pin belongs here because budgetIsMeaningful reads it. It
  // lives in the runtime store rather than in a saved config, so a preset or a
  // session-only load configuration changes it with no config write: the
  // whole-store subscription did fire, but this snapshot did not move, so
  // useSyncExternalStore suppressed the rerender and a bar stayed drawn against
  // aggregate multi-GPU VRAM after the launch had been pinned to one card.
  const pin = useChatRuntimeStore.getState();
  // The session speculative mode belongs here for the same reason: it is read
  // above and never written to a config, so nothing else would move the epoch.
  const pinSignature = `${(pin.selectedGpuIds ?? []).join(",")} ${pin.selectedGpuIndexKind ?? ""} ${pin.speculativeType ?? ""}`;
  const prefSignature = `${readPersistedGpuMemoryMode()} ${readPersistedSpeculativeType()} ${pinSignature}`;
  if (prefSignature !== lastPrefSignature) {
    lastPrefSignature = prefSignature;
    configEpoch += 1;
  }
  if (configsDirty) {
    configsDirty = false;
    const signature = JSON.stringify(listPerModelConfigs());
    if (signature !== lastConfigSignature) {
      lastConfigSignature = signature;
      configEpoch += 1;
    }
  }
  return configEpoch;
}

/**
 * The user's saved settings for this exact variant, if any.
 *
 * Variant-exact only, matching `resolveInitialConfig`: the load path looks up
 * (modelId, ggufVariant) and drops to defaults when there's no hit. Falling back
 * to a model-level entry here would size the bar from settings the variant will
 * never load with.
 *
 * Keys are stored normalized, which lower-cases hub repo ids, so comparing raw
 * strings would miss every mixed-case repo and quietly use defaults instead of
 * the context length and KV dtype the user chose.
 */
function configFor(source: ModelMemorySource): PerModelConfig | undefined {
  const wantId = normalizeModelIdentity(source.repoId);
  const wantVariant = normalizeGgufVariantIdentity(source.quant);
  return listPerModelConfigs().find(
    (e) =>
      normalizeModelIdentity(e.modelId) === wantId &&
      normalizeGgufVariantIdentity(e.ggufVariant) === wantVariant,
  )?.config;
}

/** A context the user pinned, or undefined to let the model use its own. */
function pinnedContext(config: PerModelConfig | undefined): number | undefined {
  return config?.customContextLength || config?.maxSeqLength || undefined;
}

/**
 * Whether the budget we're handed still describes where this model will load.
 *
 * Callers pass the total across visible GPUs. Pin the model to a subset, or
 * offload layers to CPU, and that total stops being the ceiling the model is
 * judged against -- charting against it would call a 30 GB quant "fits" on a
 * 2x24 GB host pinned to one card. Better to draw nothing than to mislead.
 *
 * The "at default" tests mirror `gpuFieldsAtDefault` in per-model-config: note
 * that a negative gpuLayers is the runtime's Auto value, not an override, so a
 * truthiness check here would hide the bar for ordinary saved configs. GPU mode
 * falls back to the standing preference, which is what the loader does when a
 * model has no override of its own.
 */
function budgetIsMeaningful(config: PerModelConfig | undefined): boolean {
  const mode = config?.gpuMemoryMode ?? readPersistedGpuMemoryMode();
  if (mode === "manual") return false;
  // A session pin lives in the runtime store rather than in a saved config, so
  // a user who pinned cards without ever saving a per-model override reached
  // none of the tests below and was charted against the whole multi-GPU sum.
  const sessionPin = useChatRuntimeStore.getState().selectedGpuIds;
  if (sessionPin != null && sessionPin.length > 0) return false;
  if (!config) return true;
  // Pass-through args are appended after Unsloth's own flags, so an -ngl or a
  // device pin in that box is what the launch actually uses. Reading only the
  // structured fields left the bar charting a CPU-offloaded run against every
  // GPU on the host.
  if (extraArgsOwnPlacement(config.llamaExtraArgs)) return false;
  // Same reasoning one term over: these do not move the cache, they resize it.
  // The estimate is built from the structured controls, so a --swa-full in the
  // box has the launch reserve a full-context cache while the bar priced the
  // compact sliding window, and the row reads as a comfortable fit.
  if (extraArgsShapeKvCache(config.llamaExtraArgs)) return false;
  // And these add resident files nothing here priced -- a LoRA, a control
  // vector, a hand-named drafter. The total would be short by whatever they
  // weigh, with no sign of it in the bar.
  if (extraArgsAddResidentFiles(config.llamaExtraArgs)) return false;
  return (
    config.selectedGpuIds == null &&
    (config.gpuLayers == null || config.gpuLayers < 0) &&
    (config.nCpuMoe == null || config.nCpuMoe === 0)
  );
}

/**
 * The speculative mode the load will actually use. A null per-model value means
 * "follow the standing preference", which is where the loader looks too --
 * sending nothing would omit the draft reserve until a per-model override is
 * saved.
 */
function effectiveSpeculativeType(
  config: PerModelConfig | undefined,
): string | undefined {
  // The live runtime value before the persisted one. Forced modes (mtp,
  // mtp+ngram, dspark, dflash) are deliberately session-only and are never
  // returned by readPersistedSpeculativeType, so a model loaded with one of them
  // but not remembered was priced as auto -- and auto drops MTP on an MLA target
  // while the forced load engages it and opens another full cache.
  return (
    config?.speculativeType ??
    useChatRuntimeStore.getState().speculativeType ??
    readPersistedSpeculativeType()
  );
}

async function fetchEstimate(
  cacheKey: string,
  source: ModelMemorySource,
  nCtx: number | undefined,
  config: PerModelConfig | undefined,
): Promise<Estimate> {
  const known = cached(cacheKey) ?? IN_FLIGHT.get(cacheKey);
  if (known) return known;

  const run = estimateKvCache(source.repoId, source.quant, nCtx, {
    cacheTypeKv: config?.kvCacheDtype,
    nParallel: config?.nParallel,
    speculativeType: effectiveSpeculativeType(config),
    specDraftNMax: config?.specDraftNMax,
    specDraftCacheType: config?.specDraftCacheDtype,
    ctxCheckpoints: config?.ctxCheckpoints,
    disableVision: config?.disableVision,
    nBatch: config?.nBatch,
    nUbatch: config?.nUbatch,
    tensorParallel: config?.tensorParallel,
  })
    .then((r) => {
      const estimate: Estimate = {
        kvBytes: r.kv_bytes,
        weightsBytes: r.weights_bytes,
        specBytes: r.spec_bytes,
        nCtx: r.n_ctx ?? 0,
        projectorBytes: r.projector_bytes ?? null,
        kvCheckpointBytes: r.kv_checkpoint_bytes ?? null,
        specFixedBytes: r.spec_fixed_bytes ?? null,
        computeBytes: r.compute_bytes ?? null,
        contextIsPinned: r.context_is_pinned ?? null,
        inheritedDevicePin: r.inherited_device_pin ?? null,
        gpuTotalBytes: r.gpu_bytes ?? null,
        gpuFloorBytes: r.gpu_floor_bytes ?? null,
        specUnpriced: r.spec_unpriced === true,
      };
      // A 200 that could size nothing arrives down the success path, so
      // remembering it as a success pinned the row blank for the rest of the
      // session, which is exactly what FAILURE_TTL_MS exists to stop.
      remember(cacheKey, estimate, estimateIsUnsized(estimate));
      return estimate;
    })
    // A model we can't size still draws its weights. The miss is remembered
    // briefly so a long list doesn't retry on every reopen, then expires.
    .catch(() => {
      remember(cacheKey, MISS, true);
      return MISS;
    })
    .finally(() => IN_FLIGHT.delete(cacheKey));

  IN_FLIGHT.set(cacheKey, run);
  return run;
}

/**
 * Bar geometry for one row. Pass no source for rows that aren't on disk; the
 * hook then does nothing and reports "unknown", which draws nothing.
 */
export function useModelMemory(
  source: ModelMemorySource | undefined,
  gpuGb?: number | null,
): ModelMemorySegments {
  // Held with its key so a source change invalidates the old answer by
  // comparison rather than by clearing state from inside an effect.
  const [entry, setEntry] = useState<{
    key: string;
    estimate: Estimate;
  } | null>(null);
  // Opt-in, off by default. Checked here rather than at each render site so a
  // disabled bar also costs no request.
  const enabled = useChatRuntimeStore((state) => state.showMemoryBar);

  // Read before the subscription below, because that subscription is not free:
  // subscribeToConfigChanges watches the whole chat runtime store, which ticks on
  // every streamed token. Every row in the list mounts this hook, so subscribing
  // unconditionally meant each token woke every mounted row to re-read a snapshot
  // it would not use -- including with the feature switched off entirely. Both
  // branches are module-level constants, so swapping between them resubscribes
  // once on toggle rather than on every render.
  // `source` as well as `enabled`: an unselected picker row calls this hook with
  // no source and can never draw a bar, but it still reached the subscription
  // and so re-read the epoch -- localStorage included -- on every streamed
  // token. A long list multiplied each token by the number of mounted rows.
  const watching = enabled && source != null;
  const epoch = useSyncExternalStore(
    watching ? subscribeToConfigChanges : subscribeNothing,
    watching ? readConfigEpoch : readZeroEpoch,
    () => 0,
  );

  // Whether the number the caller handed us is a dedicated VRAM pool at all.
  // On a Vulkan iGPU the backend reports free shared system RAM minus a host
  // reserve, so the same model flips between "fits" and "OOM likely" as the
  // desktop's RAM moves; on Apple the figure is the entire machine's RAM, which
  // says "fits" for almost anything Metal would refuse. Neither is a VRAM
  // ceiling, so the bar declines to draw rather than warn against the wrong
  // pool. Known gap: a ROCm APU reports its whole GTT pool and is not flagged
  // as shared, which needs a unified-memory signal in hardware.py to fix.
  const inferenceGpu = useInferenceGpuInfo();
  const budgetIsDedicatedVram =
    !inferenceGpu.sharedMemory &&
    // A ROCm APU reports the GTT/system pool, which moves with host usage and is
    // not an independent ceiling. The backend classifies these positively; it is
    // reported as its own field rather than through shared_memory, which the
    // total and free aggregates already act on.
    !inferenceGpu.unifiedMemory &&
    inferenceGpu.backend !== "mlx";

  // The loader's own admission fraction, which the user can change. Cached and
  // shared, so a long list of rows costs one request.
  const [budgetFraction, setBudgetFraction] = useState<number | null>(null);

  const repoId = source?.repoId;
  const quant = source?.quant;
  const sizeBytes = source?.sizeBytes;
  const loadId = source?.loadId;
  // Keyed on primitives rather than the object: callers build the source inline,
  // so a fresh identity each render would loop forever.
  const plan = useMemo(() => {
    // A direct .gguf selection has no quant label and does not need one: the
    // path names the weights outright and the route resolves such a file to
    // itself. Requiring a label here suppressed the bar for exactly the custom
    // and LM Studio models the direct-file support was added for.
    const isDirectGgufFile = (loadId ?? "").toLowerCase().endsWith(".gguf");
    if (!enabled || !repoId || (!quant && !isDirectGgufFile)) return null;
    // Empty rather than undefined past this point: a direct file legitimately
    // has no label, and the route resolves such a path without one, but every
    // consumer below wants a string it can key and send.
    const quantLabel = quant ?? "";
    const config = configFor({ repoId, quant: quantLabel });
    const nCtx = pinnedContext(config);
    const cacheKey = estimateCacheKey({
      repoId: loadId || repoId,
      quant: quantLabel,
      sizeBytes,
      nCtx,
      kvCacheDtype: config?.kvCacheDtype,
      speculativeType: effectiveSpeculativeType(config),
      nParallel: config?.nParallel,
      specDraftNMax: config?.specDraftNMax,
      specDraftCacheType: config?.specDraftCacheDtype,
      ctxCheckpoints: config?.ctxCheckpoints,
      disableVision: config?.disableVision,
      nBatch: config?.nBatch,
      nUbatch: config?.nUbatch,
      tensorParallel: config?.tensorParallel,
    });
    return {
      // Identity for the request is the row's own load target; the saved config
      // is still looked up by repo id, which is how it is keyed.
      source: { repoId: loadId || repoId, quant: quantLabel },
      config,
      nCtx,
      cacheKey,
      trustBudget: budgetIsMeaningful(config) && budgetIsDedicatedVram,
    };
    // `epoch` is a real dependency the linter cannot see: `configFor` reads
    // localStorage, and epoch is what changes when that storage does. Folding it
    // into the cache key instead would evict every row's answer on any save.
    // biome-ignore lint/correctness/useExhaustiveDependencies: see above
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, repoId, loadId, quant, sizeBytes, epoch, budgetIsDedicatedVram]);

  // Gated on a real plan, not merely on the feature being on. Every row in the
  // list mounts this hook, including remote and undownloaded ones that will never
  // draw a bar, and loadVramBudgetSettings folds only calls that are already in
  // flight together -- it holds no read-through cache. Gating on `enabled` alone
  // therefore turned scrolling or filtering a long list into a repeated request
  // per row that appeared.
  useEffect(() => {
    if (!plan) return;
    let alive = true;
    const unsubscribe = subscribeVramBudgetSettings((s) => {
      if (alive) setBudgetFraction(s.fraction);
    });
    void loadVramBudgetSettings()
      .then((s) => {
        // Null on a backend too old to serve the route; the fallback covers it.
        if (alive && s) setBudgetFraction(s.fraction);
      })
      .catch(() => {
        // Falls back to the shared headroom ratio, which is what the fit badge
        // beside the bar already uses.
      });
    return () => {
      alive = false;
      unsubscribe();
    };
  }, [plan]);

  useEffect(() => {
    if (!plan) return;
    let alive = true;
    void fetchEstimate(plan.cacheKey, plan.source, plan.nCtx, plan.config).then(
      (estimate) => {
        if (alive) setEntry({ key: plan.cacheKey, estimate });
      },
    );
    return () => {
      alive = false;
    };
  }, [plan]);

  // The dedicated aggregate is the only figure that is VRAM beside system RAM.
  // Falls back to the supplied total when the inventory reports no shared device
  // at all, which is the ordinary discrete-card host and the two agree there.
  const budgetGb =
    inferenceGpu.dedicatedMemoryTotalGb > 0 &&
    inferenceGpu.dedicatedMemoryTotalGb < inferenceGpu.memoryTotalGb
      ? inferenceGpu.dedicatedMemoryTotalGb
      : gpuGb;

  return useMemo(() => {
    if (!plan?.trustBudget) return computeModelMemory({});
    const estimate = entry?.key === plan.cacheKey ? entry.estimate : undefined;
    // A drafter we could not price is the launch's largest single allocation
    // (a DSpark sidecar runs to about 11 GB). Charting the rest would read as a
    // comfortable fit for a load that is nothing of the sort, so draw nothing.
    if (estimate?.specUnpriced) return computeModelMemory({});
    // The environment can pin the launch to a subset of the cards the aggregate
    // budget credits, and an automatic launch preserves that pin. budgetIsMeaningful
    // can only see browser-side pins and saved config, so this is the one placement
    // override it cannot reach: without it a 30 GiB model reads as fitting 2x24 GiB
    // while the child is confined to one of them.
    if (estimate?.inheritedDevicePin) return computeModelMemory({});
    const weights = estimate?.weightsBytes ?? source?.sizeBytes;
    return computeModelMemory({
      // The projector is resident alongside the weights, so it belongs in that
      // segment rather than as a fourth sliver the eye cannot resolve.
      weightsBytes:
        weights == null ? weights : weights + (estimate?.projectorBytes ?? 0),
      // Context checkpoints are part of the cache, but llama.cpp keeps those
      // snapshots in host heap: the load planner's GPU figure is
      // kv_bytes - kv_checkpoint_bytes. Charged against a VRAM bar they warn OOM
      // over memory that never reaches the card.
      kvBytes:
        estimate?.kvBytes == null
          ? estimate?.kvBytes
          : Math.max(0, estimate.kvBytes - (estimate.kvCheckpointBytes ?? 0)),
      specBytes: estimate?.specBytes,
      specFixedBytes: estimate?.specFixedBytes,
      computeBytes: estimate?.computeBytes,
      gpuTotalBytes: estimate?.gpuTotalBytes,
      gpuFloorBytes: estimate?.gpuFloorBytes,
      nCtx: estimate?.nCtx,
      // The dedicated-only aggregate, never the combined one. On a discrete card
      // beside a Vulkan iGPU the combined total adds the iGPU's allowance, which
      // is a capped view of free system RAM, so a model would be judged against
      // 24 GiB of real VRAM plus whatever the desktop happened not to be using.
      // `sharedMemory` cannot carry this: it is every(), so a mixed host reads
      // false and the gate above lets exactly this case through.
      gpuGb: budgetGb,
      budgetFraction,
      // With no pinned context the estimate is sized at the model's native
      // length, but a default load sends 0 and the loader auto-reduces to the
      // largest context that fits. Warning OOM for a length it would never have
      // tried is the false positive this whole bar exists to avoid, so the
      // verdict softens and only the weights can still fail outright.
      // The route's answer when it gave one: only a context nobody pinned gets
      // auto-fitted, and load_model keeps a positive inherited LLAMA_ARG_CTX_SIZE
      // rather than fitting it. Reading plan.nCtx alone said "auto-fitted" for
      // that launch, which suppressed the overage AND drew only the irreducible
      // floor, so an inherited window over budget reported a comfortable fit.
      contextIsAutoFitted:
        estimate?.contextIsPinned == null
          ? plan.nCtx == null
          : !estimate.contextIsPinned,
    });
  }, [plan, entry, source?.sizeBytes, budgetGb, budgetFraction]);
}
