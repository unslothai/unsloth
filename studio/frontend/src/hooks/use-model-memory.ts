// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Answers are cached by their inputs since `/kv-cache-estimate` reads GGUF metadata off disk. */

// Deep paths, not the `@/features/chat` barrel: the barrel imports this file back, and the cycle
// blanks the page under dev's unbundled ESM.
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

export interface ModelMemorySource {
  repoId: string;
  quant: string;
  /** So the bar draws before the estimate arrives. */
  sizeBytes?: number | null;
  /** What selecting the row loads, so a duplicate HF cache cannot price a different copy. */
  loadId?: string | null;
}

interface Estimate {
  kvBytes: number | null;
  weightsBytes: number | null;
  specBytes: number | null;
  nCtx: number;
  projectorBytes: number | null;
  /** Host-heap share of kvBytes (SWA checkpoints), never on the card. */
  kvCheckpointBytes: number | null;
  /** The drafter's weights, which no shorter context reduces. */
  specFixedBytes: number | null;
  computeBytes: number | null;
  /** False only when the loader may shrink the context to fit. */
  contextIsPinned: boolean | null;
  /** An inherited LLAMA_ARG_DEVICE confines the launch to fewer cards than the budget credits. */
  inheritedDevicePin: boolean | null;
  /** Supersedes the segment sum. */
  gpuTotalBytes: number | null;
  gpuFloorBytes: number | null;
  /** The drafter could not be priced, so the total is a floor. */
  specUnpriced: boolean;
}

const CACHE = new Map<string, Estimate>();
const IN_FLIGHT = new Map<string, Promise<Estimate>>();

/** Keys include settings, so the key space exceeds the model count. Oldest-first eviction. */
const CACHE_LIMIT = 256;

/** Expires so a briefly unavailable backend does not leave rows blank all session. */
const FAILURE_TTL_MS = 30_000;

const failedAt = new Map<string, number>();

function remember(cacheKey: string, estimate: Estimate, failed: boolean): void {
  // Re-pricing an existing key must not evict another row.
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

/** Per-model configs announce themselves; the standing mode and speculative type live in
 * localStorage, so the runtime store stands in as their notification. */
function subscribeToConfigChanges(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const onConfigWrite = () => {
    configsDirty = true;
    onChange();
  };
  window.addEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, onConfigWrite);
  // The storage event fires only in OTHER tabs; that is how a second tab's edits arrive.
  const onStorage = (event: Event) => {
    const key = (event as StorageEvent).key;
    // A null key means the whole store was cleared.
    if (key == null || watchedStorageKeys().includes(key)) onConfigWrite();
  };
  window.addEventListener("storage", onStorage);
  const unsubscribeStore = useChatRuntimeStore.subscribe(onChange);
  return () => {
    window.removeEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, onConfigWrite);
    window.removeEventListener("storage", onStorage);
    unsubscribeStore();
  };
}

/** A function, not a module array: an import cycle means chat-runtime-store's consts can be in
 * their TDZ here. */
const watchedStorageKeys = () => [
  PER_MODEL_CONFIG_STORAGE_KEY,
  CHAT_GPU_MEMORY_MODE_KEY,
  CHAT_SPECULATIVE_TYPE_KEY,
];

const subscribeNothing = () => () => {};

/** Constant, so it can never force a re-render. */
const readZeroEpoch = () => 0;

let configEpoch = 0;
let lastConfigSignature = "";
let lastPrefSignature = "";
// The store ticks per streamed token, so only re-serialise configs after a write.
let configsDirty = true;

/** Re-keys the estimate on any config change, e.g. a context edit beside these rows. */
function readConfigEpoch(): number {
  // The standing mode, speculative type and session GPU pin also change the budget without a config
  // write, so fold them in.
  const pin = useChatRuntimeStore.getState();
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

/** Variant-exact like `resolveInitialConfig`; keys are normalized (hub repo ids lower-cased). */
function configFor(source: ModelMemorySource): PerModelConfig | undefined {
  const wantId = normalizeModelIdentity(source.repoId);
  const wantVariant = normalizeGgufVariantIdentity(source.quant);
  return listPerModelConfigs().find(
    (e) =>
      normalizeModelIdentity(e.modelId) === wantId &&
      normalizeGgufVariantIdentity(e.ggufVariant) === wantVariant,
  )?.config;
}

function pinnedContext(config: PerModelConfig | undefined): number | undefined {
  return config?.customContextLength || config?.maxSeqLength || undefined;
}

/** False when a pin or CPU offload means the visible-GPU total is not the ceiling. A negative
 * gpuLayers is Auto, as in `gpuFieldsAtDefault`. */
function budgetIsMeaningful(config: PerModelConfig | undefined): boolean {
  const mode = config?.gpuMemoryMode ?? readPersistedGpuMemoryMode();
  if (mode === "manual") return false;
  // A session pin lives in the runtime store, not a saved config.
  const sessionPin = useChatRuntimeStore.getState().selectedGpuIds;
  if (sessionPin != null && sessionPin.length > 0) return false;
  if (!config) return true;
  // Pass-through args follow Unsloth's flags, so an -ngl or device pin there wins.
  if (extraArgsOwnPlacement(config.llamaExtraArgs)) return false;
  // These resize the cache (e.g. --swa-full).
  if (extraArgsShapeKvCache(config.llamaExtraArgs)) return false;
  // These add resident files nothing here priced (LoRA, control vector, drafter).
  if (extraArgsAddResidentFiles(config.llamaExtraArgs)) return false;
  return (
    config.selectedGpuIds == null &&
    (config.gpuLayers == null || config.gpuLayers < 0) &&
    (config.nCpuMoe == null || config.nCpuMoe === 0)
  );
}

/** null per-model means the standing preference, which the loader also follows. */
function effectiveSpeculativeType(
  config: PerModelConfig | undefined,
): string | undefined {
  // The live runtime value first: forced modes are session-only and never persisted.
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
      // A 200 that sized nothing is remembered as a failure so it expires.
      remember(cacheKey, estimate, estimateIsUnsized(estimate));
      return estimate;
    })
    // The miss is remembered briefly, then expires.
    .catch(() => {
      remember(cacheKey, MISS, true);
      return MISS;
    })
    .finally(() => IN_FLIGHT.delete(cacheKey));

  IN_FLIGHT.set(cacheKey, run);
  return run;
}

/** No source (not on disk) does nothing and reports "unknown". */
export function useModelMemory(
  source: ModelMemorySource | undefined,
  gpuGb?: number | null,
): ModelMemorySegments {
  // Held with its key so a source change invalidates by comparison, not by an effect.
  const [entry, setEntry] = useState<{
    key: string;
    estimate: Estimate;
  } | null>(null);
  // Opt-in; checked here so a disabled bar costs no request.
  const enabled = useChatRuntimeStore((state) => state.showMemoryBar);

  // Gate before subscribing: the store ticks per streamed token, and an unconditional subscription
  // woke every row. Both branches are module constants.
  const watching = enabled && source != null;
  const epoch = useSyncExternalStore(
    watching ? subscribeToConfigChanges : subscribeNothing,
    watching ? readConfigEpoch : readZeroEpoch,
    () => 0,
  );

  // A Vulkan iGPU reports shared RAM and Apple the whole machine, neither a VRAM ceiling, so draw
  // nothing. Known gap: a ROCm APU's GTT pool is not flagged shared.
  const inferenceGpu = useInferenceGpuInfo();
  const budgetIsDedicatedVram =
    !inferenceGpu.sharedMemory &&
    // A ROCm APU's pool moves with host usage; reported apart from shared_memory.
    !inferenceGpu.unifiedMemory &&
    inferenceGpu.backend !== "mlx";

  // Cached and shared, so a long list costs one request.
  const [budgetFraction, setBudgetFraction] = useState<number | null>(null);

  const repoId = source?.repoId;
  const quant = source?.quant;
  const sizeBytes = source?.sizeBytes;
  const loadId = source?.loadId;
  // Keyed on primitives: callers build the source inline.
  const plan = useMemo(() => {
    // A direct .gguf path names its weights and needs no quant label.
    const isDirectGgufFile = (loadId ?? "").toLowerCase().endsWith(".gguf");
    if (!enabled || !repoId || (!quant && !isDirectGgufFile)) return null;
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
      // Saved configs are keyed by repo id, the request by the row's load target.
      source: { repoId: loadId || repoId, quant: quantLabel },
      config,
      nCtx,
      cacheKey,
      trustBudget: budgetIsMeaningful(config) && budgetIsDedicatedVram,
    };
    // `epoch` tracks configFor's localStorage; keying the cache on it would evict on any save.
    // biome-ignore lint/correctness/useExhaustiveDependencies: see above
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, repoId, loadId, quant, sizeBytes, epoch, budgetIsDedicatedVram]);

  // Gated on a real plan: every row mounts this hook, and only in-flight calls fold together.
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
        // Falls back to the headroom ratio the fit badge already uses.
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

  // Only the dedicated aggregate is VRAM beside system RAM.
  const budgetGb =
    inferenceGpu.dedicatedMemoryTotalGb > 0 &&
    inferenceGpu.dedicatedMemoryTotalGb < inferenceGpu.memoryTotalGb
      ? inferenceGpu.dedicatedMemoryTotalGb
      : gpuGb;

  return useMemo(() => {
    if (!plan?.trustBudget) return computeModelMemory({});
    const estimate = entry?.key === plan.cacheKey ? entry.estimate : undefined;
    // An unpriced drafter can be the largest allocation (~11 GB), so draw nothing.
    if (estimate?.specUnpriced) return computeModelMemory({});
    // An env device pin is the one placement override budgetIsMeaningful cannot see.
    if (estimate?.inheritedDevicePin) return computeModelMemory({});
    const weights = estimate?.weightsBytes ?? source?.sizeBytes;
    return computeModelMemory({
      // The projector is resident alongside the weights.
      weightsBytes:
        weights == null ? weights : weights + (estimate?.projectorBytes ?? 0),
      // Checkpoints live in host heap: the GPU figure is kv_bytes - kv_checkpoint_bytes.
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
      // Dedicated-only: the combined total adds an iGPU's allowance, a view of free RAM. `sharedMemory`
      // is every(), so a mixed host reads false and reaches here.
      gpuGb: budgetGb,
      budgetFraction,
      // Unpinned default loads auto-fit the context, so use the route's answer; an inherited positive
      // LLAMA_ARG_CTX_SIZE is kept, not fitted.
      contextIsAutoFitted:
        estimate?.contextIsPinned == null
          ? plan.nCtx == null
          : !estimate.contextIsPinned,
    });
  }, [plan, entry, source?.sizeBytes, budgetGb, budgetFraction]);
}
