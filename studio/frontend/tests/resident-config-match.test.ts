// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Remembered configs pass without forceReload, so adopting on identity alone drops settings. */

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

const USE_CHAT_MODEL_RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");

registerBundlerResolver();
const { residentRuntimeMatchesConfig, residentSpeculativeNeedsRepair } =
  await import("../src/features/chat/lib/resident-config-match.ts");

const DEFAULT_ISH = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  mlxKvQuant: null,
  speculativeType: null,
  specDraftNMax: null,
  nParallel: null,
  nBatch: null,
  nUbatch: null,
  reasoningBudget: -1,
  reasoningBudgetMessage: "",
  tensorParallel: false,
  disableVision: false,
  chatTemplateOverride: null,
} as const;

const BLANK = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  mlxKvQuant: null,
  speculativeType: null,
  specDraftNMax: null,
  nParallel: null,
  nBatch: null,
  nUbatch: null,
  reasoningBudget: -1,
  reasoningBudgetMessage: "",
  tensorParallel: false,
  disableVision: false,
  chatTemplateOverride: null,
};

const DEFAULTS = {};

/** Four fields are not per-model: the applier resolves them, and the load sends the result. */
const STANDING = {
  speculativeType: "auto",
  gpuMemoryMode: "auto" as const,
  gpuLayers: -1,
  nCpuMoe: 0,
  reconcileGpuIds: (ids: number[] | null) => ids,
  resolveContextLength: (customContextLength: number | null) =>
    customContextLength ?? 0,
  parallelSlots: 1,
  splitRatio: null,
  // The comparator must normalize the speculative mode on both sides.
  normalizeSpeculative: (v: string | null | undefined) =>
    v == null || !String(v).trim()
      ? null
      : String(v).trim().toLowerCase() === "default"
        ? "auto"
        : String(v).trim().toLowerCase(),
};

const matches = (
  status: Parameters<typeof residentRuntimeMatchesConfig>[0],
  config: Parameters<typeof residentRuntimeMatchesConfig>[1],
  standing: Parameters<typeof residentRuntimeMatchesConfig>[2] = STANDING,
) => residentRuntimeMatchesConfig(status, config, standing);

test("no config at all adopts the resident model", () => {
  assert.equal(matches(DEFAULTS, null), true);
  assert.equal(matches(DEFAULTS, undefined), true);
});

test("a config that pins nothing adopts the resident model", () => {
  assert.equal(matches(DEFAULTS, BLANK), true);
});

test("a remembered context length the resident load does not run is a reload", () => {
  assert.equal(
    matches(
      { requested_context_length: 4096 },
      { ...BLANK, customContextLength: 32768 },
    ),
    false,
  );
});

test("a remembered context length the resident load already runs adopts it", () => {
  assert.equal(
    matches(
      { requested_context_length: 32768 },
      { ...BLANK, customContextLength: 32768 },
    ),
    true,
  );
});

const FIELDS: {
  name: string;
  config: Record<string, unknown>;
  same: Record<string, unknown>;
  differs: Record<string, unknown>;
}[] = [
  {
    name: "context length",
    config: { customContextLength: 8192 },
    same: { requested_context_length: 8192 },
    differs: { requested_context_length: 4096 },
  },
  {
    name: "KV cache dtype",
    config: { kvCacheDtype: "q8_0" },
    same: { cache_type_kv: "q8_0" },
    differs: { cache_type_kv: "f16" },
  },
  {
    name: "MLX KV quantization",
    config: { mlxKvQuant: "4" },
    same: { mlx_kv_quant_requested: "4" },
    differs: { mlx_kv_quant_requested: "tq-4" },

  },
  {
    name: "speculative mode",
    config: { speculativeType: "mtp" },
    same: { speculative_type: "mtp" },
    differs: { speculative_type: "off" },
  },
  {
    name: "draft depth",
    config: { specDraftNMax: 8 },
    same: { spec_draft_n_max: 8 },
    differs: { spec_draft_n_max: 3 },
  },
  {
    name: "parallel slots",
    config: { nParallel: 4 },
    same: { requested_parallel_slots: 4 },
    differs: { requested_parallel_slots: 1 },
  },
  {
    name: "batch size",
    config: { nBatch: 2048 },
    same: { requested_n_batch: 2048 },
    differs: { requested_n_batch: 512 },
  },
  {
    name: "micro-batch size",
    config: { nUbatch: 512 },
    same: { requested_n_ubatch: 512 },
    differs: { requested_n_ubatch: 128 },
  },
  {
    name: "tensor parallel",
    config: { tensorParallel: true },
    same: { tensor_parallel: true },
    differs: { tensor_parallel: false },
  },
  {
    name: "chat template override",
    config: { chatTemplateOverride: "{{ custom }}" },
    same: { chat_template_override: "{{ custom }}" },
    differs: { chat_template_override: null },
  },
  {
    name: "pass-through llama args",
    config: { llamaExtraArgs: ["--flash-attn", "on"] },
    same: { requested_llama_extra_args: ["--flash-attn", "on"] },
    differs: { requested_llama_extra_args: ["--flash-attn", "off"] },
  },
  {
    name: "GPU memory mode",
    config: { gpuMemoryMode: "manual" },
    same: { gpu_memory_mode: "manual" },
    differs: { gpu_memory_mode: "auto" },
  },
  {
    // The backend compares offload knobs only under Manual, MoE count only with a layer pin.
    name: "GPU layers",
    config: { gpuMemoryMode: "manual", gpuLayers: 20 },
    same: { gpu_memory_mode: "manual", gpu_layers: 20 },
    differs: { gpu_memory_mode: "manual", gpu_layers: 99 },
  },
  {
    name: "CPU MoE layers",
    config: { gpuMemoryMode: "manual", gpuLayers: 20, nCpuMoe: 12 },
    same: { gpu_memory_mode: "manual", gpu_layers: 20, n_cpu_moe: 12 },
    differs: { gpu_memory_mode: "manual", gpu_layers: 20, n_cpu_moe: 0 },
  },
  {
    name: "GPU placement",
    config: { selectedGpuIds: [0, 2] },
    same: { requested_gpu_ids: [0, 2] },
    differs: { requested_gpu_ids: [0, 1] },
  },
];

for (const field of FIELDS) {
  test(`a matching ${field.name} adopts the resident model`, () => {
    assert.equal(matches(field.same, { ...BLANK, ...field.config }), true);
  });
  test(`a differing ${field.name} is a real reload`, () => {
    assert.equal(
      matches(field.differs, {
        ...BLANK,
        ...field.config,
      }),
      false,
    );
  });
  test(`a ${field.name} the resident load never reported is a real reload`, () => {
    // A backend that does not echo the field cannot prove agreement.
    assert.equal(matches({}, { ...BLANK, ...field.config }), false);
  });
}

test("Auto adopts a resident model the backend reported as Auto", () => {
  assert.equal(matches({ mlx_kv_quant_requested: "auto" }, { ...BLANK, mlxKvQuant: null }), true);
  assert.equal(matches({ mlx_kv_quant_requested: "8" }, { ...BLANK, mlxKvQuant: null }), false);
});

test("GPU placement compares as an order, not as a set", () => {
  // Position decides which card takes the prompt, so a reorder is a different placement.
  assert.equal(
    matches(
      { requested_gpu_ids: [3, 1, 0] },
      { ...BLANK, selectedGpuIds: [0, 1, 3] },
    ),
    false,
  );
  assert.equal(
    matches(
      { requested_gpu_ids: [3, 1, 0] },
      { ...BLANK, selectedGpuIds: [3, 1, 0] },
    ),
    true,
  );
});

test("automatic placement does not adopt a load pinned to chosen GPUs", () => {
  assert.equal(
    matches({ requested_gpu_ids: [0, 1] }, { ...BLANK, selectedGpuIds: null }),
    false,
  );
  assert.equal(matches({}, { ...BLANK, selectedGpuIds: null }), true);
});

/** The three states of llamaExtraArgs are load-bearing; see PerModelConfig. */
test("llama args never read by this copy express no opinion", () => {
  assert.equal(
    matches({ requested_llama_extra_args: ["--verbose"] }, { ...BLANK }),
    true,
  );
});

test("llama args the user cleared only agree with a load that has none", () => {
  assert.equal(
    matches(
      { requested_llama_extra_args: ["--verbose"] },
      { ...BLANK, llamaExtraArgs: null },
    ),
    false,
  );
  assert.equal(matches({}, { ...BLANK, llamaExtraArgs: null }), true);
  assert.equal(
    matches(
      { requested_llama_extra_args: [] },
      { ...BLANK, llamaExtraArgs: null },
    ),
    true,
  );
});

test("llama args differing only in order are a real reload", () => {
  // argv order changes what llama-server does; unlike GPU ids these are not a set.
  assert.equal(
    matches(
      { requested_llama_extra_args: ["b", "a"] },
      { ...BLANK, llamaExtraArgs: ["a", "b"] },
    ),
    false,
  );
});

/** tensorParallel is the one non-nullable field, so it always has an opinion. */
test("tensor parallel off agrees with a status that omits the field", () => {
  assert.equal(matches({}, { ...BLANK, tensorParallel: false }), true);
});

test("tensor parallel off is a reload when the resident load split tensors", () => {
  assert.equal(
    matches({ tensor_parallel: true }, { ...BLANK, tensorParallel: false }),
    false,
  );
});

/** maxSeqLength never reaches llama-server, so it cannot force a reload. */
test("a generation cap alone still adopts the resident model", () => {
  assert.equal(matches(DEFAULTS, { ...BLANK, maxSeqLength: 2048 }), true);
});

/** Zero is a real value in this API (0 = Auto for context, 0 layers = CPU only). */
test("zero is a pinned value, not an absent one", () => {
  const manual = { ...BLANK, gpuMemoryMode: "manual" as const };
  assert.equal(
    matches(
      { gpu_memory_mode: "manual", gpu_layers: 40 },
      { ...manual, gpuLayers: 0 },
    ),
    false,
  );
  assert.equal(
    matches(
      { gpu_memory_mode: "manual", gpu_layers: 0 },
      { ...manual, gpuLayers: 0 },
    ),
    true,
  );
  assert.equal(
    matches(
      { requested_context_length: 0 },
      { ...BLANK, customContextLength: 0 },
    ),
    true,
  );
});

test("one differing field among many agreeing ones is still a reload", () => {
  const config = {
    ...BLANK,
    customContextLength: 8192,
    kvCacheDtype: "q8_0",
    nParallel: 2,
    gpuLayers: 99,
  };
  const status = {
    requested_context_length: 8192,
    cache_type_kv: "q8_0",
    requested_parallel_slots: 2,
    gpu_layers: 99,
  };
  assert.equal(matches(status, config), true);
  assert.equal(matches({ ...status, cache_type_kv: "f16" }, config), false);
});

/** The config and lease gates must be checked before the reload is decided. */
test("selectModel weighs the config and the lease before confirming a reload", () => {
  const configCheck = USE_CHAT_MODEL_RUNTIME.search(/residentRuntimeMatchesConfig\(\s*status/);
  const identityCheck = USE_CHAT_MODEL_RUNTIME.indexOf("residentModelMatchesPick(status");
  const confirmPrompt = USE_CHAT_MODEL_RUNTIME.indexOf(
    "await confirmStopRunningChatsIfNeeded(",
  );
  assert.ok(identityCheck > 0, "selectModel no longer checks residency");
  assert.ok(
    configCheck > 0,
    "selectModel adopts a resident model without weighing the pick's own config",
  );
  assert.ok(confirmPrompt > 0, "selectModel no longer confirms running chats");
  assert.ok(identityCheck < confirmPrompt);
  assert.ok(configCheck < confirmPrompt);
  // A leased native file's label can be shared and the lease is only written by a load.
  // Scoped to the adoption short-circuit, since residency is also checked when cancelling.
  const guard = USE_CHAT_MODEL_RUNTIME.lastIndexOf(
    "if (!forceReload && !nativePathToken) {",
    confirmPrompt,
  );
  assert.ok(
    guard > 0,
    "the resident short-circuit no longer excludes native-lease picks",
  );
  const adoptionIdentityCheck = USE_CHAT_MODEL_RUNTIME.indexOf(
    "residentModelMatchesPick(status",
    guard,
  );
  assert.ok(
    adoptionIdentityCheck > guard && adoptionIdentityCheck < confirmPrompt,
    "the resident short-circuit no longer wraps the identity check",
  );
});

/** Standing fields resolve from a preference or constant, so unset is not silence. */
test("an unset speculative mode is the standing preference, not silence", () => {
  assert.equal(
    matches({ ...DEFAULTS, speculative_type: "mtp" }, BLANK, {
      ...STANDING,
      speculativeType: "off",
    }),
    false,
  );
  assert.equal(
    matches({ ...DEFAULTS, speculative_type: "off" }, BLANK, {
      ...STANDING,
      speculativeType: "off",
    }),
    true,
  );
});

test("a config that names a mode still beats the standing preference", () => {
  assert.equal(
    matches(
      { ...DEFAULTS, speculative_type: "mtp" },
      { ...BLANK, speculativeType: "mtp" },
      { ...STANDING, speculativeType: "off" },
    ),
    true,
  );
});

test("the speculative mode is normalized on both sides", () => {
  assert.equal(
    matches({ ...DEFAULTS, speculative_type: "default" }, BLANK),
    true,
  );
});

test("the GPU pick is compared after reconciliation, not as saved", () => {
  // performLoad reconciles saved ids by namespace, so compare the reconciled ids.
  const dropped = { ...STANDING, reconcileGpuIds: () => null };
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [1] },
      { ...BLANK, selectedGpuIds: [1], selectedGpuIndexKind: "physical" },
      dropped,
    ),
    false,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: null },
      { ...BLANK, selectedGpuIds: [1], selectedGpuIndexKind: "physical" },
      dropped,
    ),
    true,
  );
  const kinds: (string | null | undefined)[] = [];
  matches(
    DEFAULTS,
    { ...BLANK, selectedGpuIds: [0], selectedGpuIndexKind: "vulkan" },
    {
      ...STANDING,
      reconcileGpuIds: (ids, kind) => {
        kinds.push(kind);
        return ids;
      },
    },
  );
  assert.deepEqual(kinds, ["vulkan"]);
});

test("an unset context length is resolved the way the load resolves it", () => {
  // resolveLoadMaxSeqLength gives 0 cross-model and the resident context on re-pick.
  assert.equal(
    matches({ ...DEFAULTS, requested_context_length: 0 }, BLANK),
    true,
  );
  assert.equal(
    matches({ ...DEFAULTS, requested_context_length: 32768 }, BLANK, {
      ...STANDING,
      resolveContextLength: (pin) => pin ?? 32768,
    }),
    true,
  );
  assert.equal(
    matches({ ...DEFAULTS, requested_context_length: 32768 }, BLANK),
    false,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, requested_context_length: 8192 },
      { ...BLANK, customContextLength: 4096 },
    ),
    false,
  );
});

test("an unset slot count is the server default, not null", () => {
  // _resolve_parallel_slots stores the server default, so the status never echoes null.
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 4 }, BLANK, {
      ...STANDING,
      parallelSlots: 4,
    }),
    true,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, requested_parallel_slots: 4 },
      { ...BLANK, nParallel: 8 },
      {
        ...STANDING,
        parallelSlots: 4,
      },
    ),
    false,
  );
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 8 }, BLANK, {
      ...STANDING,
      parallelSlots: 4,
    }),
    false,
  );
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 4 }, BLANK, {
      ...STANDING,
      parallelSlots: null,
    }),
    false,
  );
});

test("a pick naming the fitted subset of a wider pool is not a reload", () => {
  // matches_gpu_ids accepts the request or the fitted pool.
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [0, 1], gpu_ids: [0] },
      { ...BLANK, selectedGpuIds: [0] },
    ),
    true,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [0, 1], gpu_ids: [0] },
      { ...BLANK, selectedGpuIds: [0, 1] },
    ),
    true,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [0, 1], gpu_ids: [0] },
      { ...BLANK, selectedGpuIds: [1] },
    ),
    false,
  );
  // An absent echo is no placement, not Automatic, or unpinned picks adopt pinned servers.
  assert.equal(
    matches({ ...DEFAULTS, requested_gpu_ids: [0, 1] }, BLANK),
    false,
  );
});

test("a retry arm that records no fallback reason still declines the shortcut", () => {
  // Two retry arms leave spec_fallback_reason null, so the reason alone is not enough.
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: null, spec_dflash_retry_pending: true },
      "auto",
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: null, spec_dflash_retry_pending: true },
      "dflash",
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: null, spec_dflash_retry_pending: true },
      "mtp",
    ),
    false,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: null, spec_probe_retry_pending: true },
      "off",
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: null,
        spec_probe_retry_pending: false,
        spec_dflash_retry_pending: false,
      },
      "auto",
    ),
    false,
  );
});

test("a binary stand-down that cannot repair does not decline the shortcut", () => {
  // An identical /load only repairs after a different llama-server is installed.
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "binary_no_mtp",
        spec_fallback_binary_changed: false,
      },
      "auto",
    ),
    false,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "binary_outdated",
        spec_fallback_binary_changed: false,
      },
      "mtp",
    ),
    false,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "binary_no_mtp",
        spec_fallback_binary_changed: true,
      },
      "auto",
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: "binary_no_mtp" },
      "auto",
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "drafter_not_found",
        spec_fallback_binary_changed: false,
      },
      "auto",
    ),
    true,
  );
});

test("an MLX resident is judged on its speculative settings, a standing ngram reading as auto", () => {
  const mlx = { ...DEFAULTS, is_gguf: false, is_mlx: true, speculative_type: "auto" };
  const ngram = { ...STANDING, speculativeType: "ngram" };
  assert.equal(matches(mlx, BLANK, ngram), true);
  assert.equal(matches({ ...mlx, speculative_type: "ngram" }, BLANK, ngram), false);
  assert.equal(matches(mlx, { ...BLANK, speculativeType: "eagle3" }), false);
  const drafted = { ...BLANK, speculativeType: "auto", specDraftModel: "o/d" };
  assert.equal(matches({ ...mlx, spec_draft_model: "o/d" }, drafted), true);
  assert.equal(matches(mlx, drafted), false);
  const deep = { ...BLANK, speculativeType: "mtp", specDraftNMax: 6 };
  assert.equal(matches({ ...mlx, speculative_type: "mtp", spec_draft_n_max: 6 }, deep), true);
  assert.equal(matches({ ...mlx, speculative_type: "mtp", spec_draft_n_max: 4 }, deep), false);
  // An MLX load reads "+ngram" as its kind, which already copies; GGUF keeps the two apart.
  const copying = { ...BLANK, speculativeType: "mtp+ngram" };
  assert.equal(matches({ ...mlx, speculative_type: "mtp" }, copying), true);
  assert.equal(matches({ ...DEFAULTS, speculative_type: "mtp" }, copying), false);
  // llama.cpp's retry arms: an identical MLX load dedupes.
  const notFound = { ...mlx, spec_fallback_reason: "drafter_not_found" };
  assert.equal(residentSpeculativeNeedsRepair(notFound, "auto"), false);
});

test("a non-GGUF resident is not judged on a GGUF invocation field", () => {
  // requested_context_length is GGUF-only, so its absence on safetensors/MLX is not 0.
  assert.equal(
    matches({ ...DEFAULTS, is_gguf: false }, BLANK, {
      ...STANDING,
      resolveContextLength: () => 8192,
    }),
    true,
  );
  assert.equal(
    matches({ ...DEFAULTS, is_gguf: true }, BLANK, {
      ...STANDING,
      resolveContextLength: () => 8192,
    }),
    false,
  );
  assert.equal(
    matches({ ...DEFAULTS, is_gguf: false, cache_type_kv: "q8_0" }, BLANK),
    true,
  );
  assert.equal(
    matches({ ...DEFAULTS, is_gguf: false, mlx_kv_quant_requested: "4" }, BLANK),
    false,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, is_gguf: false, chat_template_override: "{{ bos }}" },
      BLANK,
    ),
    false,
  );
});

test("a hidden MoE count under Auto layers is not a reload", () => {
  // _runtime_matches_intent compares nCpuMoe only under Manual with a non-negative pin.
  assert.equal(
    matches({ ...DEFAULTS, n_cpu_moe: 0 }, { ...BLANK, nCpuMoe: 8 }),
    true,
  );
  const manual = { ...BLANK, gpuMemoryMode: "manual" as const };
  const running = { ...DEFAULTS, gpu_memory_mode: "manual" as const };
  assert.equal(
    matches(
      { ...running, gpu_layers: -1, n_cpu_moe: 0 },
      { ...manual, gpuLayers: -1, nCpuMoe: 8 },
    ),
    true,
  );
  assert.equal(
    matches(
      { ...running, gpu_layers: 4, n_cpu_moe: 0 },
      { ...manual, gpuLayers: 4, nCpuMoe: 8 },
    ),
    false,
  );
});

test("a standalone .gguf never reaches the drafter retry arm", () => {
  // The arm is guarded on gguf_path being None, so a directly loaded file dedupes.
  const status = {
    spec_fallback_reason: "drafter_not_found",
    spec_drafter_kind: "mtp",
  };
  assert.equal(residentSpeculativeNeedsRepair(status, "auto", true), false);
  assert.equal(residentSpeculativeNeedsRepair(status, "auto", false), true);
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "binary_no_mtp",
        spec_fallback_binary_changed: true,
      },
      "auto",
      true,
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: null, spec_probe_retry_pending: true },
      "auto",
      true,
    ),
    true,
  );
});

test("a permanently absent drafter does not decline the shortcut", () => {
  // Two drafter kinds are permanently absent, and retrying them relaunches forever.
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "drafter_not_found",
        spec_drafter_kind: "dspark",
        spec_dspark_sidecar_absent: true,
      },
      "auto",
    ),
    false,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "drafter_not_found",
        spec_drafter_kind: "dflash",
      },
      "auto",
    ),
    false,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "drafter_not_found",
        spec_drafter_kind: "dspark",
        spec_dspark_sidecar_absent: false,
      },
      "auto",
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: "drafter_not_found", spec_drafter_kind: "mtp" },
      "auto",
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "drafter_not_found",
        spec_drafter_kind: "dspark",
      },
      "auto",
    ),
    true,
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      {
        spec_fallback_reason: "drafter_not_found",
        spec_drafter_kind: "dflash",
        spec_dflash_retry_pending: true,
      },
      "auto",
    ),
    true,
  );
});

test("a tensor split the architecture gate normalized away still matches", () => {
  // The gate rewrites tensor-parallel to layer mode, so status false still matches a true request.
  const gated = {
    ...DEFAULTS,
    tensor_parallel: false,
    tensor_parallel_dropped_by_arch_gate: true,
  };
  assert.equal(matches(gated, { ...BLANK, tensorParallel: true }), true);
  assert.equal(
    matches(
      { ...DEFAULTS, tensor_parallel: false },
      { ...BLANK, tensorParallel: true },
    ),
    false,
  );
  assert.equal(
    matches(
      {
        ...DEFAULTS,
        tensor_parallel: false,
        tensor_parallel_dropped_by_arch_gate: null,
      },
      { ...BLANK, tensorParallel: true },
    ),
    false,
  );
  assert.equal(
    matches(
      {
        ...DEFAULTS,
        tensor_parallel: true,
        tensor_parallel_dropped_by_arch_gate: true,
      },
      BLANK,
    ),
    false,
  );
});

test("the arch-gate excuse reads the resolved split, not the toggle", () => {
  // --split-mode tensor in pass-through args can request the split the toggle does not.
  assert.equal(
    matches(
      {
        ...DEFAULTS,
        tensor_parallel: false,
        tensor_parallel_dropped_by_arch_gate: true,
        requested_llama_extra_args: ["--split-mode", "tensor"],
      },
      {
        ...BLANK,
        tensorParallel: false,
        llamaExtraArgs: ["--split-mode", "tensor"],
      },
    ),
    true,
  );
  assert.equal(
    matches(
      {
        ...DEFAULTS,
        tensor_parallel: true,
        requested_llama_extra_args: ["-sm", "layer"],
      },
      { ...BLANK, tensorParallel: true, llamaExtraArgs: ["-sm", "layer"] },
    ),
    false,
  );
});

test("a malformed manual layer override declines rather than normalizing away", () => {
  // parse_gpu_layers_override raises on these, so they must not fold into no override.
  const running = {
    ...DEFAULTS,
    gpu_memory_mode: "manual" as const,
    gpu_layers: 20,
    requested_llama_extra_args: [],
  };
  const manual = { ...BLANK, gpuMemoryMode: "manual" as const, gpuLayers: 20 };
  for (const bad of [["-ngl", "-2"], ["--gpu-layers=many"], ["-ngl", "20.5"]]) {
    assert.equal(matches(running, { ...manual, llamaExtraArgs: bad }), false);
  }
  assert.equal(
    matches(
      { ...running, gpu_layers: 99 },
      {
        ...manual,
        llamaExtraArgs: ["-ngl", "99"],
      },
    ),
    true,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, requested_llama_extra_args: ["-ngl", "-2"] },
      {
        ...BLANK,
        gpuMemoryMode: "auto" as const,
        llamaExtraArgs: ["-ngl", "-2"],
      },
    ),
    true,
  );
});

test("a virtualised Metal host cannot disagree about placement", () => {
  // Paravirtual hosts normalize every GGUF request to manual zero layers before comparing.
  const pv = {
    ...DEFAULTS,
    gpu_placement_paravirtual: true,
    gpu_memory_mode: "manual" as const,
    gpu_layers: 0,
    tensor_parallel: false,
  };
  assert.equal(matches(pv, BLANK), true);
  assert.equal(
    matches(pv, { ...BLANK, gpuLayers: 40, nCpuMoe: 8, tensorParallel: true }),
    true,
  );
  assert.equal(
    matches(
      { ...pv, requested_gpu_ids: null },
      { ...BLANK, selectedGpuIds: [1] },
    ),
    true,
  );
  assert.equal(matches({ ...pv, cache_type_kv: "q8_0" }, BLANK), false);
  assert.equal(
    matches({ ...DEFAULTS, gpu_memory_mode: "manual", gpu_layers: 0 }, BLANK),
    false,
  );
});

test("a diffusion resident is not judged on the chat-only invocation fields", () => {
  // The diffusion runner ignores --parallel, batch sizes and pass-through args.
  const diffusion = {
    ...DEFAULTS,
    is_diffusion: true,
    requested_parallel_slots: null,
    requested_n_batch: null,
    requested_n_ubatch: null,
    requested_llama_extra_args: null,
  };
  assert.equal(
    matches(diffusion, {
      ...BLANK,
      nParallel: 2,
      nBatch: 2048,
      nUbatch: 512,
      llamaExtraArgs: ["--flash-attn", "on"],
    }),
    true,
  );
  assert.equal(
    matches({ ...diffusion, is_diffusion: false }, { ...BLANK, nParallel: 2 }),
    false,
  );
  assert.equal(matches({ ...diffusion, cache_type_kv: "q8_0" }, BLANK), false);
});

test("a dropped diffusion split is rechecked once the shim can apply it", () => {
  // Once the shim gains --ngl support, the retained request must reload to apply the split.
  const manual = { ...BLANK, gpuMemoryMode: "manual" as const, gpuLayers: 12 };
  const dropped = {
    ...DEFAULTS,
    is_diffusion: true,
    diffusion_requested_ngl: 12,
    gpu_layers: 0,
  };
  assert.equal(
    matches({ ...dropped, diffusion_split_supported: true }, manual),
    false,
  );
  assert.equal(
    matches({ ...dropped, diffusion_split_supported: false }, manual),
    true,
  );
  assert.equal(matches(dropped, manual), true);
  assert.equal(
    matches(
      { ...dropped, gpu_layers: 12, diffusion_split_supported: true },
      manual,
    ),
    true,
  );
  assert.equal(
    matches(
      {
        ...DEFAULTS,
        is_diffusion: true,
        diffusion_requested_ngl: null,
        gpu_layers: 8,
        diffusion_split_supported: true,
      },
      BLANK,
    ),
    true,
  );
});

test("a diffusion pick is reduced to its lowest GPU, as the backend reduces it", () => {
  // A diffusion runner drives only the lowest GPU id, which is all the status reports.
  const diffusion = { ...DEFAULTS, is_diffusion: true, requested_gpu_ids: [1] };
  assert.equal(matches(diffusion, { ...BLANK, selectedGpuIds: [3, 1] }), true);
  assert.equal(matches(diffusion, { ...BLANK, selectedGpuIds: [1] }), true);
  assert.equal(matches(diffusion, { ...BLANK, selectedGpuIds: [2, 3] }), false);
  assert.equal(matches(diffusion, BLANK), false);
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [1] },
      { ...BLANK, selectedGpuIds: [3, 1] },
    ),
    false,
  );
});

test("no llama.cpp invocation field decides against a non-GGUF resident", () => {
  // Non-GGUF /load never reads llama.cpp flags, so they must not force a reload.
  const resident = { ...DEFAULTS, is_gguf: false };
  assert.equal(
    matches(resident, {
      ...BLANK,
      gpuMemoryMode: "manual",
      gpuLayers: 20,
      nCpuMoe: 8,
      tensorParallel: true,
      nBatch: 2048,
      nUbatch: 512,
      selectedGpuIds: [1],
      llamaExtraArgs: ["--flash-attn", "on"],
      kvCacheDtype: "q8_0",
    }),
    true,
  );
  assert.equal(
    matches({ ...resident, is_gguf: true }, { ...BLANK, nParallel: 4 }),
    false,
  );
});

test("a resident decoding at another width is not adopted, whichever backend", () => {
  for (const is_gguf of [true, false]) {
    assert.equal(
      matches({ ...DEFAULTS, is_gguf, requested_parallel_slots: 4 }, { ...BLANK, nParallel: 2 }),
      false,
    );
    assert.equal(
      matches({ ...DEFAULTS, is_gguf, requested_parallel_slots: 2 }, { ...BLANK, nParallel: 2 }),
      true,
    );
  }
});

test("a diffusion resident is judged on its NGL, not on the placement fields", () => {
  // An older shim that dropped a manual NGL reports Auto while the request says Manual.
  const diffusion = {
    ...DEFAULTS,
    is_diffusion: true,
    gpu_memory_mode: "auto" as const,
    diffusion_requested_ngl: null,
  };
  assert.equal(
    matches(diffusion, { ...BLANK, gpuMemoryMode: "manual", gpuLayers: -1 }),
    true,
  );
  assert.equal(
    matches(diffusion, { ...BLANK, gpuMemoryMode: "manual", gpuLayers: 12 }),
    false,
  );
  assert.equal(
    matches(
      { ...diffusion, diffusion_requested_ngl: 12 },
      { ...BLANK, gpuMemoryMode: "manual", gpuLayers: 12 },
    ),
    true,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, gpu_memory_mode: "auto" },
      { ...BLANK, gpuMemoryMode: "manual", gpuLayers: -1 },
    ),
    false,
  );
});

test("a model switch is judged on the defaults it resets to, not the outgoing settings", () => {
  // performLoad clears per-model fields on a switch, so compare against defaults, not outgoing.
  const outgoing = { ...BLANK, nParallel: 4 };
  const reset = {
    ...DEFAULT_ISH,
    kvCacheDtype: null,
    tensorParallel: false,
  };
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 1 }, reset, {
      ...STANDING,
      parallelSlots: 1,
    }),
    true,
  );
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 4 }, reset, {
      ...STANDING,
      parallelSlots: 1,
    }),
    false,
  );
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 4 }, outgoing, {
      ...STANDING,
      parallelSlots: 1,
    }),
    true,
  );
});

test("a pass-through split mode decides the tensor-parallel comparison", () => {
  // An explicit --split-mode last-wins over the toggle before the comparator sees it.
  assert.equal(
    matches(
      {
        ...DEFAULTS,
        tensor_parallel: true,
        requested_llama_extra_args: ["--split-mode", "tensor"],
      },
      {
        ...BLANK,
        tensorParallel: false,
        llamaExtraArgs: ["--split-mode", "tensor"],
      },
    ),
    true,
  );
  assert.equal(
    matches(
      {
        ...DEFAULTS,
        tensor_parallel: false,
        requested_llama_extra_args: ["-sm", "layer"],
      },
      { ...BLANK, tensorParallel: true, llamaExtraArgs: ["-sm", "layer"] },
    ),
    true,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, tensor_parallel: true },
      { ...BLANK, tensorParallel: false },
    ),
    false,
  );
});

test("a manual pass-through layer count is compared as the field it becomes", () => {
  // The route moves the last -ngl into gpu_layers and strips the flag before comparing.
  const manual = { ...BLANK, gpuMemoryMode: "manual" as const };
  const running = {
    ...DEFAULTS,
    gpu_memory_mode: "manual" as const,
    gpu_layers: 20,
    requested_llama_extra_args: ["--flash-attn", "on"],
  };
  assert.equal(
    matches(running, {
      ...manual,
      llamaExtraArgs: ["-ngl", "20", "--flash-attn", "on"],
    }),
    true,
  );
  assert.equal(
    matches(running, {
      ...manual,
      llamaExtraArgs: ["-ngl", "8", "--flash-attn", "on"],
    }),
    false,
  );
  // Auto does not own the offload flags, so an inherited -ngl is compared as written.
  assert.equal(
    matches(
      { ...DEFAULTS, requested_llama_extra_args: ["--flash-attn", "on"] },
      { ...BLANK, llamaExtraArgs: ["-ngl", "20", "--flash-attn", "on"] },
    ),
    false,
  );
});

test("a custom tensor split the config cannot carry is still a reload", () => {
  assert.equal(
    matches({ ...DEFAULTS, tensor_split: [0.7, 0.3] }, BLANK),
    false,
  );
  assert.equal(
    matches({ ...DEFAULTS, tensor_split: [0.7, 0.3] }, BLANK, {
      ...STANDING,
      splitRatio: [0.7, 0.3],
    }),
    true,
  );
  assert.equal(matches({ ...DEFAULTS, tensor_split: null }, BLANK), true);
});

test("a preserved Vulkan CPU fallback is not a placement disagreement", () => {
  // _preserve_cpu_fallback_intent rewrites an eligible Auto request before comparing.
  const fallback = {
    ...DEFAULTS,
    gpu_memory_mode: "manual" as const,
    gpu_layers: 0,
    cpu_fallback_reason: "vulkan_startup_crash" as const,
  };
  assert.equal(matches(fallback, BLANK), true);
  assert.equal(matches({ ...fallback, cache_type_kv: "q8_0" }, BLANK), false);
  assert.equal(matches(fallback, { ...BLANK, selectedGpuIds: [0] }), false);
  assert.equal(matches(fallback, { ...BLANK, tensorParallel: true }), false);
  assert.equal(matches(fallback, { ...BLANK, nCpuMoe: 4 }), false);
  assert.equal(
    matches(fallback, { ...BLANK, llamaExtraArgs: ["--device", "Vulkan0"] }),
    false,
  );
  assert.equal(
    matches({ ...fallback, cpu_fallback_reason: null }, BLANK),
    false,
  );
});

test("an unset GPU memory mode is the standing preference, not silence", () => {
  assert.equal(
    matches({ ...DEFAULTS, gpu_memory_mode: "manual" }, BLANK),
    false,
  );
  assert.equal(
    matches({ ...DEFAULTS, gpu_memory_mode: "manual" }, BLANK, {
      ...STANDING,
      gpuMemoryMode: "manual",
    }),
    true,
  );
});

test("unset GPU layers and CPU MoE layers resolve to Auto and 0", () => {
  const manual = { ...BLANK, gpuMemoryMode: "manual" as const };
  const running = { ...DEFAULTS, gpu_memory_mode: "manual" as const };
  assert.equal(matches({ ...running, gpu_layers: 20 }, manual), false);
  assert.equal(matches({ ...running, gpu_layers: -1 }, manual), true);
  assert.equal(
    matches(
      { ...running, gpu_layers: 4, n_cpu_moe: 12 },
      { ...manual, gpuLayers: 4 },
    ),
    false,
  );
  assert.equal(
    matches(
      { ...running, gpu_layers: 4, n_cpu_moe: 0 },
      { ...manual, gpuLayers: 4 },
    ),
    true,
  );
  assert.equal(
    matches({ ...DEFAULTS, gpu_layers: 20, n_cpu_moe: 12 }, BLANK),
    true,
  );
});

test("no config at all still adopts, whatever the resident runtime is", () => {
  // The load path reads the live runtime, which was hydrated from the resident model.
  assert.equal(
    matches(
      {
        speculative_type: "mtp",
        gpu_layers: 20,
        gpu_memory_mode: "manual",
        n_cpu_moe: 12,
      },
      null,
    ),
    true,
  );
});

/** The applier resolves unset nullable fields to null, so a blank pick asks for the default. */
test("unset nullable settings ask for the default, not for the resident value", () => {
  const pinnedResident = {
    ...DEFAULTS,
    requested_context_length: 8192,
    cache_type_kv: "q8_0",
    mlx_kv_quant_requested: "4",
    requested_parallel_slots: 4,
    requested_n_batch: 2048,
    requested_n_ubatch: 512,
    chat_template_override: "{{ bos }}",
  };
  assert.equal(matches(pinnedResident, BLANK), false);
  for (const [key, value] of Object.entries({
    requested_context_length: 8192,
    cache_type_kv: "q8_0",
    mlx_kv_quant_requested: "4",
    requested_parallel_slots: 4,
    requested_n_batch: 2048,
    requested_n_ubatch: 512,
    chat_template_override: "{{ bos }}",
  })) {
    assert.equal(
      matches({ ...DEFAULTS, [key]: value }, BLANK),
      false,
      `${key} pinned on the resident load must not be adopted by a blank config`,
    );
  }
  assert.equal(matches({ ...DEFAULTS, spec_draft_n_max: 16 }, BLANK), false);
  assert.equal(
    matches(
      { ...DEFAULTS, spec_draft_n_max: 16 },
      { ...BLANK, specDraftNMax: 8 },
    ),
    false,
  );
  assert.equal(matches(DEFAULTS, BLANK), true);
});

test("a blank chat template agrees with a load that has none", () => {
  // The applier and the load both send whitespace-only templates as null.
  assert.equal(
    matches({ ...DEFAULTS, chat_template_override: "" }, BLANK),
    true,
  );
  assert.equal(
    matches(
      { ...DEFAULTS, chat_template_override: null },
      {
        ...BLANK,
        chatTemplateOverride: "   ",
      },
    ),
    true,
  );
});

/** Retryable drafter failures must reload; permanent downgrades must not prompt each pick. */
test("a retryable drafter failure declines the shortcut", () => {
  for (const reason of [
    "drafter_not_found",
    "binary_no_mtp",
    "binary_outdated",
  ]) {
    for (const mode of ["auto", "mtp", "mtp+ngram", "dspark", "dflash"]) {
      assert.equal(
        residentSpeculativeNeedsRepair({ spec_fallback_reason: reason }, mode),
        true,
        `${reason} under ${mode} must reload`,
      );
    }
  }
});

test("a repaired drafter has to reach the backend to be re-checked", () => {
  // The remedy replaces the sidecar in place, and adoption would skip the re-check.
  for (const mode of ["auto", "mtp", "mtp+ngram"]) {
    assert.equal(
      residentSpeculativeNeedsRepair(
        { spec_fallback_reason: "drafter_unloadable", spec_drafter_kind: "mtp" },
        mode,
      ),
      true,
      `drafter_unloadable under ${mode} must reload`,
    );
  }
  // No sendsGgufPath exclusion: this re-check is in the drafter comparison, not the refetch.
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: "drafter_unloadable", spec_drafter_kind: "mtp" },
      "auto",
      true,
    ),
    true,
    "a standalone .gguf load must still reload for a repaired drafter",
  );
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: "drafter_unloadable", spec_drafter_kind: "mtp" },
      "off",
    ),
    false,
    "spec off asked for no drafter, so there is nothing to repair",
  );
});

test("an Auto-mode policy downgrade is not a repair the load can make", () => {
  for (const reason of [
    "drafter_no_vram",
    "mla_mtp_disabled",
    "runtime_error",
  ]) {
    assert.equal(
      residentSpeculativeNeedsRepair({ spec_fallback_reason: reason }, "auto"),
      false,
      `${reason} must not reload`,
    );
  }
});

test("a healthy runtime and a pick wanting no drafter both stay on the shortcut", () => {
  assert.equal(residentSpeculativeNeedsRepair({}, "auto"), false);
  assert.equal(
    residentSpeculativeNeedsRepair({ spec_fallback_reason: null }, "mtp"),
    false,
  );
  for (const mode of ["off", "none", "ngram"]) {
    assert.equal(
      residentSpeculativeNeedsRepair(
        { spec_fallback_reason: "drafter_not_found" },
        mode,
      ),
      false,
      `${mode} must not reload`,
    );
  }
  // A null resolved mode is Auto, which is a speculative mode.
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: "drafter_not_found" },
      null,
    ),
    true,
  );
});

/** No status echoes maxSeqLength, so the shortcut must carry the picked model's own cap. */
test("the resident shortcut keeps the picked model's own sequence cap", () => {
  const configCheck = USE_CHAT_MODEL_RUNTIME.search(/residentRuntimeMatchesConfig\(\s*status/);
  const rollback = USE_CHAT_MODEL_RUNTIME.indexOf("restorePreviousConfig();", configCheck);
  const reapply = USE_CHAT_MODEL_RUNTIME.indexOf("pickedMaxSeqLength", configCheck);
  assert.ok(
    rollback > 0,
    "the shortcut no longer rolls the staged config back",
  );
  assert.ok(
    reapply > rollback,
    "the shortcut leaves the outgoing model's maxSeqLength in place",
  );
  const confirmPrompt = USE_CHAT_MODEL_RUNTIME.indexOf(
    "await confirmStopRunningChatsIfNeeded(",
  );
  assert.ok(reapply < confirmPrompt, "the re-apply escaped the shortcut");
  // applyPerModelConfigToRuntime resolves an absent cap to the default, not the outgoing cap.
  assert.match(
    USE_CHAT_MODEL_RUNTIME.slice(reapply, reapply + 260),
    /\?\?\s*defaultInferenceParams\.maxSeqLength/,
    "an absent cap no longer resolves to the default",
  );
});

/** load_model re-probes audio only on its fast path, so an unprobed model must not skip /load. */
test("an outstanding audio probe keeps the shortcut from skipping the load", () => {
  const identity = USE_CHAT_MODEL_RUNTIME.search(/residentModelMatchesPick\(\s*status/);
  const probe = USE_CHAT_MODEL_RUNTIME.indexOf("status.audio_probe_pending !== true", identity);
  // The route derives gguf_path from the identifier, and the drafter retry is guarded on it.
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /\(loadPath \?\? modelId\)\.toLowerCase\(\)\.endsWith\("\.gguf"\)/,
    "the repair check no longer knows whether the pick sends a path",
  );
  assert.ok(
    probe > identity,
    "the shortcut adopts a model whose audio probe never finished",
  );
  const decision = USE_CHAT_MODEL_RUNTIME.indexOf("const confirmedStatus", identity);
  assert.ok(probe < decision, "the probe check escaped the residency verdict");
  // Only an explicit true declines: a backend too old to report it behaves as before.
  assert.match(USE_CHAT_MODEL_RUNTIME.slice(probe - 40, probe + 40), /!== true/);
});

/** Another tab can swap the resident model during the awaits, so re-read before adopting. */
test("the shortcut re-reads and re-judges the status before adopting", () => {
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /const adoptable = \(status: InferenceStatusResponse\) =>\s*\(status\.loading\?\.length \?\? 0\) === 0 &&/,
    "the residency verdict is no longer callable against a second status",
  );
  const decision = USE_CHAT_MODEL_RUNTIME.search(
    /const confirmedStatus = await readPickStatus\(\)/,
  );
  assert.ok(
    decision > 0,
    "the shortcut adopts the status it opened with, across every await above it",
  );
  assert.match(
    USE_CHAT_MODEL_RUNTIME.slice(decision, decision + 200),
    /if \(confirmedStatus && adoptable\(confirmedStatus\)\)/,
    "the re-read is not judged, only fetched",
  );
  const adopt = USE_CHAT_MODEL_RUNTIME.indexOf("applyActiveModelStatusToStore(", decision);
  assert.match(
    USE_CHAT_MODEL_RUNTIME.slice(adopt, adopt + 60),
    /applyActiveModelStatusToStore\(confirmedStatus/,
  );
  // Skipping /load skips the backend's pool update, so the pick's GPU selection must be restored.
  const restore = USE_CHAT_MODEL_RUNTIME.indexOf("selectedGpuIds: picked", decision);
  const hydrate = USE_CHAT_MODEL_RUNTIME.indexOf("applyActiveModelStatusToStore(", decision);
  assert.ok(
    restore > hydrate,
    "the adopted pick no longer keeps its own GPU selection",
  );
  assert.ok(
    USE_CHAT_MODEL_RUNTIME.includes(
      "getInferenceStatus(undefined, modelId).catch(() => null);",
    ),
    "the re-read no longer tolerates a failed status",
  );
});

/** With no saved config compare what /load would send: defaults on a switch, else the live store. */
test("with no saved config the gate compares what the load would send", () => {
  const configCheck = USE_CHAT_MODEL_RUNTIME.search(/residentRuntimeMatchesConfig\(\s*status/);
  assert.ok(
    configCheck > 0 &&
      /const comparedConfig =\s*\n\s*pendingConfig \?\?/.test(USE_CHAT_MODEL_RUNTIME),
    "the gate takes an absent config as a wildcard again",
  );
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /resetsPerModelSettings\s*\?\s*\{\s*\n\s*\.\.\.DEFAULT_PER_MODEL_CONFIG/,
    "a model switch no longer compares against the defaults it would send",
  );
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /: currentRuntimePerModelConfig\(\)\)/,
    "a re-pick that switches nothing no longer compares against the live runtime",
  );
});

test("adopting reseeds the slot and batch controls the rollback left behind", () => {
  // The rollback restores the outgoing model's config, so slot and batch controls must reseed.
  const hydrator = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
  assert.match(
    hydrator.replace(/\s+/g, " "),
    /const slotsModelChanged = hydratingExistingModel;/,
  );
  assert.equal(hydrator.includes("readoptingSameModel"), false);
  assert.equal(USE_CHAT_MODEL_RUNTIME.includes("readoptingSameModel"), false);
});

test("selectModel asks about a repairable drafter before adopting", () => {
  const repairCheck = USE_CHAT_MODEL_RUNTIME.indexOf("residentSpeculativeNeedsRepair(");
  const confirmPrompt = USE_CHAT_MODEL_RUNTIME.indexOf(
    "await confirmStopRunningChatsIfNeeded(",
  );
  assert.ok(
    repairCheck > 0,
    "selectModel no longer reloads a degraded drafter",
  );
  assert.ok(repairCheck < confirmPrompt);
});

/**
 * Must agree with the backend's _LEGACY_SPEC_MODE_MAP. The store cannot be imported here,
 * so the function is lifted from source and evaluated.
 */
test("the speculative normalizer reads llama.cpp's disable spellings as off", () => {
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  const start = store.indexOf("export function normalizeSpeculativeType");
  assert.ok(start > 0, "normalizeSpeculativeType moved; follow it here");
  const source = store
    .slice(start, store.indexOf("\n}\n", start) + 2)
    .replace("export function", "function")
    .replace("  v: string | null | undefined,\n): string | null {", "v) {");
  const normalize = new Function(
    `${source}; return normalizeSpeculativeType;`,
  )() as (v: string | null | undefined) => string | null;

  for (const spelling of [
    "off",
    "none",
    "None",
    "NONE",
    "  none  ",
    "disable",
    "Disabled",
    "disabled",
  ]) {
    assert.equal(normalize(spelling), "off", `${spelling} must read as off`);
  }
  assert.equal(normalize("default"), "auto");
  assert.equal(normalize("draft-mtp"), "mtp");
  assert.equal(normalize("mtp+ngram"), "mtp+ngram");
  assert.equal(normalize("bogus"), "auto");
  assert.equal(normalize(null), null);
});

/** Forced ngram-mod stand-down is invisible to settings, so only the repair check reloads. */
test("a forced ngram stand-down reloads onto an updated binary", () => {
  const stoodDown = {
    spec_fallback_reason: "binary_outdated",
    spec_fallback_binary_changed: true,
  };
  assert.equal(residentSpeculativeNeedsRepair(stoodDown, "ngram"), true);
  assert.equal(
    residentSpeculativeNeedsRepair(
      { ...stoodDown, spec_fallback_binary_changed: false },
      "ngram",
    ),
    false,
  );
  assert.equal(
    residentSpeculativeNeedsRepair({ spec_fallback_reason: null }, "ngram"),
    false,
  );
});

/** MTP-free recovery nulls the draft depth while the backend still compares against 8. */
test("a runtime_error resident does not claim its draft depth is the default", () => {
  const recovered = { ...DEFAULTS, spec_fallback_reason: "runtime_error" };
  assert.equal(matches(recovered, BLANK), false);
  assert.equal(matches(recovered, { ...BLANK, specDraftNMax: 8 }), false);
  for (const reason of [null, "binary_no_mtp", "drafter_not_found", "mtp_partial_offload"]) {
    assert.equal(
      matches({ ...DEFAULTS, spec_fallback_reason: reason }, BLANK),
      true,
      `${reason} must still adopt a default-against-default pick`,
    );
  }
});

/** In auto mode the store never holds a split, while the server legitimately reports one. */
test("an auto tensor-parallel server that reports a split still adopts", () => {
  assert.equal(
    matches(
      { gpu_memory_mode: "auto", tensor_parallel: true, tensor_split: [0.75, 0.25] },
      { ...BLANK, tensorParallel: true },
    ),
    true,
  );
});

/** A pending ratio set under Manual survives the switch to Auto and is still sent. */
test("a pending ratio the auto resident is not running is still a reload", () => {
  assert.equal(
    matches(
      { gpu_memory_mode: "auto", tensor_parallel: true, tensor_split: [0.75, 0.25] },
      { ...BLANK, tensorParallel: true },
      { ...STANDING, splitRatio: [0.5, 0.5] },
    ),
    false,
  );
});

test("a pending ratio the auto resident IS running adopts", () => {
  assert.equal(
    matches(
      { gpu_memory_mode: "auto", tensor_parallel: true, tensor_split: [0.75, 0.25] },
      { ...BLANK, tensorParallel: true },
      { ...STANDING, splitRatio: [0.75, 0.25] },
    ),
    true,
  );
});

test("a remembered manual split the resident load does not run is still a reload", () => {
  assert.equal(
    matches(
      { gpu_memory_mode: "manual", tensor_split: [0.5, 0.5] },
      { ...BLANK, gpuMemoryMode: "manual" as const, gpuLayers: 99 },
      { ...STANDING, splitRatio: [0.75, 0.25] },
    ),
    false,
  );
});


test("inherited reasoning defaults do not reload an unchanged resident", () => {
  assert.equal(matches({
    reasoning_budget: 32,
    reasoning_budget_message: "Conclude now.",
    requested_reasoning_budget: -1,
    requested_reasoning_budget_message: "",
  }, BLANK), true);
});

test("pinning the effective reasoning value changes the resident request", () => {
  assert.equal(matches({
    reasoning_budget: 32,
    requested_reasoning_budget: -1,
  }, { ...BLANK, reasoningBudget: 32 }), false);
  assert.equal(matches({
    reasoning_budget_message: "Conclude now.",
    requested_reasoning_budget_message: "",
  }, { ...BLANK, reasoningBudgetMessage: "Conclude now." }), false);
});

test("an explicit zero reasoning request can reuse the resident", () => {
  assert.equal(matches({
    reasoning_budget: 0,
    requested_reasoning_budget: 0,
  }, { ...BLANK, reasoningBudget: 0 }), true);
});

test("legacy status without reasoning request echoes keeps its comparison", () => {
  assert.equal(matches({
    reasoning_budget: 32,
    reasoning_budget_message: "Conclude now.",
  }, { ...BLANK, reasoningBudget: 32, reasoningBudgetMessage: "Conclude now." }), true);
});

test("a pick asks the status about its own model and keeps or replaces the others per the box", () => {
  const CONFIRM = readSrc("features/chat/utils/confirm-stop-running-chats.ts");
  assert.equal(USE_CHAT_MODEL_RUNTIME.match(/await readPickStatus\(\)/g)?.length, 2);
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /let keepsOthers =\s*keepModelsLoaded && !forceReload && \(paramsNow\.engine \?\? "auto"\) === "auto";[\s\S]*?const touchesOnlySelected =\s*forceReload && !isExternalModelId\(paramsNow\.checkpoint\) && loadedNow\.length > 1;/,
  );
  assert.match(USE_CHAT_MODEL_RUNTIME, /touchesOnlySelected \? \(paramsNow\.checkpoint \?\? undefined\) : undefined,/);
  assert.match(USE_CHAT_MODEL_RUNTIME, /stopQueuedRuns\(stopDecision, keepsOthers \|\| touchesOnlySelected\);/);
  assert.match(USE_CHAT_MODEL_RUNTIME, /if \(!keepsOthers && !touchesOnlySelected\) \{\s*requestLocalPromptQueueStop\(\);/);
  assert.match(USE_CHAT_MODEL_RUNTIME, /if \(currentCheckpoint && !keepsOthers\)/);
  assert.match(USE_CHAT_MODEL_RUNTIME, /if \(!forceCancelActive && !touchesOnlySelected\) \{/);
  assert.equal(
    USE_CHAT_MODEL_RUNTIME.match(/alongside: keepModelsLoaded \|\| touchesOnlySelected,/g)?.length,
    2,
  );
  assert.match(CONFIRM, /let running = model\s*\?\s*\[\]/);
  assert.match(CONFIRM, /await getActiveGenerations\(model\)/);
});

test("ejects stop only the ejected model's chats; eject all asks once and unloads the others first", () => {
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /function stopQueuedRuns\(decision: StopRunningChatsDecision, scoped: boolean\): void \{\s*if \(scoped\) \{\s*requestScopedLocalPromptQueueStop\(decision\.promptQueueThreadIds\);\s*return;\s*\}\s*cancelPreStreamRunReservations\(decision\.preStreamRunTokens\);\s*requestLocalPromptQueueStop\(decision\.promptQueueThreadIds\);/,
  );
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /confirmStopRunningChatsIfNeeded\("Unloading this model", "unload", keptId\)[\s\S]{0,80}?if \(!decision\.proceed\) return false;\s*stopQueuedRuns\(decision, true\);/,
  );
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /const scope =\s*!confirmed && useChatRuntimeStore\.getState\(\)\.loadedModels\.length > 1\s*\?\s*params\.checkpoint\s*:\s*undefined;\s*const stopDecision =\s*confirmed \?\?\s*\(await confirmStopRunningChatsIfNeeded\(\s*"Unloading the model",\s*"unload",\s*scope,\s*\)\);/,
  );
  assert.match(USE_CHAT_MODEL_RUNTIME, /stopQueuedRuns\(stopDecision, Boolean\(scope\)\);/);
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /"Unloading every model",\s*"unload",\s*\);\s*if \(!decision\.proceed\) return false;\s*\/\/ Before any unload[^\n]*\n\s*stopQueuedRuns\(decision, false\);\s*\/\/ Others first/,
  );
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /others\.map\(\(id\) =>\s*unloadModel\(\{ model_path: id, force_cancel_active: decision\.forceCancelActive \}\),\s*\),\s*\);\s*if \(selectedLocal && !\(await ejectModel\(undefined, decision\)\)\) return false;\s*await refresh\(\);/,
  );
});

test("cancelling a load clears the selection unless kept models stay loaded and this run unloaded none", () => {
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /if \(!preserveCheckpoint\) \{[\s\S]{0,160}?if \(!useChatRuntimeStore\.getState\(\)\.keepModelsLoaded \|\| run\.residentModelUnloaded\) \{\s*clearCheckpoint\(\);\s*\}\s*await refresh\(\);/,
  );
});

test("reloading one of several stays in its own slot; a new pick with the setting off replaces as before", () => {
  // /load replaces the model in its own slot, so no preliminary unload is needed.
  assert.doesNotMatch(USE_CHAT_MODEL_RUNTIME, /replacesOneOfSeveral/);
  assert.match(USE_CHAT_MODEL_RUNTIME, /const touchesOnlySelected =\s*forceReload &&/);
});


test("a remembered split matches the resident without relying on another model's store ratio", () => {
  const config = {
    ...BLANK,
    gpuMemoryMode: "manual" as const,
    gpuLayers: 66,
    selectedGpuIds: [1, 2, 0],
    selectedGpuIndexKind: "physical" as const,
    tensorSplit: [30, 20, 16],
  };
  const running = {
    ...DEFAULTS,
    gpu_memory_mode: "manual" as const,
    gpu_layers: 66,
    gpu_ids: [1, 2, 0],
    requested_gpu_ids: [1, 2, 0],
    tensor_split: [30, 20, 16],
  };
  assert.equal(matches(running, config), true);
  assert.equal(matches({ ...running, tensor_split: [22, 22, 22] }, config), false);
  assert.equal(matches(running, { ...config, tensorSplit: null }), false);
  assert.equal(matches(
    { ...running, gpu_ids: null, requested_gpu_ids: null, tensor_split: null },
    config,
    { ...STANDING, reconcileGpuIds: () => null },
  ), true);
});
