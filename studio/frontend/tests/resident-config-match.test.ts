// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * #8893's fix skips the reload when the picked model is already resident. Identity is not
 * the whole of a load, though: the picker and Hub's Run button both pass a REMEMBERED
 * config without forceReload (chat-page.tsx stageOrLoad, hub-page.tsx handleRun), and the
 * backend reloads for any of those settings changing. Adopting on identity alone dropped
 * them silently, because the same path rolls the panel back to the resident model and so
 * looks consistent either way.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { residentRuntimeMatchesConfig, residentSpeculativeNeedsRepair } =
  await import("../src/features/chat/lib/resident-config-match.ts");

/** Every field unset: what a model the user never configured would carry. */
const BLANK = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  mlxKvBits: null,
  speculativeType: null,
  specDraftNMax: null,
  nParallel: null,
  nBatch: null,
  nUbatch: null,
  tensorParallel: false,
  chatTemplateOverride: null,
};

/** A resident llama-server running nothing but defaults. */
const DEFAULTS = {};

/**
 * What the applier fills an unset field with. Four fields are not per-model, so leaving
 * them out of a config is not silence: applyPerModelConfigToRuntime resolves them from a
 * standing preference or a constant, and the load sends the result.
 */
const STANDING = {
  speculativeType: "auto",
  gpuMemoryMode: "auto" as const,
  gpuLayers: -1,
  nCpuMoe: 0,
  // Identity by default: reconciliation is exercised on its own below.
  reconcileGpuIds: (ids: number[] | null) => ids,
  // Auto resolves to 0 here; the resident-repick branch is exercised on its own below.
  resolveContextLength: (customContextLength: number | null) =>
    customContextLength ?? 0,
  parallelSlots: 1,
  splitRatio: null,
  // Enough of normalizeSpeculativeType for these cases; the real mapping is the store's
  // and is tested there. What matters here is that the comparator USES it on both sides.
  normalizeSpeculative: (v: string | null | undefined) =>
    v == null || !String(v).trim()
      ? null
      : String(v).trim().toLowerCase() === "default"
        ? "auto"
        : String(v).trim().toLowerCase(),
};

/** Shorthand: the comparator always needs the standing defaults. */
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

/** The regression this file exists for: the setting must reach the server. */
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

/** Every field the backend's _runtime_matches_intent reloads for, one per row. */
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
    name: "MLX KV bits",
    config: { mlxKvBits: 4 },
    same: { mlx_kv_bits_requested: 4 },
    differs: { mlx_kv_bits_requested: 8 },
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
    name: "GPU layers",
    config: { gpuLayers: 20 },
    same: { gpu_layers: 20 },
    differs: { gpu_layers: 99 },
  },
  {
    name: "CPU MoE layers",
    config: { nCpuMoe: 12 },
    same: { n_cpu_moe: 12 },
    differs: { n_cpu_moe: 0 },
  },
  {
    name: "GPU placement",
    config: { selectedGpuIds: [0, 2] },
    same: { requested_gpu_ids: [2, 0] },
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
    // An older backend that does not echo the field cannot prove it agrees, and guessing
    // that it does is the one answer that loses a setting with nothing on screen to say so.
    assert.equal(matches({}, { ...BLANK, ...field.config }), false);
  });
}

/** Ordering is the backend's to choose: it narrows and reorders placement at fit time. */
test("GPU placement compares as a set, not as an order", () => {
  assert.equal(
    matches(
      { requested_gpu_ids: [3, 1, 0] },
      { ...BLANK, selectedGpuIds: [0, 1, 3] },
    ),
    true,
  );
});

test("automatic placement does not adopt a load pinned to chosen GPUs", () => {
  // Automatic is what the applier resolves an unset selection to, and it is what the load
  // would then send, so it disagrees with a server placed on a chosen pool.
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
  assert.equal(matches({ gpu_layers: 40 }, { ...BLANK, gpuLayers: 0 }), false);
  assert.equal(matches({ gpu_layers: 0 }, { ...BLANK, gpuLayers: 0 }), true);
  assert.equal(
    matches(
      { requested_context_length: 0 },
      { ...BLANK, customContextLength: 0 },
    ),
    // 0 is Auto and PerModelConfig stores "unset" as null, so a stored 0 is a real pin.
    true,
  );
});

/** Several pins at once, which is what a configured model actually carries. */
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

/**
 * The comparator only helps if the pick is tested against it BEFORE the reload is decided,
 * and the two gates beside it are invariants rather than preferences: a staged config
 * carries forceReload, and a native pick carries a lease this path cannot adopt.
 */
test("selectModel weighs the config and the lease before confirming a reload", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/hooks/use-chat-model-runtime.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  // Newline-tolerant: the call wraps once its argument list grows.
  const configCheck = source.search(
    /residentRuntimeMatchesConfig\(\s*residentStatus/,
  );
  const identityCheck = source.indexOf(
    "residentModelMatchesPick(residentStatus",
  );
  const confirmPrompt = source.indexOf(
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
  // A leased native file is named by a label two files can share, and the lease itself is
  // written only by a completed load, so this path must not adopt one.
  // Widened as the gate's preamble grows: what matters is that the guard opens the block
  // the identity check sits in, not how many reads it makes first.
  const guard = source.lastIndexOf(
    "if (!forceReload && !nativePathToken) {",
    identityCheck,
  );
  assert.ok(
    guard > 0,
    "the resident short-circuit no longer excludes native-lease picks",
  );
});

/**
 * The four fields that are NOT per-model. Leaving one out of a config is not silence:
 * applyPerModelConfigToRuntime resolves it from a standing preference or a constant and
 * the load sends that, so reading it as unpinned let a pick adopt a runtime it did not
 * ask for. Reported on this file by review, reproduced against the applier, fixed here.
 */
test("an unset speculative mode is the standing preference, not silence", () => {
  // Standing preference is "off"; the resident model is running MTP. The load would send
  // "off", so this is a real reload even though the config names no mode.
  assert.equal(
    matches({ ...DEFAULTS, speculative_type: "mtp" }, BLANK, {
      ...STANDING,
      speculativeType: "off",
    }),
    false,
  );
  // Same standing preference, and the resident model already runs it.
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
  // "default" and "auto" are the same mode; a spelling difference is not a reload.
  assert.equal(
    matches({ ...DEFAULTS, speculative_type: "default" }, BLANK),
    true,
  );
});

test("the GPU pick is compared after reconciliation, not as saved", () => {
  // performLoad sends reconcilePersistedGpuIds(ids, kind): a pick saved in another index
  // namespace, or naming GPUs that are gone, leaves as Automatic. Comparing the saved ids
  // let a physical [1] adopt a server pinned to Vulkan device 1.
  const dropped = { ...STANDING, reconcileGpuIds: () => null };
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [1] },
      { ...BLANK, selectedGpuIds: [1], selectedGpuIndexKind: "physical" },
      dropped,
    ),
    false,
  );
  // The reverse, so the reconciler is not merely refusing everything: once the pick is
  // Automatic it agrees with a server that was placed automatically.
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: null },
      { ...BLANK, selectedGpuIds: [1], selectedGpuIndexKind: "physical" },
      dropped,
    ),
    true,
  );
  // And the kind reaches the reconciler, which is the only thing that can use it.
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
  // resolveLoadMaxSeqLength answers 0 for a cross-model GGUF pick and the resident context
  // when re-picking the same one. Comparing null against either was a reload the backend
  // would have deduplicated.
  assert.equal(
    matches({ ...DEFAULTS, requested_context_length: 0 }, BLANK),
    true,
  );
  assert.equal(
    matches({ ...DEFAULTS, requested_context_length: 32768 }, BLANK, {
      ...STANDING,
      // The re-pick branch: ggufContextLength, not 0.
      resolveContextLength: (pin) => pin ?? 32768,
    }),
    true,
  );
  // Still a reload when the resolved value really differs.
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
  // _resolve_parallel_slots fills an omitted --parallel from the server-wide default and
  // stores THAT as requested_parallel_slots, so the status never echoes null. Comparing
  // null against it reloaded every default pick, which is the common case.
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 4 }, BLANK, {
      ...STANDING,
      parallelSlots: 4,
    }),
    true,
  );
  // A pick that names a different count is still a reload.
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
  // And a resident load pinned above the default is not adopted by an unset pick.
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 8 }, BLANK, {
      ...STANDING,
      parallelSlots: 4,
    }),
    false,
  );
  // Unknown default: reload, the safe direction.
  assert.equal(
    matches({ ...DEFAULTS, requested_parallel_slots: 4 }, BLANK, {
      ...STANDING,
      parallelSlots: null,
    }),
    false,
  );
});

test("a pick naming the fitted subset of a wider pool is not a reload", () => {
  // matches_gpu_ids accepts the request or the effective pool: fitting narrows [0, 1] to
  // [0] when that is the smallest subset holding the model, and asking for [0] dedupes.
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [0, 1], gpu_ids: [0] },
      { ...BLANK, selectedGpuIds: [0] },
    ),
    true,
  );
  // The raw request still answers for itself.
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [0, 1], gpu_ids: [0] },
      { ...BLANK, selectedGpuIds: [0, 1] },
    ),
    true,
  );
  // A pool neither of them names is still a reload.
  assert.equal(
    matches(
      { ...DEFAULTS, requested_gpu_ids: [0, 1], gpu_ids: [0] },
      { ...BLANK, selectedGpuIds: [1] },
    ),
    false,
  );
  // An absent echo is no placement, not Automatic: reading it as Automatic would let an
  // unpinned pick adopt every pinned server.
  assert.equal(
    matches({ ...DEFAULTS, requested_gpu_ids: [0, 1] }, BLANK),
    false,
  );
});

test("a retry arm that records no fallback reason still declines the shortcut", () => {
  // _dflash_retry_needed and the capability-probe arm both reject an identical load while
  // leaving spec_fallback_reason null, so reading only the reason adopted a runtime the
  // backend would have rebuilt, and nothing else would ever retry it.
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
  // The backend gates that arm on those two modes; a pick asking for MTP is not it.
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: null, spec_dflash_retry_pending: true },
      "mtp",
    ),
    false,
  );
  // The probe arm has no mode gate at all.
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: null, spec_probe_retry_pending: true },
      "off",
    ),
    true,
  );
  // Neither pending, and no reason: nothing to repair.
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
  // spec_binary_fallback_can_retry needs a different llama-server installed before an
  // identical /load repairs anything; without one it dedupes and the prompt was for nothing.
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
  // Updated since launch: the repair the update was for must still go through.
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
  // A backend too old to report it keeps the coarser answer rather than suppress a repair.
  assert.equal(
    residentSpeculativeNeedsRepair(
      { spec_fallback_reason: "binary_no_mtp" },
      "auto",
    ),
    true,
  );
  // The flag says nothing about a reason that was never about the binary.
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

test("a custom tensor split the config cannot carry is still a reload", () => {
  // applyPerModelConfigToRuntime clears splitRatio, so a remembered config asks for the
  // default distribution while the resident manual load runs a custom one.
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
  // _preserve_cpu_fallback_intent rewrites an eligible Auto request into the resident
  // manual/zero-layer intent before the comparison, so /load would report already-loaded.
  const fallback = {
    ...DEFAULTS,
    gpu_memory_mode: "manual" as const,
    gpu_layers: 0,
    cpu_fallback_reason: "vulkan_startup_crash" as const,
  };
  assert.equal(matches(fallback, BLANK), true);
  // Only placement is exempt: a real setting difference still reloads.
  assert.equal(matches({ ...fallback, cache_type_kv: "q8_0" }, BLANK), false);
  // And only for a request the backend would actually rewrite. _cpu_fallback_request_eligible
  // refuses one that pins its own placement.
  assert.equal(matches(fallback, { ...BLANK, selectedGpuIds: [0] }), false);
  assert.equal(matches(fallback, { ...BLANK, tensorParallel: true }), false);
  assert.equal(matches(fallback, { ...BLANK, nCpuMoe: 4 }), false);
  assert.equal(
    matches(fallback, { ...BLANK, llamaExtraArgs: ["--device", "Vulkan0"] }),
    false,
  );
  // A fallback from another cause is not this one.
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
  // GPU_LAYERS_AUTO is -1; a resident manual pin is a real reload.
  assert.equal(matches({ ...DEFAULTS, gpu_layers: 20 }, BLANK), false);
  assert.equal(matches({ ...DEFAULTS, gpu_layers: -1 }, BLANK), true);
  assert.equal(matches({ ...DEFAULTS, n_cpu_moe: 12 }, BLANK), false);
  assert.equal(matches({ ...DEFAULTS, n_cpu_moe: 0 }, BLANK), true);
});

test("no config at all still adopts, whatever the resident runtime is", () => {
  // Nothing to send means nothing can differ: the load path reads the live runtime, which
  // was hydrated from the resident model.
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

/**
 * The nullable per-model settings are pinned for the same reason the standing four are, and
 * the reason is the applier rather than the field's type. `applyModelLoadConfigToRuntime`
 * writes the config over the runtime store before `selectModel` runs (`chat-page.tsx:3242`,
 * `hub-page.tsx:1329`) and resolves each of these with `?? null`, so the snapshot
 * `performLoad` takes reads null rather than inheriting the resident model's value. A pick
 * that leaves the box empty is asking for the default, not for silence.
 */
test("unset nullable settings ask for the default, not for the resident value", () => {
  const pinnedResident = {
    ...DEFAULTS,
    requested_context_length: 8192,
    cache_type_kv: "q8_0",
    mlx_kv_bits_requested: 4,
    spec_draft_n_max: 16,
    requested_parallel_slots: 4,
    requested_n_batch: 2048,
    requested_n_ubatch: 512,
    chat_template_override: "{{ bos }}",
  };
  assert.equal(matches(pinnedResident, BLANK), false);
  // Field by field, so a regression names itself rather than reporting one false.
  for (const [key, value] of Object.entries({
    requested_context_length: 8192,
    cache_type_kv: "q8_0",
    mlx_kv_bits_requested: 4,
    spec_draft_n_max: 16,
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
  // And the default-against-default case still adopts, which is the whole point of #8893.
  assert.equal(matches(DEFAULTS, BLANK), true);
});

test("a blank chat template agrees with a load that has none", () => {
  // Both ends trim: the applier's cleanTemplate and the load both send "" as null, so an
  // all-whitespace override is not a difference.
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

/**
 * The speculative repair window. `_runtime_matches_intent` answers False for a retryable
 * drafter failure so the next identical load fixes it, which is the only case where
 * skipping the load is not free. Reading a permanent downgrade as repairable is the
 * opposite failure: it would prompt to stop running chats on every re-pick, which is #8893
 * again, and repair nothing.
 */
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
  // Speculation off: the retry arms are all guarded on a speculative mode, so a pick that
  // asks for none has nothing to repair.
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

/**
 * maxSeqLength is the one setting the comparator deliberately ignores, because no status
 * field echoes it, and that is exactly why the shortcut has to carry it: the rollback that
 * makes the panel agree with the resident server is the last word on a client-only cap, and
 * it speaks for the OUTGOING model. Structural because the write happens inside selectModel
 * against the live store rather than in this leaf.
 */
test("the resident shortcut keeps the picked model's own sequence cap", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/hooks/use-chat-model-runtime.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const configCheck = source.search(
    /residentRuntimeMatchesConfig\(\s*residentStatus/,
  );
  const rollback = source.indexOf("restorePreviousConfig();", configCheck);
  const reapply = source.indexOf("pickedMaxSeqLength", configCheck);
  assert.ok(
    rollback > 0,
    "the shortcut no longer rolls the staged config back",
  );
  assert.ok(
    reapply > rollback,
    "the shortcut leaves the outgoing model's maxSeqLength in place",
  );
  const confirmPrompt = source.indexOf(
    "await confirmStopRunningChatsIfNeeded(",
  );
  assert.ok(reapply < confirmPrompt, "the re-apply escaped the shortcut");
  // An absent cap is not "leave it alone": applyPerModelConfigToRuntime resolves it to the
  // default, so a pick without one must land on the default rather than the outgoing cap.
  assert.match(
    source.slice(reapply, reapply + 260),
    /\?\?\s*defaultInferenceParams\.maxSeqLength/,
    "an absent cap no longer resolves to the default",
  );
});

/**
 * The comparison must be against what /load would send, and with no saved config that is
 * the live runtime store: the caller ran applyModelLoadConfigToRuntime(null) first, which
 * resets it to DEFAULT_PER_MODEL_CONFIG, and performLoad reads the store for every field
 * the config does not carry. Passing the absent config straight through made the gate a
 * wildcard, adopting whatever another tab or API client had left running.
 */
test("with no saved config the gate compares the live runtime, not nothing", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/hooks/use-chat-model-runtime.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const configCheck = source.search(
    /residentRuntimeMatchesConfig\(\s*residentStatus/,
  );
  assert.match(
    source.slice(configCheck, configCheck + 200),
    /pendingConfig\s*\?\?\s*currentRuntimePerModelConfig\(\)/,
    "the gate takes an absent config as a wildcard again",
  );
});

/** The repair window is only useful if it is consulted before the reload is decided. */
test("selectModel asks about a repairable drafter before adopting", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/hooks/use-chat-model-runtime.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const repairCheck = source.indexOf("residentSpeculativeNeedsRepair(");
  const confirmPrompt = source.indexOf(
    "await confirmStopRunningChatsIfNeeded(",
  );
  assert.ok(
    repairCheck > 0,
    "selectModel no longer reloads a degraded drafter",
  );
  assert.ok(repairCheck < confirmPrompt);
});
