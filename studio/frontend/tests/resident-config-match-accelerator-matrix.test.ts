// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `residentRuntimeMatchesConfig` across the accelerators Studio runs on, crossed with every
 * setting a remembered config can pin.
 *
 * The two failures are not symmetric. A wrong FALSE costs one reload, which is what
 * happened before #8893. A wrong TRUE leaves the user on a server invoked differently from
 * what they asked for, with the panel rolled back so nothing says so. Hence both directions
 * for every field rather than sampling.
 *
 * The accelerator axis is real even though no paths are compared, because the fields read
 * are the GPU ones: CUDA and ROCm report a placement pool and an offload mode, a CPU-only
 * host reports `manual` with zero layers, MLX reports none of them and a KV width instead.
 * "The status does not carry this field" must never read as agreement.
 *
 * The structural test at the bottom is the one that lasts: it fails when a field is added
 * to `PerModelConfig` without being classified here.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { residentRuntimeMatchesConfig: matchesWithStanding } = await import(
  "../src/features/chat/lib/resident-config-match.ts"
);

/**
 * What the applier resolves an unset field to. Four fields are not per-model, so leaving
 * them out of a config is not silence; the caller passes what `/load` would actually send.
 * These are the shipped defaults: speculation off, memory mode "auto", `GPU_LAYERS_AUTO`
 * and no CPU MoE offload.
 */
const STANDING = {
  speculativeType: null,
  gpuMemoryMode: "auto" as const,
  gpuLayers: -1,
  nCpuMoe: 0,
  // Identity: the sweep is about the fields, not about a stale pick. The reconciler's own
  // effect is covered in resident-config-match.test.ts.
  reconcileGpuIds: (ids: number[] | null) => ids,
  // Auto resolves to 0 here; the resident-repick branch is exercised on its own below.
  resolveContextLength: (customContextLength: number | null) =>
    customContextLength ?? 0,
  parallelSlots: 1,
  splitRatio: null,
  normalizeSpeculative: (value: string | null | undefined) =>
    value == null || value === "" || value === "none" ? null : value,
};

const residentRuntimeMatchesConfig = (
  status: Parameters<typeof matchesWithStanding>[0],
  config: Parameters<typeof matchesWithStanding>[1],
) => matchesWithStanding(status, config, STANDING);

/** A config that pins nothing: what a model the user never configured carries. */
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

/** What `/api/inference/status` reports per host, measured against a running Studio rather
 * than copied from the type: a default CUDA load answers auto / -1 / 0 / null / false. */
const ACCELERATORS: Record<string, Record<string, unknown>> = {
  "nvidia-cuda": {
    gpu_memory_mode: "auto",
    gpu_layers: -1,
    n_cpu_moe: 0,
    // A default load requests no particular GPU, so the echo is null. The set comparison
    // on a real placement pool has its own test below; here a non-null pool would make
    // BLANK disagree, since an unset pick resolves to Automatic rather than to "any".
    requested_gpu_ids: null,
    tensor_parallel: false,
  },
  "amd-rocm": {
    gpu_memory_mode: "auto",
    gpu_layers: -1,
    n_cpu_moe: 0,
    // Left unrequested for the same reason as above; the ROCm-shaped pool of several
    // physical indices, compared as a SET, has its own test below.
    requested_gpu_ids: null,
    tensor_parallel: false,
  },
  // A host with no GPU still reports the invocation it was ASKED for, not what llama.cpp
  // settled on: a default load sends Auto, so the echo is "auto" / -1 exactly as on a CUDA
  // box, with no placement pool. A manual zero-layer load is a different thing and is
  // covered on its own below, since an unset config resolves to Auto and so differs from it.
  "cpu-only": {
    gpu_memory_mode: "auto",
    gpu_layers: -1,
    n_cpu_moe: 0,
    requested_gpu_ids: null,
    tensor_parallel: false,
  },
  // An MLX server records a KV width and none of the llama.cpp placement fields at all. A
  // default load leaves the width unrequested; a pinned width is swept below like any other
  // field, and is not part of the base for the same reason placement is not.
  "apple-mlx": {
    mlx_kv_bits_requested: null,
  },
};

/** One field a remembered config can pin, and a value that is NOT what the status runs. */
type FieldCase = {
  key: string;
  statusKey: string;
  same: unknown;
  different: unknown;
  /**
   * What must hold on both sides for the field to be live at all. The backend compares
   * the offload knobs only under Manual, and the MoE count only with a layer pin beside
   * it, so sweeping them under Auto would assert a comparison that does not happen.
   */
  live?: { config: Record<string, unknown>; status: Record<string, unknown> };
};

const MANUAL_MODE = {
  config: { gpuMemoryMode: "manual" },
  status: { gpu_memory_mode: "manual" },
};

const FIELDS: FieldCase[] = [
  {
    key: "customContextLength",
    statusKey: "requested_context_length",
    same: 32768,
    different: 8192,
  },
  {
    key: "kvCacheDtype",
    statusKey: "cache_type_kv",
    same: "q8_0",
    different: "f16",
  },
  {
    key: "mlxKvBits",
    statusKey: "mlx_kv_bits_requested",
    same: 8,
    different: 4,
  },
  {
    key: "speculativeType",
    statusKey: "speculative_type",
    same: "ngram",
    different: "auto",
  },
  {
    key: "specDraftNMax",
    statusKey: "spec_draft_n_max",
    same: 16,
    different: 8,
  },
  {
    key: "nParallel",
    statusKey: "requested_parallel_slots",
    same: 4,
    different: 2,
  },
  {
    key: "nBatch",
    statusKey: "requested_n_batch",
    same: 2048,
    different: 1024,
  },
  {
    key: "nUbatch",
    statusKey: "requested_n_ubatch",
    same: 512,
    different: 256,
  },
  {
    key: "chatTemplateOverride",
    statusKey: "chat_template_override",
    same: "{{ bos }}",
    different: "{{ eos }}",
  },
  {
    key: "llamaExtraArgs",
    statusKey: "requested_llama_extra_args",
    same: ["--threads", "8"],
    different: ["--threads", "4"],
  },
  {
    key: "gpuMemoryMode",
    statusKey: "gpu_memory_mode",
    same: "manual",
    different: "auto",
  },
  {
    key: "gpuLayers",
    statusKey: "gpu_layers",
    same: 20,
    different: 10,
    live: MANUAL_MODE,
  },
  {
    key: "nCpuMoe",
    statusKey: "n_cpu_moe",
    same: 8,
    different: 4,
    live: {
      config: { ...MANUAL_MODE.config, gpuLayers: 20 },
      status: { ...MANUAL_MODE.status, gpu_layers: 20 },
    },
  },
  {
    key: "selectedGpuIds",
    statusKey: "requested_gpu_ids",
    same: [1, 0],
    different: [0, 2],
  },
  {
    key: "tensorParallel",
    statusKey: "tensor_parallel",
    same: true,
    different: false,
  },
];

for (const [accelerator, base] of Object.entries(ACCELERATORS)) {
  test(`[${accelerator}] a config pinning nothing adopts the resident load`, () => {
    assert.equal(residentRuntimeMatchesConfig(base, BLANK), true);
    assert.equal(residentRuntimeMatchesConfig(base, null), true);
  });

  for (const field of FIELDS) {
    test(`[${accelerator}] ${field.key} the resident load already runs is adopted`, () => {
      assert.equal(
        residentRuntimeMatchesConfig(
          { ...base, ...field.live?.status, [field.statusKey]: field.same },
          { ...BLANK, ...field.live?.config, [field.key]: field.same },
        ),
        true,
      );
    });

    test(`[${accelerator}] ${field.key} the resident load does not run is a reload`, () => {
      assert.equal(
        residentRuntimeMatchesConfig(
          { ...base, ...field.live?.status, [field.statusKey]: field.different },
          { ...BLANK, ...field.live?.config, [field.key]: field.same },
        ),
        false,
      );
    });

    test(`[${accelerator}] ${field.key} pinned against a status that omits it is a reload`, () => {
      // Never agreement: a field the server cannot report is one this cannot verify.
      const status = { ...base, ...field.live?.status } as Record<
        string,
        unknown
      >;
      delete status[field.statusKey];
      assert.equal(
        residentRuntimeMatchesConfig(status, {
          ...BLANK,
          ...field.live?.config,
          [field.key]: field.same,
        }),
        // tensorParallel has no unset state: false is a real request for a layer split,
        // and a status omitting the flag ran without one, so they agree.
        field.key === "tensorParallel" ? field.same === false : false,
      );
    });
  }
}

test("placement compares as a set on a multi-GPU host, not as an order", () => {
  // The backend narrows and reorders the pool at fit time, so only membership counts.
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["amd-rocm"], requested_gpu_ids: [0, 1] },
      { ...BLANK, selectedGpuIds: [1, 0] },
    ),
    true,
  );
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["amd-rocm"], requested_gpu_ids: [0, 1] },
      { ...BLANK, selectedGpuIds: [0, 1, 2] },
    ),
    false,
  );
});

test("automatic placement is a request of its own, not a wildcard", () => {
  // The applier turns an absent or null selection into a null selection, and the load sends
  // that as Automatic. Automatic is not "whatever is running": a server the user placed on
  // four specific GPUs was invoked differently and has to be reloaded.
  for (const ids of [null, undefined]) {
    assert.equal(
      residentRuntimeMatchesConfig(
        { ...ACCELERATORS["nvidia-cuda"], requested_gpu_ids: [0, 1, 2, 3] },
        { ...BLANK, selectedGpuIds: ids },
      ),
      false,
    );
    assert.equal(
      residentRuntimeMatchesConfig(ACCELERATORS["nvidia-cuda"], {
        ...BLANK,
        selectedGpuIds: ids,
      }),
      true,
    );
  }
});

test("a CPU-only host distinguishes zero offloaded layers from automatic", () => {
  // gpu_layers 0 under manual is "all on the CPU", -1 is "let llama.cpp size it": two
  // different loads, and 0 must not read as absent.
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["cpu-only"], gpu_memory_mode: "manual", gpu_layers: 0 },
      { ...BLANK, gpuMemoryMode: "manual", gpuLayers: 0 },
    ),
    true,
  );
  // The other direction, which is the one the standing defaults buy: a config that pins
  // neither field resolves to Auto and so does NOT adopt a manual zero-layer server.
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["cpu-only"], gpu_memory_mode: "manual", gpu_layers: 0 },
      BLANK,
    ),
    false,
  );
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["nvidia-cuda"] },
      { ...BLANK, gpuMemoryMode: "manual", gpuLayers: 0 },
    ),
    false,
  );
});

test("a config stored by an older Studio does not throw and does not over-adopt", () => {
  // Blobs written before a field existed lack the key. Optional ones express no opinion
  // and adopt; a missing tensorParallel cannot be confirmed and reloads. Neither throws.
  // Cast at the boundary on purpose: these come off localStorage, so the point is the
  // shapes the current type says cannot occur.
  const legacyBlobs = [
    // Pre-GPU-controls, pre-extra-args, pre-MLX.
    {
      customContextLength: null,
      maxSeqLength: null,
      kvCacheDtype: null,
      speculativeType: null,
      specDraftNMax: null,
      nParallel: null,
      nBatch: null,
      nUbatch: null,
      tensorParallel: false,
      chatTemplateOverride: null,
    },
    // A blob so old it carries only what the very first version stored.
    { customContextLength: null, kvCacheDtype: null },
    // Corrupt-but-parseable: the key is there with the wrong emptiness.
    { ...BLANK, selectedGpuIds: [] },
  ];
  for (const blob of legacyBlobs) {
    for (const base of Object.values(ACCELERATORS)) {
      assert.equal(
        typeof residentRuntimeMatchesConfig(base, blob as PerModelConfig),
        "boolean",
      );
    }
  }
  assert.equal(
    residentRuntimeMatchesConfig(
      ACCELERATORS["nvidia-cuda"],
      legacyBlobs[1] as PerModelConfig,
    ),
    // tensorParallel absent vs status false: reloads, the direction that loses nothing.
    false,
  );
});

test("an empty pinned pool is Automatic, not a demand for no GPUs", () => {
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["nvidia-cuda"], requested_gpu_ids: [0, 1] },
      { ...BLANK, selectedGpuIds: [] },
    ),
    // set() treats [] as pinned, so this reloads. Recorded, not endorsed: the panel
    // writes null for Automatic.
    false,
  );
});

test("every PerModelConfig field is either compared or deliberately excluded", () => {
  // A new setting not classified here is one an adopted pick would drop silently.
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/model-config/per-model-config.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const body = source.slice(
    source.indexOf("export interface PerModelConfig {"),
    source.indexOf("export const DEFAULT_PER_MODEL_CONFIG"),
  );
  const declared = new Set(
    [...body.matchAll(/^\s{2}(\w+)\??:/gm)].map((match) => match[1]),
  );
  assert.ok(declared.size > 10, "failed to parse PerModelConfig");

  const compared = new Set(FIELDS.map((field) => field.key));
  const excluded = new Set([
    // A client-side generation cap: no status echoes it, so it cannot force a reload.
    "maxSeqLength",
    // Qualifies selectedGpuIds rather than adding a dimension of its own: it is read, as
    // the reconciler's namespace argument, but /status has no field to compare it against.
    "selectedGpuIndexKind",
  ]);
  const unclassified = [...declared].filter(
    (field) => !compared.has(field) && !excluded.has(field),
  );
  assert.deepEqual(unclassified, []);
  // And the reverse, so a removed field leaves no dead check here.
  const stale = [...compared, ...excluded].filter(
    (field) => !declared.has(field),
  );
  assert.deepEqual(stale, []);
});
