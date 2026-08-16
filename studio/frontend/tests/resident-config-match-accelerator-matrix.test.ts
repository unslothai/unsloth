// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `residentRuntimeMatchesConfig` across the accelerators Studio runs on, and across every
 * setting a remembered config can pin.
 *
 * Two different failures live here and they are not symmetric. A wrong FALSE costs one
 * reload, which is what happened before #8893 was fixed. A wrong TRUE leaves the user on a
 * server invoked differently from what they asked for, with the panel rolled back to the
 * resident model so nothing on screen says so. The sweep below therefore checks both
 * directions for every field rather than sampling.
 *
 * The accelerator axis is real for this function even though it compares no paths: the
 * fields it reads are the GPU ones. A CUDA or ROCm host reports a placement pool and an
 * offload mode; a CPU-only host reports `manual` with zero layers; an Apple MLX server
 * reports none of them and a KV width instead. A check written against one shape has to
 * behave on the others, and "the status does not carry this field" must never read as
 * agreement.
 *
 * The structural test at the bottom is the one that matters over time: it fails when a
 * field is added to `PerModelConfig` without being classified here, so a new setting
 * cannot be silently dropped by an adopted pick.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { residentRuntimeMatchesConfig } = await import(
  "../src/features/chat/lib/resident-config-match.ts"
);

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

/**
 * What `/api/inference/status` reports on each host, as measured against a running Studio
 * rather than copied from the type: a default CUDA load answers gpu_memory_mode "auto",
 * gpu_layers -1, n_cpu_moe 0, requested_gpu_ids null, tensor_parallel false.
 */
const ACCELERATORS: Record<string, Record<string, unknown>> = {
  "nvidia-cuda": {
    gpu_memory_mode: "auto",
    gpu_layers: -1,
    n_cpu_moe: 0,
    requested_gpu_ids: [0],
    tensor_parallel: false,
  },
  "amd-rocm": {
    gpu_memory_mode: "auto",
    gpu_layers: -1,
    n_cpu_moe: 0,
    // ROCm placement is physical indices, and more than one of them is the common case
    // this function has to compare as a SET. The split flag stays off here so BLANK agrees
    // with every base; tensor_parallel true is covered by the sweep below.
    requested_gpu_ids: [0, 1],
    tensor_parallel: false,
  },
  "cpu-only": {
    gpu_memory_mode: "manual",
    gpu_layers: 0,
    n_cpu_moe: 0,
    requested_gpu_ids: null,
    tensor_parallel: false,
  },
  // An MLX server records a KV width and none of the llama.cpp placement fields at all.
  "apple-mlx": {
    mlx_kv_bits_requested: 8,
  },
};

/** One field a remembered config can pin, and a value that is NOT what the status runs. */
type FieldCase = {
  key: string;
  statusKey: string;
  same: unknown;
  different: unknown;
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
  { key: "gpuLayers", statusKey: "gpu_layers", same: 20, different: 10 },
  { key: "nCpuMoe", statusKey: "n_cpu_moe", same: 8, different: 4 },
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
          { ...base, [field.statusKey]: field.same },
          { ...BLANK, [field.key]: field.same },
        ),
        true,
      );
    });

    test(`[${accelerator}] ${field.key} the resident load does not run is a reload`, () => {
      assert.equal(
        residentRuntimeMatchesConfig(
          { ...base, [field.statusKey]: field.different },
          { ...BLANK, [field.key]: field.same },
        ),
        false,
      );
    });

    test(`[${accelerator}] ${field.key} pinned against a status that omits it is a reload`, () => {
      // The direction that must never be read as agreement: a field the running server
      // cannot report is a field this function cannot verify.
      const status = { ...base } as Record<string, unknown>;
      delete status[field.statusKey];
      assert.equal(
        residentRuntimeMatchesConfig(status, {
          ...BLANK,
          [field.key]: field.same,
        }),
        // tensorParallel is the one field with no unset state: false is a real request for
        // a layer split, and a status omitting the flag ran without one, so they agree.
        field.key === "tensorParallel" ? field.same === false : false,
      );
    });
  }
}

test("placement compares as a set on a multi-GPU host, not as an order", () => {
  // The backend narrows and reorders the pool at fit time, so an order difference is not a
  // difference. A membership difference is.
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

test("automatic placement pins nothing, so it adopts any pool", () => {
  for (const ids of [null, undefined]) {
    assert.equal(
      residentRuntimeMatchesConfig(
        { ...ACCELERATORS["nvidia-cuda"], requested_gpu_ids: [0, 1, 2, 3] },
        { ...BLANK, selectedGpuIds: ids },
      ),
      true,
    );
  }
});

test("a CPU-only host distinguishes zero offloaded layers from automatic", () => {
  // gpu_layers 0 under manual is "keep it all on the CPU"; -1 is "let llama.cpp size it".
  // They are different loads, and 0 must not be read as an absent value.
  assert.equal(
    residentRuntimeMatchesConfig(ACCELERATORS["cpu-only"], {
      ...BLANK,
      gpuMemoryMode: "manual",
      gpuLayers: 0,
    }),
    true,
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
  // Blobs written before a field existed simply lack the key. The optional ones express no
  // opinion, which adopts; tensorParallel has no unset state, so an old blob missing it
  // reads as a demand the status cannot confirm and reloads. Neither may throw.
  // Typed as the parameter and cast at the boundary: these blobs come off localStorage,
  // where a version that predates a field simply has no key for it, so the point of the
  // test is precisely the shapes the current type says cannot occur.
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
    // tensorParallel absent, status false: `undefined === false` is false, so it reloads.
    // Conservative, and the direction that cannot lose a setting.
    false,
  );
});

test("an empty pinned pool is Automatic, not a demand for no GPUs", () => {
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["nvidia-cuda"], requested_gpu_ids: [0, 1] },
      { ...BLANK, selectedGpuIds: [] },
    ),
    // set() treats [] as pinned, and sameGpuSet([], [0,1]) is false, so this reloads.
    // Recorded rather than asserted as desirable: the panel writes null for Automatic.
    false,
  );
});

test("every PerModelConfig field is either compared or deliberately excluded", () => {
  // The guard that survives this PR. A new setting added to the config and not classified
  // here is a setting an adopted pick would drop silently.
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
    // A client-side generation cap. No status field echoes it and it never reaches
    // llama-server's invocation, so it cannot be a reason to reload.
    "maxSeqLength",
    // Qualifies selectedGpuIds rather than adding a dimension: the kind is decided by the
    // running llama.cpp build, not by the pick, so a resident server and a pick evaluated
    // against it always share it. /status publishes no field to check it against either.
    "selectedGpuIndexKind",
  ]);
  const unclassified = [...declared].filter(
    (field) => !compared.has(field) && !excluded.has(field),
  );
  assert.deepEqual(unclassified, []);
  // And the reverse, so a field removed from the config does not leave a dead check here.
  const stale = [...compared, ...excluded].filter(
    (field) => !declared.has(field),
  );
  assert.deepEqual(stale, []);
});
