// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A wrong TRUE silently runs a different server, so every field is tested both ways.
 * The structural test fails when a PerModelConfig field is added unclassified.
 */

import assert from "node:assert/strict";
import test from "node:test";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { residentRuntimeMatchesConfig: matchesWithStanding } = await import(
  "../src/features/chat/lib/resident-config-match.ts"
);

/** Four fields are not per-model; unset resolves to the shipped standing defaults. */
const STANDING = {
  speculativeType: null,
  gpuMemoryMode: "auto" as const,
  gpuLayers: -1,
  nCpuMoe: 0,
  reconcileGpuIds: (ids: number[] | null) => ids,
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

/** Measured against a running Unsloth: a default CUDA load reports auto / -1 / 0 / null / false. */
const ACCELERATORS: Record<string, Record<string, unknown>> = {
  "nvidia-cuda": {
    gpu_memory_mode: "auto",
    gpu_layers: -1,
    n_cpu_moe: 0,
    // An unset pick resolves to Automatic, so a non-null pool would make BLANK disagree.
    requested_gpu_ids: null,
    tensor_parallel: false,
  },
  "amd-rocm": {
    gpu_memory_mode: "auto",
    gpu_layers: -1,
    n_cpu_moe: 0,
    requested_gpu_ids: null,
    tensor_parallel: false,
  },
  // A CPU host echoes the requested invocation, so a default load still reports auto / -1.
  "cpu-only": {
    gpu_memory_mode: "auto",
    gpu_layers: -1,
    n_cpu_moe: 0,
    requested_gpu_ids: null,
    tensor_parallel: false,
  },
  "apple-mlx": {
    mlx_kv_quant_requested: null,
  },
};

type FieldCase = {
  key: string;
  statusKey: string;
  same: unknown;
  different: unknown;
  /** The backend compares offload knobs only under Manual, MoE count only with a layer pin. */
  live?: { config: Record<string, unknown>; status: Record<string, unknown> };
};

const MANUAL_MODE = {
  config: { gpuMemoryMode: "manual" },
  status: { gpu_memory_mode: "manual" },
};

const FIELDS: FieldCase[] = [
  {
    key: "engine",
    statusKey: "engine",
    same: "vllm",
    different: "sglang",
  },
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
    key: "mlxKvQuant",
    statusKey: "mlx_kv_quant_requested",
    same: "8",
    different: "tq-4",
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
  { key: "specDraftModel", statusKey: "spec_draft_model", same: "org/d", different: "org/e" },
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
    key: "loadMode",
    statusKey: "requested_load_mode",
    same: "mmap+mlock",
    different: "mmap",
  },
  {
    key: "specDraftCacheDtype",
    statusKey: "requested_spec_draft_cache_type",
    same: "q8_0",
    different: "f16",
  },
  {
    key: "ctxCheckpoints",
    statusKey: "requested_ctx_checkpoints",
    same: 8,
    different: 32,
  },
  {
    key: "cacheRam",
    statusKey: "requested_cache_ram",
    same: 4096,
    different: 8192,
  },
  {
    key: "reasoningBudget",
    statusKey: "reasoning_budget",
    same: 1024,
    different: 512,
  },
  {
    key: "reasoningBudgetMessage",
    statusKey: "reasoning_budget_message",
    same: "Wrap up.",
    different: "Stop here.",
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
  {
    key: "disableVision",
    statusKey: "disable_vision",
    same: true,
    different: false,
  },
  {
    key: "mlxInt8Prefill",
    statusKey: "mlx_int8_prefill_requested",
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
          {
            ...base,
            ...field.live?.status,
            [field.statusKey]: field.different,
          },
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
        // tensorParallel has no unset state: false agrees with a status omitting the flag.
        field.key === "tensorParallel" ? field.same === false : false,
      );
    });
  }
}

test("placement compares as an order on a multi-GPU host, not as a set", () => {
  // Picker order is the order the backend pins, so a reorder must restart the runner.
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["amd-rocm"], requested_gpu_ids: [0, 1] },
      { ...BLANK, selectedGpuIds: [1, 0] },
    ),
    false,
  );
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["amd-rocm"], requested_gpu_ids: [0, 1] },
      { ...BLANK, selectedGpuIds: [0, 1] },
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
  // Automatic is not "whatever is running": a server on chosen GPUs must reload.
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
  // Under manual, gpu_layers 0 means all-CPU and -1 means auto, so 0 is not absent.
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["cpu-only"], gpu_memory_mode: "manual", gpu_layers: 0 },
      { ...BLANK, gpuMemoryMode: "manual", gpuLayers: 0 },
    ),
    true,
  );
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

test("a config stored by an older Unsloth does not throw and does not over-adopt", () => {
  // Old localStorage blobs lack keys: optional ones adopt, missing tensorParallel reloads.
  const legacyBlobs = [
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
    { customContextLength: null, kvCacheDtype: null },
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
    false,
  );
});

test("an empty pinned pool is Automatic, not a demand for no GPUs", () => {
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...ACCELERATORS["nvidia-cuda"], requested_gpu_ids: [0, 1] },
      { ...BLANK, selectedGpuIds: [] },
    ),
    // set() treats [] as pinned; recorded, not endorsed, since the panel writes null.
    false,
  );
});

test("every PerModelConfig field is either compared or deliberately excluded", () => {
  // A new setting not classified here is one an adopted pick would drop silently.
  const source = readSrc(
    "features/model-picker/model-config/per-model-config.ts",
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
    "enginePrecision",
    "engineParallelism",
    // Read as the reconciler's namespace, but /status has no field to compare it against.
    "selectedGpuIndexKind",
    // Compared through standing.splitRatio, which the caller seeds from it.
    "tensorSplit",
  ]);
  const unclassified = [...declared].filter(
    (field) => !compared.has(field) && !excluded.has(field),
  );
  assert.deepEqual(unclassified, []);
  const stale = [...compared, ...excluded].filter(
    (field) => !declared.has(field),
  );
  assert.deepEqual(stale, []);
});

for (const engine of ["vllm", "sglang"] as const) {
  test(`${engine}: GPU and context changes require reloading the resident engine`, () => {
    const status = {
      engine,
      is_gguf: false,
      requested_gpu_ids: [1],
      gpu_ids: [1],
      requested_context_length: 4096,
    };
    const config = {
      ...BLANK,
      engine,
      selectedGpuIds: [1],
      selectedGpuIndexKind: "physical" as const,
      maxSeqLength: 4096,
    };
    assert.equal(residentRuntimeMatchesConfig(status, config), true);
    assert.equal(
      residentRuntimeMatchesConfig(status, { ...config, selectedGpuIds: [0] }),
      false,
    );
    assert.equal(
      residentRuntimeMatchesConfig(status, { ...config, maxSeqLength: 8192 }),
      false,
    );
    assert.equal(
      residentRuntimeMatchesConfig(status, { ...config, engine: "auto" }),
      false,
    );
  });
}

for (const engine of ["vllm", "sglang"] as const) {
  test(`${engine}: an unpicked GPU matches the backend's first visible GPU`, () => {
    // Studio restricted to GPUs 2 and 3: the backend loads an unpicked engine on 2, not 0.
    const status = { engine, is_gguf: false, requested_gpu_ids: [2], gpu_ids: [2] };
    const config = { ...BLANK, engine };
    const withDefault = (defaultEngineGpuIds: number[]) =>
      matchesWithStanding(status, config, { ...STANDING, defaultEngineGpuIds });
    assert.equal(withDefault([2]), true);
    assert.equal(withDefault([0]), false);
  });
}

for (const engine of ["vllm", "sglang"] as const) {
  test(`${engine}: changing a tensor-parallel GPU group requires a reload`, () => {
    const status = {
      engine,
      is_gguf: false,
      gpu_ids: [1, 0],
      requested_gpu_ids: [1, 0],
      tensor_parallel: true,
      requested_context_length: 4096,
    };
    const config = {
      ...BLANK,
      engine,
      selectedGpuIds: [1, 0],
      selectedGpuIndexKind: "physical" as const,
      maxSeqLength: 4096,
    };
    assert.equal(residentRuntimeMatchesConfig(status, config), true);
    for (const ids of [[1], [0], [0, 1]]) {
      assert.equal(
        residentRuntimeMatchesConfig(status, { ...config, selectedGpuIds: ids }),
        false,
      );
    }
  });
}

for (const engine of ["vllm", "sglang"] as const) {
  test(`${engine}: precision changes require a reload`, () => {
    const status = { engine, engine_precision: "int4" as const, gpu_ids: [0] };
    const config = { ...BLANK, engine, selectedGpuIds: [0], enginePrecision: "int4" as const };
    assert.equal(residentRuntimeMatchesConfig(status, config), true);
    assert.equal(residentRuntimeMatchesConfig(status, { ...config, enginePrecision: "int8" }), false);
  });
}

for (const engine of ["vllm", "sglang"] as const) {
  for (const mode of ["tensor", "pipeline", "data"] as const) {
    test(`${engine}: ${mode} mode is compared before reusing a resident model`, () => {
      const status = { engine, engine_parallelism: mode, gpu_ids: [0, 1] };
      const config = { ...BLANK, engine, engineParallelism: mode, selectedGpuIds: [0, 1] };
      assert.equal(residentRuntimeMatchesConfig(status, config), true);
      const other = mode === "tensor" ? "pipeline" : "tensor";
      assert.equal(residentRuntimeMatchesConfig(status, { ...config, engineParallelism: other }), false);
    });
  }
}
