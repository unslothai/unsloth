// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { countNonDefaultAdvancedSettings } = await import(
  "../src/features/studio/wizard/advanced-settings-summary.ts"
);

const LLAMA_TARGETS = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
];

async function waitForModelDefaults(model: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const state = useTrainingConfigStore.getState();
    if (
      !state.isLoadingModelDefaults &&
      state.modelDefaultsAppliedFor === model
    ) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("model defaults did not settle");
}

function deferLfmDefaults(): () => void {
  let resolveModelConfig!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveModelConfig = resolve;
      }),
  );
  return () =>
    resolveModelConfig(
      Response.json({
        id: "LiquidAI/LFM2-1.2B",
        config: { lora: { target_modules: ["all-linear"] } },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    );
}

after(() => setAuthFetchHandler(null));

test("leaving CPT after a model switch restores the new model targets", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({
    selectedModel: "old/llama",
    modelDefaultsAppliedFor: "old/llama",
    trainingMethod: "cpt",
    targetModules: [...LLAMA_TARGETS, "embed_tokens", "lm_head"],
    trainingMethodProvenance: {
      learningRateManuallySet: false,
      modelAdapterLearningRate: null,
      datasetFormatBeforeCpt: "chatml",
      targetModulesBeforeCpt: [...LLAMA_TARGETS],
      loraRankBeforeCpt: null,
      loraAlphaBeforeCpt: null,
      loraVariantBeforeCpt: null,
      trainOnCompletionsBeforeCpt: null,
    },
  });

  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "LiquidAI/LFM2-1.2B",
        config: { lora: { target_modules: ["all-linear"] } },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
  ]);
});

test("model targets apply when CPT is selected during the defaults request", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
});

test("an explicit target edit still wins during the defaults request", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore
    .getState()
    .setTargetModules(["q_proj", "embed_tokens", "lm_head"]);
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "q_proj",
    "embed_tokens",
    "lm_head",
  ]);
});

test("a target edit before entering CPT still wins", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTargetModules(["q_proj"]);
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    ...LLAMA_TARGETS,
    "embed_tokens",
    "lm_head",
  ]);
  assert.deepEqual(
    useTrainingConfigStore.getState().trainingMethodProvenance
      .targetModulesBeforeCpt,
    ["q_proj"],
  );
  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.deepEqual(useTrainingConfigStore.getState().targetModules, ["q_proj"]);
});

test("an unrelated edit does not block the model targets", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().setBatchSize(3);
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.equal(useTrainingConfigStore.getState().batchSize, 3);
  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
});

test("an unrelated edit does not block targets when CPT was already active", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setBatchSize(3);
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.equal(useTrainingConfigStore.getState().batchSize, 3);
  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
});

test("targets imported during the defaults request still win", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().applyConfigPatch({
    lora: { target_modules: ["q_proj"] },
  });
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, ["q_proj"]);
});

test("leaving CPT after a model switch restores the new model adapter params", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "old/llama",
        config: {
          lora: {
            lora_r: 8,
            lora_alpha: 8,
            target_modules: [...LLAMA_TARGETS],
          },
        },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");
  assert.equal(useTrainingConfigStore.getState().loraRank, 8);

  useTrainingConfigStore.getState().setTrainingMethod("cpt");

  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "LiquidAI/LFM2-1.2B",
        config: {
          lora: {
            lora_r: 64,
            lora_alpha: 128,
            use_dora: true,
            target_modules: ["all-linear"],
          },
        },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");
  assert.equal(useTrainingConfigStore.getState().loraRank, 128);

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 64);
  assert.equal(state.loraAlpha, 128);
  assert.equal(state.loraVariant, "dora");
  assert.deepEqual(state.targetModules, ["all-linear"]);
});

test("an unrelated edit does not strand the previous model adapter params", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "old/llama",
        config: {
          lora: {
            lora_r: 8,
            lora_alpha: 8,
            target_modules: [...LLAMA_TARGETS],
          },
        },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");

  let resolveModelConfig!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveModelConfig = resolve;
      }),
  );
  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setBatchSize(3);
  resolveModelConfig(
    Response.json({
      id: "LiquidAI/LFM2-1.2B",
      config: {
        lora: {
          lora_r: 64,
          lora_alpha: 128,
          use_dora: true,
          target_modules: ["all-linear"],
        },
      },
      is_vision: false,
      is_embedding: false,
      is_audio: false,
      audio_type_known: true,
      is_lora: false,
      model_type: "text",
      model_size_bytes: null,
      max_position_embeddings: 32768,
    }),
  );
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 64);
  assert.equal(state.loraAlpha, 128);
  assert.equal(state.loraVariant, "dora");
  assert.deepEqual(state.targetModules, ["all-linear"]);
});

test("a LoRA edit before entering CPT still wins", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setLoraRank(64);
  useTrainingConfigStore.getState().setLoraAlpha(64);
  useTrainingConfigStore.getState().setLoraVariant("dora");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.equal(useTrainingConfigStore.getState().loraRank, 128);
  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 64);
  assert.equal(state.loraAlpha, 64);
  assert.equal(state.loraVariant, "dora");
});

test("LoRA params imported before entering CPT still win", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore
    .getState()
    .applyConfigPatch({ lora: { lora_r: 64, lora_alpha: 64 } });
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 64);
  assert.equal(state.loraAlpha, 64);
});

test("editing one LoRA field does not freeze the other two on a model switch", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "old/llama",
        config: {
          lora: {
            lora_r: 8,
            lora_alpha: 8,
            target_modules: [...LLAMA_TARGETS],
          },
        },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");

  let resolveModelConfig!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveModelConfig = resolve;
      }),
  );
  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setLoraRank(32);
  resolveModelConfig(
    Response.json({
      id: "LiquidAI/LFM2-1.2B",
      config: {
        lora: {
          lora_r: 64,
          lora_alpha: 128,
          use_dora: true,
          target_modules: ["all-linear"],
        },
      },
      is_vision: false,
      is_embedding: false,
      is_audio: false,
      audio_type_known: true,
      is_lora: false,
      model_type: "text",
      model_size_bytes: null,
      max_position_embeddings: 32768,
    }),
  );
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 8);
  assert.equal(state.loraAlpha, 128);
  assert.equal(state.loraVariant, "dora");
});

test("a target-modules edit does not strand the previous model adapter params", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "old/llama",
        config: {
          lora: {
            lora_r: 8,
            lora_alpha: 8,
            target_modules: [...LLAMA_TARGETS],
          },
        },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");

  let resolveModelConfig!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveModelConfig = resolve;
      }),
  );
  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore
    .getState()
    .setTargetModules([...LLAMA_TARGETS, "embed_tokens"]);
  resolveModelConfig(
    Response.json({
      id: "LiquidAI/LFM2-1.2B",
      config: {
        lora: {
          lora_r: 64,
          lora_alpha: 128,
          use_dora: true,
          target_modules: ["all-linear"],
        },
      },
      is_vision: false,
      is_embedding: false,
      is_audio: false,
      audio_type_known: true,
      is_lora: false,
      model_type: "text",
      model_size_bytes: null,
      max_position_embeddings: 32768,
    }),
  );
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 64);
  assert.equal(state.loraAlpha, 128);
  assert.equal(state.loraVariant, "dora");
});

test("leaving CPT does not report untouched adapter params as modified", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "old/llama",
        config: {
          lora: {
            lora_r: 8,
            lora_alpha: 8,
            target_modules: [...LLAMA_TARGETS],
          },
        },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");

  const inCpt = useTrainingConfigStore.getState();
  assert.equal(
    countNonDefaultAdvancedSettings(inCpt, inCpt.advancedSettingsBaseline),
    0,
  );

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 8);
  assert.equal(state.loraAlpha, 8);
  assert.equal(
    countNonDefaultAdvancedSettings(state, state.advancedSettingsBaseline),
    0,
  );
});

// load_model_defaults returns {} on a YAML failure, so cptTargetModules falls back to live
// state; the baseline must follow or the summary reports a phantom edit.
test("an empty model config does not invent a modified target-modules setting", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore
    .getState()
    .setTargetModules(["all-linear", "embed_tokens", "lm_head"]);
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "org/no-defaults",
        config: {},
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
  useTrainingConfigStore.getState().selectTrainingModel("org/no-defaults", "text");
  await waitForModelDefaults("org/no-defaults");

  const inCpt = useTrainingConfigStore.getState();
  assert.deepEqual(inCpt.targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
  assert.equal(
    countNonDefaultAdvancedSettings(inCpt, inCpt.advancedSettingsBaseline),
    0,
  );

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const out = useTrainingConfigStore.getState();
  assert.equal(
    countNonDefaultAdvancedSettings(out, out.advancedSettingsBaseline),
    0,
  );
});

// Cache reconciliation restarts the request with applyTrainingDefaults: false.
async function cacheRestartInsideCpt(beforeRestart: () => void): Promise<void> {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "old/llama",
        config: {
          lora: {
            lora_r: 8,
            lora_alpha: 8,
            target_modules: [...LLAMA_TARGETS],
          },
        },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");

  let resolveModelConfig!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveModelConfig = resolve;
      }),
  );
  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  beforeRestart();

  const lfmDefaults = () =>
    Response.json({
      id: "LiquidAI/LFM2-1.2B",
      config: {
        lora: {
          lora_r: 64,
          lora_alpha: 128,
          use_dora: true,
          target_modules: ["all-linear"],
        },
      },
      is_vision: false,
      is_embedding: false,
      is_audio: false,
      audio_type_known: true,
      is_lora: false,
      model_type: "text",
      model_size_bytes: null,
      max_position_embeddings: 32768,
    });
  setAuthFetchHandler(() => Promise.resolve(lfmDefaults()));
  useTrainingConfigStore
    .getState()
    .setSelectedModelCacheReference("LiquidAI/LFM2-1.2B", {
      localPath: "/models/lfm2",
      modelFormat: "safetensors",
    });
  resolveModelConfig(lfmDefaults());
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");
}

test("a cache-reference restart still refreshes the pre-CPT LoRA params", async () => {
  await cacheRestartInsideCpt(() => {
    useTrainingConfigStore.getState().setBatchSize(3);
  });

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 64);
  assert.equal(state.loraAlpha, 128);
  assert.equal(state.loraVariant, "dora");
});

test("a cache-reference restart does not forget an edit made before it", async () => {
  await cacheRestartInsideCpt(() => {
    useTrainingConfigStore.getState().setLoraRank(40);
  });

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 8);
  assert.equal(state.loraAlpha, 128);
  assert.equal(state.loraVariant, "dora");
});

test("a reload inside CPT keeps the pre-CPT LoRA params the session saved", async () => {
  useTrainingConfigStore.getState().reset();
  let calls = 0;
  setAuthFetchHandler(() => {
    calls += 1;
    return Promise.resolve(
      Response.json({
        id: "org/reload-model",
        config: {
          lora: {
            lora_r: 8,
            lora_alpha: 8,
            target_modules: [...LLAMA_TARGETS],
          },
        },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    );
  });

  useTrainingConfigStore.setState({
    selectedModel: "org/reload-model",
    modelDefaultsAppliedFor: "org/reload-model",
    trainingMethod: "cpt",
    loraRank: 128,
    loraAlpha: 32,
    loraVariant: "rslora",
    trainingMethodProvenance: {
      learningRateManuallySet: false,
      modelAdapterLearningRate: null,
      datasetFormatBeforeCpt: null,
      targetModulesBeforeCpt: null,
      loraRankBeforeCpt: 64,
      loraAlphaBeforeCpt: 64,
      loraVariantBeforeCpt: "dora",
      trainOnCompletionsBeforeCpt: null,
    },
  });

  useTrainingConfigStore.getState().ensureModelDefaultsLoaded();
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (
      calls > 0 &&
      !useTrainingConfigStore.getState().isLoadingModelDefaults
    ) {
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.ok(calls > 0, "the mount-time defaults request never ran");

  const provenance = useTrainingConfigStore.getState().trainingMethodProvenance;
  assert.equal(provenance.loraRankBeforeCpt, 64);
  assert.equal(provenance.loraAlphaBeforeCpt, 64);
  assert.equal(provenance.loraVariantBeforeCpt, "dora");

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  const state = useTrainingConfigStore.getState();
  assert.equal(state.loraRank, 64);
  assert.equal(state.loraAlpha, 64);
  assert.equal(state.loraVariant, "dora");
});

test("leaving CPT after a model switch does not restore the old model's train on completions", async () => {
  useTrainingConfigStore.getState().reset();
  const defaultsWith = (id: string, trainOnCompletions: boolean) => () =>
    Promise.resolve(
      Response.json({
        id,
        config: { training: { train_on_completions: trainOnCompletions } },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    );
  setAuthFetchHandler(defaultsWith("old/llama", true));
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, true);

  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  setAuthFetchHandler(defaultsWith("new/model", false));
  useTrainingConfigStore.getState().selectTrainingModel("new/model", "text");
  await waitForModelDefaults("new/model");

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});

test("leaving CPT after switching to a model whose default is on turns train on completions back on", async () => {
  const defaultsWith = (id: string, trainOnCompletions: boolean) => () =>
    Promise.resolve(
      Response.json({
        id,
        config: { training: { train_on_completions: trainOnCompletions } },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    );
  for (const oldDefault of [false, true]) {
    useTrainingConfigStore.getState().reset();
    setAuthFetchHandler(defaultsWith("old/llama", oldDefault));
    useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
    await waitForModelDefaults("old/llama");

    useTrainingConfigStore.getState().setTrainingMethod("cpt");
    setAuthFetchHandler(defaultsWith("new/model", true));
    useTrainingConfigStore.getState().selectTrainingModel("new/model", "text");
    await waitForModelDefaults("new/model");

    useTrainingConfigStore.getState().setTrainingMethod("qlora");
    assert.equal(useTrainingConfigStore.getState().trainOnCompletions, true);
  }
});

test("leaving CPT while the new model's defaults are loading does not restore the old model's train on completions", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() =>
    Response.json({
      id: "old/llama",
      config: { training: { train_on_completions: true } },
      is_vision: false,
      is_embedding: false,
      is_audio: false,
      audio_type_known: true,
      is_lora: false,
      model_type: "text",
      model_size_bytes: null,
      max_position_embeddings: 32768,
    }),
  );
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");

  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  setAuthFetchHandler(() => new Promise<Response>(() => {}));
  useTrainingConfigStore.getState().selectTrainingModel("new/audio", "audio");
  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});

test("an image dataset picked inside CPT keeps train on completions off for a vision model", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler((input, init) => {
    if (input === "/api/hub/datasets/check-format") {
      const { dataset_name } = JSON.parse(String(init?.body));
      return Response.json({
        columns: ["messages"],
        detected_format: "chatml",
        is_audio: false,
        is_image: dataset_name === "org/images",
        requires_manual_mapping: false,
      });
    }
    return Response.json({
      id: "org/vision",
      config: { training: { train_on_completions: true } },
      is_vision: true,
      is_embedding: false,
      is_audio: false,
      audio_type_known: true,
      is_lora: false,
      model_type: "vision",
      model_size_bytes: null,
      max_position_embeddings: 32768,
    });
  });
  const datasetChecked = async (): Promise<void> => {
    for (let attempt = 0; attempt < 100; attempt += 1) {
      if (!useTrainingConfigStore.getState().isCheckingDataset) return;
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    throw new Error("dataset check did not settle");
  };
  useTrainingConfigStore.getState().selectTrainingModel("org/vision", "vision");
  await waitForModelDefaults("org/vision");
  useTrainingConfigStore.getState().setDataset("org/text");
  await datasetChecked();
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, true);

  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().setDataset("org/images");
  await datasetChecked();
  assert.equal(useTrainingConfigStore.getState().isDatasetImage, true);

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});

const completionDefaults = (id: string, isAudio: boolean) =>
  Response.json({
    id,
    config: { training: { train_on_completions: true } },
    is_vision: false,
    is_embedding: false,
    is_audio: isAudio,
    audio_type_known: true,
    is_lora: false,
    model_type: isAudio ? "audio" : "text",
    model_size_bytes: null,
    max_position_embeddings: 32768,
  });

function deferModelDefaults(): (response: Response) => void {
  let resolveDefaults!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveDefaults = resolve;
      }),
  );
  return (response) => resolveDefaults(response);
}

test("entering CPT before the new model's defaults load restores that model's train on completions", async () => {
  for (const isAudio of [true, false]) {
    useTrainingConfigStore.getState().reset();
    setAuthFetchHandler(() => completionDefaults("old/llama", false));
    useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
    await waitForModelDefaults("old/llama");
    assert.equal(useTrainingConfigStore.getState().trainOnCompletions, true);

    const resolveDefaults = deferModelDefaults();
    useTrainingConfigStore
      .getState()
      .selectTrainingModel("new/model", isAudio ? "audio" : "text");
    useTrainingConfigStore.getState().setTrainingMethod("cpt");
    resolveDefaults(completionDefaults("new/model", isAudio));
    await waitForModelDefaults("new/model");

    useTrainingConfigStore.getState().setTrainingMethod("qlora");
    assert.equal(
      useTrainingConfigStore.getState().trainOnCompletions,
      !isAudio,
    );
  }
});

test("an edit while the new model's defaults load inside CPT still restores its train on completions", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  const resolveDefaults = deferModelDefaults();
  useTrainingConfigStore.getState().selectTrainingModel("new/model", "text");
  useTrainingConfigStore.getState().setBatchSize(8);
  resolveDefaults(completionDefaults("new/model", false));
  await waitForModelDefaults("new/model");

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, true);
});

test("restored train on completions is not counted as a modified setting", async () => {
  const defaults = () =>
    Response.json({
      id: "org/chat",
      config: {
        lora: { target_modules: [...LLAMA_TARGETS] },
        training: { train_on_completions: true },
      },
      is_vision: false,
      is_embedding: false,
      is_audio: false,
      audio_type_known: true,
      is_lora: false,
      model_type: "text",
      model_size_bytes: null,
      max_position_embeddings: 32768,
    });
  const count = () => {
    const state = useTrainingConfigStore.getState();
    return countNonDefaultAdvancedSettings(
      state,
      state.advancedSettingsBaseline,
    );
  };
  for (const chooseInsideCpt of [false, true]) {
    useTrainingConfigStore.getState().reset();
    setAuthFetchHandler(defaults);
    if (chooseInsideCpt) {
      useTrainingConfigStore.getState().setTrainingMethod("cpt");
    }
    useTrainingConfigStore.getState().selectTrainingModel("org/chat", "text");
    await waitForModelDefaults("org/chat");
    useTrainingConfigStore.getState().setTrainingMethod("cpt");
    assert.equal(count(), 0);

    useTrainingConfigStore.getState().setTrainingMethod("qlora");
    assert.equal(useTrainingConfigStore.getState().trainOnCompletions, true);
    assert.equal(count(), 0);
  }
});

test("a cache refetch inside CPT keeps the train on completions the user turned off", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler((input) => {
    if (input === "/api/hub/datasets/check-format") {
      return Response.json({
        columns: ["messages"],
        detected_format: "chatml",
        is_audio: false,
        is_image: false,
        requires_manual_mapping: false,
      });
    }
    return completionDefaults("org/chat", false);
  });
  useTrainingConfigStore.getState().selectTrainingModel("org/chat", "text");
  await waitForModelDefaults("org/chat");
  useTrainingConfigStore.getState().setTrainOnCompletions(false);
  useTrainingConfigStore.getState().setDataset("org/text");
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (!useTrainingConfigStore.getState().isCheckingDataset) break;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }

  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().setSelectedModelCacheReference("org/chat", {
    localPath: "/cache/org/chat",
    modelFormat: null,
  });
  await waitForModelDefaults("org/chat");

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});

test("a config imported inside CPT decides train on completions after leaving", async () => {
  for (const imported of [false, true]) {
    useTrainingConfigStore.getState().reset();
    setAuthFetchHandler(() => completionDefaults("org/chat", false));
    useTrainingConfigStore.getState().selectTrainingModel("org/chat", "text");
    await waitForModelDefaults("org/chat");
    useTrainingConfigStore.getState().setTrainOnCompletions(!imported);

    useTrainingConfigStore.getState().setTrainingMethod("cpt");
    useTrainingConfigStore
      .getState()
      .applyConfigPatch({ training: { train_on_completions: imported } });
    useTrainingConfigStore.getState().setTrainingMethod("qlora");
    assert.equal(
      useTrainingConfigStore.getState().trainOnCompletions,
      imported,
    );
  }
});

const completionModel = (id: string, isVision: boolean) =>
  Response.json({
    id,
    config: { training: { train_on_completions: true } },
    is_vision: isVision,
    is_embedding: false,
    is_audio: false,
    audio_type_known: true,
    is_lora: false,
    model_type: isVision ? "vision" : "text",
    model_size_bytes: null,
    max_position_embeddings: 32768,
  });

const datasetCheck = (isImage: boolean) =>
  Response.json({
    columns: ["messages"],
    detected_format: "chatml",
    is_audio: false,
    is_image: isImage,
    requires_manual_mapping: false,
  });

async function settleStore(): Promise<void> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const state = useTrainingConfigStore.getState();
    if (
      !state.isLoadingModelDefaults &&
      !state.isCheckingDataset &&
      state.modelDefaultsAppliedFor === state.selectedModel
    ) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("store did not settle");
}

test("a model found to be vision inside CPT keeps completions off for an image dataset", async () => {
  useTrainingConfigStore.getState().reset();
  let isVision = false;
  setAuthFetchHandler((input) =>
    input === "/api/hub/datasets/check-format"
      ? datasetCheck(true)
      : completionModel("org/model", isVision),
  );
  useTrainingConfigStore.getState().selectTrainingModel("org/model", "text");
  await settleStore();
  useTrainingConfigStore.getState().setDataset("org/images");
  await settleStore();
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, true);

  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  isVision = true;
  useTrainingConfigStore
    .getState()
    .setSelectedModelCacheReference("org/model", {
      localPath: "/cache/org/model",
      modelFormat: null,
    });
  await settleStore();
  assert.equal(useTrainingConfigStore.getState().isVisionModel, true);

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});

test("an opt-out made while defaults load survives a CPT round trip", async () => {
  useTrainingConfigStore.getState().reset();
  let resolveDefaults!: (response: Response) => void;
  setAuthFetchHandler((input) =>
    input === "/api/hub/datasets/check-format"
      ? datasetCheck(false)
      : new Promise<Response>((resolve) => {
          resolveDefaults = resolve;
        }),
  );
  useTrainingConfigStore.getState().selectTrainingModel("org/model", "text");
  useTrainingConfigStore.getState().setTrainOnCompletions(false);
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().setDataset("org/text");
  await new Promise((resolve) => setTimeout(resolve, 5));
  resolveDefaults(completionModel("org/model", false));
  await settleStore();

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});

test("an image split picked inside CPT keeps completions off after manual toggles", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler((input, init) =>
    input === "/api/hub/datasets/check-format"
      ? datasetCheck(JSON.parse(String(init?.body)).train_split === "images")
      : completionModel("org/vision", true),
  );
  useTrainingConfigStore.getState().selectTrainingModel("org/vision", "vision");
  await settleStore();
  useTrainingConfigStore.getState().setDataset("org/mixed");
  await settleStore();
  useTrainingConfigStore.getState().setTrainOnCompletions(false);
  useTrainingConfigStore.getState().setTrainOnCompletions(true);

  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().setDatasetSplit("images");
  await settleStore();
  assert.equal(useTrainingConfigStore.getState().isDatasetImage, true);

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});

test("a manual toggle on the previous model does not decide the next model's CPT restore", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() => completionModel("old/model", false));
  useTrainingConfigStore.getState().selectTrainingModel("old/model", "text");
  await settleStore();
  useTrainingConfigStore.getState().setTrainOnCompletions(true);

  let resolveDefaults!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveDefaults = resolve;
      }),
  );
  useTrainingConfigStore.getState().selectTrainingModel("new/model", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().setBatchSize(8);
  resolveDefaults(
    Response.json({
      id: "new/model",
      config: { training: { train_on_completions: false } },
      is_vision: false,
      is_embedding: false,
      is_audio: false,
      audio_type_known: true,
      is_lora: false,
      model_type: "text",
      model_size_bytes: null,
      max_position_embeddings: 32768,
    }),
  );
  await settleStore();

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});

test("a config imported while defaults load survives entering and leaving CPT", async () => {
  useTrainingConfigStore.getState().reset();
  let resolveDefaults!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveDefaults = resolve;
      }),
  );
  useTrainingConfigStore.getState().selectTrainingModel("org/model", "text");
  useTrainingConfigStore
    .getState()
    .applyConfigPatch({ training: { train_on_completions: false } });
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  resolveDefaults(completionModel("org/model", false));
  await settleStore();

  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.equal(useTrainingConfigStore.getState().trainOnCompletions, false);
});
