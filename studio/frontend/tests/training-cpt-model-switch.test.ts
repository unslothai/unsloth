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
  // The rank edit is protected; the two untouched fields follow the new model.
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
  // Chosen inside CPT, so the baseline freezes there: the regressing case.
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().selectTrainingModel("old/llama", "text");
  await waitForModelDefaults("old/llama");

  // Inside CPT the summary reads CPT's own values, so nothing is modified there.
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
  // The edit predates the restart, so its guard has to survive the new request.
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

  // The shape rehydration leaves behind: CPT active, the pre-CPT values the user
  // configured persisted, defaults already recorded as applied for this model.
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
