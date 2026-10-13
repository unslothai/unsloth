// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test, { after } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const yaml = await import("js-yaml");
const { DEFAULT_HYPERPARAMS, LR_DEFAULT_DECISION_FULL, LR_DEFAULT_LORA } =
  await import("../src/config/training.ts");
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { DatasetFormatError } = await import(
  "../src/features/training/api/datasets-api.ts"
);
const { validateTrainingConfig } = await import(
  "../src/features/training/lib/validation.ts"
);
const { deriveTrainingReadiness } = await import(
  "../src/features/training/hooks/use-training-readiness.ts"
);
const { checkDecisionDatasetColumns, missingDecisionColumns } = await import(
  "../src/features/training/lib/decision-dataset.ts"
);
const { trainingModelTypeFlagsFromMetadata } = await import(
  "../src/features/training/lib/model-type-inference.ts"
);
const { inferTrainingModelTypeFromFlags } = await import(
  "../src/features/training/lib/model-type-capabilities.ts"
);
const { isTrainingModelTypeSupportedOnDevice } = await import(
  "../src/features/training/lib/training-methods.ts"
);
const { countNonDefaultAdvancedSettings } = await import(
  "../src/features/studio/wizard/advanced-settings-summary.ts"
);

type TrainingState = ReturnType<typeof useTrainingConfigStore.getState>;

const LAYA = "convaiinnovations/laya";

const LAYA_YAML = yaml.load(
  readFileSync(
    new URL(
      "../../backend/assets/configs/model_defaults/decision/convaiinnovations_laya.yaml",
      import.meta.url,
    ),
    "utf8",
  ),
) as {
  training: Record<string, number>;
  lora: Record<string, number>;
};

const LAYA_CONFIG = {
  id: LAYA,
  config: LAYA_YAML,
  is_vision: false,
  is_embedding: false,
  is_decision: true,
  decision_checkpoints: [
    { name: "laya-multilingual", subfolder: "multilingual", description: "" },
    { name: "laya-english", subfolder: null, description: "" },
    {
      name: "laya-typed-decisions",
      subfolder: "typed-decisions",
      description: "",
    },
  ],
  is_audio: false,
  audio_type_known: true,
  is_lora: false,
  model_type: "decision",
  model_size_bytes: 600_000_000,
  max_position_embeddings: null,
};

const LLM = "unsloth/Qwen3-0.6B";

const LLM_CONFIG = {
  id: LLM,
  config: { training: { max_steps: 60 } },
  is_vision: false,
  is_embedding: false,
  is_audio: false,
  audio_type_known: true,
  is_lora: false,
  model_type: "text",
  model_size_bytes: 1_200_000_000,
  max_position_embeddings: 32768,
};

async function waitFor(done: (state: TrainingState) => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (done(useTrainingConfigStore.getState())) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("the training config did not settle");
}

const waitForModelDefaults = (model: string) =>
  waitFor(
    (state) =>
      !state.isLoadingModelDefaults && state.modelDefaultsAppliedFor === model,
  );

function serveConfigs(laya: object = LAYA_CONFIG): string[] {
  const requested: string[] = [];
  setAuthFetchHandler((input) => {
    requested.push(input);
    if (input.startsWith("/api/models/config/")) {
      return Response.json(input.includes("laya") ? laya : LLM_CONFIG);
    }
    return Response.json({});
  });
  return requested;
}

function serveConfigsLater(): () => void {
  let release: (() => void) | undefined;
  const loaded = new Promise<void>((resolve) => {
    release = resolve;
  });
  setAuthFetchHandler(async (input) => {
    await loaded;
    return Response.json(input.includes("laya") ? LAYA_CONFIG : LLM_CONFIG);
  });
  return () => release?.();
}

async function selectModel(
  model: string,
  laya: object = LAYA_CONFIG,
): Promise<string[]> {
  const requested = serveConfigs(laya);
  useTrainingConfigStore.getState().selectTrainingModel(model, "text");
  await waitForModelDefaults(model);
  return requested;
}

const selectLaya = () => selectModel(LAYA);

const HF_DECISION_DATASET = {
  datasetSource: "huggingface" as const,
  dataset: "LocalLLaMA/typed-decisions",
  datasetKnownCached: false,
  manualDatasetOptionsValid: true,
};

after(() => setAuthFetchHandler(null));

test("picking a decision model starts a LoRA run with the model's defaults", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({
    trainingMethod: "qlora",
    packing: true,
    datasetStreaming: true,
  });

  const requested = await selectLaya();

  assert.deepEqual(
    requested.filter((url) => !url.startsWith("/api/models/config/")),
    [],
    "a decision model must not ask the hardware for a LoRA/QLoRA pick",
  );
  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelType, "decision");
  assert.equal(state.trainingMethod, "lora");
  assert.equal(state.learningRate, LAYA_YAML.training.learning_rate);
  assert.equal(state.epochs, LAYA_YAML.training.num_epochs);
  assert.equal(state.maxSteps, LAYA_YAML.training.max_steps);
  assert.equal(state.batchSize, LAYA_YAML.training.batch_size);
  assert.equal(
    state.gradientAccumulation,
    LAYA_YAML.training.gradient_accumulation_steps,
  );
  assert.equal(state.loraRank, LAYA_YAML.lora.lora_r);
  assert.equal(state.loraAlpha, LAYA_YAML.lora.lora_alpha);
  assert.equal(state.datasetStreaming, false);
  assert.equal(state.decisionCheckpoints?.length, 3);
  assert.equal(state.modelSubfolder, "multilingual");
});

test("leaving CPT for a decision model does not keep CPT's learning rate", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.getState().setTrainingMethod("cpt");

  await selectLaya();

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainingMethod, "lora");
  assert.equal(state.learningRate, LAYA_YAML.training.learning_rate);
});

test("a fresh decision pick defaults to LoRA even from full fine-tuning", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.getState().setTrainingMethod("full");

  await selectLaya();

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainingMethod, "lora");
  assert.equal(state.learningRate, LAYA_YAML.training.learning_rate);
});

test("LoRA takes the model's learning rate and full fine-tuning the decision one", async () => {
  useTrainingConfigStore.getState().reset();
  await selectModel(LAYA, {
    ...LAYA_CONFIG,
    config: {
      ...LAYA_YAML,
      training: { ...LAYA_YAML.training, learning_rate: 5e-4 },
    },
  });
  assert.equal(useTrainingConfigStore.getState().learningRate, 5e-4);

  useTrainingConfigStore.getState().setTrainingMethod("full");
  let payload = buildTrainingStartPayload(
    useTrainingConfigStore.getState(),
    null,
  );
  assert.equal(payload.training_type, "Full Finetuning");
  assert.equal(payload.use_lora, false);
  assert.equal(payload.learning_rate, String(LR_DEFAULT_DECISION_FULL));

  useTrainingConfigStore.getState().setTrainingMethod("lora");
  payload = buildTrainingStartPayload(useTrainingConfigStore.getState(), null);
  assert.equal(payload.training_type, "LoRA/QLoRA");
  assert.equal(payload.use_lora, true);
  assert.equal(payload.learning_rate, "0.0005");
});

test("a checkpoint pick survives defaults reloads and resets on a model change", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLaya();
  useTrainingConfigStore.getState().setModelSubfolder(null);

  useTrainingConfigStore.getState().ensureModelDefaultsLoaded();
  await waitForModelDefaults(LAYA);
  assert.equal(useTrainingConfigStore.getState().modelSubfolder, null);

  useTrainingConfigStore.getState().setSelectedModelCacheReference(LAYA, {
    localPath: "/cache/convaiinnovations/laya",
    modelFormat: null,
  });
  await waitForModelDefaults(LAYA);
  assert.equal(useTrainingConfigStore.getState().modelSubfolder, null);

  await selectModel(LLM);
  await selectLaya();
  assert.equal(
    useTrainingConfigStore.getState().modelSubfolder,
    "multilingual",
  );
});

test("Reset to defaults keeps the chosen checkpoint", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLaya();

  for (const subfolder of ["typed-decisions", null]) {
    useTrainingConfigStore.getState().setModelSubfolder(subfolder);
    useTrainingConfigStore.getState().setBatchSize(4);
    useTrainingConfigStore.getState().resetToModelDefaults();
    await waitForModelDefaults(LAYA);

    const state = useTrainingConfigStore.getState();
    assert.equal(state.modelSubfolder, subfolder);
    assert.equal(state.batchSize, LAYA_YAML.training.batch_size);
  }
});

test("a method picked while a decision model loads still gets the model's defaults", async () => {
  useTrainingConfigStore.getState().reset();
  await selectModel(LLM);
  const release = serveConfigsLater();

  useTrainingConfigStore.getState().selectTrainingModel(LAYA, "text");
  useTrainingConfigStore.getState().setTrainingMethod("full");
  release();
  await waitForModelDefaults(LAYA);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainingMethod, "full");
  assert.equal(state.learningRate, LR_DEFAULT_DECISION_FULL);
  assert.equal(state.epochs, LAYA_YAML.training.num_epochs);
  assert.equal(state.maxSteps, LAYA_YAML.training.max_steps);
  assert.equal(state.batchSize, LAYA_YAML.training.batch_size);
  assert.equal(state.modelSubfolder, "multilingual");
});

test("leaving a decision model restores the method and streaming it overrode", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({
    datasetSource: "huggingface",
    dataset: "roneneldan/TinyStories",
    maxSteps: 60,
    datasetStreaming: true,
  });
  useTrainingConfigStore.getState().setTrainingMethod("cpt");

  await selectLaya();
  assert.equal(useTrainingConfigStore.getState().trainingMethod, "lora");
  assert.equal(useTrainingConfigStore.getState().datasetStreaming, false);
  useTrainingConfigStore.getState().setTrainingMethod("full");

  await selectModel(LLM);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelType, "text");
  assert.equal(state.trainingMethod, "cpt");
  assert.equal(state.learningRate, 5e-5);
  assert.equal(state.datasetStreaming, true);
  assert.equal(state.settingsBeforeDecision, null);
});

test("an edit made while leaving a decision model does not keep its recipe", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLaya();
  const release = serveConfigsLater();

  useTrainingConfigStore.getState().selectTrainingModel(LLM, "text");
  useTrainingConfigStore.getState().setBatchSize(2);
  release();
  await waitForModelDefaults(LLM);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainingMethod, "qlora");
  assert.equal(state.maxSteps, LLM_CONFIG.config.training.max_steps);
  assert.equal(state.batchSize, 2);
  assert.equal(state.settingsBeforeDecision, null);
});

test("leaving a decision model for one whose config fails drops the decision recipe", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLaya();
  useTrainingConfigStore.getState().setModelSubfolder("typed-decisions");
  setAuthFetchHandler(() => new Response("upstream error", { status: 502 }));

  useTrainingConfigStore.getState().selectTrainingModel(LLM, "text");
  await waitFor(
    (state) =>
      state.modelDefaultsError !== null &&
      !state.isLoadingModelDefaults &&
      !state.isCheckingVision,
  );

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainingMethod, "qlora");
  assert.equal(state.learningRate, LR_DEFAULT_LORA);
  assert.equal(state.epochs, DEFAULT_HYPERPARAMS.epochs);
  assert.equal(state.maxSteps, DEFAULT_HYPERPARAMS.maxSteps);
  assert.equal(state.batchSize, DEFAULT_HYPERPARAMS.batchSize);
  assert.equal(state.loraRank, DEFAULT_HYPERPARAMS.loraRank);
  assert.equal(state.optimizerType, DEFAULT_HYPERPARAMS.optimizerType);
  assert.equal(state.modelSubfolder, null);
  assert.equal(state.decisionCheckpoints, null);
  assert.equal(state.settingsBeforeDecision, null);
  const payload = buildTrainingStartPayload(state, null);
  assert.equal(payload.is_decision, false);
  assert.equal(payload.load_in_4bit, true);
});

test("a decision model whose config fails cannot start until it loads", async () => {
  useTrainingConfigStore.getState().reset();
  setAuthFetchHandler(() => new Response("upstream error", { status: 502 }));

  useTrainingConfigStore.getState().selectTrainingModel(LAYA, "decision");
  await waitFor(
    (state) =>
      state.modelDefaultsError !== null &&
      !state.isLoadingModelDefaults &&
      !state.isCheckingVision,
  );
  useTrainingConfigStore.setState(HF_DECISION_DATASET);

  const failed = useTrainingConfigStore.getState();
  assert.equal(failed.modelType, "decision");
  assert.equal(failed.modelSubfolder, null);
  const blocked = deriveTrainingReadiness(failed, "cuda", true);
  assert.equal(blocked.isReady, false);
  assert.equal(blocked.configValidation.ok, true);
  assert.notEqual(blocked.modelError, null);

  serveConfigs();
  useTrainingConfigStore.getState().ensureModelDefaultsLoaded();
  await waitForModelDefaults(LAYA);
  const loaded = useTrainingConfigStore.getState();
  assert.equal(loaded.modelDefaultsError, null);
  assert.equal(loaded.modelSubfolder, "multilingual");
  assert.equal(loaded.learningRate, LAYA_YAML.training.learning_rate);
  assert.equal(deriveTrainingReadiness(loaded, "cuda", true).isReady, true);
});

test("a decision model last loaded as a chat model gets its defaults on reload", async () => {
  useTrainingConfigStore.getState().reset();
  // What a build without decision models persisted for Laya: a text model on default.yaml.
  useTrainingConfigStore.setState({
    selectedModel: LAYA,
    modelType: "text",
    modelDefaultsAppliedFor: LAYA,
    trainingMethod: "qlora",
    maxSteps: 30,
    batchSize: 2,
  });
  serveConfigs();

  useTrainingConfigStore.getState().ensureModelDefaultsLoaded();
  await waitFor(
    (state) => !state.isLoadingModelDefaults && state.modelType === "decision",
  );

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainingMethod, "lora");
  assert.equal(state.maxSteps, LAYA_YAML.training.max_steps);
  assert.equal(state.batchSize, LAYA_YAML.training.batch_size);
  assert.equal(state.modelSubfolder, "multilingual");
});

test("a decision run sends the decision fields and switches off the LLM-only options", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLaya();
  useTrainingConfigStore.getState().setTrainingMethod("lora");
  useTrainingConfigStore.getState().setModelSubfolder(null);
  useTrainingConfigStore.setState({
    datasetSource: "huggingface",
    dataset: "LocalLLaMA/typed-decisions",
    datasetStreaming: true,
    packing: true,
    trainOnCompletions: true,
    loraVariant: "dora",
    optimizerType: "paged_adamw_8bit",
    contextLength: 32768,
    datasetManualMapping: { state: "user", gold: "assistant" },
  });

  const payload = buildTrainingStartPayload(
    useTrainingConfigStore.getState(),
    null,
  );

  assert.equal(payload.is_decision, true);
  assert.equal(payload.model_subfolder, null);
  assert.equal(payload.is_embedding, false);
  assert.equal(payload.training_type, "LoRA/QLoRA");
  assert.equal(payload.load_in_4bit, false);
  assert.equal(payload.use_lora, true);
  assert.equal(payload.lora_r, LAYA_YAML.lora.lora_r);
  assert.equal(payload.use_dora, false);
  assert.equal(payload.packing, false);
  assert.equal(payload.train_on_completions, false);
  assert.equal(payload.dataset_streaming, false);
  assert.equal(payload.custom_format_mapping, undefined);
  assert.equal(payload.optim, "adamw_torch");
  assert.equal(payload.max_seq_length, 1024);
  assert.equal(payload.learning_rate, String(LAYA_YAML.training.learning_rate));
});

test("an LLM run keeps its method and sends no decision fields", () => {
  useTrainingConfigStore.getState().reset();
  const payload = buildTrainingStartPayload(
    {
      ...useTrainingConfigStore.getState(),
      selectedModel: LLM,
      modelType: "text",
      trainingMethod: "qlora",
      modelSubfolder: "multilingual",
      optimizerType: "paged_adamw_8bit",
      contextLength: 32768,
    },
    null,
  );

  assert.equal(payload.is_decision, false);
  assert.equal(payload.model_subfolder, null);
  assert.equal(payload.training_type, "LoRA/QLoRA");
  assert.equal(payload.load_in_4bit, true);
  assert.equal(payload.optim, "paged_adamw_8bit");
  assert.equal(payload.max_seq_length, 32768);
});

test("decision training starts on Apple Silicon and on CUDA", () => {
  const config = {
    ...useTrainingConfigStore.getState(),
    ...HF_DECISION_DATASET,
    selectedModel: LAYA,
    modelType: "decision" as const,
    trainingMethod: "full" as const,
  };

  assert.deepEqual(validateTrainingConfig(config, "mac"), {
    ok: true,
    errorKey: null,
  });
  assert.deepEqual(validateTrainingConfig(config, "cuda"), {
    ok: true,
    errorKey: null,
  });
});

test("the model picker knows Laya is a decision model before its config loads", () => {
  const flags = trainingModelTypeFlagsFromMetadata({
    pipelineTag: "text-classification",
    tags: ["transformers", "safetensors", "laya", "system-one"],
    identifiers: [LAYA],
  });
  const modelType = inferTrainingModelTypeFromFlags(flags);

  assert.equal(modelType, "decision");
  assert.equal(isTrainingModelTypeSupportedOnDevice(modelType, "mac"), true);
  assert.equal(isTrainingModelTypeSupportedOnDevice(modelType, "cuda"), true);
});

test("only the Studio owner can start a decision run", () => {
  const config = {
    ...useTrainingConfigStore.getState(),
    ...HF_DECISION_DATASET,
    selectedModel: LAYA,
    modelType: "decision" as const,
    trainingMethod: "lora" as const,
  };

  assert.deepEqual(validateTrainingConfig(config, "cuda", false), {
    ok: false,
    errorKey: "studio.training.validation.decisionOwnerOnly",
  });
  assert.deepEqual(
    validateTrainingConfig(
      { ...config, selectedModel: LLM, modelType: "text" },
      "cuda",
      false,
    ),
    { ok: true, errorKey: null },
  );
});

test("the start check names the decision columns a dataset lacks", () => {
  assert.deepEqual(missingDecisionColumns(["state", "questions", "gold"]), []);
  assert.deepEqual(
    missingDecisionColumns(["state", "questions", "answers", "id"]),
    [],
  );
  assert.deepEqual(missingDecisionColumns(["messages"]), [
    "state",
    "questions",
    "gold",
  ]);
  assert.deepEqual(missingDecisionColumns(["state", "gold"]), ["questions"]);
});

test("a decision start skips the column check only for an uploaded JSON file it cannot read", async () => {
  const unreadable = () =>
    Promise.reject(
      new DatasetFormatError(
        "JSON parse error: Column(/gold/q1) changed from string to object",
        400,
      ),
    );
  assert.deepEqual(
    await checkDecisionDatasetColumns(unreadable, "0a1b_decisions.jsonl"),
    [],
  );
  await assert.rejects(
    checkDecisionDatasetColumns(unreadable, "0a1b_decisions.csv"),
    DatasetFormatError,
  );
  await assert.rejects(
    checkDecisionDatasetColumns(
      () => Promise.reject(new DatasetFormatError("Dataset is gated.", 401)),
      null,
    ),
    DatasetFormatError,
  );
  await assert.rejects(
    checkDecisionDatasetColumns(
      () => Promise.reject(new TypeError("Failed to fetch")),
      "0a1b_decisions.jsonl",
    ),
    TypeError,
  );
  assert.deepEqual(
    await checkDecisionDatasetColumns(
      () => Promise.resolve({ columns: ["messages"] }),
      null,
    ),
    ["state", "questions", "gold"],
  );
  assert.equal(
    await checkDecisionDatasetColumns(() => Promise.resolve(null), null),
    null,
  );
});

test("the run summary does not count settings the decision trainer ignores", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLaya();
  useTrainingConfigStore.getState().setTrainingMethod("lora");
  useTrainingConfigStore.setState({
    packing: true,
    trainOnCompletions: true,
    saveSteps: 50,
    optimizerType: "paged_adamw_8bit",
    loraVariant: "dora",
    targetModules: ["q_proj"],
  });
  const state = useTrainingConfigStore.getState();

  assert.equal(
    countNonDefaultAdvancedSettings(state, state.advancedSettingsBaseline),
    0,
  );
  assert.equal(buildTrainingStartPayload(state, null).optim, "adamw_torch");
  assert.equal(
    countNonDefaultAdvancedSettings(
      { ...state, modelType: "text" },
      state.advancedSettingsBaseline,
    ),
    6,
  );
});

const CLEF = "Cloudflare/clef-flash";

const CLEF_YAML = yaml.load(
  readFileSync(
    new URL(
      "../../backend/assets/configs/model_defaults/decision/Cloudflare_clef-flash.yaml",
      import.meta.url,
    ),
    "utf8",
  ),
) as {
  training: Record<string, number>;
  lora: Record<string, number>;
};

async function selectClef(): Promise<void> {
  setAuthFetchHandler((input) =>
    Response.json(
      input.startsWith("/api/models/config/")
        ? {
            ...LAYA_CONFIG,
            id: CLEF,
            config: CLEF_YAML,
            decision_checkpoints: null,
            decision_layout: "clef",
            model_size_bytes: 19_000_000_000,
          }
        : {},
    ),
  );
  useTrainingConfigStore.getState().selectTrainingModel(CLEF, "text");
  await waitForModelDefaults(CLEF);
}

test("a Clef run keeps QLoRA, sends 4-bit and the recipe's context, and has no checkpoint picker", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingMethod: "qlora" });
  await selectClef();
  useTrainingConfigStore.setState({
    ...HF_DECISION_DATASET,
    loraVariant: "dora",
  });

  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelType, "decision");
  assert.equal(state.decisionLayout, "clef");
  assert.equal(state.trainingMethod, "qlora");
  assert.equal(state.decisionCheckpoints, null);
  assert.equal(state.modelSubfolder, null);
  const payload = buildTrainingStartPayload(state, null);
  assert.equal(payload.is_decision, true);
  assert.equal(payload.training_type, "LoRA/QLoRA");
  assert.equal(payload.load_in_4bit, true);
  assert.equal(payload.use_dora, false);
  assert.equal(payload.model_subfolder, null);
  assert.equal(payload.max_seq_length, CLEF_YAML.training.max_seq_length);
  assert.equal(payload.lora_r, CLEF_YAML.lora.lora_r);
  assert.equal(payload.optim, "adamw_8bit");

  useTrainingConfigStore.getState().setTrainingMethod("lora");
  const lora = buildTrainingStartPayload(
    useTrainingConfigStore.getState(),
    null,
  );
  assert.equal(lora.load_in_4bit, false);
});

test("a fresh Clef pick from full fine-tuning starts as QLoRA, a Laya pick as LoRA", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingMethod: "full" });
  await selectClef();
  assert.equal(useTrainingConfigStore.getState().trainingMethod, "qlora");

  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingMethod: "qlora" });
  await selectLaya();
  assert.equal(useTrainingConfigStore.getState().trainingMethod, "lora");
  assert.equal(useTrainingConfigStore.getState().decisionLayout, "laya");
});
