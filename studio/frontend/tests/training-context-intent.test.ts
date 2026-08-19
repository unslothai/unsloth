import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { mergeTrainingConfig } = await import(
  "../src/features/training/stores/training-config-persistence.ts"
);
const { selectTrainingMethodForHardware } = await import(
  "../src/features/training/stores/training-method-hardware-policy.ts"
);

const MODEL_DEFAULTS: Record<string, number> = {
  "org/first": 8192,
  "org/second": 16384,
};

function responseFor(input: string): Response {
  if (input === "/api/system/hardware") {
    return Response.json({ gpu: { vram_free_gb: 10 } });
  }
  const model = decodeURIComponent(input.split("/api/models/config/")[1]);
  return Response.json({
    id: model,
    model_name: model,
    config: { training: { max_seq_length: MODEL_DEFAULTS[model] } },
    is_vision: false,
    is_embedding: false,
    is_audio: false,
    audio_type_known: true,
    is_lora: false,
    model_type: "text",
  });
}

async function settle(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve));
  await new Promise<void>((resolve) => setImmediate(resolve));
}

test.beforeEach(() => {
  setAuthFetchHandler((input) => responseFor(input));
  useTrainingConfigStore.getState().reset();
});

test.after(() => setAuthFetchHandler(null));

test("explicit context survives same-model defaults and cache refresh", async () => {
  useTrainingConfigStore.getState().setSelectedModel("org/first");
  await settle();
  assert.equal(useTrainingConfigStore.getState().contextLength, 8192);

  useTrainingConfigStore.getState().setContextLength(4096);
  useTrainingConfigStore.getState().setSelectedModelCacheReference("org/first", {
    localPath: "/cache/first",
    modelFormat: null,
  });
  await settle();

  const state = useTrainingConfigStore.getState();
  assert.equal(state.contextLength, 4096);
  assert.equal(state.contextLengthManuallySet, true);
});

test("rehydrated context ownership survives same-model model defaults", async () => {
  const rehydrated = mergeTrainingConfig(
    {
      selectedModel: "org/first",
      contextLength: 4096,
      contextLengthManuallySet: true,
      modelDefaultsAppliedFor: "org/first",
      advancedSettingsBaseline: { contextLength: 8192 },
    },
    useTrainingConfigStore.getState(),
  );
  useTrainingConfigStore.setState({
    ...rehydrated,
    modelDefaultsAppliedFor: null,
    advancedSettingsBaseline: null,
  });
  useTrainingConfigStore.getState().setSelectedModel("org/first");
  await settle();

  const state = useTrainingConfigStore.getState();
  assert.equal(state.contextLength, 4096);
  assert.equal(state.contextLengthManuallySet, true);
});

test("untouched defaults apply, while a true switch clears intent first", async () => {
  useTrainingConfigStore.getState().setSelectedModel("org/first");
  await settle();
  assert.equal(useTrainingConfigStore.getState().contextLengthManuallySet, false);

  useTrainingConfigStore.getState().setContextLength(4096);
  useTrainingConfigStore.getState().setSelectedModel("org/second");
  assert.equal(useTrainingConfigStore.getState().contextLengthManuallySet, false);
  await settle();

  assert.equal(useTrainingConfigStore.getState().contextLength, 16384);
});

test("a stale defaults response cannot overwrite the selected model", async () => {
  let releaseFirst: (() => void) | undefined;
  setAuthFetchHandler((input) => {
    if (input.includes("/api/models/config/org%2Ffirst")) {
      return new Promise<Response>((resolve) => {
        releaseFirst = () => resolve(responseFor(input));
      });
    }
    return responseFor(input);
  });

  useTrainingConfigStore.getState().setSelectedModel("org/first");
  useTrainingConfigStore.getState().setSelectedModel("org/second");
  await settle();
  releaseFirst?.();
  await settle();

  assert.equal(useTrainingConfigStore.getState().selectedModel, "org/second");
  assert.equal(useTrainingConfigStore.getState().contextLength, 16384);
});

test("clear, full reset, and reset-to-model-defaults clear context intent", async () => {
  useTrainingConfigStore.getState().setSelectedModel("org/first");
  await settle();
  useTrainingConfigStore.getState().setContextLength(4096);
  useTrainingConfigStore.getState().resetToModelDefaults();
  assert.equal(useTrainingConfigStore.getState().contextLengthManuallySet, false);
  await settle();
  assert.equal(useTrainingConfigStore.getState().contextLength, 8192);

  useTrainingConfigStore.getState().setContextLength(4096);
  useTrainingConfigStore.getState().setSelectedModel(null);
  assert.equal(useTrainingConfigStore.getState().contextLengthManuallySet, false);
  useTrainingConfigStore.getState().setContextLength(4096);
  useTrainingConfigStore.getState().reset();
  assert.equal(useTrainingConfigStore.getState().contextLengthManuallySet, false);
});

test("hardware selection uses the effective context length", async () => {
  const signal = new AbortController().signal;
  assert.equal(await selectTrainingMethodForHardware(4 * 1024 ** 3, 4096, signal), "lora");
  assert.equal(await selectTrainingMethodForHardware(4 * 1024 ** 3, 32768, signal), "qlora");
});
