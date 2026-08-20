// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The config plumbing reads the platform store, and config files travel between
// machines, so pin the value mapping against every device type and every shipped
// model config rather than the one config the round-trip test seeds from.

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const yaml = await import("js-yaml");
const { usePlatformStore } = await import("@/config/env");
const { mapBackendModelConfigToTrainingPatch } = await import(
  "../src/features/training/lib/model-defaults.ts"
);
const { parseYamlConfig, serializeConfigToYaml } = await import(
  "../src/features/training/lib/yaml-config.ts"
);
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);

// main.py maps sys.platform, so WSL arrives as "linux" and anything exotic
// arrives verbatim; the browser fallback in env.ts only ever guesses these three.
const DEVICE_TYPES = ["linux", "windows", "mac", "freebsd"];

const MODEL_DEFAULTS_DIR = new URL(
  "../../backend/assets/configs/model_defaults/",
  import.meta.url,
);

const SHIPPED_CONFIGS = readdirSync(MODEL_DEFAULTS_DIR, { recursive: true })
  .map(String)
  .filter((name) => name.endsWith(".yaml"))
  .sort();

function setDeviceType(deviceType: string): void {
  (
    usePlatformStore as unknown as {
      setState: (next: { deviceType: string }) => void;
    }
  ).setState({ deviceType });
}

function patchFor(training: Record<string, unknown>): Record<string, unknown> {
  return mapBackendModelConfigToTrainingPatch({ training } as never) as Record<
    string,
    unknown
  >;
}

test("gradient_checkpointing only ever maps to a value the picker can show", () => {
  // Whatever a hand-written or shipped file holds, the store must not end up
  // with a value <Select> cannot render.
  const rawValues: unknown[] = [
    true,
    false,
    "true",
    "none",
    "unsloth",
    "mlx",
    "None",
    "TRUE",
    " true ",
    1,
    0,
    null,
    "",
    "yes",
    [],
    {},
  ];
  for (const deviceType of DEVICE_TYPES) {
    setDeviceType(deviceType);
    for (const value of rawValues) {
      const mapped = patchFor({
        gradient_checkpointing: value,
      }).gradientCheckpointing;
      if (mapped !== undefined) {
        assert.ok(
          ["none", "true", "unsloth", "mlx"].includes(mapped as string),
          `${deviceType}: ${JSON.stringify(value)} mapped to ${String(mapped)}`,
        );
      }
      if (typeof value === "boolean") {
        assert.equal(mapped, value ? "true" : "none", deviceType);
      }
    }
  }
});

test("Unsloth GC is still never selected on a Mac", () => {
  setDeviceType("mac");
  assert.equal(
    patchFor({ gradient_checkpointing: "unsloth" }).gradientCheckpointing,
    "mlx",
  );
  // The boolean path must not sneak past that remap either.
  assert.equal(
    patchFor({ gradient_checkpointing: true }).gradientCheckpointing,
    "true",
  );
  assert.equal(
    patchFor({ gradient_checkpointing: false }).gradientCheckpointing,
    "none",
  );
});

test("every shipped config's checkpointing survives a save and a reload", () => {
  assert.ok(SHIPPED_CONFIGS.length > 50, "expected the shipped config set");
  let booleanConfigs = 0;
  for (const deviceType of DEVICE_TYPES) {
    setDeviceType(deviceType);
    for (const file of SHIPPED_CONFIGS) {
      const config = yaml.load(
        readFileSync(new URL(file, MODEL_DEFAULTS_DIR), "utf8"),
      ) as { training?: { gradient_checkpointing?: unknown } };
      const shipped = config.training?.gradient_checkpointing;
      const seeded = mapBackendModelConfigToTrainingPatch(config as never);
      if (typeof shipped === "boolean") {
        booleanConfigs++;
        assert.equal(
          seeded.gradientCheckpointing,
          shipped ? "true" : "none",
          file,
        );
      }
      useTrainingConfigStore.setState(seeded);
      const selected = useTrainingConfigStore.getState().gradientCheckpointing;
      const saved = serializeConfigToYaml(
        useTrainingConfigStore.getState(),
        false,
      );
      useTrainingConfigStore
        .getState()
        .applyConfigPatch(parseYamlConfig(saved));
      assert.equal(
        useTrainingConfigStore.getState().gradientCheckpointing,
        selected,
        `${file} on ${deviceType}`,
      );
    }
  }
  assert.ok(
    booleanConfigs > 0,
    "some shipped configs still decode as booleans; this is what covers them",
  );
});

test("a blank value is ignored for every numeric field, a real 0 is not", () => {
  setDeviceType("linux");
  const numericFields = [
    ["max_seq_length", "contextLength"],
    ["num_epochs", "epochs"],
    ["learning_rate", "learningRate"],
    ["embedding_learning_rate", "embeddingLearningRate"],
    ["batch_size", "batchSize"],
    ["gradient_accumulation_steps", "gradientAccumulation"],
    ["warmup_steps", "warmupSteps"],
    ["max_steps", "maxSteps"],
    ["save_steps", "saveSteps"],
    ["eval_steps", "evalSteps"],
    ["weight_decay", "weightDecay"],
    ["random_seed", "randomSeed"],
  ] as const;
  for (const [yamlKey, stateKey] of numericFields) {
    for (const blank of ["", "   ", "\t"]) {
      assert.ok(
        !Object.hasOwn(patchFor({ [yamlKey]: blank }), stateKey),
        `${yamlKey}: ${JSON.stringify(blank)} must not be applied`,
      );
    }
    assert.equal(patchFor({ [yamlKey]: 0 })[stateKey], 0, yamlKey);
  }
});

test("a config file written on another OS still imports", () => {
  setDeviceType("windows");
  useTrainingConfigStore.setState({ epochs: 3, gradientCheckpointing: "none" });
  const saved = serializeConfigToYaml(useTrainingConfigStore.getState(), false);
  // A file saved on Windows, or one an editor wrote with a byte order mark.
  const bom = "\uFEFF";
  const variants: [string, string][] = [
    ["CRLF", saved.replace(/\n/g, "\r\n")],
    ["UTF-8 BOM", `${bom}${saved}`],
    ["BOM and CRLF", `${bom}${saved.replace(/\n/g, "\r\n")}`],
  ];
  for (const [name, text] of variants) {
    const patch = mapBackendModelConfigToTrainingPatch(parseYamlConfig(text));
    assert.equal(patch.epochs, 3, name);
    assert.equal(patch.gradientCheckpointing, "none", name);
  }
});

test("the WandB token never reaches the file, whatever the state", () => {
  for (const deviceType of DEVICE_TYPES) {
    setDeviceType(deviceType);
    for (const enableWandb of [true, false]) {
      useTrainingConfigStore.setState({
        enableWandb,
        wandbToken: "wandb-secret-value",
        wandbProject: "p",
      });
      const saved = serializeConfigToYaml(
        useTrainingConfigStore.getState(),
        false,
      );
      assert.ok(!saved.includes("wandb-secret-value"), deviceType);
      assert.ok(!saved.includes("wandb_token"), deviceType);
    }
  }
});

test("saving a reloaded config reproduces the same file", () => {
  setDeviceType("linux");
  useTrainingConfigStore.setState({
    embeddingLearningRate: 3e-5,
    enableWandb: true,
    wandbProject: "my-project",
    enableTensorboard: true,
    tensorboardDir: "my-runs",
    logFrequency: 25,
  });
  const first = serializeConfigToYaml(useTrainingConfigStore.getState(), false);
  useTrainingConfigStore.getState().applyConfigPatch(parseYamlConfig(first));
  const second = serializeConfigToYaml(
    useTrainingConfigStore.getState(),
    false,
  );
  assert.equal(second, first);
});
