// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildCachedInventoryRow,
  buildLocalInventoryRows,
} from "../src/features/inventory/view-models.ts";

test("local normalization trusts backend GGUF format over names", () => {
  const [row] = buildLocalInventoryRows([
    {
      id: "/models/foo",
      inventory_id: "local:gguf:%2Fmodels%2Ffoo",
      load_id: "/models/foo",
      display_name: "foo",
      path: "/models/foo",
      source: "models_dir",
      model_format: "gguf",
      runtime: "llama_cpp",
      capabilities: {
        can_train: false,
        can_chat: true,
        requires_variant: true,
      },
    },
  ]);

  assert.equal(row.id, "local:gguf:%2Fmodels%2Ffoo");
  assert.equal(row.loadId, "/models/foo");
  assert.equal(row.modelFormat, "gguf");
  assert.equal(row.isGguf, true);
  assert.equal(row.capabilities.canTrain, false);
  assert.equal(row.capabilities.requiresVariant, true);
});

test("local normalization does not mark backend safetensors rows as GGUF by suffix", () => {
  const [row] = buildLocalInventoryRows([
    {
      id: "/models/foo-GGUF",
      inventory_id: "local:safetensors:%2Fmodels%2Ffoo-GGUF",
      load_id: "/models/foo-GGUF",
      display_name: "foo-GGUF",
      path: "/models/foo-GGUF",
      source: "models_dir",
      model_format: "safetensors",
      runtime: "transformers",
      capabilities: {
        can_train: true,
        can_chat: true,
      },
    },
  ]);

  assert.equal(row.modelFormat, "safetensors");
  assert.equal(row.isGguf, false);
  assert.equal(row.capabilities.canTrain, true);
});

test("partial HF-cache safetensors rows keep format but stay non-runnable", () => {
  const [row] = buildLocalInventoryRows([
    {
      id: "unsloth/Partial",
      inventory_id: "hf_cache:safetensors:unsloth%2FPartial",
      load_id: "unsloth/Partial",
      display_name: "Partial",
      path: "/hf/models--unsloth--Partial",
      source: "hf_cache",
      model_id: "unsloth/Partial",
      model_format: "safetensors",
      runtime: "transformers",
      partial: true,
      capabilities: {
        can_train: false,
        can_chat: false,
      },
    },
  ]);

  assert.equal(row.repoId, "unsloth/Partial");
  assert.equal(row.modelFormat, "safetensors");
  assert.equal(row.runtime, "transformers");
  assert.equal(row.partial, true);
  assert.equal(row.capabilities.canTrain, false);
  assert.equal(row.capabilities.canChat, false);
});

test("cached mixed repo rows use distinct inventory ids and shared load id", () => {
  const gguf = buildCachedInventoryRow(
    {
      repo_id: "Org/Mixed",
      inventory_id: "cache:gguf:Org%2FMixed",
      load_id: "Org/Mixed",
      model_format: "gguf",
      runtime: "llama_cpp",
      size_bytes: 100,
      capabilities: {
        can_train: false,
        can_chat: true,
        requires_variant: true,
      },
    },
    "gguf",
  );
  const safetensors = buildCachedInventoryRow(
    {
      repo_id: "Org/Mixed",
      inventory_id: "cache:safetensors:Org%2FMixed",
      load_id: "Org/Mixed",
      model_format: "safetensors",
      runtime: "transformers",
      size_bytes: 200,
      capabilities: {
        can_train: true,
        can_chat: true,
      },
    },
    "safetensors",
  );

  assert.equal(gguf.loadId, safetensors.loadId);
  assert.notEqual(gguf.id, safetensors.id);
  assert.equal(gguf.modelFormat, "gguf");
  assert.equal(safetensors.modelFormat, "safetensors");
  assert.equal(gguf.capabilities.canTrain, false);
  assert.equal(safetensors.capabilities.canTrain, true);
});

test("cached model endpoint fallback restores runnable safetensors rows", () => {
  const row = buildCachedInventoryRow(
    {
      repo_id: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
      inventory_id: "cache:unknown:unsloth%2FQwen2.5-3B-Instruct-bnb-4bit",
      load_id: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
      model_format: "unknown",
      runtime: "unknown",
      size_bytes: 200,
      capabilities: {
        can_train: false,
        can_chat: false,
      },
    },
    "safetensors",
  );

  assert.equal(
    row.id,
    "cache:safetensors:unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
  );
  assert.equal(row.modelFormat, "safetensors");
  assert.equal(row.runtime, "transformers");
  assert.equal(row.capabilities.canChat, true);
  assert.equal(row.capabilities.canTrain, true);
});

test("custom adapter rows keep local identity and expose HF base model separately", () => {
  const [row] = buildLocalInventoryRows([
    {
      id: "/outputs/checkpoint-60",
      inventory_id: "custom:adapter:%2Foutputs%2Fcheckpoint-60",
      load_id: "/outputs/checkpoint-60",
      display_name: "checkpoint-60",
      path: "/outputs/checkpoint-60",
      source: "custom",
      model_id: "outputs/checkpoint-60",
      model_format: "adapter",
      runtime: "adapter",
      base_model: "unsloth/Llama-3.2-3B",
      base_model_source: "huggingface",
      adapter_type: "LORA",
      training_method: "qlora",
      capabilities: {
        can_train: false,
        can_chat: true,
      },
    },
  ]);

  assert.equal(row.repoId, null);
  assert.equal(row.modelFormat, "adapter");
  assert.equal(row.baseModel, "unsloth/Llama-3.2-3B");
  assert.equal(row.baseModelHubId, "unsloth/Llama-3.2-3B");
  assert.equal(row.baseModelSource, "huggingface");
  assert.equal(row.adapterType, "LORA");
  assert.equal(row.trainingMethod, "qlora");
});

test("custom adapter local base path is not treated as a HF repo", () => {
  const [row] = buildLocalInventoryRows([
    {
      id: "/outputs/checkpoint-60",
      inventory_id: "custom:adapter:%2Foutputs%2Fcheckpoint-60",
      load_id: "/outputs/checkpoint-60",
      display_name: "checkpoint-60",
      path: "/outputs/checkpoint-60",
      source: "custom",
      model_format: "adapter",
      runtime: "adapter",
      base_model: "outputs/checkpoint-40",
      base_model_source: "local",
      capabilities: {
        can_train: false,
        can_chat: true,
      },
    },
  ]);

  assert.equal(row.repoId, null);
  assert.equal(row.baseModel, "outputs/checkpoint-40");
  assert.equal(row.baseModelHubId, null);
  assert.equal(row.baseModelSource, "local");
});
