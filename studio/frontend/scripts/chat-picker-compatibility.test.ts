// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  cachedInventoryRowCanChat,
  localInventoryRowCanChat,
} from "../src/components/assistant-ui/model-selector/chat-picker-compatibility.ts";
import {
  buildCachedInventoryRow,
  buildLocalInventoryRows,
} from "../src/features/inventory/view-models.ts";

test("chat picker rejects cached rows when backend marks chat unavailable", () => {
  const row = buildCachedInventoryRow(
    {
      repo_id: "Org/TokenizerOnly",
      inventory_id: "cache:safetensors:Org%2FTokenizerOnly",
      load_id: "Org/TokenizerOnly",
      model_format: "safetensors",
      runtime: "transformers",
      size_bytes: 2_000_000,
      capabilities: {
        can_chat: false,
        can_train: false,
      },
    },
    "safetensors",
  );

  assert.equal(cachedInventoryRowCanChat(row), false);
});

test("chat picker does not reject valid local folders by title or path substrings", () => {
  const [row] = buildLocalInventoryRows([
    {
      id: "/models/awq/archive/ValidModel",
      inventory_id: "local:safetensors:%2Fmodels%2Fawq%2Farchive%2FValidModel",
      load_id: "/models/awq/archive/ValidModel",
      display_name: "Valid ONNX Import",
      path: "/models/awq/archive/ValidModel",
      source: "custom",
      model_format: "safetensors",
      runtime: "transformers",
      capabilities: {
        can_chat: true,
        can_train: true,
      },
    },
  ]);

  assert.equal(localInventoryRowCanChat(row), true);
});

test("chat picker does not reject valid local rows by local base path substrings", () => {
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
      base_model: "/models/awq/base",
      base_model_source: "local",
      capabilities: {
        can_chat: true,
        can_train: false,
      },
    },
  ]);

  assert.equal(localInventoryRowCanChat(row), true);
});

test("chat picker rejects local rows with unsupported quant metadata", () => {
  const [row] = buildLocalInventoryRows([
    {
      id: "/models/Quantized",
      inventory_id: "local:safetensors:%2Fmodels%2FQuantized",
      load_id: "/models/Quantized",
      display_name: "Quantized",
      path: "/models/Quantized",
      source: "custom",
      model_id: "Org/Quantized",
      model_format: "safetensors",
      runtime: "transformers",
      quant_method: "awq",
      capabilities: {
        can_chat: true,
        can_train: true,
      },
    },
  ]);

  assert.equal(localInventoryRowCanChat(row), false);
});
