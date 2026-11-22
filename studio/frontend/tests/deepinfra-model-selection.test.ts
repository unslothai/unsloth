// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);
const {
  formatPricePerMillion,
  formatContextTokens,
  parseModelIds,
  parseDeepInfraCatalog,
  filterModels,
  sortModels,
} = await import("../src/features/chat/deepinfra-model-catalog.ts");

test("formatPricePerMillion formats cents per token to dollars per million tokens ($/Mt)", () => {
  assert.equal(formatPricePerMillion(null), "—");
  assert.equal(formatPricePerMillion(undefined), "—");
  assert.equal(formatPricePerMillion(0), "Free");
  // 0.0000005 cents/token = $0.0050/Mt
  assert.equal(formatPricePerMillion(0.0000005), "$0.0050/Mt");
  // 0.000001 cents/token = $0.01/Mt
  assert.equal(formatPricePerMillion(0.000001), "$0.01/Mt");
  // 0.00005 cents/token = $0.50/Mt
  assert.equal(formatPricePerMillion(0.00005), "$0.50/Mt");
  // 0.00025 cents/token = $2.50/Mt
  assert.equal(formatPricePerMillion(0.00025), "$2.50/Mt");
});

test("formatContextTokens formats token limits to concise readable strings", () => {
  assert.equal(formatContextTokens(null), "—");
  assert.equal(formatContextTokens(undefined), "—");
  assert.equal(formatContextTokens(0), "—");
  assert.equal(formatContextTokens(512), "512");
  assert.equal(formatContextTokens(8192), "8k");
  assert.equal(formatContextTokens(32768), "32k");
  assert.equal(formatContextTokens(131072), "128k");
  assert.equal(formatContextTokens(1048576), "1024k");
  assert.equal(formatContextTokens(2000000), "2M");
});

test("parseModelIds extracts trimmed unique IDs from comma or newline separated strings", () => {
  assert.deepEqual(parseModelIds(""), []);
  assert.deepEqual(parseModelIds("   "), []);
  assert.deepEqual(
    parseModelIds("deepseek-ai/DeepSeek-V3, meta-llama/Llama-3.3-70B-Instruct"),
    ["deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.3-70B-Instruct"],
  );
  assert.deepEqual(
    parseModelIds(
      "deepseek-ai/DeepSeek-V3\nmeta-llama/Llama-3.3-70B-Instruct\ndeepseek-ai/DeepSeek-V3",
    ),
    ["deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.3-70B-Instruct"],
  );
});

test("parseDeepInfraCatalog maps raw API items and excludes deprecated and anthropic models", () => {
  const raw = [
    {
      model_name: "anthropic/claude-3-haiku",
      type: "text-generation",
      max_tokens: 200000,
    },
    {
      model_name: "legacy/old-model",
      type: "text-generation",
      deprecated: true,
      max_tokens: 4096,
    },
    {
      model_name: "meta-llama/Llama-3.3-70B-Instruct",
      type: "text-generation",
      description: "State-of-the-art 70B open model",
      tags: ["flex", "priority", "featured"],
      featured: true,
      pricing: {
        cents_per_input_token: 0.000023,
        cents_per_output_token: 0.00004,
        discount: 0.2,
      },
      max_tokens: 131072,
      quantization: "bfloat16",
    },
    {
      model_name: "BAAI/bge-m3",
      type: "embeddings",
      description: "Multi-lingual embeddings",
      tags: ["priority"],
      pricing: {
        cents_per_input_token: 0.000001,
        cents_per_output_token: 0,
      },
      max_tokens: 8192,
      quantization: "fp32",
    },
  ];

  const parsed = parseDeepInfraCatalog(raw);
  assert.equal(parsed.length, 2);

  const llama = parsed.find(
    (m: { id: string }) => m.id === "meta-llama/Llama-3.3-70B-Instruct",
  );
  assert.ok(llama);
  assert.equal(llama.isFeatured, true);
  assert.equal(llama.hasFlex, true);
  assert.equal(llama.hasPriority, true);
  assert.equal(llama.hasDiscount, true);
  assert.equal(llama.discountPercent, 20);
  assert.equal(llama.maxTokens, 131072);
  assert.equal(llama.quantization, "bfloat16");
  assert.equal(llama.pricePerMillionInput, 0.23);
  assert.equal(llama.pricePerMillionOutput, 0.4);

  const bge = parsed.find((m: { id: string }) => m.id === "BAAI/bge-m3");
  assert.ok(bge);
  assert.equal(bge.isFeatured, false);
  assert.equal(bge.hasFlex, false);
  assert.equal(bge.hasPriority, true);
  assert.equal(bge.hasDiscount, false);
  assert.equal(bge.type, "embeddings");
});

test("filterModels filters by search, task type, quantization, context and tier tags", () => {
  const models = [
    {
      id: "deepseek-ai/DeepSeek-R1",
      name: "deepseek-ai/DeepSeek-R1",
      type: "text-generation",
      maxTokens: 131072,
      quantization: "fp8",
      tags: ["flex", "priority", "featured"],
      isFeatured: true,
      hasFlex: true,
      hasPriority: true,
      hasDiscount: false,
      discountPercent: null,
      centsPerInputToken: 0.00005,
      centsPerOutputToken: 0.00021,
      pricePerMillionInput: 0.55,
      pricePerMillionOutput: 2.19,
      description: "Reasoning model",
    },
    {
      id: "BAAI/bge-large-en-v1.5",
      name: "BAAI/bge-large-en-v1.5",
      type: "embeddings",
      maxTokens: 512,
      quantization: null,
      tags: ["priority"],
      isFeatured: false,
      hasFlex: false,
      hasPriority: true,
      hasDiscount: false,
      discountPercent: null,
      centsPerInputToken: 0.000001,
      centsPerOutputToken: 0,
      pricePerMillionInput: 0.01,
      pricePerMillionOutput: 0,
      description: "English embeddings",
    },
    {
      id: "Qwen/Qwen2.5-Coder-32B-Instruct",
      name: "Qwen/Qwen2.5-Coder-32B-Instruct",
      type: "text-generation",
      maxTokens: 131072,
      quantization: "bfloat16",
      tags: ["flex", "discount"],
      isFeatured: false,
      hasFlex: true,
      hasPriority: false,
      hasDiscount: true,
      discountPercent: 15,
      centsPerInputToken: 0.00001,
      centsPerOutputToken: 0.00002,
      pricePerMillionInput: 0.1,
      pricePerMillionOutput: 0.2,
      description: "Coding specialist",
    },
  ];

  // Search filter
  assert.equal(filterModels(models, { searchQuery: "coder" }).length, 1);
  assert.equal(filterModels(models, { searchQuery: "qwen" })[0].id, "Qwen/Qwen2.5-Coder-32B-Instruct");

  // Type filter
  assert.equal(filterModels(models, { selectedType: "embeddings" }).length, 1);
  assert.equal(filterModels(models, { selectedType: "text-generation" }).length, 2);

  // Quant filter
  assert.equal(filterModels(models, { selectedQuant: "fp8" }).length, 1);
  assert.equal(filterModels(models, { selectedQuant: "bfloat16" }).length, 1);

  // Context filter
  assert.equal(filterModels(models, { minContext: 32000 }).length, 2);
  assert.equal(filterModels(models, { minContext: 200000 }).length, 0);

  // Tag button filters
  assert.equal(filterModels(models, { featuredOnly: true }).length, 1);
  assert.equal(filterModels(models, { flexOnly: true }).length, 2);
  assert.equal(filterModels(models, { priorityOnly: true }).length, 2);
  assert.equal(filterModels(models, { discountOnly: true }).length, 1);
});

test("sortModels sorts items by name, context, quant, and in/out prices", () => {
  const models = [
    {
      id: "b",
      name: "B",
      type: "text-generation",
      maxTokens: 131072,
      quantization: "fp8",
      tags: [],
      isFeatured: false,
      hasFlex: false,
      hasPriority: false,
      hasDiscount: false,
      discountPercent: null,
      centsPerInputToken: 0.00002,
      centsPerOutputToken: 0.00008,
      pricePerMillionInput: 0.2,
      pricePerMillionOutput: 0.8,
    },
    {
      id: "a",
      name: "A",
      type: "text-generation",
      maxTokens: 8192,
      quantization: "bfloat16",
      tags: [],
      isFeatured: false,
      hasFlex: false,
      hasPriority: false,
      hasDiscount: false,
      discountPercent: null,
      centsPerInputToken: 0.00005,
      centsPerOutputToken: 0.00004,
      pricePerMillionInput: 0.5,
      pricePerMillionOutput: 0.4,
    },
  ];

  // Name asc / desc
  assert.equal(sortModels(models, "name", "asc")[0].id, "a");
  assert.equal(sortModels(models, "name", "desc")[0].id, "b");

  // Context asc / desc
  assert.equal(sortModels(models, "context", "asc")[0].id, "a");
  assert.equal(sortModels(models, "context", "desc")[0].id, "b");

  // PriceIn asc / desc
  assert.equal(sortModels(models, "priceIn", "asc")[0].id, "b");
  assert.equal(sortModels(models, "priceIn", "desc")[0].id, "a");

  // PriceOut asc / desc
  assert.equal(sortModels(models, "priceOut", "asc")[0].id, "a");
  assert.equal(sortModels(models, "priceOut", "desc")[0].id, "b");
});

test("chat-providers-dialog renders DeepInfraModelSelectionPanel before isCuratedModelList", async () => {
  const { readSrc } = await import("./helpers/kit.ts");
  const dialogSrc = readSrc("features/chat/chat-providers-dialog.tsx");
  const deepinfraIndex = dialogSrc.indexOf('providerType === "deepinfra"');
  const curatedIndex = dialogSrc.indexOf(
    "isCuratedModelList && (availableModels.length > 0 || modelIdsEditable)",
  );

  assert.ok(deepinfraIndex > 0, "deepinfra providerType branch must exist");
  assert.ok(curatedIndex > 0, "curated model list branch must exist");
  assert.ok(
    deepinfraIndex < curatedIndex,
    "DeepInfra panel branch must be evaluated before curated checklist fallback",
  );
});
