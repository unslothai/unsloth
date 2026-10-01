// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);
const { providerLogoPath } = await import("../src/features/chat/provider-logo-path.ts");
const { PROVIDER_LOGOS } = await import("../src/features/hub/lib/provider-logos.ts");

test("connections use the model hub's canonical provider images", () => {
  for (const [connection, hubId] of [
    ["openai", "openai"],
    ["openai_codex", "openai"],
    ["deepseek", "deepseek-ai"],
    ["mistral", "mistralai"],
    ["huggingface", "huggingface"],
    ["kimi", "moonshotai"],
    ["qwen", "qwen"],
  ]) {
    const hubLogo = PROVIDER_LOGOS.find(({ id }) => id === hubId);
    assert.ok(hubLogo, hubId);
    assert.equal(providerLogoPath(connection), hubLogo.logoPath, connection);
  }
});

test("Gemini keeps its own logo, separate from Google's hub image", () => {
  assert.equal(providerLogoPath("gemini"), "/provider-logos/gemini.svg");
  assert.notEqual(
    providerLogoPath("gemini"),
    PROVIDER_LOGOS.find(({ id }) => id === "google")?.logoPath,
  );
});

test("every supported connection resolves to an existing public asset", () => {
  for (const type of [
    "openai", "openai_codex", "deepseek", "mistral", "huggingface", "kimi",
    "qwen", "gemini", "anthropic", "openrouter", "vllm", "ollama", "llama_cpp",
  ]) {
    const path = providerLogoPath(type);
    assert.ok(path?.startsWith("/"), type);
    assert.ok(existsSync(new URL(`../public${path}`, import.meta.url)), `${type}: ${path}`);
  }
});

test("custom, unknown, and absent providers leave rendering to the fallback", () => {
  for (const type of [null, undefined, "", "custom", "unknown", "toString", "constructor"]) {
    assert.equal(providerLogoPath(type), undefined, String(type));
  }
});
