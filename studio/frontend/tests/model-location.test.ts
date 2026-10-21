// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ModelPickTarget } from "../src/features/model-picker/components/model-selector/types.ts";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveModelLocation } = await import(
  "../src/features/model-picker/model-config/model-location.ts"
);

function target(overrides: Partial<ModelPickTarget> = {}): ModelPickTarget {
  return {
    id: "unsloth/Repo-GGUF",
    displayName: "Repo",
    isGguf: true,
    meta: { source: "hub", isLora: false, isDownloaded: true },
    ...overrides,
  };
}

test("a local pick's own path wins over its inventory id", () => {
  const picked = target({
    id: "models_dir--gemma-3-12b",
    isGguf: false,
    meta: {
      source: "local",
      isLora: false,
      isDownloaded: true,
      localPath: String.raw`C:\models\gemma-3-12b`,
    },
  });
  assert.deepEqual(resolveModelLocation(picked), {
    kind: "path",
    path: String.raw`C:\models\gemma-3-12b`,
  });
});

test("a path-shaped id is the location: standalone GGUF, snapshot, resident local folder", () => {
  for (const id of [
    "/home/u/models/x.gguf",
    String.raw`C:\models\x.gguf`,
    "~/models/x.gguf",
  ]) {
    const resident = target({
      id,
      meta: { source: "local", isLora: false, isDownloaded: true },
    });
    assert.deepEqual(resolveModelLocation(resident), { kind: "path", path: id });
  }
});

test("a downloaded cached repo keys by repo id and quant for the cache lookup", () => {
  const cached = target({
    id: "unsloth/Qwen3.6-35B-A3B-GGUF",
    ggufVariant: "Q4_K_M",
  });
  assert.deepEqual(resolveModelLocation(cached), {
    kind: "cachedRepo",
    repoId: "unsloth/Qwen3.6-35B-A3B-GGUF",
    variant: "Q4_K_M",
  });
});

test("a cached non-GGUF repo resolves without a variant", () => {
  const cached = target({ id: "unsloth/Llama-3.2-3B", isGguf: false });
  assert.deepEqual(resolveModelLocation(cached), {
    kind: "cachedRepo",
    repoId: "unsloth/Llama-3.2-3B",
  });
});

test("a snapshot-pinned repo shows the pinned snapshot, not a cache lookup", () => {
  const pinned = target({
    id: "unsloth/Repo-GGUF",
    ggufVariant: "Q4_K_M",
    meta: {
      source: "hub",
      isLora: false,
      isDownloaded: true,
      loadId: "/home/u/.cache/huggingface/hub/models--unsloth--Repo-GGUF/snapshots/abc123",
    },
  });
  assert.deepEqual(resolveModelLocation(pinned), {
    kind: "path",
    path: "/home/u/.cache/huggingface/hub/models--unsloth--Repo-GGUF/snapshots/abc123",
  });
});

test("an MLX precision-folder pick looks up its repo without a GGUF variant", () => {
  const mlx = target({
    id: "mlx-community/Qwen3.5-2B",
    isGguf: false,
    ggufVariant: "6-bit",
  });
  assert.deepEqual(resolveModelLocation(mlx), {
    kind: "cachedRepo",
    repoId: "mlx-community/Qwen3.5-2B",
  });
});

test("the settings id is preferred when the load id only spells the quant", () => {
  const cached = target({
    id: "unsloth/Repo-GGUF:Q4_K_M",
    configId: "unsloth/Repo-GGUF",
    ggufVariant: "Q4_K_M",
  });
  assert.deepEqual(resolveModelLocation(cached), {
    kind: "cachedRepo",
    repoId: "unsloth/Repo-GGUF",
    variant: "Q4_K_M",
  });
});

test("not-yet-downloaded and external picks have no location", () => {
  const pending = target({
    meta: { source: "hub", isLora: false, isDownloaded: false },
  });
  assert.equal(resolveModelLocation(pending), null);

  const external = target({
    id: "gpt-5.2",
    isGguf: false,
    meta: { source: "external", isLora: false },
  });
  assert.equal(resolveModelLocation(external), null);
});

test("Ollama models are skipped even when the id is a link path", () => {
  for (const id of [
    "ollama-manifest:llama3.2",
    String.raw`C:\Users\u\.unsloth\studio\.studio_links\llama3.2`,
    "/home/u/.cache/ollama_links/llama3.2",
  ]) {
    const ollama = target({
      id,
      isGguf: false,
      meta: { source: "local", isLora: false, isDownloaded: true },
    });
    assert.equal(resolveModelLocation(ollama), null);
  }
});

test("a blank id with no picked path resolves to nothing", () => {
  const blank = target({
    id: "   ",
    isGguf: false,
    meta: { source: "local", isLora: false, isDownloaded: true },
  });
  assert.equal(resolveModelLocation(blank), null);
});
