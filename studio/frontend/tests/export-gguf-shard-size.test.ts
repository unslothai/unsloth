// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import {
  ggufShardSaveDirectory,
  isValidGgufShardSize,
  normalizeGgufShardSize,
} from "../src/features/export/lib/gguf-shard-size.ts";

register("./helpers/export-store-resolver.mjs", import.meta.url);

const stub = await import("./helpers/export-api-stub.mjs");
const { useExportRuntimeStore } = await import(
  "../src/features/export/stores/export-runtime-store.ts"
);

test("shard sizes match backend normalization", () => {
  for (const [value, expected] of [
    ["500m", "500MB"],
    [" 4 GB ", "4GB"],
    ["0002gb", "2GB"],
    ["1MB", "1MB"],
  ] as const) {
    assert.equal(normalizeGgufShardSize(value), expected);
    assert.equal(isValidGgufShardSize(value), true);
  }
});

test("invalid and empty split sizes fail validation", () => {
  for (const value of [
    "",
    "0",
    "none",
    "0MB",
    "0GB",
    "1.5GB",
    "512",
    "64KB",
    "-2GB",
    "4TB",
    "4GBx",
  ]) {
    assert.equal(normalizeGgufShardSize(value), null, value);
    assert.equal(isValidGgufShardSize(value), false, value);
  }
});

test("split defaults use a separate save directory", () => {
  assert.equal(
    ggufShardSaveDirectory("Qwen-GGUF", "512MB"),
    "Qwen-GGUF-split-512MB",
  );
  assert.equal(ggufShardSaveDirectory("Qwen-GGUF", "0"), "Qwen-GGUF");
  assert.equal(ggufShardSaveDirectory("Qwen-GGUF", null), "Qwen-GGUF");
});

test("split defaults keep the final path component within 255 characters", () => {
  for (const base of [
    `models/${"m".repeat(250)}-GGUF`,
    `C:\\models\\${"m".repeat(250)}-GGUF`,
  ]) {
    const directory = ggufShardSaveDirectory(base, "512MB");
    const separator = Math.max(
      directory.lastIndexOf("/"),
      directory.lastIndexOf("\\"),
    );
    const component = directory.slice(separator + 1);
    assert.equal(component.length, 255);
    assert.equal(component.endsWith("-split-512MB"), true);
  }
});

function params(ggufShardSize: string | null) {
  return {
    sourceMode: "model",
    checkpointPath: null,
    source: "unsloth/Qwen3-0.6B",
    modelSource: "hf",
    trustRemoteCode: false,
    exportMethod: "gguf",
    isAdapter: false,
    quantLevels: ["f16"],
    ggufShardSize,
    saveDirectory: "out",
    destination: "local",
    privateRepo: false,
    summary: {},
  } as unknown as Parameters<
    ReturnType<typeof useExportRuntimeStore.getState>["runExport"]
  >[0];
}

async function request(ggufShardSize: string | null) {
  stub.resetStub();
  await useExportRuntimeStore.getState().runExport(params(ggufShardSize));
  const call = stub.calls.find((entry) => entry.name === "exportGGUF");
  if (!call) {
    throw new Error("no gguf export request was made");
  }
  return call.args[0] as Record<string, unknown>;
}

test("the runtime preserves split, single-file, and legacy values", async () => {
  assert.equal((await request("512MB")).gguf_shard_size, "512MB");
  assert.equal((await request("0")).gguf_shard_size, "0");
  assert.equal((await request(null)).gguf_shard_size, null);
});
