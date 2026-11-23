// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import * as pickerModel from "../src/components/assistant-ui/composer-model-picker/picker-model.ts";
import { readSrc } from "./helpers/kit.ts";

const hit = (id: string, isGguf = false) => ({ id, isGguf });

test("Discover sizes only the Hub results that are its rows", () => {
  const { hubSizeIds } = pickerModel as unknown as {
    hubSizeIds?: (
      results: readonly { id: string; isGguf: boolean }[],
      recommended: readonly { id: string }[],
    ) => string[];
  };
  assert.equal(typeof hubSizeIds, "function", "picker-model has no hubSizeIds");
  if (!hubSizeIds) {
    return;
  }
  const results = [
    hit("unsloth/Chat-8B"),
    hit("unsloth/Chat-8B-GGUF", true),
    hit("unsloth/Speech-TTS"),
    hit("unsloth/Embed-0.6B"),
    hit("unsloth/Chat-8B-MLX-4bit"),
  ];
  const recommended = [
    { id: "unsloth/chat-8b" },
    { id: "unsloth/Chat-8B-GGUF" },
    { id: "unsloth/Chat-8B-MLX-4bit" },
  ];
  assert.deepEqual(hubSizeIds(results, recommended), [
    "unsloth/Chat-8B",
    "unsloth/Chat-8B-MLX-4bit",
  ]);
  assert.deepEqual(hubSizeIds(results, []), []);
  assert.deepEqual(hubSizeIds([], [{ id: "unsloth/Chat-8B" }]), []);
});

test("the picker lists sizes for its Discover rows, not every Hub result", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  const call = picker.slice(
    picker.indexOf("const hubSizes = useHubRepoSizes("),
  );
  assert.match(
    call,
    /^const hubSizes = useHubRepoSizes\(\s+useMemo\(\s+\(\) =>\s+hubSizeIds\(hubResults, recommended, \(r\) =>\s+isMlxId\(r\.id, hasMlxCardTag\(r\)\),\s+\),\s+\[hubResults, recommended\],/,
  );
  assert.ok(
    picker.indexOf("const recommended = useMemo") <
      picker.indexOf("const hubSizes = useHubRepoSizes("),
  );
});

test("a plain checkpoint the listing already sized costs no file listing; MLX still does", () => {
  const { hubSizeIds } = pickerModel;
  const results = [
    { id: "unsloth/Chat-8B", isGguf: false, estimatedSizeBytes: 16e9 },
    { id: "unsloth/Chat-8B-MLX-4bit", isGguf: false, estimatedSizeBytes: 16e9 },
    { id: "unsloth/Chat-4B", isGguf: false },
  ];
  const recommended = results.map((r) => ({ id: r.id }));
  assert.deepEqual(
    hubSizeIds(results, recommended, (r) => r.id.includes("MLX")),
    ["unsloth/Chat-8B-MLX-4bit", "unsloth/Chat-4B"],
  );
});

test("the Safetensors filter lists plain checkpoints on any host that runs them", () => {
  const { discoverListsCheckpoints } = pickerModel;
  // isMac, formatFilter, chatOnly, isMlx, servesMlx, chatOnlyKnown
  assert.equal(discoverListsCheckpoints(true, "all", false, false, true, false), true);
  assert.equal(discoverListsCheckpoints(true, "all", false, true, true, false), true);
  assert.equal(discoverListsCheckpoints(false, "all", false, false, false, false), false);
  assert.equal(discoverListsCheckpoints(false, "gguf", false, false, false, false), false);
  assert.equal(
    discoverListsCheckpoints(false, "safetensors", false, false, false, false),
    true,
  );
  assert.equal(
    discoverListsCheckpoints(false, "safetensors", false, true, false, false),
    false,
  );
  assert.equal(
    discoverListsCheckpoints(false, "safetensors", true, false, false, true),
    false,
  );
});

test("a Chat only Mac lists checkpoints in Discover only where it serves MLX", () => {
  const { discoverListsCheckpoints } = pickerModel;
  assert.equal(discoverListsCheckpoints(true, "all", true, true, true, true), true);
  assert.equal(discoverListsCheckpoints(true, "all", true, false, true, true), true);
  assert.equal(discoverListsCheckpoints(true, "all", true, true, false, true), false);
  assert.equal(discoverListsCheckpoints(true, "all", true, false, false, true), false);
  // Every Mac guesses Chat only until measured; that guess must not hide checkpoints.
  assert.equal(discoverListsCheckpoints(true, "all", true, false, true, false), true);
});

test("the picker gates Discover checkpoints on the format filter and chat-only state", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  const memo = picker.slice(picker.indexOf("const recommended = useMemo"));
  assert.match(
    memo,
    /discoverListsCheckpoints\(\s+isMac,\s+formatFilter,\s+chatOnly,\s+isMlxId\(r\.id, hasMlxCardTag\(r\)\),\s+servesMlx,\s+chatOnlyKnown,/,
  );
  assert.ok(
    picker.indexOf('useState<FormatFilter>("all")') <
      picker.indexOf("const recommended = useMemo"),
  );
});

test("a failed size listing waits before it is asked again", () => {
  const src = readSrc(
    "components/assistant-ui/composer-model-picker/hub-repo-size.ts",
  );
  assert.match(src, /failedAt\.set\(key, Date\.now\(\)\)/);
  assert.match(
    src,
    /!sizes\.has\(keyOf\(id\)\) && !coolingDown\(keyOf\(id\)\)/,
  );
});

test("a Chat only host lists cached checkpoints only where it serves MLX", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /const servesMlx = usePlatformStore\(\(s\) =>\s*isServedByMlx\(false, s\.deviceType, s\.chatOnlyReason, s\.mlxInference\),/,
  );
  const loop = picker.slice(picker.indexOf("for (const repo of inventory.cachedModels)"));
  assert.match(
    loop.slice(0, loop.indexOf("device.push")),
    /if \(!audio && chatOnlyKnown && !servesMlx\) \{\s*continue;/,
  );
});

test("a Chat only host lists local checkpoints beyond GGUF only where it serves MLX", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  const loop = picker.slice(picker.indexOf("for (const lora of loraModels ?? [])"));
  assert.match(
    loop.slice(0, loop.indexOf("device.push")),
    /lora\.source === "local" &&\s*!lora\.audioType &&\s*chatOnlyKnown &&\s*format !== "gguf" &&\s*!servesMlx\s*\)/,
  );
});
