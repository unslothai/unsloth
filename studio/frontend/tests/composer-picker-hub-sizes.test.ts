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
  // A GGUF row needs no listing; a result that is no row is never listed.
  assert.deepEqual(hubSizeIds(results, recommended), [
    "unsloth/Chat-8B",
    "unsloth/Chat-8B-MLX-4bit",
  ]);
  assert.deepEqual(hubSizeIds(results, []), []);
  // The chat fallback's rows are not Hub results: nothing to list.
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
    /^const hubSizes = useHubRepoSizes\(\s+useMemo\(\s+\(\) => hubSizeIds\(hubResults, recommended\),\s+\[hubResults, recommended\],/,
  );
  assert.ok(
    picker.indexOf("const recommended = useMemo") <
      picker.indexOf("const hubSizes = useHubRepoSizes("),
  );
});
