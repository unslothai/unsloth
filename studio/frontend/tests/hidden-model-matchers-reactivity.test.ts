// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  getHiddenModelMatchersSnapshot,
  replaceHiddenModelMatchers,
  subscribeHiddenModelMatchers,
} from "../src/features/hub/lib/hidden-model-matcher-store.ts";

const MATCHER_REPLACEMENT_RE = /replaceHiddenModelMatchers\(\{/;
const MATCHER_SUBSCRIPTION_RE =
  /const hiddenModelMatchers = useSyncExternalStore\(\s*subscribeHiddenModelMatchers,\s*getHiddenModelMatchersSnapshot,\s*getHiddenModelMatchersSnapshot,\s*\);/;
const REACTIVE_FILTER_CALL_RE =
  /!isHiddenModelIdWithMatchers\(\s*hiddenModelMatchers,/;

test("hidden model matcher replacements publish a stable external-store snapshot", () => {
  const revisions: number[] = [];
  const unsubscribe = subscribeHiddenModelMatchers(() => {
    revisions.push(getHiddenModelMatchersSnapshot().revision);
  });
  const initial = getHiddenModelMatchersSnapshot();

  replaceHiddenModelMatchers({
    needles: ["embedder"],
    exactIds: ["org/embedder"],
    exactPaths: ["/models/embedder"],
  });
  const replaced = getHiddenModelMatchersSnapshot();
  replaceHiddenModelMatchers({
    needles: ["embedder"],
    exactIds: ["org/embedder"],
    exactPaths: ["/models/embedder"],
  });
  unsubscribe();

  assert.deepEqual(revisions, [initial.revision + 1]);
  assert.equal(replaced.ready, true);
  assert.equal(replaced.revision, initial.revision + 1);
  assert.equal(getHiddenModelMatchersSnapshot(), replaced);
});

test("Hub filters subscribe to successful hidden matcher replacements", () => {
  const hiddenModels = readFileSync(
    new URL("../src/features/hub/lib/hidden-models.ts", import.meta.url),
    "utf8",
  );
  const hubPage = readFileSync(
    new URL("../src/features/hub/hub-page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(hiddenModels, MATCHER_REPLACEMENT_RE);
  assert.match(hubPage, MATCHER_SUBSCRIPTION_RE);
  assert.match(hubPage, REACTIVE_FILTER_CALL_RE);
});
