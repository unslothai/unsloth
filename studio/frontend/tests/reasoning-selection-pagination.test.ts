// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const REASONING = readText("../src/components/assistant-ui/reasoning.tsx");
const INTERSECTION_HELPER =
  /function selectionIntersectsElement\([\s\S]*?selection\.getRangeAt\(index\)\.intersectsNode\(element\)/;
const CONTENT_REF = /<ReasoningContent[\s\S]*?ref=\{reasoningContentRef\}/;
const SCOPED_SELECTION_CHECK =
  /selectionIntersectsElement\(\s*window\.getSelection\(\),\s*reasoningContentRef\.current,?\s*\)/g;

test("pagination defers only for a selection intersecting this reasoning block", () => {
  assert.match(REASONING, INTERSECTION_HELPER);
  assert.match(REASONING, CONTENT_REF);
  assert.equal(REASONING.match(SCOPED_SELECTION_CHECK)?.length, 2);
});
