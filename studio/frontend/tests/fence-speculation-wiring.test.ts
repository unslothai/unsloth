// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const DEFER = readSrc("components/assistant-ui/code-fence-defer.tsx");
const MARKDOWN = readSrc("components/assistant-ui/markdown-text.tsx");
const THREAD = readSrc("components/assistant-ui/thread.tsx");
const PLUGIN = readSrc("components/assistant-ui/code-plugin.ts");

test("the speculator seeds the same cache key the demanded path reads, and only fences the main thread has warmed", () => {
  assert.match(DEFER, /eligible: \(gate\) => grammarsWarmed\.has\(grammarOf\(gate\)\),/);
  assert.match(DEFER, /enabled: \(\) =>\s*fenceMode\(\) !== "off" && highlightWorkerState\(\) !== "unavailable",/);
  assert.match(DEFER, /busy: \(\) => streamActive\(\) \|\| highlightWorkerState\(\) === "stalled",/);
  assert.match(DEFER, /maxChars: MAX_HIGHLIGHT_CHARS,\s*budgetChars: MAX_CACHED_CHARACTERS \/ 2,\s*budgetFences: MAX_FENCES \/ 2,/);
  assert.match(DEFER, /budgetFences: MAX_FENCES \/ 2,\s*maxInFlight: SPECULATION_IN_FLIGHT,/, "a few requests stay queued in the worker so speculation is not paced by main-thread idle slots");
  assert.match(DEFER, /const SPECULATION_IN_FLIGHT = 3;/);
  assert.match(DEFER, /busy: \(\) => streamActive\(\) \|\| highlightWorkerState\(\) === "stalled",/, "a worker that is late pauses speculation instead of dropping every candidate, so it resumes on the ready message");
  assert.match(DEFER, /fenceMode\(\) !== "off" && highlightWorkerState\(\) !== "unavailable",/, "only a worker that failed turns speculation off");
  assert.match(DEFER, /const rect = gate\.node\.getBoundingClientRect\(\);/);
  assert.equal(DEFER.includes("gate.anchor"), false);
  const speculate = MARKDOWN.slice(MARKDOWN.indexOf("const speculate = useCallback("), MARKDOWN.indexOf("const reached = useFenceReached("));
  assert.ok(speculate.indexOf("code.cached(options) !== null") < speculate.indexOf("requestFullHighlight("));
  assert.match(speculate, /requestFullHighlight\(\s*body,\s*languageToken,\s*\(result\) => \{\s*if \(result !== null\) code\.seed\(options, result\);\s*settle\(result !== null\);\s*\},\s*true,\s*\);/);
  assert.match(MARKDOWN, /useFenceReached\(\s*host,\s*mode !== "off",\s*Boolean\(isIncomplete\),[\s\S]{0,200}?trimmedLength\(source\),\s*source,\s*warm,\s*speculate,\s*\);/);
});

test("every gate joins and leaves the speculator with its registration, and a latch withdraws it first", () => {
  assert.match(DEFER, /scheduleGrammarWarm\(\);\s*fenceSpeculator\.add\(registered\);/);
  assert.match(DEFER, /fenceSpeculator\.remove\(registered\);\s*unreached\.delete\(registered\);/);
  assert.match(DEFER, /unreached\.delete\(gate\);\s*fenceSpeculator\.remove\(gate\);\s*gate\.warm\(true\);/);
  assert.match(DEFER, /latchNow\(arrived\);\s*fenceSpeculator\.rerank\(\);/);
  assert.match(PLUGIN, /const reachInto = \(fence: Fence, code: string\): number => \{\s*if \(fence\.result === null \|\| fence\.seeded\) return -1;/, "a seeded entry is an exact hit only, never a prefix anchor");
  assert.match(PLUGIN, /export const MAX_FENCES = 512;\s*export const MAX_CACHED_CHARACTERS = 512_000;/);
});

test("the grammar pre-warm runs once per session from the thread root, idle and quiet only, over untrimmed samples", () => {
  assert.match(THREAD, /let grammarPrewarm: GrammarPrewarm \| null = null;/);
  assert.match(THREAD, /if \(grammarPrewarm !== null\) return;/);
  assert.match(THREAD, /\(code, language, late\) => highlightFenceSource\(code, language, late\),/);
  assert.match(THREAD, /quiet: \(\) =>\s*!streamActive\(\)\s*&& performance\.now\(\) - lastInputAt >= PREWARM_INPUT_QUIET_MS,/);
  assert.match(THREAD, /skip: grammarWarmed,/);
  assert.match(THREAD, /useEffect\(\(\) => \{\s*startGrammarPrewarm\(\);\s*\}, \[\]\);/);
  assert.match(MARKDOWN, /export const highlightFenceSource = \([\s\S]*?\): FenceTokens \| null =>\s*code\.highlight\(fenceHighlightOptions\(body, language\), late\);/);
});
