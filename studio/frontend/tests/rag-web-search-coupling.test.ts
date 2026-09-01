// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #9947: the Search pill gates web_search, not the attached documents. The policy tests
// run the production function rather than matching its source text, because the bug
// being guarded against is a wrong value on the wire, which reading the source cannot
// see.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveRagAutoinject } = await import(
  "../src/features/chat/api/rag-autoinject.ts"
);

// 9B is the Auto cutoff, so these bracket it on both sides plus the unknown-size case.
const SMALL = "Qwen3.8-9B-Distill-uncensored-heretic.Q6_K.gguf";
const LARGE = "Llama-3.3-70B-Instruct";
const UNKNOWN = "some-local-checkpoint";
const CHECKPOINTS = [SMALL, LARGE, UNKNOWN];

test("Auto honours the size cutoff, including for project-only scopes", () => {
  // An earlier revision forced pre-retrieval on for a project-only scope at every size,
  // so the Search pill could not block grounding. Measured against main over 1260
  // generations, that override only ever fired above the cutoff -- below it Auto is
  // already true -- and above the cutoff it cost a redundant second retrieval on 26.7%
  // of turns and dropped distractor disambiguation from 90% to 37%, buying no accuracy.
  assert.equal(resolveRagAutoinject("auto", SMALL), true);
  assert.equal(resolveRagAutoinject("auto", UNKNOWN), true);
  assert.equal(resolveRagAutoinject("auto", LARGE), false);
});

test("an explicit Auto-retrieve Off is never overridden", () => {
  // The control's own copy says "On and Off force it either way", and its threshold
  // slider is disabled when Off, so a forced retrieval could be neither stopped nor
  // tuned. Off is a user decision, not a default to improve on.
  for (const checkpoint of CHECKPOINTS) {
    assert.equal(
      resolveRagAutoinject("off", checkpoint),
      false,
      `off must win for ${checkpoint}`,
    );
  }
});

test("On forces retrieval for every model", () => {
  for (const checkpoint of CHECKPOINTS) {
    assert.equal(resolveRagAutoinject("on", checkpoint), true);
  }
});

test("every emitted autoinject is a boolean, never a mode string", () => {
  // The deep-research request builder used to put the raw mode on the wire, where the
  // backend read "off" as a non-empty string and enabled retrieval.
  for (const mode of ["auto", "on", "off"] as const) {
    for (const checkpoint of CHECKPOINTS) {
      assert.equal(typeof resolveRagAutoinject(mode, checkpoint), "boolean");
    }
  }
});

// The request bodies have no importable seam: they are built inline inside the adapter,
// whose module graph cannot be loaded here. Asserted against the source, like the other
// chat-adapter tests.
const SOURCE = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

test("the adapter no longer derives a project-only auto-inject override", () => {
  // The override is gone from the policy, so an adapter still computing the flag would
  // be dead code that a future edit could wire back up without going through a test.
  assert.doesNotMatch(
    SOURCE,
    /projectRagEnabled &&\s+(?:ragProjectId|researchProjectId) &&\s+!(?:runtime\.)?ragEnabled/,
  );
});

test("local enabled_tools still lists search_knowledge_base without web_search", () => {
  const start = SOURCE.indexOf("...(ragEnabled || projectRagEnabled");
  assert.ok(start > 0, "enabled_tools list moved");
  const slice = SOURCE.slice(start, start + 400);
  assert.match(slice, /search_knowledge_base/);
  assert.match(slice, /toolsEnabled \? \["web_search"\]/);
});

test("every rag_scope the adapter builds routes through the shared policy", () => {
  // Four call sites: two chat bodies and the deep-research KB / non-KB branches. A new
  // one that inlines its own rule is how the deep-research string leak happened.
  const calls = SOURCE.match(/autoinject: resolveRagAutoinject\(/g) ?? [];
  assert.equal(calls.length, 4);
  assert.doesNotMatch(SOURCE, /autoinject: runtime\.ragAutoInject/);
  assert.doesNotMatch(SOURCE, /autoinject: ragAutoInject\b/);
});
