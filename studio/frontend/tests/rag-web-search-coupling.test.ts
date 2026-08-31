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

test("a project-only scope pre-retrieves under Auto whatever the model size", () => {
  // The Search pill is off in this case, so nothing else would ground the answer.
  for (const checkpoint of CHECKPOINTS) {
    assert.equal(resolveRagAutoinject("auto", checkpoint, true), true);
  }
});

test("an explicit Auto-retrieve Off is never overridden", () => {
  // The control's own copy says "On and Off force it either way", and its threshold
  // slider is disabled when Off, so a forced retrieval could be neither stopped nor
  // tuned. Off is a user decision, not a default to improve on.
  for (const checkpoint of CHECKPOINTS) {
    for (const projectOnlyScope of [true, false]) {
      assert.equal(
        resolveRagAutoinject("off", checkpoint, projectOnlyScope),
        false,
        `off must win for ${checkpoint} projectOnly=${projectOnlyScope}`,
      );
    }
  }
});

test("mixed and KB scopes fall through to the size heuristic", () => {
  assert.equal(resolveRagAutoinject("auto", LARGE, false), false);
  assert.equal(resolveRagAutoinject("auto", SMALL, false), true);
  assert.equal(resolveRagAutoinject("auto", UNKNOWN, false), true);
});

test("On forces retrieval for every scope and model", () => {
  for (const checkpoint of CHECKPOINTS) {
    for (const projectOnlyScope of [true, false]) {
      assert.equal(
        resolveRagAutoinject("on", checkpoint, projectOnlyScope),
        true,
      );
    }
  }
});

test("every emitted autoinject is a boolean, never a mode string", () => {
  // The deep-research request builder used to put the raw mode on the wire, where the
  // backend read "off" as a non-empty string and enabled retrieval.
  for (const mode of ["auto", "on", "off"] as const) {
    for (const checkpoint of CHECKPOINTS) {
      for (const projectOnlyScope of [true, false]) {
        assert.equal(
          typeof resolveRagAutoinject(mode, checkpoint, projectOnlyScope),
          "boolean",
        );
      }
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

test("mixed thread/project scopes preserve Auto-retrieve Off", () => {
  const projectOnlyChecks = SOURCE.match(
    /projectRagEnabled &&\s+(?:ragProjectId|researchProjectId) &&\s+!(?:runtime\.)?ragEnabled/g,
  );
  assert.equal(projectOnlyChecks?.length, 3);
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
