// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The bar abstains when a pass-through arg means the estimate no longer
// describes the launch. That policy is only as good as its spelling list, and
// llama.cpp accepts several spellings for most of these flags.
//
// Three review rounds in a row found an alias missing here -- -dev, --cpu-moe,
// --draft-max, -ctkd, --swa-checkpoints -- and each was fixed by appending one
// more string, which is a fix that lasts until the next alias. So this reads the
// backend's own frozensets, which are what the launch actually honours, and
// fails when one of them contains a flag the frontend would not act on.
//
// It deliberately does NOT require the two sides to be equal. The frontend lists
// are a superset by design: they also carry flags no single backend set groups
// together. The invariant is one-directional -- everything the backend parses in
// these groups must be recognised here.

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  PLACEMENT_OWNING_ARGS,
  KV_SHAPING_ARGS,
  RESIDENT_ADDING_ARGS,
  extraArgsOwnPlacement,
  extraArgsShapeKvCache,
  extraArgsAddResidentFiles,
} = await import("../src/lib/model-memory.ts");

const BACKEND = new URL("../../backend/", import.meta.url);

/**
 * The string literals of a `frozenset({...})` assignment in the backend source.
 *
 * Parsed rather than imported because there is no Python runtime here, and
 * duplicated rather than skipped because a test that silently passes when it
 * cannot find its subject is worse than no test: the miss it is guarding
 * against looks exactly like success.
 */
function frozensetLiterals(source: string, name: string): string[] {
  const start = source.indexOf(name);
  assert.ok(start >= 0, `${name} is gone from the backend; update this test`);
  const open = source.indexOf("frozenset(", start);
  assert.ok(open >= 0, `${name} is no longer a frozenset literal`);
  const close = source.indexOf(")", open);
  const body = source.slice(open, close);
  const found = [...body.matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  assert.ok(found.length > 0, `${name} parsed as empty`);
  return found;
}

const SERVER_ARGS = readFileSync(
  new URL("core/inference/llama_server_args.py", BACKEND),
  "utf8",
);
const LLAMA_CPP = readFileSync(
  new URL("core/inference/llama_cpp.py", BACKEND),
  "utf8",
);

test("every placement spelling the launch honours makes the bar abstain", () => {
  const groups = [
    ["_DEVICE_FLAGS", SERVER_ARGS],
    ["_GPU_LAYER_FLAGS", SERVER_ARGS],
    ["_MOE_OFFLOAD_FLAGS", SERVER_ARGS],
    ["_DRAFT_GPU_LAYER_FLAGS", LLAMA_CPP],
    ["_MMPROJ_OFFLOAD_FLAGS", LLAMA_CPP],
  ] as const;
  for (const [name, source] of groups) {
    for (const flag of frozensetLiterals(source, name)) {
      assert.ok(
        PLACEMENT_OWNING_ARGS.includes(flag),
        `${flag} (${name}) is parsed by the launch but not by PLACEMENT_OWNING_ARGS, ` +
          `so a config using that spelling is charted against a budget it does not use`,
      );
    }
  }
});

test("every draft-memory spelling the launch honours makes the bar abstain", () => {
  const groups = [
    "_SPEC_DRAFT_CACHE_K_FLAGS",
    "_SPEC_DRAFT_CACHE_V_FLAGS",
    "_CTX_CHECKPOINTS_FLAGS",
  ];
  for (const name of groups) {
    for (const flag of frozensetLiterals(SERVER_ARGS, name)) {
      assert.ok(
        KV_SHAPING_ARGS.includes(flag),
        `${flag} (${name}) resizes the cache at launch but is not in KV_SHAPING_ARGS`,
      );
    }
  }
});

test("the batch and ubatch spellings are covered", () => {
  for (const name of ["_BATCH_FLAGS", "_UBATCH_FLAGS"]) {
    for (const flag of frozensetLiterals(SERVER_ARGS, name)) {
      assert.ok(
        KV_SHAPING_ARGS.includes(flag),
        `${flag} (${name}) scales the compute buffers but is not in KV_SHAPING_ARGS`,
      );
    }
  }
});

test("the three policies stay disjoint in intent", () => {
  // A flag that both moves the load and resizes its cache would be ambiguous
  // about why the bar abstained. They abstain identically today, so this is
  // about keeping the reasons legible rather than about behaviour.
  const resident = new Set(RESIDENT_ADDING_ARGS);
  for (const flag of PLACEMENT_OWNING_ARGS) {
    assert.ok(
      !resident.has(flag),
      `${flag} is classified as both placement-owning and resident-adding`,
    );
  }
});

test("every drafter-selecting spelling makes the bar abstain", () => {
  // A hand-named drafter is weights plus a cache nothing here priced, and the
  // flags are last-wins, so any accepted spelling means the launch opens one.
  for (const name of ["_LOCAL_DRAFT_FLAGS", "_HF_DRAFT_FLAGS"]) {
    for (const flag of frozensetLiterals(LLAMA_CPP, name)) {
      assert.ok(
        RESIDENT_ADDING_ARGS.includes(flag),
        `${flag} (${name}) selects a drafter at launch but is not in RESIDENT_ADDING_ARGS`,
      );
    }
  }
});

test("underscore spellings are classified like their dashed twins", () => {
  // llama.cpp accepts both, and the backend normalises underscores to dashes
  // before parsing. Comparing raw tokens let a single spelling slip past ALL
  // THREE predicates at once, which is a hole in the policy rather than a
  // missing entry in one list -- so this checks the normalisation, not a list.
  const cases: [string, (a: string[]) => boolean][] = [
    ["--gpu_layers", extraArgsOwnPlacement],
    ["--ctx_size", extraArgsShapeKvCache],
    ["--spec_draft_hf", extraArgsAddResidentFiles],
  ];
  for (const [flag, predicate] of cases) {
    assert.ok(predicate([flag]), `${flag} was not recognised`);
    assert.ok(predicate([`${flag}=4`]), `${flag}=4 was not recognised`);
    // The dashed twin must still work, so normalisation did not replace one
    // spelling with the other.
    const dashed = flag.replace(/_/g, "-");
    assert.ok(predicate([dashed]), `${dashed} stopped being recognised`);
  }
  // A short option keeps its underscores: only long options are normalised, so
  // this must not start matching things that are not flags at all.
  assert.equal(extraArgsOwnPlacement(["not_a_flag"]), false);
  assert.equal(extraArgsOwnPlacement(["--temp", "0.7"]), false);
});
