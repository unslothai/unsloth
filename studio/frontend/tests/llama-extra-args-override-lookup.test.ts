// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which stored row the picker reads its pass-through arguments from.
//
// This has to agree with the server, because the two act on the same data from
// different ends: the panel hydrates from a row and then sends what it found as an
// EXPLICIT list, while an API auto-switch resolves the row itself. Where they
// disagree, a model launches with one set of flags from the picker and another from
// the API, which is the kind of difference nobody thinks to look for.
//
// The rules mirrored here are resolve_model_override_key and _folded_override_matches
// in utils/openai_auto_switch_settings.py.

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { resolveStoredExtraArgs } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);

const ARGS = ["--numa", "distribute"];

test("an exact key wins", () => {
  assert.deepEqual(
    resolveStoredExtraArgs(
      { "unsloth/Model-GGUF:q4_k_m": { llama_extra_args: ARGS } },
      ["unsloth/Model-GGUF:q4_k_m"],
    ),
    ARGS,
  );
});

test("a repo id and its quant fold by case", () => {
  // The browser lowercases the quant before storing, so a row written with the
  // upstream spelling still has to be found.
  assert.deepEqual(
    resolveStoredExtraArgs(
      { "unsloth/Model-GGUF:Q4_K_M": { llama_extra_args: ARGS } },
      ["unsloth/model-gguf:q4_k_m"],
    ),
    ARGS,
  );
});

test("a POSIX path stays case-sensitive", () => {
  // /models/Foo.gguf and /models/foo.gguf are two real files, and folding them
  // would replay one model's arguments on the other.
  assert.deepEqual(
    resolveStoredExtraArgs({ "/models/Foo.gguf": { llama_extra_args: ARGS } }, [
      "/models/foo.gguf",
    ]),
    [],
  );
});

test("a colon inside a POSIX filename is not a quant", () => {
  assert.deepEqual(
    resolveStoredExtraArgs(
      { "/models/foo:bar.gguf": { llama_extra_args: ARGS } },
      ["/models/foo:Bar.gguf"],
    ),
    [],
  );
});

test("a Windows path folds", () => {
  assert.deepEqual(
    resolveStoredExtraArgs(
      { "C:\\Models\\Foo.gguf": { llama_extra_args: ARGS } },
      ["c:\\models\\foo.gguf"],
    ),
    ARGS,
  );
});

test("a separator and a trailing slash do not make a different key", () => {
  // _fold_case_insensitive_path replaces backslashes, then trims trailing
  // separators down to the root, so all of these name one file to the server.
  for (const key of [
    "C:\\Models\\Foo.gguf",
    "c:/models/foo.gguf",
    "C:/Models/Foo.gguf/",
  ]) {
    assert.deepEqual(
      resolveStoredExtraArgs({ [key]: { llama_extra_args: ARGS } }, [
        "c:\\models\\foo.gguf",
      ]),
      ARGS,
      key,
    );
  }
});

test("a UNC share folds however it is spelled", () => {
  // Written with forward slashes it still starts "//", which is the shape the
  // server tests; reading it as an ordinary POSIX path made it case-sensitive.
  assert.deepEqual(
    resolveStoredExtraArgs(
      { "\\\\Server\\Share\\Foo.gguf": { llama_extra_args: ARGS } },
      ["//server/share/foo.gguf"],
    ),
    ARGS,
  );
});

test("a WSL drive mount folds like the Windows volume it is", () => {
  // _fold_case_insensitive_path treats /mnt/<letter> as a Windows path, because it
  // is one seen through Linux. Leaving it under the POSIX rule stranded an override
  // the server does apply, and a cold picker load then omitted the arguments.
  assert.deepEqual(
    resolveStoredExtraArgs(
      { "/mnt/c/models/foo.gguf": { llama_extra_args: ARGS } },
      ["/mnt/C/Models/Foo.gguf"],
    ),
    ARGS,
  );
  // Not every /mnt path: /mnt/storage is an ordinary POSIX mount point.
  assert.deepEqual(
    resolveStoredExtraArgs(
      { "/mnt/storage/models/foo.gguf": { llama_extra_args: ARGS } },
      ["/mnt/storage/models/Foo.gguf"],
    ),
    [],
  );
});

test("two keys that fold together resolve to nothing", () => {
  // resolve_model_override_key returns None here on purpose: picking one of them at
  // enumeration order applies another model's settings half the time.
  assert.deepEqual(
    resolveStoredExtraArgs(
      {
        "unsloth/Model-GGUF": { llama_extra_args: ARGS },
        "unsloth/model-gguf": { llama_extra_args: ["--top-k", "20"] },
      },
      ["unsloth/MODEL-gguf"],
    ),
    [],
  );
});

test("the first entry that exists is the one read, fields and all", () => {
  // The auto-switch loader breaks on the first non-empty override and reads its
  // fields from there. Falling through to the bare repo because the variant row
  // happens to carry no arguments would launch the picker with flags an API load
  // would not use.
  assert.deepEqual(
    resolveStoredExtraArgs(
      {
        "unsloth/model-gguf:q4_k_m": { max_seq_length: 4096 },
        "unsloth/model-gguf": { llama_extra_args: ARGS },
      },
      ["unsloth/model-gguf:q4_k_m", "unsloth/model-gguf"],
    ),
    [],
  );
});

test("an empty entry is skipped rather than stopping the search", () => {
  // `if override: break` on the server: a row with no fields is not a match.
  assert.deepEqual(
    resolveStoredExtraArgs(
      {
        "unsloth/model-gguf:q4_k_m": {},
        "unsloth/model-gguf": { llama_extra_args: ARGS },
      },
      ["unsloth/model-gguf:q4_k_m", "unsloth/model-gguf"],
    ),
    ARGS,
  );
});

test("no row at all is no arguments, not an error", () => {
  assert.deepEqual(resolveStoredExtraArgs({}, ["unsloth/model-gguf"]), []);
});
