// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// The module under test resolves the way vite and tsconfig resolve, not the way
// bare node does.
register("./bundler-resolver.mjs", import.meta.url);

const { modelStorageKey, splitQuantSuffix } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);

// Every answer below is the one split_quant_suffix in
// studio/backend/utils/openai_auto_switch_settings.py gives for the same key. The
// backfill folds a stored key with this before comparing it against the server's,
// so a suffix this splits and the backend does not collapses two models onto one
// key on the browser side only.
const CASES: [string, [string, string] | null][] = [
  // A known quant label, with and without the optional bpw modifier.
  ["org/Repo-GGUF:Q4_K_M", ["org/Repo-GGUF", "Q4_K_M"]],
  ["org/Repo-GGUF:IQ4_XS-3.53bpw", ["org/Repo-GGUF", "IQ4_XS-3.53bpw"]],
  ["org/Repo-GGUF:UD-Q4_K_XL", ["org/Repo-GGUF", "UD-Q4_K_XL"]],
  // A .gguf with no quant token in its name is labelled by its stem, and storage
  // lowercases the label while the scanner keeps the filename's casing.
  ["/models/CustomModel.gguf:custommodel", ["/models/CustomModel.gguf", "custommodel"]],
  ["/models/CustomModel.gguf:CustomModel", ["/models/CustomModel.gguf", "CustomModel"]],
  ["C:\\models\\CustomModel.gguf:custommodel", ["C:\\models\\CustomModel.gguf", "custommodel"]],
  // A shard suffix is not part of the label.
  [
    "/models/Custom-00001-of-00003.gguf:custom",
    ["/models/Custom-00001-of-00003.gguf", "custom"],
  ],
  ["/models/Custom-00001-of-00003.gguf:custom-00001-of-00003", null],
  // An extensionless .gguf still has a label.
  ["/models/.gguf:gguf", ["/models/.gguf", "gguf"]],
  // A quant token inside the filename wins over the stem.
  ["/models/tinyllama-Q4_K_M.gguf:q4_k_m", ["/models/tinyllama-Q4_K_M.gguf", "q4_k_m"]],
  ["/models/tinyllama-Q4_K_M.gguf:tinyllama-q4_k_m", null],
  // Only the basename is labelled, never the directories above it.
  [
    "/models/dir/CustomModel.gguf:custommodel",
    ["/models/dir/CustomModel.gguf", "custommodel"],
  ],
  ["/models/dir/CustomModel.gguf:dir/custommodel", null],
  // A colon is legal in a POSIX filename. Neither of these is a variant, and
  // reading them as one folds two real files onto a single key.
  ["/models/foo:Bar.gguf", null],
  ["/models/foo:bar.gguf", null],
  ["/models/llama.gguf:Bar.gguf", null],
  ["/models/llama.gguf:bar.gguf", null],
  ["/models/CustomModel.gguf:othermodel", null],
  ["/models/model.gguf:notalabel", null],
  ["/models/plain.gguf:plain:extra", null],
  // A Windows drive letter is not a separator either.
  ["C:\\models\\foo.gguf", null],
  ["C:/models/foo.gguf", null],
  // Nothing to split.
  ["org/Repo-GGUF", null],
  ["/models/foo.gguf", null],
  ["org/Repo:", null],
  [":Q4_K_M", null],
];

test("splitQuantSuffix answers exactly as the backend's split_quant_suffix", () => {
  for (const [value, expected] of CASES) {
    assert.deepEqual(splitQuantSuffix(value), expected, value);
  }
});

test("a .gguf filename carrying a colon is not folded into a variant", () => {
  // Two real, distinct files: POSIX allows a colon in a name and is case
  // sensitive, so the one-time backfill has to keep their settings apart. The
  // variant half of an override key is stored lowercased, so folding these makes
  // one key and strands whichever file the backfill reaches second.
  const upper = "/models/llama.gguf:Bar.gguf";
  const lower = "/models/llama.gguf:bar.gguf";
  assert.equal(splitQuantSuffix(upper), null);
  assert.equal(splitQuantSuffix(lower), null);
  assert.notEqual(modelStorageKey(upper, null), modelStorageKey(lower, null));
});
