// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The arm `residentModelMatchesPick` takes when the status reports no `model_identifier`.
 *
 * The raw-identifier arm above settles which revision is resident. This one has no raw
 * identifier to compare, so all that is left is a public id every snapshot of the repo
 * shares, and adopting on it leaves the user on weights they did not pick.
 *
 * Reachable from a backend predating the field, and from a native lease, which withholds
 * the raw path by design. Found by fuzzing this function against the server's own
 * already-loaded test, `LlamaCppBackend.matches_load_source`.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { residentModelMatchesPick } = await import(
  "../src/features/chat/lib/resident-model-match.ts"
);

const REPO = "unsloth/Qwen3-0.6B-GGUF";
const Q = "Q4_K_M";
const RESIDENT_SHA = "50968a4468ef4233ed78cd7c3de230dd1d61a56b";
const NEWER_SHA = "d7f544eead698dbd1f15126ef60b45a1e1933222";

/** The cache dir holding one repo, spelled the way each host spells it. */
const CACHE_ROOTS: Record<string, string> = {
  linux:
    "/home/dev/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/",
  // A second disk at /mnt/<letter> is ordinary on Linux and indistinguishable from WSL
  // to a string comparator, so it is its own host here.
  linuxMounted: "/mnt/d/hf/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/",
  mac: "/Users/dev/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/",
  wsl: "/mnt/c/Users/dev/hf/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/",
  windows: "D:\\hf\\hub\\models--unsloth--Qwen3-0.6B-GGUF\\snapshots\\",
  windowsForwardSlashes:
    "D:/hf/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/",
  unc: "\\\\nas\\hf\\hub\\models--unsloth--Qwen3-0.6B-GGUF\\snapshots\\",
};

for (const [host, root] of Object.entries(CACHE_ROOTS)) {
  const resident = `${root}${RESIDENT_SHA}`;
  const newer = `${root}${NEWER_SHA}`;

  test(`[${host}] a status with no raw identifier does not adopt a different snapshot`, () => {
    assert.equal(
      residentModelMatchesPick(
        // What an install predating model_identifier publishes: the repo id, the same
        // string for every revision in the cache.
        { active_model: REPO, gguf_variant: Q },
        { id: REPO, loadPath: newer, ggufVariant: Q },
      ),
      false,
    );
  });

  test(`[${host}] the same shape still adopts the snapshot that is actually resident`, () => {
    // The control: without it the test above passes for a function that never adopts.
    assert.equal(
      residentModelMatchesPick(
        { active_model: resident, gguf_variant: Q },
        { id: REPO, loadPath: resident, ggufVariant: Q },
      ),
      true,
    );
  });

  test(`[${host}] a null identifier is treated the same as an absent one`, () => {
    assert.equal(
      residentModelMatchesPick(
        { active_model: REPO, model_identifier: null, gguf_variant: Q },
        { id: REPO, loadPath: newer, ggufVariant: Q },
      ),
      false,
    );
  });

  test(`[${host}] the raw identifier, when present, still settles it`, () => {
    assert.equal(
      residentModelMatchesPick(
        { active_model: REPO, model_identifier: resident, gguf_variant: Q },
        { id: REPO, loadPath: resident, ggufVariant: Q },
      ),
      true,
    );
    assert.equal(
      residentModelMatchesPick(
        { active_model: REPO, model_identifier: resident, gguf_variant: Q },
        { id: REPO, loadPath: newer, ggufVariant: Q },
      ),
      false,
    );
  });
}

test("an unpinned pick is its own load id, so the public id still answers", () => {
  // No loadPath: the repo id is what the server was given, so there is no revision the
  // comparison could be wrong about.
  assert.equal(
    residentModelMatchesPick(
      { active_model: REPO, gguf_variant: Q },
      { id: REPO, ggufVariant: Q },
    ),
    true,
  );
});

test("a native lease keeps matching on the label it was granted under", () => {
  // What the public-id arm exists for: a bare label is not a cache snapshot, so the
  // refusal above must not reach it.
  assert.equal(
    residentModelMatchesPick(
      { active_model: "Qwen3-0.6B-Q4_K_M.gguf", model_identifier: null },
      { id: "Qwen3-0.6B-Q4_K_M.gguf" },
    ),
    true,
  );
});

test("a standalone .gguf is not adopted on its stem alone", () => {
  // Two directories can each hold a Qwen3-0.6B-Q4_K_M.gguf, so the stem is shareable and
  // residentModelIdMatches takes its public pass only for a namespaced repo id.
  assert.equal(
    residentModelMatchesPick(
      { active_model: "Qwen3-0.6B-Q4_K_M", gguf_variant: Q },
      { id: "/srv/models/Qwen3-0.6B-Q4_K_M.gguf" },
    ),
    false,
  );
  // With the raw identifier there is nothing to guess about, and it adopts.
  assert.equal(
    residentModelMatchesPick(
      {
        active_model: "Qwen3-0.6B-Q4_K_M",
        model_identifier: "/srv/models/Qwen3-0.6B-Q4_K_M.gguf",
        gguf_variant: Q,
      },
      { id: "/srv/models/Qwen3-0.6B-Q4_K_M.gguf" },
    ),
    true,
  );
});

test("a models-- dir that is not the cache layout is not treated as a snapshot", () => {
  // hf_cache_repo_id only recognises models--*/snapshots/*, so a blobs dir has no revision
  // to be wrong about and the public-id arm still applies.
  assert.equal(
    residentModelMatchesPick(
      { active_model: "unsloth/Qwen3-0.6B-GGUF", gguf_variant: Q },
      {
        id: "/hf/hub/models--unsloth--Qwen3-0.6B-GGUF/blobs/abc",
        ggufVariant: Q,
      },
    ),
    false,
  );
});
