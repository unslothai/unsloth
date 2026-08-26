// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { routableToMediaPage } from "../src/features/hub/lib/local-path.ts";
import { studioPageForTask } from "../src/features/hub/lib/unsloth-support.ts";

// What the inspector must decide before enabling Run, and what runSelectedModel requires
// before navigating. They have to agree: the handler falls through to the chat loader for a
// row it cannot route, and that unloads the resident model for a load that can only fail.
const runsOnMediaPage = (
  task: string | null | undefined,
  kind: "discover" | "cache" | "local",
  localSource?: string | null,
) => studioPageForTask(task) !== undefined && routableToMediaPage(kind, localSource);

// The backend tags a local non-GGUF diffusers checkpoint text-to-image (_local_model_task),
// and it does that for every local row whatever its source, so these rows are real.
test("a filesystem diffusion row never counts as running on a media page", () => {
  assert.equal(runsOnMediaPage("text-to-image", "local", "models_dir"), false);
  assert.equal(runsOnMediaPage("text-to-image", "local", "lmstudio"), false);
  assert.equal(runsOnMediaPage("text-to-image", "local", "ollama"), false);
  assert.equal(runsOnMediaPage("text-to-video", "local", "custom"), false);
  assert.equal(runsOnMediaPage("text-to-image", "local", undefined), false);
});

test("hub-backed diffusion rows stay runnable on their page", () => {
  // An hf_cache row is a complete Hub snapshot, so it routes like a cached repo.
  assert.equal(runsOnMediaPage("text-to-image", "local", "hf_cache"), true);
  assert.equal(runsOnMediaPage("text-to-image", "cache"), true);
  assert.equal(runsOnMediaPage("image-text-to-video", "discover"), true);
});

test("a chat task is unaffected by the media gate", () => {
  assert.equal(runsOnMediaPage("text-generation", "cache"), false);
  assert.equal(runsOnMediaPage(null, "discover"), false);
  assert.equal(runsOnMediaPage("text-generation", "local", "hf_cache"), false);
});
