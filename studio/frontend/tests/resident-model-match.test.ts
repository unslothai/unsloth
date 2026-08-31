// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * #8893: picking the model that is already loaded raised "Stop 1 running chat?" and reloaded
 * it. The pick and the status name one model with different strings -- a cached row pinned to
 * a snapshot dir loads by path while its picker row keeps the repo id -- so comparing the row
 * id against the status checkpoint alone read a resident model as a different one.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { residentModelMatchesPick } = await import(
  "../src/features/chat/lib/resident-model-match.ts"
);

const REPO_ID = "unsloth/Qwen3.5-9B-GGUF";
const SNAPSHOT =
  "D:\\models\\hub\\models--unsloth--Qwen3.5-9B-GGUF\\snapshots\\a1b2c3";

/** What /api/inference/status publishes for a pinned cached row: the clean public id next to
 * the raw path the load actually ran as. */
const pinnedStatus = {
  active_model: REPO_ID,
  model_identifier: SNAPSHOT,
  gguf_variant: "Q4_K_M",
};

test("the picker row id names the resident model behind a snapshot path", () => {
  assert.equal(
    residentModelMatchesPick(pinnedStatus, {
      id: REPO_ID,
      loadPath: SNAPSHOT,
      ggufVariant: "Q4_K_M",
    }),
    true,
  );
});

test("the load path alone names the resident model", () => {
  assert.equal(
    residentModelMatchesPick(
      { active_model: REPO_ID, model_identifier: SNAPSHOT },
      { id: "Qwen3.5-9B", loadPath: SNAPSHOT },
    ),
    true,
  );
});

// windows reports the same directory under either separator and either case
test("a path naming the same file matches whatever its separators", () => {
  assert.equal(
    residentModelMatchesPick(
      { active_model: SNAPSHOT, model_identifier: SNAPSHOT },
      { id: SNAPSHOT.replace(/\\/g, "/").toLowerCase() },
    ),
    true,
  );
});

test("a different quant of the same repo is a real reload", () => {
  assert.equal(
    residentModelMatchesPick(pinnedStatus, {
      id: REPO_ID,
      loadPath: SNAPSHOT,
      ggufVariant: "Q8_0",
    }),
    false,
  );
});

/** A local file load derives its quant label from the filename, so the status reports one the
 * picker row never carries. Comparing them left the reported bug in place for On Device files. */
test("a standalone .gguf matches the label the backend derived for it", () => {
  assert.equal(
    residentModelMatchesPick(
      {
        active_model: "/Users/dev/models/Qwen3-8B-Q4_K_M.gguf",
        model_identifier: "/Users/dev/models/Qwen3-8B-Q4_K_M.gguf",
        gguf_variant: "Q4_K_M",
      },
      { id: "/Users/dev/models/Qwen3-8B-Q4_K_M.gguf" },
    ),
    true,
  );
});

/** The exemption is for the file's own derived label, not for picking a different quant. */
test("a repo row still has to name the resident quant", () => {
  assert.equal(
    residentModelMatchesPick(pinnedStatus, {
      id: REPO_ID,
      ggufVariant: undefined,
    }),
    false,
  );
});

/** A repo that advanced to a newer snapshot reports the same public id for both, and the
 * inventory repoints load_id at the newest, so matching on the id alone kept stale weights. */
test("a newer snapshot of the resident repo is a real reload", () => {
  assert.equal(
    residentModelMatchesPick(pinnedStatus, {
      id: REPO_ID,
      loadPath: `${SNAPSHOT.slice(0, SNAPSHOT.lastIndexOf("\\"))}\\d4e5f6`,
      ggufVariant: "Q4_K_M",
    }),
    false,
  );
});

test("another model does not match the resident one", () => {
  assert.equal(
    residentModelMatchesPick(pinnedStatus, {
      id: "unsloth/gemma-4-12b-GGUF",
      ggufVariant: "Q4_K_M",
    }),
    false,
  );
});

test("nothing resident matches nothing", () => {
  assert.equal(
    residentModelMatchesPick(
      { active_model: null, model_identifier: SNAPSHOT },
      { id: REPO_ID, loadPath: SNAPSHOT },
    ),
    false,
  );
});

/** The rules above only help if the pick is tested before the confirmation is raised: the
 * dialog is what the report is about, and /load answers already_loaded without stopping
 * anything. */
test("selectModel checks residency before prompting to stop running chats", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/hooks/use-chat-model-runtime.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const residencyCheck = source.indexOf(
    "residentModelMatchesPick(status",
  );
  const confirmPrompt = source.indexOf(
    "await confirmStopRunningChatsIfNeeded(",
  );
  assert.ok(residencyCheck > 0, "selectModel no longer checks residency");
  assert.ok(confirmPrompt > 0, "selectModel no longer confirms running chats");
  assert.ok(residencyCheck < confirmPrompt);
  // gating the check on an external checkpoint is the shape that left #8893 open
  assert.doesNotMatch(
    source.slice(Math.max(0, residencyCheck - 500), residencyCheck),
    /isExternalModelId\(selectedCheckpoint\)/,
    "the residency check is gated on an external checkpoint again",
  );
});
