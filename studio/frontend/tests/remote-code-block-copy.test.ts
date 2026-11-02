// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { blockCopy } from "../src/features/security/lib/block-copy.ts";
import { readSrc } from "./helpers/kit.ts";

test("only a scan finding is described as CRITICAL", () => {
  assert.equal(
    blockCopy("critical_findings", "ModelScope", false).note,
    "remoteCode.blockedNote",
  );
  const others = ["unsafe_files", "unscanned_weights", "managed_refused", "not_on_source", "unscannable", null] as const;
  for (const cause of others) {
    const copy = blockCopy(cause, "ModelScope", false);
    assert.notEqual(copy.note, "remoteCode.blockedNote");
    assert.notEqual(copy.title, "remoteCode.blockedTitle");
  }
});

test("each cause names its source's way forward", () => {
  const body = (source: string) => blockCopy("not_on_source", source, true).body;
  assert.equal(body("ModelScope"), "remoteCode.notOnModelScopeBody");
  assert.equal(body("Hugging Face"), "remoteCode.notFoundBody");
  const note = (cause: "unscanned_weights" | "unscannable", source: string) =>
    blockCopy(cause, source, false).note;
  assert.equal(note("unscanned_weights", "ModelScope"), "remoteCode.switchSourceNote");
  assert.equal(note("unscanned_weights", "Hugging Face"), "remoteCode.reconnectNote");
  assert.equal(note("unscannable", "ModelScope"), "remoteCode.switchSourceNote");
  assert.equal(note("unscannable", "Hugging Face"), undefined);
  assert.equal(
    blockCopy("unsafe_files", "Hugging Face", true).body,
    "remoteCode.unsafeBodyFromProvider",
  );
});

test("the dialog takes its blocked copy from the cause and the blocked repo", () => {
  const dialog = readSrc("features/security/components/remote-code-consent-dialog.tsx");
  assert.match(dialog, /blockCopy\(scan\?\.blockCause/);
  assert.match(dialog, /scan\?\.blockTarget \?\? scan\?\.modelName/);
});
