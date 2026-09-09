// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  editFileChangeLabel,
  editFileResultIsError,
  summarizeEditFileArgs,
} from "../src/components/assistant-ui/edit-file-tool-summary.ts";
import { canRememberToolApproval } from "../src/components/assistant-ui/tool-approval-policy.ts";

const EDIT_FILE_CARD_RE = /edit_file:\s*EditFileToolUIConfirmable/;
const UNSUPPORTED_BLOCKED_RE = /Unsupported operations stay blocked/;
const editWithReplaceAll = (replaceAll: unknown): Record<string, unknown> =>
  Object.fromEntries([["replace_all", replaceAll]]);

test("mutating local tools require approval for each call", () => {
  for (const toolName of ["edit_file", "python", "terminal"]) {
    assert.equal(canRememberToolApproval(toolName), false, toolName);
  }
  assert.equal(canRememberToolApproval("web_search"), true);
});

test("edit_file approval summary distinguishes create and replace requests", () => {
  const create = summarizeEditFileArgs(
    JSON.parse(
      '{"path":"src/new.ts","edits":[{"old_string":"","new_string":"export {};"}]}',
    ),
  );
  assert.deepEqual(create, {
    path: "src/new.ts",
    editCount: 1,
    mode: "create",
    replaceAllCount: 0,
  });
  assert.equal(editFileChangeLabel(create), "Create file");

  const replace = summarizeEditFileArgs(
    JSON.parse(
      '{"path":"src/existing.ts","edits":[{"old_string":"one","new_string":"two"},{"old_string":"x","new_string":"y","replace_all":true}]}',
    ),
  );
  assert.deepEqual(replace, {
    path: "src/existing.ts",
    editCount: 2,
    mode: "replace",
    replaceAllCount: 1,
  });
  assert.equal(editFileChangeLabel(replace), "2 replacements");
});

test("edit_file approval summary is safe for malformed model arguments", () => {
  assert.deepEqual(summarizeEditFileArgs({ path: 42, edits: "invalid" }), {
    path: "42",
    editCount: 0,
    mode: "replace",
    replaceAllCount: 0,
  });
  assert.doesNotThrow(() =>
    summarizeEditFileArgs(
      JSON.parse('{"path":{"toString":null},"edits":[null,42]}'),
    ),
  );
});

test("edit_file approval summary mirrors backend replace_all coercion", () => {
  const edits = [
    editWithReplaceAll(true),
    editWithReplaceAll(" true "),
    editWithReplaceAll("1"),
    editWithReplaceAll("YES"),
    editWithReplaceAll(1),
    editWithReplaceAll(-2),
    editWithReplaceAll(false),
    editWithReplaceAll("false"),
    editWithReplaceAll("0"),
    editWithReplaceAll("no"),
    editWithReplaceAll(""),
    editWithReplaceAll(0),
  ];

  assert.equal(summarizeEditFileArgs({ edits }).replaceAllCount, 6);
});

test("edit_file replace_all counting ignores malformed values", () => {
  const malformed: unknown[] = [
    null,
    1.5,
    [],
    {},
    { toString: null },
    ["true"],
  ];
  const edits: unknown[] = malformed.map(editWithReplaceAll);
  edits.push(null, 42);

  assert.doesNotThrow(() => summarizeEditFileArgs({ edits }));
  assert.equal(summarizeEditFileArgs({ edits }).replaceAllCount, 0);
});

test("edit_file error results cannot render as successful edits", () => {
  assert.equal(editFileResultIsError("Error: target changed"), true);
  assert.equal(editFileResultIsError("  Error: unsafe path"), true);
  assert.equal(editFileResultIsError("Edited module.py"), false);
});

test("edit_file has a dedicated confirmable card", async () => {
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.match(thread, EDIT_FILE_CARD_RE);
});

test("automatic mode tells users unsupported operations stay blocked", async () => {
  const permissions = await readFile(
    new URL("../src/features/chat/permission-mode-select.tsx", import.meta.url),
    "utf8",
  );
  assert.match(permissions, UNSUPPORTED_BLOCKED_RE);
});
