// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  editFileChangeLabel,
  editFileIncompleteTitle,
  editFileResultIsError,
  editFileResultWasDeclined,
  summarizeEditFileArgs,
} from "../src/components/assistant-ui/edit-file-tool-summary.ts";
import {
  MAX_SERIALISED_LENGTH,
  toolArgText,
} from "../src/components/assistant-ui/tool-arg-text.ts";
import { canApproveToolArguments } from "../src/components/assistant-ui/tool-argument-visibility.ts";

const EDIT_FILE_CARD_RE = /edit_file:\s*EditFileToolUIConfirmable/;
const editWithReplaceAll = (replaceAll: unknown): Record<string, unknown> =>
  Object.fromEntries([["replace_all", replaceAll]]);

test("truncated or unrepresentable edit requests cannot be approved", () => {
  const small = {
    path: "file.py",
    edits: [{ old_string: "", new_string: "x" }],
  };
  assert.equal(canApproveToolArguments("edit_file", small), true);
  assert.equal(toolArgText(small), JSON.stringify(small));
  const large = {
    ...small,
    edits: [{ new_string: "x".repeat(MAX_SERIALISED_LENGTH) }],
  };
  assert.equal(toolArgText(large).endsWith("…"), true);
  assert.equal(canApproveToolArguments("edit_file", large), false);
  const cycle: { self?: unknown } = {};
  cycle.self = cycle;
  assert.equal(canApproveToolArguments("edit_file", cycle), false);
  assert.equal(canApproveToolArguments("edit_file", undefined), false);
  assert.equal(canApproveToolArguments("web_search", large), true);
});

test("incomplete edit statuses cannot render as successful edits", () => {
  assert.equal(
    editFileIncompleteTitle({ type: "incomplete", reason: "cancelled" }),
    "Edit cancelled",
  );
  assert.equal(
    editFileIncompleteTitle({
      type: "incomplete",
      reason: "error",
      error: "connection lost",
    }),
    "Edit incomplete",
  );
  assert.equal(editFileIncompleteTitle({ type: "complete" }), null);
  assert.equal(editFileIncompleteTitle({ type: "running" }), null);
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
  const denied = "The user declined to run this tool call.";
  assert.equal(editFileResultIsError(denied), true);
  assert.equal(editFileResultWasDeclined(denied), true);
  assert.equal(editFileResultWasDeclined("Edited module.py"), false);
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
