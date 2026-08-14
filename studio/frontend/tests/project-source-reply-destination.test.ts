// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Save to project sources" on a reply used to upload into useProjectStore's
// activeProjectId. That id is the project the *sidebar* is showing, and it lags
// a thread switch: opening a chat with no project query parameter leaves the
// previous project selected, so the item stayed enabled and the reply went into
// a project the chat has nothing to do with. One user's project then holds
// another's content. The destination has to come from the thread being read.
//
// thread.tsx is 6k lines of TSX that node cannot load, so this reads the source,
// the way rag-availability-marker.test.ts does for the same file.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const src = await readFile(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);

/** The onSelect body of the "Save to project sources" action bar item. */
function saveHandler(): string {
  const marker = src.indexOf("Save to project sources");
  assert.ok(marker > 0, "the reply action is gone or was renamed");
  const start = src.lastIndexOf("<ActionBarMorePrimitive.Item", marker);
  assert.ok(start > 0 && start < marker, "could not find the item's opening tag");
  return src.slice(start, marker);
}

test("the reply is saved to its own thread's project, not the selected one", () => {
  const handler = saveHandler();
  assert.match(
    handler,
    /await getStoredChatThread\(remoteId\)/,
    "the destination is not read from the stored thread",
  );
  assert.match(
    handler,
    /saveMarkdownAsProjectSource\(\n\s*thread\.projectId,/,
    "the upload does not address the thread's own project",
  );
  assert.ok(
    !/saveMarkdownAsProjectSource\(\s*activeProjectId/.test(handler),
    "the reply is still uploaded into whatever project the sidebar has selected",
  );
});

test("a thread with no project is refused rather than sent somewhere", () => {
  assert.match(
    saveHandler(),
    /if \(!thread\?\.projectId\) \{\n\s*toast\.info\("This chat isn't in a project\."\);\n\s*return;\n\s*\}/,
    "a chat that is not in a project falls through to a save with no destination",
  );
});

test("the thread id is resolved before the menu can close", () => {
  const handler = saveHandler();
  const idRead = handler.indexOf("aui.threadListItem().getState()");
  const firstAwait = handler.indexOf("await");
  assert.ok(idRead > 0, "the thread list item is no longer read at all");
  assert.ok(
    firstAwait === -1 || idRead < firstAwait,
    "the message and thread context is read after an await, by which point the " +
      "menu has closed and the assistant-ui context can be gone",
  );
  assert.match(
    handler,
    /const remoteId =\n\s*state\.remoteId \|\|\n\s*useChatRuntimeStore\.getState\(\)\.activeThreadId;/,
    "a thread whose list item has no remoteId yet has no destination to resolve",
  );
});

// The action is still hidden outside a project, so the reply menu does not grow
// an item that can only refuse itself.
test("the item is only rendered while a project is selected", () => {
  const marker = src.indexOf("Save to project sources");
  const before = src.slice(Math.max(0, marker - 2200), marker);
  assert.match(
    before,
    /\{activeProjectId && \(/,
    "the reply action is offered outside a project",
  );
});
