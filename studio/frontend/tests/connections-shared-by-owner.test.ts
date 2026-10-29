// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
const sync = readSrc("features/chat/sync-external-providers.ts");
const autoReload = readSrc("features/chat/llama-cpp-auto-reload.ts");

test("only the owner sees the share toggle, and only with another account or a shared row", () => {
  assert.match(
    dialog,
    /const showShareToggle =\s*isAccountOwner &&\s*!usesOAuth &&\s*\(loginMode === "multi" \|\| editingProvider\?\.shared === true\);/,
  );
  assert.match(dialog, /\{showShareToggle \? \(/);
});

test("a single-account save sends no shared field", () => {
  assert.match(dialog, /\.\.\.\(showShareToggle && sharedDraft \? \{ shared: true \} : \{\}\),/);
  assert.match(
    dialog,
    /\.\.\.\(showShareToggle && sharedDraft !== \(existing\.shared === true\)\s*\? \{ shared: sharedDraft \}\s*: \{\}\),/,
  );
});

test("a row shared by the owner has no edit or delete action", () => {
  assert.equal([...dialog.matchAll(/\{provider\.sharedByOwner \? null : \(/g)].length, 2);
  assert.match(dialog, /if \(provider\.sharedByOwner\) \{\s*toast\.info\(/);
});

test("sync maps the flags and never writes back to a row shared by the owner", () => {
  assert.match(sync, /shared: config\.shared === true,/);
  assert.match(sync, /sharedByOwner: config\.shared_by_owner === true,/);
  assert.match(sync, /if \(!config\.shared_by_owner && \(needsModelBackfill \|\| needsAvailableBackfill\)\)/);
  assert.match(autoReload, /p\.autoReloadModels === true && !p\.sharedByOwner/);
});
