// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/chat/chat-providers-dialog.tsx");

test("an empty connection list stays on the list, where the provider gallery is", () => {
  assert.doesNotMatch(dialog, /syncedProviders\.length === 0[^\n]*\{\s*setPage\("form"\)/);
  assert.doesNotMatch(dialog, /autoOpenedAddFormRef/);
  assert.match(
    dialog,
    /const selectableRegistry = registryRows\.filter\(\(entry\) => !entry\.hidden\);\s*setRegistry\(selectableRegistry\);/,
  );
  assert.match(dialog, /No connections yet/);
  assert.match(
    dialog,
    /<ConnectionGallery\s*registry=\{menuRegistry\}[\s\S]{0,200}?onPick=\{\(type\) => openAddProvider\(type\)\}/,
  );
});

test("only the user's own navigation moves between the list and the form", () => {
  const calls = [...dialog.matchAll(/setPage\("(list|form)"\)/g)].map((call) => {
    const before = dialog.slice(0, call.index);
    const owner = [...before.matchAll(/\n {2}(?:async )?function (\w+)\(/g)].at(-1)?.[1];
    return `${owner}:${call[1]}`;
  });
  assert.deepEqual(calls, [
    "openAddProvider:form",
    "closeForm:list",
    "addProvider:list",
    "saveProviderEdits:list",
    "editProvider:form",
  ]);
});

test("the form's back arrow still returns to the list", () => {
  assert.match(dialog, /function closeForm\(\) \{\s*resetForm\(\);\s*setPage\("list"\);/);
  assert.match(dialog, /onClick=\{closeForm\}\s*aria-label="Back to connections"/);
});
