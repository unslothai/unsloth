// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// Dialogs scroll their own rounded box. A scrollbar running its full height covered the right-hand
// corners (Edit project, New project, MCP servers), so the track starts and ends where they do.

test("dialog scrollbars stop short of the rounded corners", () => {
  const css = readSrc("index.css");
  const dialogs = String.raw`:is\(\[data-slot="dialog-content"\], \[data-slot="alert-dialog-content"\]\)`;
  assert.match(
    css,
    new RegExp(`${dialogs}::-webkit-scrollbar-track \\{\\s*margin-block: var\\(--radius-4xl\\);`),
  );
  // The thin standard scrollbar has no track to inset, so Chromium is switched to the styled one.
  assert.match(
    css,
    new RegExp(
      String.raw`@supports selector\(::-webkit-scrollbar\) \{\s*` +
        `${dialogs} \\{\\s*scrollbar-width: auto;\\s*scrollbar-color: auto;`,
    ),
  );
  // Both primitives scroll a box with that radius.
  for (const file of ["components/ui/dialog.tsx", "components/ui/alert-dialog.tsx"]) {
    const source = readSrc(file);
    assert.match(source, /overflow-y-auto/, file);
    assert.match(source, /rounded-4xl/, file);
  }
});
