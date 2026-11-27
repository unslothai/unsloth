// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("a link to a protected backend file opens through the session, not a bare navigation", () => {
  // Saved-voice file links 401 without the Bearer token; each step stays in its own function.
  const sessionLink =
    /\nfunction MarkdownLink\((?:(?!\nfunction )[\s\S])*?const backend = href \? protectedBackendPath\(href\) : null;(?:(?!\nfunction )[\s\S])*?<BackendFileLink href=\{backend\}(?:(?!\nfunction )[\s\S])*?backendPathUrl\(href\)[\s\S]*?\nfunction BackendFileLink\((?:(?!\nfunction )[\s\S])*?loadBackendFile\(href\)(?:(?!\nfunction )[\s\S])*?event\.preventDefault\(\);\s*openFile\(\);/;
  const markdown = readSrc("components/assistant-ui/markdown-text.tsx");
  assert.match(markdown, sessionLink);
  const menu = readSrc("components/assistant-ui/link-context-menu.tsx");
  assert.match(menu, /loadBackendFile\(path[^}]*authFetch\(path\)/);
});
