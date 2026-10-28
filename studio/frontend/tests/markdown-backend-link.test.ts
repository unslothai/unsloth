// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const MARKDOWN = readSrc("components/assistant-ui/markdown-text.tsx");
const MENU = readSrc("components/assistant-ui/link-context-menu.tsx");

function body(source: string, name: string): string {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, name);
  const next = source.indexOf("\nfunction ", start + 1);
  return source.slice(start, next < 0 ? undefined : next);
}

test("a link to a protected backend file opens through the session, not a bare navigation", () => {
  // A saved-voice link from the chat's tools (/v1/audio/voices/<id>/file) 401s without the Bearer token.
  const link = body(MARKDOWN, "MarkdownLink");
  const guard = link.indexOf(
    "const backend = href ? protectedBackendPath(href) : null;",
  );
  assert.ok(guard >= 0 && guard < link.indexOf("backendPathUrl(href)"));
  assert.match(link, /<BackendFileLink href=\{backend\}/);
  const file = body(MARKDOWN, "BackendFileLink");
  assert.match(file, /loadBackendFile\(href\)/);
  assert.match(file, /event\.preventDefault\(\);\s*openFile\(\);/);
  assert.match(body(MENU, "loadBackendFile"), /authFetch\(path\)/);
});
