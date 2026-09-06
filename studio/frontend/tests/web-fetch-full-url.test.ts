// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/components/assistant-ui/tool-ui-web-search.tsx",
    import.meta.url,
  ),
  "utf8",
);
const EXPANDED_CONTENT_RE =
  /<ToolFallbackContent>([\s\S]*?)<\/ToolFallbackContent>/;
const URL_BLOCK_RE = /<div\s+data-slot="tool-web-fetch-url"[\s\S]*?<\/div>/;
const RAW_URL_RE = /\{url\}/;
const WRAPPING_RE = /break-all/;
const LTR_RE = /dir="ltr"/;
const LINK_RE = /href=/;

test("expanded web fetch cards show the complete raw URL as text", () => {
  const expandedContent = source.match(EXPANDED_CONTENT_RE)?.[1];
  const urlBlock = expandedContent?.match(URL_BLOCK_RE)?.[0];

  assert.ok(urlBlock, "missing the expanded-card URL detail");
  assert.match(
    urlBlock,
    RAW_URL_RE,
    "the detail must render the full raw argument",
  );
  assert.match(urlBlock, WRAPPING_RE, "long URLs must wrap inside the card");
  assert.match(urlBlock, LTR_RE, "URLs must keep a stable reading direction");
  assert.doesNotMatch(
    urlBlock,
    LINK_RE,
    "an untrusted tool argument must not become a clickable link",
  );
});
