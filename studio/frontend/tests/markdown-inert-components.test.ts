// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";
import { INERT_MARKDOWN_COMPONENTS } from "../src/components/markdown/inert-components.ts";

const MARKDOWN = "See [the docs](https://unsloth.ai) and <https://example.com>.\n\n- [x] done\n- [ ] todo\n";

function render(components?: typeof INERT_MARKDOWN_COMPONENTS): string {
  return renderToStaticMarkup(
    React.createElement(Streamdown, { mode: "static", controls: false, components }, MARKDOWN),
  );
}

test("inert markdown renders links and task checkboxes as text", () => {
  const plain = render();
  assert.match(plain, /<(?:a|button)[\s>]/);
  assert.match(plain, /<input/);
  const inert = render(INERT_MARKDOWN_COMPONENTS);
  assert.doesNotMatch(inert, /<a[\s>]|<input|<button/);
  assert.match(inert, /the docs/);
  assert.match(inert, /☑ /);
  assert.match(inert, /☐ /);
});
