// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// Below lg the toolbar lives in the catalog pane, so clearing an NPU search swaps the pane and
// remounts the search field; focus has to follow it or the next keystrokes are lost.
test("clearing a compact NPU search keeps the search field focused", async () => {
  const page = await readSrcAsync("features/hub/hub-page.tsx");
  const handler = page.slice(
    page.indexOf("const handleQueryChange = useCallback("),
  );
  assert.match(
    handler.slice(0, handler.indexOf("if (next.trim()")),
    /focusSearchOnMount\.current =\s*isCompactViewport &&\s*document\.activeElement\?\.closest\('\[data-tour="hub-search"\]'\)\s*\?\s*next\s*:\s*null;/,
  );
  assert.match(
    page,
    /useLayoutEffect\(\(\) => \{\s*if \(focusSearchOnMount\.current !== query\) return;[\s\S]*?'input\[name="hub-search"\]'[\s\S]*?input\.focus\(\{ preventScroll: true \}\);\s*\}\s*\}, \[query\]\);/,
  );

  const toolbar = await readSrcAsync("features/hub/catalog/models-toolbar.tsx");
  assert.match(toolbar, /data-tour="hub-search"/);
  assert.match(toolbar, /name="hub-search"/);
});
