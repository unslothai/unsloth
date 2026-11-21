// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("the View menu zooms a native page that holds the keys, wherever activeElement is", () => {
  // A click into a native page view leaves activeElement in the app (often the composer), so the
  // menu asks the scopes first and only then falls back to activeElement.
  const actions = readSrc("features/interface-zoom/lib/zoom-actions.ts");
  assert.match(
    actions,
    /const scope = \[\.\.\.scopes\]\.find\(\(candidate\) => candidate\.holdsKeys\?\.\(\)\) \?\? zoomScopeFor\(document\.activeElement\);/,
  );
  // The browser panel's scope claims the keys while its page view has them.
  assert.match(
    readSrc("features/browser/browser-panel.tsx"),
    /holdsKeys: nativePageHasKeys,/,
  );
  assert.match(
    readSrc("features/browser/native-view.ts"),
    /export function nativePageHasKeys\(\): boolean \{\s*return shownView !== null && !document\.hasFocus\(\);/,
  );
});
