// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const page = readFileSync(
  new URL("../src/features/translate/translate-page.tsx", import.meta.url),
  "utf8",
);

test("the model menu is memoized", () => {
  assert.match(page, /const ModelMenu = memo\(function ModelMenu\(/);
  assert.doesNotMatch(page, /^function ModelMenu\(/m);
  assert.match(page, /import \{[^}]*\bmemo\b[^}]*\} from "react";/);
});

test("the page passes only stable props, so memo bails out on keystrokes and streamed frames", () => {
  const uses = page.match(/<ModelMenu\b[\s\S]*?\/>/g) ?? [];
  assert.equal(uses.length, 1);
  const props = [...uses[0].matchAll(/(\w+)=\{([^}]*)\}/g)].map(
    ([, name, value]) => [name, value.trim()],
  );
  assert.deepEqual(props, [
    ["value", "prefs.modelId"],
    ["custom", "prefs.customModel"],
    ["onChange", "pickTranslateModel"],
    ["onPickCustom", "pickCustomTranslateModel"],
    ["downloaded", "downloaded"],
    ["offered", "offeredModels"],
  ]);
  assert.match(page, /const offeredModels = useMemo\(/);
  for (const name of ["pickTranslateModel", "pickCustomTranslateModel"]) {
    assert.match(
      page,
      new RegExp(`const ${name} = useCallback\\([\\s\\S]*?\\[setPrefs\\],\\s*\\);`),
    );
  }
});
