// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

test("any failure without its own copy reads as unreachable, with the ModelScope offer", () => {
  const states = read("features/hub/catalog/catalog-states.tsx");
  assert.doesNotMatch(states, /discovery feed couldn't load/);
  assert.doesNotMatch(states, /Couldn't reach \$\{hub\}/);
  const fallback =
    states.match(
      /case "unknown":\s*default:\s*break;\s*\}([\s\S]*?)\n\}/,
    )?.[1] ?? "";
  assert.match(fallback, /title: `Can't reach \$\{hub\}`/);
  assert.match(fallback, /try ModelScope instead/);
  assert.doesNotMatch(states, /(shown|failure) && !offlineLike/);
  assert.equal(
    states.match(/!offlineLike && !tokenRejected \? <UseModelScopeButton \/>/g)
      ?.length,
    2,
  );
});

test("the Discover toast and the picker toasts use the same unreachable title", () => {
  const discover = read("features/hub/hooks/use-discover-search.ts");
  assert.doesNotMatch(discover, /Couldn't reach/);
  assert.match(discover, /default:\s*return `Can't reach \$\{hub\}`;/);
  const en = read("i18n/locales/en.ts");
  assert.equal(
    en.match(/hubUnreachable: "Can't reach Hugging Face"/g)?.length,
    2,
  );
});
