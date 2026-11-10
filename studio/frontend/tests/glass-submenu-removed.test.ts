// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Window material had two controls in the same Appearance tab: a Material submenu inside the
// palette picker, and the Glass theme section's switch plus style select. The section is the one
// that can express "off" as a switch, keeps the related knobs (strength, lens, menus), and is
// where the search index points, so the palette picker should not carry a second copy.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PALETTE = "features/settings/components/color-theme-select.tsx";
const SEARCH = "features/settings/settings-search.ts";

test("the palette picker does not carry window material", () => {
  const source = readSrc(PALETTE);

  assert.doesNotMatch(source, /GlassThemeSub|GlassChip/, "material belongs to the Glass theme section");
  assert.doesNotMatch(source, /useMaterialStore/, "the palette picker must not write the material store");
  assert.doesNotMatch(source, /chatBackground\.material\b/, "the material label is not this menu's");
});

test("the settings search index has no scroll target for the removed label", () => {
  const source = readSrc(SEARCH);

  assert.doesNotMatch(
    source,
    /"chatBackground\.material",/,
    "every indexed key needs a rendered data-settings-label",
  );
  assert.match(
    source,
    /"chatBackground\.materialStyle",/,
    "the surviving control in the Glass theme section stays searchable",
  );
});
