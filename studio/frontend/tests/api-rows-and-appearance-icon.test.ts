// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("Appearance uses the brush-cleaning icon wherever it is listed", () => {
  for (const file of [
    "features/settings/settings-dialog.tsx",
    "features/settings/components/sidebar-menu-customizer.tsx",
    "components/app-sidebar.tsx",
  ]) {
    const src = readSrc(file);
    assert.match(
      src,
      /import \{ BrushCleaningIcon \} from "@\/lib\/brush-cleaning-icon";/,
      file,
    );
    assert.doesNotMatch(src, /PaintBrush02Icon/, file);
  }
  assert.match(
    readSrc("lib/brush-cleaning-icon.ts"),
    /export const BrushCleaningIcon: IconSvgElement/,
  );
});

test("served-model facts hug their content with one gap, icons sized from their text", () => {
  const page = readSrc("features/api-monitor/api-monitor-page.tsx");
  assert.match(
    page,
    /@3xl\/served:grid-cols-\[minmax\(0,max-content\)_auto_auto_auto_auto_1fr\][^"]*@3xl\/served:gap-x-10/,
  );
  assert.match(page, /@3xl\/served:grid-cols-subgrid @3xl\/served:gap-x-10/);
  assert.match(page, /className="size-\[1em\] shrink-0 text-muted-foreground"/);
  assert.match(
    page,
    /<Fact label="Total time, average per request" icon=\{Clock01Icon\}>/,
  );
  assert.doesNotMatch(page, /Timer01Icon/);
  assert.match(
    page,
    /fitContent=\{true\}[\s\S]{0,140}className="-mx-2 -my-1 h-6 font-sans text-ui-12 tabular-nums/,
  );
  const chip = readSrc("features/chat/components/context-usage-bar.tsx");
  assert.match(
    chip,
    /fitContent \? "flex items-center" : "grid shrink! items-center"/,
  );
  assert.match(chip, /fitContent \? "gap-1\.5" : "w-\(--full\) gap-2"/);
});

test("the served-models line points to Model memory while one model at a time", () => {
  const page = readSrc("features/api-monitor/api-monitor-page.tsx");
  assert.match(
    page,
    /if \(!known \|\| keepLoaded \|\| servedCount > 1\) return null;/,
  );
  assert.match(page, /Enable loading multiple models in\{" "\}/);
  assert.match(
    page,
    /openDialog\("resources", \{\s*scrollTarget: "resources-model-memory",\s*\}\)/,
  );
  assert.match(
    page,
    /isOwner \? \([\s\S]{0,700}\) : \(\s*"One model at a time"\s*\)/,
  );
  assert.match(page, /<MultiModelHint servedCount=\{0\} \/>/);
  assert.match(page, /<MultiModelHint servedCount=\{served\.length\} \/>/);
});
