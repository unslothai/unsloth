// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("rare tool cards remain async, with a visible fallback and stable component identities", () => {
  const source = readSrc("components/assistant-ui/thread.tsx");
  for (const [module, component] of [
    ["tool-ui-image-generation", "ImageGenerationToolUI"],
    ["tool-ui-render-html", "RenderHtmlToolUI"],
  ]) {
    assert.doesNotMatch(
      source,
      new RegExp('^import \\{ ' + component + ' \\} from [^\\n]*' + module, "m"),
    );
    assert.match(source, new RegExp('import\\("[^"]*' + module + '"\\)\\.then\\('));
    assert.match(source, new RegExp('<Deferred' + component + ' \\{\\.\\.\\.props\\} \\/>'));
  }
  assert.match(source, /<Suspense fallback=\{<ToolFallback \{\.\.\.props\} \/>\}>/);
  assert.match(source, /const ASSISTANT_PART_COMPONENTS = \{/);
  assert.match(source, /image_generation: ImageGenerationToolUIConfirmable/);
  assert.match(source, /render_html: RenderHtmlToolUIConfirmable/);
});
