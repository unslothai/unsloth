import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const BOUNDED_TEXT_COLUMN =
  /<div className="[^"]*w-full[^"]*max-w-\[calc\(32rem\*var\(--ui-space-scale,1\)\)\][^"]*flex-col[^"]*">\s*<span className="text-sm font-medium text-foreground">/;

test("settings row titles and descriptions share a bounded text column", async () => {
  const source = await readSrcAsync(
    "features/settings/components/settings-row.tsx",
  );

  assert.match(source, BOUNDED_TEXT_COLUMN);
});

test("a settings row's control is capped too, unless the row opts out for a value it must not clip", async () => {
  const source = await readSrcAsync(
    "features/settings/components/settings-row.tsx",
  );
  assert.match(
    source,
    /!wideControl &&\s*"max-w-\[min\(100%,calc\(20rem\*var\(--ui-space-scale,1\)\)\)\] \[&>\*\]:max-w-full"/,
  );
  const llama = await readSrcAsync(
    "features/settings/components/llama-backend-section.tsx",
  );
  assert.match(llama, /customPath\.hint"\)\}\s*wideControl=\{true\}/);
});

test("a settings row's info icon sits inline on the label's baseline, like a glyph", async () => {
  const source = await readSrcAsync(
    "features/settings/components/settings-row.tsx",
  );

  assert.match(source, /aria-label=\{hint\}\s*className="[^"]*\binline-flex align-baseline\b/);
});
