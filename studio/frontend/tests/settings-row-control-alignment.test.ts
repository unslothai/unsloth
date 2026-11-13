import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const ALIGN_BY_DESCRIPTION = /description \? "items-start" : "items-center"/;
const TITLE_LINE_CONTROL =
  /data-title-line-control=\{description \? true : undefined\}\s*className=\{cn\(\s*"flex [^"]*",\s*!description && "items-center",?(?:\s*![\w]+ &&\s*"[^"]*",?)?\s*\)\}/;

for (const path of [
  "features/settings/components/settings-row.tsx",
  "features/settings/components/storage-overview.tsx",
]) {
  test(`${path}: a row with a description keeps its control by the title`, async () => {
    const source = await readSrcAsync(path);
    assert.match(source, ALIGN_BY_DESCRIPTION);
    assert.match(source, TITLE_LINE_CONTROL);
  });
}

test("every control's top edge meets the title's cap top, whatever the control", async () => {
  const css = await readSrcAsync("index.css");
  assert.match(
    css,
    /--ui-title-cap-inset: calc\(0\.875rem \* 0\.35 \* var\(--ui-font-scale, 1\)\);/,
  );
  assert.match(
    css,
    /\[data-title-line-control\] \{\s*display: flex;\s*align-items: flex-start;\s*padding-top: var\(--ui-title-cap-inset\);\s*\}/,
  );
  // A per-control-type rule misclassified a dropdown whose trigger replaced Button's data-slot.
  assert.doesNotMatch(css, /\[data-title-line-control\]:has\(/);
  assert.match(
    css,
    /\[data-title-line-control\]:not\(:has\(button, input, select, textarea, \[role\]\)\) \{\s*padding-top: 0;\s*\}/,
  );
});
