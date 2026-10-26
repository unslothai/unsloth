import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const ALIGN_BY_DESCRIPTION = /description \? "items-start" : "items-center"/;

for (const path of [
  "features/settings/components/settings-row.tsx",
  "features/settings/components/storage-overview.tsx",
]) {
  test(`${path}: a row with a description keeps its control by the title`, async () => {
    const source = await readSrcAsync(path);
    assert.match(source, ALIGN_BY_DESCRIPTION);
    assert.match(source, /description \? <TitleLineStrut\b/);
  });
}

test("the strut is exactly one label line tall and takes no width", async () => {
  const source = await readSrcAsync(
    "features/settings/components/settings-row.tsx",
  );
  assert.match(
    source,
    /aria-hidden=\{true\}\s*className=\{cn\("w-0 select-none text-sm"/,
  );
});
