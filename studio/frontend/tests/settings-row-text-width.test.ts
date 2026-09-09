import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const BOUNDED_TEXT_COLUMN =
  /<div className="[^"]*w-full[^"]*max-w-lg[^"]*flex-col[^"]*">\s*\{\/\* Flex only when hinted/;

test("settings row titles and descriptions share a bounded text column", async () => {
  const source = await readSrcAsync(
    "features/settings/components/settings-row.tsx",
  );

  assert.match(source, BOUNDED_TEXT_COLUMN);
});
