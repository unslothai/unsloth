import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const BOUNDED_TEXT_COLUMN =
  /<div className="[^"]*w-full[^"]*max-w-lg[^"]*flex-col[^"]*">\s*\{\/\* Flex only when hinted/;

test("settings row titles and descriptions share a bounded text column", async () => {
  const source = await readFile(
    new URL(
      "../src/features/settings/components/settings-row.tsx",
      import.meta.url,
    ),
    "utf8",
  );

  assert.match(source, BOUNDED_TEXT_COLUMN);
});
