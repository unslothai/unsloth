// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/components/assistant-ui/tool-confirmation-controls.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("pending tool confirmation announces the required decision", () => {
  assert.match(
    source,
    /<p role="alert" className="sr-only">\s*Tool call \{toolName\} needs your approval\./,
  );
  assert.match(source, /Choose Allow, Always allow, or\s*Deny\./);
  assert.match(
    source,
    /<span role="alert" className="text-xs text-destructive">\s*Could not send your decision\./,
  );
});
