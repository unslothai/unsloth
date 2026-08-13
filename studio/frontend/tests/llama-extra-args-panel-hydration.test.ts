// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two rules the panel has to keep while it hydrates the stored arguments, both about
// what happens in the window between the request going out and its answer arriving.
// Source-level, because both live inside effects of a component this suite has no
// renderer for; the behaviour they guard was found by review, not by a red test.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PANEL = readFileSync(
  path.join(
    HERE,
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);

test("a list typed while hydration was in flight is not sanitized", () => {
  // The local list is sanitized because it can be legacy stored data this build now
  // refuses. But if the user typed during the window, that value is live input:
  // rewriting it cleared the box on "--agent" instead of showing the error, and could
  // trim a long paste behind the cursor. The snapshot tells the two apart.
  assert.match(PANEL, /const localAtStart = configRef\.current\.llamaExtraArgs;/);
  assert.match(PANEL, /local !== null && local\.length > 0 && local === localAtStart|local != null && local\.length > 0 && local === localAtStart/);
});

test("a collapsed section is re-judged once the catalogue lands", () => {
  // The row keeps its verdict when Advanced settings collapse, because the tokens
  // still go out with the load. A verdict reached before the probe answered did not
  // know which flags this build documents, and collapsing froze it: a bare --threads
  // kept Load enabled and failed at llama-server startup instead.
  assert.match(PANEL, /if \(showAdvanced \|\| !target\.isGguf \|\| resolvedIsDiffusion\) \{/);
  assert.match(PANEL, /loadLlamaFlagCatalog\(\)\.then\(\(catalog\) => \{/);
});
