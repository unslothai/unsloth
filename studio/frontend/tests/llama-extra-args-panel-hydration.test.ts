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

const ADAPTER = readFileSync(
  path.join(HERE, "..", "src/features/chat/api/chat-adapter.ts"),
  "utf8",
);

test("a background auto-load hydrates a server-only override", () => {
  // resolveInitialConfig reads local storage, so an override written through the API
  // or from another browser leaves the field undefined. Nothing is resident at
  // startup, so /load has nothing to inherit from and the model came up without the
  // arguments saved for it.
  assert.match(ADAPTER, /let resolvedExtraArgs = config\.llamaExtraArgs;/);
  assert.match(ADAPTER, /const stored = await fetchLoadExtraArgs\(/);
  // Sanitized, because this becomes an explicit list that /load validates strictly.
  assert.match(ADAPTER, /sanitizeStoredExtraArgs\(tokens, managed\?\.managed/);
  // The local copy too: it was written by whatever build was running then, so a
  // flag added to the managed set since would be sent explicitly and 400.
  assert.match(ADAPTER, /const cleaned = clean\(resolvedExtraArgs\);/);
  // Resolved under the advertised repository id as well as the load path: cached
  // inventory can hand back a different loadId, and the row was written under
  // whichever of the two was on screen.
  assert.match(ADAPTER, /candidate\.id,\n\s*candidate\.ggufVariant \?\? null,/);
  // And both the preflight and the load send what was resolved, not the raw config.
  assert.equal(
    ADAPTER.match(/llama_extra_args: resolvedExtraArgs \?\? \[\]/g)?.length,
    2,
  );
  // A diffusion GGUF takes none of them, so it is not fetched for either.
  assert.match(ADAPTER, /candidate\.kind === "gguf" &&\s*\n?\s*!isDiffusion/);
});

const HUB = readFileSync(
  path.join(HERE, "..", "src/features/hub/hub-page.tsx"),
  "utf8",
);

test("applying from the Hub settings page carries the arguments into the load", () => {
  // applyPerModelConfigToRuntime does not store llamaExtraArgs, so a selection made
  // without the config left the field undefined: the load omitted it, the route kept
  // the resident server's old list, and an edit or a clear did nothing.
  assert.match(HUB, /forceReload: true,\n(\s*\/\/.*\n)*\s*config,/);
});

test("a collapsed section stops objecting once nothing is left to object to", () => {
  // Reset with Advanced collapsed clears the list, and the row is not mounted to
  // withdraw the verdict it left standing, so Load stayed disabled.
  assert.match(PANEL, /setExtraArgsLoadable\(true\);\n\s*return;/);
});

const COMPOSER = readFileSync(
  path.join(HERE, "..", "src/features/chat/shared-composer.tsx"),
  "utf8",
);

test("a compare pane sanitizes the local list as well as the fetched one", () => {
  // A config saved by an older build can still name a flag that is managed now, and
  // the pane sends whichever list it holds as an explicit /load argument, so the
  // comparison came back 400 instead of running.
  assert.match(COMPOSER, /const local = ownConfig\.llamaExtraArgs;/);
  assert.match(COMPOSER, /const cleaned = clean\(local\);/);
});

test("the hidden revalidation can only tighten the verdict", () => {
  // It judges the TOKENS, and formatExtraArgs quotes them back into a balanced
  // string, so an unclosed quote the row objected to reads as fine here. Raising the
  // verdict would re-enable Load over the value the user is still typing.
  assert.match(PANEL, /if \(!loadable\) \{\n\s*setExtraArgsLoadable\(false\);/);
});

test("a mounted row re-reads the catalogue when the binary changes", () => {
  // Updating llama.cpp from the banner replaces the binary while the panel stays
  // open; the row would otherwise judge arity against the old build's help.
  assert.match(PANEL, /subscribeLlamaFlagCatalog\(\(\) => setCatalogEpoch/);
  assert.match(PANEL, /\}, \[catalogEpoch\]\);/);
});
