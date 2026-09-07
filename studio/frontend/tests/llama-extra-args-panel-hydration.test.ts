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

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  DEFAULT_PER_MODEL_CONFIG,
  deletePerModelConfig,
  perModelConfigStorageChanged,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

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
  assert.match(PANEL, /const localAtStart = configAtStart\.llamaExtraArgs;/);
  assert.match(
    PANEL,
    /local !== null && local\.length > 0 && local === localAtStart|local != null && local\.length > 0 && local === localAtStart/,
  );
});

test("a collapsed section is re-judged once the catalogue lands", () => {
  // The row keeps its verdict when Advanced settings collapse, because the tokens
  // still go out with the load. A verdict reached before the probe answered did not
  // know which flags this build documents, and collapsing froze it: a bare --threads
  // kept Load enabled and failed at llama-server startup instead.
  assert.match(
    PANEL,
    /if \(showAdvanced \|\| !target\.isGguf \|\| resolvedIsDiffusion\) \{/,
  );
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

test("the hidden validation re-runs when the binary changes", () => {
  // The row's own subscription cannot help here: it is unmounted whenever this
  // check is the one running, so an in-app llama.cpp update with Advanced collapsed
  // left a verdict reached against the previous binary standing.
  assert.match(
    PANEL,
    /subscribeLlamaFlagCatalog\(\(\) =>\s*\n?\s*setHiddenCatalogEpoch/,
  );
  assert.match(PANEL, /hiddenCatalogEpoch,\n\s*\]\);/);
});

const OVERRIDES = readFileSync(
  path.join(HERE, "..", "src/features/model-picker/api/model-overrides.ts"),
  "utf8",
);

test("the legacy overrides fallback searches the caller's own identities", () => {
  // A backend that predates the resolved field answers with the whole map, and the
  // caller has to say which keys to look under. The auto-load and compare callers
  // pass none, so the default of [] searched nothing and they launched without the
  // stored arguments against an older server.
  assert.match(OVERRIDES, /fallbackKeys\.length > 0\s*\n?\s*\? fallbackKeys/);
  assert.match(OVERRIDES, /modelOverrideKey\(loadId, ggufVariant\)/);
  assert.match(OVERRIDES, /modelOverrideKey\(aliasId, ggufVariant\)/);
});

test("the hidden hydration check knows the slot floor", () => {
  // The floor is already fetched here, and the backend refuses a batch below it
  // deterministically, so releasing Load without it opens a window where a click
  // reaches that 400.
  assert.match(
    PANEL,
    /serverConfig\?\.nParallel \?\? configRef\.current\.nParallel/,
  );
});

test("the panel adopts a shared server config without overwriting a live edit", () => {
  assert.match(PANEL, /fetchLoadModelOverride\(/);
  assert.match(
    PANEL,
    /const serverConfig = resolvedRow\s*\n?\s*\? fromApiOverride\(resolvedRow, \{/,
  );
  // The panel's config is the merge base for what it SHOWS, so a field the row does
  // not carry keeps the value this browser holds instead of coming back as an app
  // default. It travels sanitized: a row that says nothing about arguments leaves
  // the local list standing, and a flag this build refuses would re-enable Load for
  // a 400.
  assert.match(
    PANEL,
    /\.\.\.configAtStart,\s*\n?\s*llamaExtraArgs: sanitizedLocal,/,
  );
  assert.match(PANEL, /let sanitizedLocal = localAtStart;/);
  assert.match(
    PANEL,
    /sanitizedLocal = cleaned\.length > 0 \? cleaned : null;/,
  );
  // Whitespace-tolerant: the call carries an eviction list now, so it spans lines.
  assert.match(
    PANEL,
    /savePerModelConfig\(\s*configId,\s*target\.ggufVariant,\s*rememberedConfig,/,
  );
  assert.match(PANEL, /configRef\.current === configAtStart/);
  assert.match(PANEL, /rememberRef\.current === rememberAtStart/);
  assert.match(PANEL, /setConfig\(serverConfig\);/);
  assert.match(PANEL, /setRemember\(true\);/);
  assert.match(PANEL, /setSavedRemember\(true\);/);
});

test("hydration detects a newer save or forget", () => {
  const modelId = "unsloth/Hydration-Race-GGUF";
  const variant = "Q4_K_M";
  assert.ok(
    savePerModelConfig(modelId, variant, {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 2048,
    }),
  );
  const atStart = resolveInitialConfig(modelId, variant);

  assert.ok(
    savePerModelConfig(modelId, variant, {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 4096,
    }),
  );
  assert.equal(
    perModelConfigStorageChanged(
      atStart,
      resolveInitialConfig(modelId, variant),
    ),
    true,
  );

  assert.ok(deletePerModelConfig(modelId, variant));
  assert.equal(
    perModelConfigStorageChanged(
      atStart,
      resolveInitialConfig(modelId, variant),
    ),
    true,
  );
  assert.equal(
    perModelConfigStorageChanged(atStart, {
      config: { ...atStart.config },
      remembered: atStart.remembered,
    }),
    false,
  );
});

test("the hydration write-back rejects a stale server response", () => {
  const requestStart = PANEL.indexOf("Promise.all([");
  const storageSnapshot = PANEL.indexOf(
    "const storedAtStart = resolveInitialConfig(configId, target.ggufVariant);",
  );
  const responseStart = PANEL.indexOf(
    ".then(([resolvedOverride, managed]) => {",
    requestStart,
  );
  const adoptionStart = PANEL.indexOf(
    "if (\n          resolvedRow &&",
    responseStart,
  );
  const writeBackEnd = PANEL.indexOf(
    "setSavedRemember(hydrationSaved);",
    adoptionStart,
  );
  assert.ok(
    requestStart >= 0 &&
      storageSnapshot >= 0 &&
      storageSnapshot < requestStart &&
      responseStart > requestStart &&
      adoptionStart > responseStart &&
      writeBackEnd > adoptionStart,
    "the storage snapshot must precede the request and guard its write-back",
  );

  const writeBack = PANEL.slice(adoptionStart, writeBackEnd);
  assert.match(
    writeBack,
    /const storedConfig = resolveInitialConfig\(\s*configId,\s*target\.ggufVariant,\s*\);\s*if \(perModelConfigStorageChanged\(storedAtStart, storedConfig\)\) \{\s*return;\s*\}[\s\S]*const rememberedConfig = fromApiOverride\(\s*resolvedRow,\s*storedConfig\.config,\s*\);[\s\S]*savePerModelConfig\(\s*configId,\s*target\.ggufVariant,\s*rememberedConfig,/,
  );
});

test("a build that serves one slot does not raise the floor", () => {
  // The published default is already effective, but an EXPLICIT Slots value is not:
  // a build without --kv-unified serves one slot however many are asked for, so
  // Slots 4 with "--batch-size 2" was refused here while the backend, which clamps
  // to one, accepts exactly that command. One helper at all three call sites, so the
  // row and the two hidden checks cannot drift apart.
  assert.match(PANEL, /if \(limits\?\.parallelSlotsClamped\) \{\n\s*return 2;/);
  assert.equal(
    PANEL.match(/effectiveBatchFloor\(/g)?.length,
    4,
    "one definition and three call sites",
  );
});

const ADAPTER_AUTOLOAD = ADAPTER;

test("an auto-load records what it launched with", () => {
  // The model-loading lease is held for the whole of this load, which is the guard
  // that stops the status applier writing the baseline, so nothing else records it:
  // an immediate switch would snapshot the previous model's list, and a failed
  // switch would then restore this one with the wrong arguments.
  assert.match(
    ADAPTER_AUTOLOAD,
    /loadedLlamaExtraArgs:\s*\n?\s*loadResp\.requested_llama_extra_args !== undefined/,
  );
  // And a non-GGUF load clears it rather than leaving a GGUF's list standing.
  assert.match(ADAPTER_AUTOLOAD, /loadedLlamaExtraArgs: null,/);
});

test("a compare pane records what it launched with", () => {
  assert.match(
    COMPOSER,
    /loadedLlamaExtraArgs:\s*\n?\s*resp\.requested_llama_extra_args !== undefined/,
  );
});

test("a stored empty list hydrates as a clear, not as nothing stored", () => {
  // The settings page writes an explicit [] when the box is cleared for a quant
  // whose bare-repository row still carries arguments: it is the tombstone that
  // stops the server's lookup there. Read as an absence, all three hydrating
  // callers left llamaExtraArgs undefined, omitted the field on /load, and the
  // route carried the resident model's arguments over, the ones just cleared.
  assert.match(OVERRIDES, /explicit: Array\.isArray\(tokens\)/);
  assert.match(OVERRIDES, /\{ tokens: \[\], explicit: false \}/);
  assert.match(PANEL, /resolvedArgs\.explicit && local === undefined/);
  assert.match(ADAPTER, /\} else if \(stored\.explicit\) \{/);
  assert.match(COMPOSER, /\} else if \(resolvedArgs\.explicit\) \{/);
});

const CHAT_PAGE = readFileSync(
  path.join(HERE, "..", "src/features/chat/chat-page.tsx"),
  "utf8",
);

test("a launch that only applies the remembered config still carries its arguments", () => {
  // applyPerModelConfigToRuntime has no field for the launch flags, and /load only
  // inherits them from the SAME resident model, so a cold launch or a switch from
  // another model ran without the arguments this model was remembered with. Both
  // paths that load through the runtime alone now pass the config itself.
  assert.match(
    HUB,
    /\.\.\.\(rememberedConfig \? \{ config: rememberedConfig \} : \{\}\),/,
  );
  assert.match(
    CHAT_PAGE,
    /const remembered = rememberedConfigFor\(selection\);/,
  );
  assert.match(
    CHAT_PAGE,
    /\.\.\.\(remembered \? \{ config: remembered \} : \{\}\),/,
  );
  // Nothing is invented when there is no remembered config: the field stays absent,
  // which is what lets /load keep a resident model's own flags.
  assert.doesNotMatch(HUB, /config: rememberedConfig \?\? null/);
});
