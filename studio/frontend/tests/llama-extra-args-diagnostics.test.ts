// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { LlamaFlagCatalog } from "../src/features/model-picker/api/llama-flags.ts";
import {
  diagnoseExtraArgs,
  extraArgsAreLoadable,
} from "../src/features/model-picker/model-config/llama-extra-args.ts";

// The row is the backend's judgement shown early. Where these disagree, the panel
// accepts an argument the load then refuses, or warns about one that works.

const CATALOG: LlamaFlagCatalog = {
  flags: {
    "--top-k": "top-k sampling",
    "--numa": "NUMA policy",
    "--ctx-size": "context size",
    "--rope-scaling": "RoPE scaling",
  },
  // The names the backend actually returns, aliases included: --n-parallel is in
  // its denylist, and leaving it out here would test a catalogue that cannot exist.
  managed: new Set([
    "--parallel",
    "--n-parallel",
    "-np",
    "--model",
    "--api-key",
    "--agent",
    "--ctx-size",
  ]),
  probeOk: true,
};

const levels = (input: string, catalog: LlamaFlagCatalog | null = CATALOG) =>
  diagnoseExtraArgs(input, catalog).map((d) => d.level);

const messages = (input: string, catalog: LlamaFlagCatalog | null = CATALOG) =>
  diagnoseExtraArgs(input, catalog)
    .map((d) => d.message)
    .join(" | ");

test("a well-formed argument the binary knows says nothing", () => {
  assert.deepEqual(diagnoseExtraArgs("--numa distribute", CATALOG), []);
  assert.deepEqual(diagnoseExtraArgs("", CATALOG), []);
});

test("a managed flag is an error that names the control that owns it", () => {
  const text = messages("--parallel 8");
  assert.match(text, /--parallel/);
  assert.equal(levels("--parallel 8")[0], "error");
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs("--parallel 8", CATALOG)),
    false,
  );
});

test("a managed flag with no control says who owns it instead", () => {
  // --api-key is not a row in this panel, so pointing at one would be a lie.
  const text = messages("--api-key secret");
  assert.match(text, /managed by Unsloth Studio/);
  assert.doesNotMatch(text, /above/);
});

test("an attached or equals form is caught the same way", () => {
  // The backend normalises these before checking, so the row has to as well.
  assert.equal(levels("-np8")[0], "error");
  assert.equal(levels("--parallel=8")[0], "error");
  assert.equal(levels("--n_parallel 8")[0], "error");
});

test("a flag a control also sets is a note, not a refusal", () => {
  // Deliberate: the backend appends extras last and reconciles the ones that move
  // its own sizing, and the CLI has always allowed this. Say who wins, do not block.
  const diagnostics = diagnoseExtraArgs("--batch-size 512", CATALOG);
  assert.deepEqual(
    diagnostics.map((d) => d.level),
    ["note"],
  );
  assert.match(diagnostics[0].message, /Batch Size/);
  assert.equal(extraArgsAreLoadable(diagnostics), true);
});

test("a flag missing from this build warns but still loads", () => {
  const diagnostics = diagnoseExtraArgs("--tempp 0.7", CATALOG);
  assert.deepEqual(
    diagnostics.map((d) => d.level),
    ["warning"],
  );
  assert.match(diagnostics[0].message, /--tempp/);
  assert.match(diagnostics[0].message, /still be passed/);
  assert.equal(extraArgsAreLoadable(diagnostics), true);
});

test("nothing is called unknown when the probe failed", () => {
  // The failure mode this exists to prevent: an unverifiable build would mark every
  // correct flag as a typo, which is worse than saying nothing.
  const unverified: LlamaFlagCatalog = {
    flags: {},
    managed: CATALOG.managed,
    probeOk: false,
  };
  assert.deepEqual(
    diagnoseExtraArgs("--tempp 0.7 --numa distribute", unverified),
    [],
  );
  // A managed flag is still refused: that judgement needs no binary.
  assert.equal(levels("--parallel 8", unverified)[0], "error");
});

test("an older backend with no catalogue still refuses nothing and warns nothing", () => {
  assert.deepEqual(diagnoseExtraArgs("--tempp 0.7", null), []);
});

test("sampling flags are noted as launch defaults", () => {
  // They work, but the chat settings send sampling per request, so a value set here
  // is not what a conversation will use.
  const text = messages("--top-k 20");
  assert.match(text, /chat settings/);
  assert.equal(levels("--top-k 20")[0], "note");
});

test("an unclosed quote is an error", () => {
  const diagnostics = diagnoseExtraArgs('--chat-template "a b', CATALOG);
  assert.equal(diagnostics[0].level, "error");
  assert.match(diagnostics[0].message, /Unclosed double quote/);
  assert.equal(extraArgsAreLoadable(diagnostics), false);
});

test("too many arguments is an error at the backend's own limit", () => {
  const diagnostics = diagnoseExtraArgs("--verbose ".repeat(257), CATALOG);
  assert.equal(diagnostics[0].level, "error");
  assert.match(diagnostics[0].message, /limit 256/);
});

test("a payload over the byte limit is an error, even within the token cap", () => {
  // One long token is the realistic shape: a grammar or a JSON schema. The backend
  // refuses this on size, so the row has to say so rather than let the load start.
  const huge = `--grammar ${"a".repeat(40_000)}`;
  const diagnostics = diagnoseExtraArgs(huge, CATALOG);

  assert.equal(diagnostics[0].level, "error");
  assert.match(diagnostics[0].message, /limit 32768/);
  assert.equal(extraArgsAreLoadable(diagnostics), false);
  // Multi-byte characters count as their UTF-8 length, which is what the backend
  // measures; counting characters would let a CJK grammar through.
  const multibyte = `--grammar ${"\u65e5".repeat(11_000)}`;
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs(multibyte, CATALOG)),
    false,
  );
  // And an ordinary line is nowhere near it.
  assert.deepEqual(diagnoseExtraArgs("--numa distribute", CATALOG), []);
});

test("each flag is reported once however often it appears", () => {
  assert.equal(
    diagnoseExtraArgs("--tempp 1 --tempp 2 --tempp 3", CATALOG).length,
    1,
  );
});

test("several unknown flags share one line", () => {
  const diagnostics = diagnoseExtraArgs("--aaa 1 --bbb 2", CATALOG);
  assert.equal(diagnostics.length, 1);
  assert.match(diagnostics[0].message, /--aaa, --bbb/);
});

test("values are never mistaken for flags", () => {
  // "-1" and "0.7" are values; treating them as flags would warn about every
  // negative number the user types.
  assert.deepEqual(
    diagnoseExtraArgs("--numa distribute --top-k -1", CATALOG).map(
      (d) => d.level,
    ),
    ["note"],
  );
});

// --- the panel wiring, asserted on source ------------------------------------
// The harness has no DOM renderer, so the row's contract is pinned the way the
// sibling model-config tests do it.

const pageSource = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("the row stores argv tokens, not the typed string", () => {
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const body = row.slice(0, row.indexOf("\n}\n")).replace(/\s+/g, " ");
  // The wire format is one token per entry; storing the raw string would make the
  // backend split it, which it does not do.
  assert.match(
    body,
    /update\(\{ llamaExtraArgs: tokens\.length > 0 \? tokens : null \}\)/,
  );
  // Cleared reads as null here and becomes an explicit [] at the API boundary.
  assert.match(body, /const \{ tokens \} = parseExtraArgs\(next\)/);
});

test("the box is filled from the stored flags, not left looking empty", () => {
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const body = row.slice(0, row.indexOf("\n}\n")).replace(/\s+/g, " ");
  // The overrides API can set these with no UI involved, and this panel's config
  // comes from local storage, so the only way the box can show what is actually
  // set is to ask. An empty box would read as "no flags" and the first edit would
  // submit a list that dropped them.
  assert.match(body, /fetchModelOverrides\(\)/);
  // Through the resolver, not a literal lookup: the backend folds identities and
  // falls back from repo:QUANT to the bare repo before it reads a row.
  assert.match(
    body,
    /resolveStoredExtraArgs\( overrides, latest\.current\.overrideKeys, \)/,
  );
  // Into the config, not only the textarea. The load sends what the config holds,
  // and the route's omission path inherits from a resident process rather than
  // from this stored override, so a box that filled without the config would show
  // flags the launch did not use.
  assert.match(body, /latest\.current\.update\(\{ llamaExtraArgs: stored \}\)/);
  // And the key is marked only once a response is in hand, or StrictMode's replayed
  // effect cancels the first fetch and skips the second.
  const marked = body.indexOf("hydrated.current = keyIdentity");
  assert.ok(
    marked > body.indexOf("if (cancelled) { return; }"),
    "mark after the response",
  );
});

test("the row does not withdraw its objection when it unmounts", () => {
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const body = row.slice(0, row.indexOf("\n}\n")).replace(/\s+/g, " ");
  // Collapsing Advanced settings unmounts the row while its tokens stay in the
  // config and still go out with the load, so a cleanup that reset the flag would
  // re-enable Load for a request the backend refuses.
  const effect = body.slice(body.indexOf("onLoadableChange(loadable)"));
  assert.doesNotMatch(
    effect.slice(0, effect.indexOf("const commit")),
    /return \(\) => onLoadableChange/,
  );
  // The panel retires it on a model change instead.
  assert.match(
    pageSource.replace(/\s+/g, " "),
    /useEffect\(\(\) => \{ setExtraArgsLoadable\(true\); \}, \[configId, target\.ggufVariant\]\)/,
  );
});

test("a config that never read the stored value is not sent as a clear", () => {
  const overrides = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/api/model-overrides.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  ).replace(/\s+/g, " ");
  // The route preserves llama_extra_args when omitted, which is what kept CLI-set
  // flags alive while this panel had no control. Sending [] for a config that never
  // loaded them would wipe them on the first save.
  assert.match(
    overrides,
    /if \(config\.llamaExtraArgs !== undefined\) \{ payload\.llama_extra_args = config\.llamaExtraArgs \?\? \[\]; \}/,
  );
});

test("the load sends the flags only once they are known", () => {
  const composer = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    ),
    "utf8",
  ).replace(/\s+/g, " ");
  assert.match(
    composer,
    /ownConfig\.llamaExtraArgs !== undefined \? .* \{ llama_extra_args: ownConfig\.llamaExtraArgs \?\? \[\] \} : \{\}/,
  );
});

test("the panel's own Load goes through the runtime, which sends them too", () => {
  // Found by loading a model from the panel and reading the emitted command: the
  // flags were in the config and absent from the argv, because this hook is the
  // path the Load button takes and it built the payload field by field.
  const runtime = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/hooks/use-chat-model-runtime.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  ).replace(/\s+/g, " ");
  assert.match(
    runtime,
    /const loadLlamaExtraArgs = pendingLoadConfig\?\.llamaExtraArgs/,
  );
  assert.match(
    runtime,
    /isGguf && loadLlamaExtraArgs !== undefined \? \{ llama_extra_args: loadLlamaExtraArgs \?\? \[\] \} : \{\}/,
  );
});
