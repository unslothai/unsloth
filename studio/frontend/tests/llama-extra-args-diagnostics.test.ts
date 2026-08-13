// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { LlamaFlagCatalog } from "../src/features/model-picker/api/llama-flags.ts";
import {
  diagnoseExtraArgs,
  dropManagedExtraArgs,
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
    "--device": "device list",
    "-ngl": "layers to offload",
    "--batch-size": "logical batch size",
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

test("a device flag is called removed, not winning, when GPUs are picked", () => {
  // The launch strips these whenever gpu_ids is set (_strip_device_extra_args), so
  // the ordinary "passed last, yours wins" note would be a lie for them.
  const withPick = diagnoseExtraArgs("--device CUDA0", CATALOG, true);
  assert.equal(withPick[0].level, "warning");
  assert.match(withPick[0].message, /--device will be removed/);
  assert.match(withPick[0].message, /GPU selection/);
  // With no GPU picked the flag is the user's own and nothing is stripped.
  assert.deepEqual(diagnoseExtraArgs("--device CUDA0", CATALOG, false), []);
});

test("a flag that takes a number rejects a value that is not one", () => {
  // parse_ctx_override and parse_gpu_layers_override raise before the load starts,
  // so the row has to say so rather than enable a request that cannot succeed.
  // -ngl rather than --ctx-size: this build's catalogue has --ctx-size on the
  // managed list, and that refusal would fire first and prove nothing.
  const bad = diagnoseExtraArgs("-ngl many", CATALOG);
  assert.equal(bad[0].level, "error");
  assert.match(bad[0].message, /takes a number/);
  assert.equal(extraArgsAreLoadable(bad), false);
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs("--batch-size=abc", CATALOG)),
    false,
  );
  // A real value is fine, negative included: -1 means every layer.
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs("-ngl -1", CATALOG)),
    true,
  );
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs("--batch-size 512", CATALOG)),
    true,
  );
});

test("a control character is an error, as it is at the backend", () => {
  // The usual way one arrives is a command copied out of coloured terminal output.
  const diagnostics = diagnoseExtraArgs("--grammar a\u001b[0mb", CATALOG);
  assert.equal(diagnostics[0].level, "error");
  assert.match(diagnostics[0].message, /control characters/);
  assert.equal(extraArgsAreLoadable(diagnostics), false);
  // Tab and newline are separators here, not control characters to refuse.
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs("--top-k\t20", CATALOG)),
    true,
  );
});

// --- the panel wiring, asserted on source ------------------------------------
test("a repeated numeric flag is checked at every occurrence", () => {
  // llama.cpp reads the last one, and so does the backend's parse_gpu_layers_override,
  // so a check that stopped at the first occurrence left Load enabled for a request
  // that comes back 400.
  const out = diagnoseExtraArgs("-ngl 20 -ngl many", CATALOG);
  assert.ok(
    out.some(
      (d) => d.level === "error" && d.message.includes('"many" is not one'),
    ),
    JSON.stringify(out),
  );
});

test("the same bad value is reported once, not once per copy", () => {
  const out = diagnoseExtraArgs("-ngl many -ngl many", CATALOG);
  assert.equal(
    out.filter((d) => d.message.includes('"many" is not one')).length,
    1,
  );
});

test("a numeric flag with nothing after it is an error", () => {
  // parse_ctx_override raises on a missing value rather than reading the next flag
  // as one, so leaving Load enabled here only moves the failure to the backend.
  for (const input of ["--ctx-size", "--ctx-size=", "--ctx-size --numa"]) {
    const out = diagnoseExtraArgs(input, CATALOG);
    assert.ok(
      out.some((d) => d.level === "error" && d.message.includes("needs a number")),
      `${input}: ${JSON.stringify(out)}`,
    );
  }
});

test("a numeric flag outside its range is an error", () => {
  // The two ranges the backend's parsers actually enforce: a context cannot be
  // negative, and -1 is the lowest meaningful layer count (all of them).
  assert.ok(
    diagnoseExtraArgs("--ctx-size -1", CATALOG).some(
      (d) => d.level === "error" && d.message.includes("cannot be negative"),
    ),
  );
  assert.ok(
    diagnoseExtraArgs("-ngl -2", CATALOG).some(
      (d) => d.level === "error" && d.message.includes("-1 or more"),
    ),
  );
  // And -1 itself is fine, or the editor would refuse the ordinary way of asking
  // for every layer.
  assert.ok(
    !diagnoseExtraArgs("-ngl -1", CATALOG).some((d) => d.level === "error"),
  );
});

test("a stored flag this build refuses is dropped with its value", () => {
  // Hydration turns a stored list into an explicit request, which /load validates
  // strictly instead of dropping the flag the way the carry-over paths do. Leaving
  // the value behind would hand llama.cpp a bare positional it reads as a model.
  const managed = new Set(["--log-file", "--agent"]);
  assert.deepEqual(
    dropManagedExtraArgs(
      ["--log-file", "/var/log/llama.log", "--numa", "distribute"],
      managed,
    ),
    ["--numa", "distribute"],
  );
  assert.deepEqual(
    dropManagedExtraArgs(["--agent", "--numa", "distribute"], managed),
    ["--numa", "distribute"],
  );
  assert.deepEqual(
    dropManagedExtraArgs(["--log-file=/x", "--top-k=20"], managed),
    ["--top-k=20"],
  );
  // Nothing to drop is the list unchanged, and an empty denylist changes nothing.
  const clean = ["--numa", "distribute"];
  assert.deepEqual(dropManagedExtraArgs(clean, managed), clean);
  assert.deepEqual(dropManagedExtraArgs(clean, new Set<string>()), clean);
});

// --- which stored row a model reads ---------------------------------------------
// The module itself imports through the "@/" alias, which the node runner does not
// resolve, so this is pinned the way the sibling tests pin such modules.

const overridesSource = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/model-picker/api/model-overrides.ts",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("only a real quant suffix folds, never a colon inside a path", () => {
  const fold = overridesSource.slice(
    overridesSource.indexOf("function foldOverrideKey("),
  );
  const body = fold.slice(0, fold.indexOf("\n}\n")).replace(/\s+/g, " ");
  // "/models/foo:Bar.gguf" and "/models/foo:bar.gguf" are two real files on a
  // case-sensitive filesystem. Splitting on the last colon folded them onto one key,
  // and the panel would then hydrate from the wrong row and send that file's
  // arguments on Load. splitQuantSuffix is the check the backend mirrors.
  assert.match(body, /splitQuantSuffix\(key\)/);
  assert.doesNotMatch(body, /lastIndexOf\(":"\)/);
  // And a POSIX path still does not fold as a whole, only its quant.
  assert.match(body, /POSIX_PATH\.test\(id\)/);
});

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
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // The overrides API can set these with no UI involved, and this panel's config
  // comes from local storage, so the only way the box can show what is actually
  // set is to ask. An empty box would read as "no flags" and the first edit would
  // submit a list that dropped them.
  assert.match(body, /fetchModelOverrides\(\)/);
  // Through the resolver, not a literal lookup: the backend folds identities and
  // falls back from repo:QUANT to the bare repo before it reads a row.
  assert.match(body, /resolveStoredExtraArgs\(overrides, keys\)/);
  // And through the denylist first: hydrating makes the stored list an explicit
  // request, which /load validates strictly rather than dropping a newly denied
  // flag the way the carry-over paths do.
  assert.match(body, /dropManagedExtraArgs\( resolveStoredExtraArgs/);
  // Into the config, not only the textarea. The load sends what the config holds,
  // and the route's omission path inherits from a resident process rather than
  // from this stored override, so a box that filled without the config would show
  // flags the launch did not use.
  assert.match(body, /llamaExtraArgs: stored/);
  // And the key is marked only once a response is in hand, or StrictMode's replayed
  // effect cancels the first fetch and skips the second.
  const marked = body.indexOf("extraArgsHydrated.current = identity");
  assert.ok(
    marked > body.indexOf("if (cancelled) { return; }"),
    "mark after the response",
  );
});

test("hydration is not gated behind the advanced disclosure", () => {
  // The row that displays these lives inside GgufAdvancedSettings, which is only
  // rendered while the section is open. A panel opened with it collapsed would
  // never fetch, and a cold load would then launch without the stored arguments,
  // so the fetch belongs in the parent that always mounts.
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const rowBody = row.slice(0, row.indexOf("\n}\n"));
  assert.doesNotMatch(rowBody, /fetchModelOverrides/);
  const advanced = pageSource.slice(
    pageSource.indexOf("function GgufAdvancedSettings("),
    pageSource.indexOf("export function ModelConfigPage("),
  );
  assert.doesNotMatch(advanced, /fetchModelOverrides/);
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
    /setExtraArgsLoadable\(true\); setExtraArgsHydrating\(true\); \}, \[configId, target\.ggufVariant\]\)/,
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

test("the box follows a config change it did not make", () => {
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const body = row.slice(0, row.indexOf("\n}\n")).replace(/\s+/g, " ");
  // Reset and the parent's hydration both replace llamaExtraArgs while this row is
  // mounted. Without this the textarea keeps its old text and disagrees with what
  // Load sends; with a plain re-seed on every change it re-quotes a half-typed line.
  assert.match(body, /if \(external === selfWritten\.current\) \{ return; \}/);
  assert.match(body, /selfWritten\.current = formatExtraArgs\(/);
});

test("load waits for the stored arguments to be read", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // A click that beats the fetch would launch a cold model without them, and /load
  // cannot inherit from a process that is not running.
  assert.match(body, /extraArgsHydrating \|\|/);
  // But never for good: a failed or hanging overrides read releases the gate.
  assert.match(body, /\.finally\(\(\) => \{ .*setExtraArgsHydrating\(false\)/);
  assert.match(body, /setTimeout\(\(\) => setExtraArgsHydrating\(false\), \d+\)/);
});
