// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import type { LlamaFlagCatalog } from "../src/features/model-picker/api/llama-flags.ts";
import {
  diagnoseExtraArgs,
  dropManagedExtraArgs,
  extraArgsAreLoadable,
  parseExtraArgs,
  sanitizeStoredExtraArgs,
} from "../src/features/model-picker/model-config/llama-extra-args.ts";

import { readSrc } from "./helpers/kit.ts";

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
  // --n-parallel is in the backend denylist, so aliases are included.
  managed: new Set([
    "--parallel",
    "--n-parallel",
    "-np",
    "--model",
    "--api-key",
    "--agent",
    "--ctx-size",
  ]),
  switches: new Set(["--verbose"]),
  maxBytes: 32 * 1024,
  windowsCommandBudget: 0,
  defaultParallelSlots: 0,
  parallelSlotsClamped: false,
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

test("parallel aliases point at the supported control", () => {
  for (const catalog of [CATALOG, null]) {
    for (const input of [
      "--parallel 8",
      "--parallel=8",
      "--n-parallel 8",
      "--n_parallel 8",
      "-np 8",
      "-np8",
    ]) {
      const diagnostics = diagnoseExtraArgs(input, catalog);
      assert.equal(diagnostics.length, 1);
      assert.match(
        diagnostics[0]?.message ?? "",
        /is set by Parallel Slots above and cannot be passed here\.$/,
      );
      assert.equal(diagnostics[0]?.level, "error");
      assert.equal(extraArgsAreLoadable(diagnostics), false);
    }
  }
});

test("a managed flag with no control says who owns it instead", () => {
  const text = messages("--api-key secret");
  assert.match(text, /managed by Unsloth/);
  assert.doesNotMatch(text, /above/);
});

test("an attached or equals form is caught the same way", () => {
  assert.equal(levels("-np8")[0], "error");
  assert.equal(levels("--parallel=8")[0], "error");
  assert.equal(levels("--n_parallel 8")[0], "error");
});

test("a flag a control also sets is a note, not a refusal", () => {
  // The backend appends extras last, so this is a note about who wins, not a block.
  for (const catalog of [CATALOG, null]) {
    const diagnostics = diagnoseExtraArgs("--batch-size 512", catalog);
    assert.deepEqual(
      diagnostics.map((d) => d.level),
      ["note"],
    );
    assert.match(diagnostics[0].message, /Batch Size/);
    assert.equal(extraArgsAreLoadable(diagnostics), true);
  }
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
  // An unverifiable build must not mark every correct flag as a typo.
  const unverified: LlamaFlagCatalog = {
    flags: {},
    managed: CATALOG.managed,
    switches: new Set<string>(),
    maxBytes: 32 * 1024,
    windowsCommandBudget: 0,
    defaultParallelSlots: 0,
    parallelSlotsClamped: false,
    probeOk: false,
  };
  assert.deepEqual(
    diagnoseExtraArgs("--tempp 0.7 --numa distribute", unverified),
    [],
  );
  assert.equal(levels("--parallel 8", unverified)[0], "error");
});

test("an older backend with no catalogue still refuses nothing and warns nothing", () => {
  assert.deepEqual(diagnoseExtraArgs("--tempp 0.7", null), []);
});

test("sampling flags are noted as launch defaults", () => {
  // Chat settings send sampling per request, so a value set here is not what a conversation uses.
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
  const huge = `--grammar ${"a".repeat(40_000)}`;
  const diagnostics = diagnoseExtraArgs(huge, CATALOG);

  assert.equal(diagnostics[0].level, "error");
  assert.match(diagnostics[0].message, /limit 32768/);
  assert.equal(extraArgsAreLoadable(diagnostics), false);
  // Counted as UTF-8 bytes, which is what the backend measures.
  const multibyte = `--grammar ${"\u65e5".repeat(11_000)}`;
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs(multibyte, CATALOG)),
    false,
  );
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
  // "-1" and "0.7" are values, not flags.
  assert.deepEqual(
    diagnoseExtraArgs("--numa distribute --top-k -1", CATALOG).map(
      (d) => d.level,
    ),
    ["note"],
  );
});

test("a device flag is called removed, not winning, when GPUs are picked", () => {
  // The launch strips these whenever gpu_ids is set (_strip_device_extra_args).
  const withPick = diagnoseExtraArgs("--device CUDA0", CATALOG, { gpuSelectionActive: true });
  assert.equal(withPick[0].level, "warning");
  assert.match(withPick[0].message, /--device will be removed/);
  assert.match(withPick[0].message, /GPU selection/);
  assert.deepEqual(diagnoseExtraArgs("--device CUDA0", CATALOG, {}), []);
});

test("a flag that takes a number rejects a value that is not one", () => {
  // -ngl rather than --ctx-size: --ctx-size is managed here, and that refusal would fire first.
  const bad = diagnoseExtraArgs("-ngl many", CATALOG);
  assert.equal(bad[0].level, "error");
  assert.match(bad[0].message, /takes a number/);
  assert.equal(extraArgsAreLoadable(bad), false);
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs("--batch-size=abc", CATALOG)),
    false,
  );
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
  const diagnostics = diagnoseExtraArgs("--grammar a\u001b[0mb", CATALOG);
  assert.equal(diagnostics[0].level, "error");
  assert.match(diagnostics[0].message, /control characters/);
  assert.equal(extraArgsAreLoadable(diagnostics), false);
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs("--top-k\t20", CATALOG)),
    true,
  );
});

test("a repeated numeric flag is checked at every occurrence", () => {
  // llama.cpp and parse_gpu_layers_override read the last occurrence.
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
  for (const input of ["--ctx-size", "--ctx-size=", "--ctx-size --numa"]) {
    const out = diagnoseExtraArgs(input, CATALOG);
    assert.ok(
      out.some((d) => d.level === "error" && d.message.includes("needs a number")),
      `${input}: ${JSON.stringify(out)}`,
    );
  }
});

test("a numeric flag outside its range is an error", () => {
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
  assert.ok(
    !diagnoseExtraArgs("-ngl -1", CATALOG).some((d) => d.level === "error"),
  );
});

test("a stored flag this build refuses is dropped with its value", () => {
  // /load validates an explicit list strictly; a leftover value becomes a bare positional (model).
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
  const clean = ["--numa", "distribute"];
  assert.deepEqual(dropManagedExtraArgs(clean, managed), clean);
  assert.deepEqual(dropManagedExtraArgs(clean, new Set<string>()), clean);
});

test("a value-taking flag with nothing after it is an error too", () => {
  // _last_flag_value raises for these groups inside validate_extra_args.
  for (const input of ["--cache-type-k=", "--top-k 20 -sm", "--split-mode="]) {
    const out = diagnoseExtraArgs(input, CATALOG);
    assert.ok(
      out.some((d) => d.level === "error" && d.message.includes("needs a value")),
      `${input}: ${JSON.stringify(out)}`,
    );
  }
  assert.ok(
    !diagnoseExtraArgs("--cache-type-k q8_0", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("the stored sanitizer removes everything this build would refuse", () => {
  // Each mirrors a case pinned against drop_managed_flags in the backend suite.
  const managed = new Set(["--log-file"]);
  const control = `${String.fromCharCode(0x1b)}[2Jx`;
  const surrogate = String.fromCharCode(0xd800);

  assert.deepEqual(
    sanitizeStoredExtraArgs(
      ["--log-file", "/var/log/llama.log", "--numa", "distribute"],
      managed,
    ),
    ["--numa", "distribute"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--chat-template", control, "--top-k", "20"], managed),
    ["--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(
      ["--chat-template", surrogate, "--top-k", "20"],
      managed,
    ),
    ["--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs([`--grammar${control}`, "root", "--top-k", "20"], managed),
    ["--top-k", "20"],
  );
  assert.equal(
    sanitizeStoredExtraArgs(new Array(300).fill("--verbose"), managed).length,
    256,
  );
  const clean = ["--numa", "distribute"];
  assert.deepEqual(sanitizeStoredExtraArgs(clean, managed), clean);
});

test("the sanitizer trims to the HOST's bounds, not the constants", () => {
  // A Windows server takes 24 KiB, not 32, plus a quoted-command budget.
  const managed = new Set<string>();
  const stored = ["--grammar", "x".repeat(30000), "--top-k", "20"];
  assert.deepEqual(sanitizeStoredExtraArgs(stored, managed), stored);
  assert.deepEqual(
    sanitizeStoredExtraArgs(stored, managed, { maxBytes: 24 * 1024 }),
    [],
  );
  // The quoted length: quoting doubles backslash runs before quotes.
  const quoted = ["--grammar", `${"\\".repeat(10)}" `.repeat(400)];
  assert.deepEqual(sanitizeStoredExtraArgs(quoted, managed), quoted);
  assert.deepEqual(
    sanitizeStoredExtraArgs(quoted, managed, {
      maxBytes: 24 * 1024,
      windowsCommandBudget: 8192,
    }),
    [],
  );
  // Zero means "not known": an older server answers without either field.
  assert.deepEqual(
    sanitizeStoredExtraArgs(stored, managed, {
      maxBytes: 0,
      windowsCommandBudget: 0,
    }),
    stored,
  );
});

test("a two-value flag is shed whole when the bounds bite", () => {
  // Mirrors drop_managed_flags: dropping END alone leaves START looking like a value.
  const managed = new Set<string>();
  assert.deepEqual(
    sanitizeStoredExtraArgs(
      ["--top-k", "20", "--control-vector-layer-range", "1", "x".repeat(40000)],
      managed,
    ),
    ["--top-k", "20"],
  );
  const whole = ["--control-vector-layer-range", "1", "10"];
  assert.deepEqual(sanitizeStoredExtraArgs(whole, managed), whole);
});

test("a whole surrogate pair is a character, not a fault", () => {
  // Only half a surrogate pair is refused; Python encodes a whole emoji fine.
  const emoji = String.fromCodePoint(0x1f600);
  const lone = String.fromCharCode(0xd800);
  const managed = new Set<string>();

  assert.deepEqual(
    sanitizeStoredExtraArgs(["--chat-template", `hi ${emoji}`, "--top-k", "20"], managed),
    ["--chat-template", `hi ${emoji}`, "--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--chat-template", lone, "--top-k", "20"], managed),
    ["--top-k", "20"],
  );
  assert.ok(
    !diagnoseExtraArgs(`--chat-template ${emoji}`, CATALOG).some(
      (d) => d.level === "error",
    ),
  );
  assert.ok(
    diagnoseExtraArgs(`--chat-template ${lone}`, CATALOG).some(
      (d) => d.level === "error" && d.message.includes("incomplete character"),
    ),
  );
});

test("a multi-line quoted value is not a control-character fault", () => {
  // _has_control_characters allows tab and newline: grammars and templates are multi-line.
  const grammar = "--grammar 'root ::= [0-9]+\n  | \"x\"'";
  assert.ok(
    !diagnoseExtraArgs(grammar, CATALOG).some((d) => d.level === "error"),
    JSON.stringify(diagnoseExtraArgs(grammar, CATALOG)),
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--grammar", "a\nb\tc", "--top-k", "20"], new Set<string>()),
    ["--grammar", "a\nb\tc", "--top-k", "20"],
  );
  assert.ok(
    diagnoseExtraArgs(`--grammar ${String.fromCharCode(0x1b)}x`, CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("a value with no flag in front of it is an error", () => {
  const out = diagnoseExtraArgs("--top-k 20 /models/other.gguf", CATALOG);
  assert.ok(
    out.some((d) => d.level === "error" && d.message.includes("belongs to no flag")),
    JSON.stringify(out),
  );
  assert.ok(
    !diagnoseExtraArgs("--control-vector-layer-range 1 10", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
  assert.ok(
    !diagnoseExtraArgs("--numa distribute", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("a value after a switch is the typo llama-server calls it", () => {
  const out = diagnoseExtraArgs("--verbose foo", CATALOG);
  assert.ok(
    out.some((d) => d.level === "error" && d.message.includes("belongs to no flag")),
    JSON.stringify(out),
  );
  assert.ok(
    !diagnoseExtraArgs("--numa distribute", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("the size limits are the host's, not this file's", () => {
  // Windows caps extras lower: the whole command line shares one 32767 character budget.
  const windows: LlamaFlagCatalog = {
    ...CATALOG,
    maxBytes: 24 * 1024,
    windowsCommandBudget: 24575,
  };
  const big = `--grammar ${"x".repeat(25 * 1024)}`;
  assert.ok(
    diagnoseExtraArgs(big, windows).some(
      (d) => d.level === "error" && d.message.includes("24576"),
    ),
  );
  assert.ok(!diagnoseExtraArgs(big, CATALOG).some((d) => d.level === "error"));
});

test("what the quoting makes of an argument counts on Windows", () => {
  // list2cmdline doubles a backslash run before a quote, so bytes alone do not decide fit.
  const windows: LlamaFlagCatalog = {
    ...CATALOG,
    maxBytes: 24 * 1024,
    windowsCommandBudget: 24575,
  };
  const escaped = `${"\\".repeat(10)}"`.repeat(2000);
  assert.ok(escaped.length < 24 * 1024);
  assert.ok(
    diagnoseExtraArgs(`--grammar ${JSON.stringify(escaped)}`, windows).some(
      (d) => d.level === "error" && d.message.includes("after quoting"),
    ),
  );
});

// No DOM renderer, so the row's contract is pinned on source.

const pageSource = readSrc("features/model-picker/components/model-config-page.tsx");

test("the row stores argv tokens, not the typed string", () => {
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const body = row.slice(0, row.indexOf("\n}\n")).replace(/\s+/g, " ");
  // The wire format is one token per entry; the backend does not split strings.
  assert.match(
    body,
    /update\(\{ llamaExtraArgs: tokens\.length > 0 \? tokens : null \}\)/,
  );
  assert.match(body, /const \{ tokens \} = parseExtraArgs\(next\)/);
});

test("the box is filled from the stored flags, not left looking empty", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // The overrides API can set these without the UI, so the box must fetch what is actually set.
  assert.match(
    body,
    /fetchLoadModelOverride\(loadId, configId, target\.ggufVariant, keys\)/,
  );
  // Through the resolver: the backend folds identities and falls back from repo:QUANT to repo.
  // Candidate keys still travel for backends predating the resolving parameter.
  assert.match(body, /modelOverrideKey\(loadId, target\.ggufVariant\)/);
  assert.match(body, /sanitizeStoredExtraArgs\( resolvedArgs\.tokens,/);
  // Into the config, not only the textarea: the load sends what the config holds.
  assert.match(body, /llamaExtraArgs: stored/);
  // Mark only once a response is in hand, or StrictMode's replayed effect skips the second fetch.
  const marked = body.indexOf("markExtraArgsHydratedForDraft(draftKey)");
  assert.ok(
    marked > body.indexOf("if (cancelled) { return; }"),
    "mark after the response",
  );
});

test("hydration is not gated behind the advanced disclosure", () => {
  // The row only renders while Advanced is open, so the fetch lives in the always-mounted parent.
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const rowBody = row.slice(0, row.indexOf("\n}\n"));
  assert.doesNotMatch(rowBody, /fetchLoadExtraArgs/);
  const advanced = pageSource.slice(
    pageSource.indexOf("function GgufAdvancedSettings("),
    pageSource.indexOf("export function ModelConfigPage("),
  );
  assert.doesNotMatch(advanced, /fetchLoadExtraArgs/);
});

test("the row does not withdraw its objection when it unmounts", () => {
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const body = row.slice(0, row.indexOf("\n}\n")).replace(/\s+/g, " ");
  // Collapsing unmounts the row while its tokens still go out, so cleanup must not reset the flag.
  const effect = body.slice(body.indexOf("onLoadableChange(loadable)"));
  assert.doesNotMatch(
    effect.slice(0, effect.indexOf("const commit")),
    /return \(\) => onLoadableChange/,
  );
  assert.match(
    pageSource.replace(/\s+/g, " "),
    /setExtraArgsLoadable\(true\); setExtraArgsHydrating\(!isDiffusion\); \}, \[configId, target\.ggufVariant, target\.isGguf, isDiffusion\]\)/,
  );
});

test("a config that never read the stored value is not sent as a clear", () => {
  const overrides = readSrc("features/model-picker/api/model-overrides.ts").replace(/\s+/g, " ");
  // The route preserves llama_extra_args when omitted; sending [] unloaded would wipe CLI flags.
  assert.match(
    overrides,
    /if \(config\.llamaExtraArgs !== undefined\) \{ payload\.llama_extra_args = config\.llamaExtraArgs \?\? \[\]; \}/,
  );
});

test("the load sends the flags only once they are known", () => {
  const composer = readSrc("features/chat/shared-composer.tsx").replace(/\s+/g, " ");
  assert.match(
    composer,
    /ownConfig\.llamaExtraArgs !== undefined \? .* \{ llama_extra_args: ownConfig\.llamaExtraArgs \?\? \[\] \} : \{\}/,
  );
});

test("the panel's own Load goes through the runtime, which sends them too", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts").replace(/\s+/g, " ");
  assert.match(
    runtime,
    /const loadLlamaExtraArgs = pendingLoadConfig\?\.llamaExtraArgs/,
  );
  assert.match(
    runtime,
    /isGguf && !targetIsDiffusion && loadLlamaExtraArgs !== undefined \? \{ llama_extra_args: loadLlamaExtraArgs \?\? \[\] \} : \{\}/,
  );
});

test("the box follows a config change it did not make", () => {
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const body = row.slice(0, row.indexOf("\n}\n")).replace(/\s+/g, " ");
  // Re-seed on external replacement, but not on every change, or a half-typed line gets re-quoted.
  assert.match(
    body,
    /const text = edit && edit\.source === external \? edit\.text : external;/,
  );
  assert.match(body, /source: formatExtraArgs\(tokens\.length > 0 \? tokens : null\),/);
});

test("the two editors type into one box", () => {
  const row = pageSource.slice(pageSource.indexOf("function ExtraArgsRow("));
  const body = row.slice(0, row.indexOf("\n}\n")).replace(/\s+/g, " ");
  // Edits are shared per draft so the other editor does not judge a re-quoted line loadable.
  assert.match(body, /readExtraArgsEditForDraft\(draftKey\)/);
  assert.match(body, /setExtraArgsEditForDraft\(draftKey, \{/);
  assert.doesNotMatch(body, /useState\(\(\) => formatExtraArgs/);
});

test("load waits for the stored arguments to be read", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // /load cannot inherit from a process that is not running.
  assert.match(body, /extraArgsHydrating \|\|/);
  assert.match(body, /\.finally\(\(\) => \{ .*setExtraArgsHydrating\(false\)/);
  assert.match(body, /setTimeout\(\(\) => setExtraArgsHydrating\(false\), \d+\)/);
  // The short deadline is on the catalogue, not the gate; the gate waits on the denylist.
  assert.match(body, /loadManagedLlamaFlags\(\)/);
  assert.doesNotMatch(body, /loadLlamaFlagCatalog\(\)[^;]*Promise\.all/);
});

test("Load waits for the server row on every model but diffusion", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // Diffusion models run through the shim, which appends no llama-server flags.
  assert.match(
    body,
    /if \(resolvedIsDiffusion\) \{ [^}]*setExtraArgsHydrating\(false\); return; \}/,
  );
  assert.match(body, /useState\( \(\) => !isDiffusion, \)/);
  assert.match(body, /setExtraArgsHydrating\(!isDiffusion\);/);
  assert.doesNotMatch(body, /setExtraArgsHydrating\(true\)/);
});

test("a diffusion classification retires the argument objection", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  assert.match(
    body,
    /if \(resolvedIsDiffusion\) \{ setExtraArgsLoadable\(true\); \}/,
  );
});

test("hydration asks under the keys the load path uses", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // The auto-switch loader reads the path-qualified key first for snapshot-path GGUFs.
  assert.match(body, /modelOverrideKey\(loadId, target\.ggufVariant\), modelOverrideKey\(configId, target\.ggufVariant\), loadId,/);
  assert.match(body, /fileVariant \? \[`\$\{loadId\}:\$\{fileVariant\}`\] : \[\]/);
});

test("a rollback restores the previous model with its arguments", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts").replace(/\s+/g, " ");
  assert.match(
    runtime,
    /rollbackState\.loadedLlamaExtraArgs != null \? \{ llama_extra_args: rollbackState\.loadedLlamaExtraArgs \}/,
  );
  assert.match(
    runtime,
    /loadedLlamaExtraArgs: loadResponse\.requested_llama_extra_args !== undefined/,
  );
  assert.match(runtime, /: loadLlamaExtraArgs !== undefined/);
});

test("a hydrated list is judged even when the row cannot be", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // With Advanced collapsed the row never mounts, so the adopted list must be judged here.
  assert.match(
    body,
    /const hydratedArgs = serverConfig\?\.llamaExtraArgs \?\? stored;/,
  );
  assert.match(
    body,
    /const hydratedIsLoadable = !target\.isGguf \|\| hydratedArgs\.length === 0 \? true : extraArgsAreLoadable\( diagnoseExtraArgs\( formatExtraArgs\(hydratedArgs\)/,
  );
  assert.match(
    body,
    /if \(configRef\.current\.llamaExtraArgs !== undefined\) \{ .*return; \} setExtraArgsLoadable\(hydratedIsLoadable\)/,
  );
});

test("the runtime preflight is sized with the arguments the load sends", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts").replace(/\s+/g, " ");
  // A --ctx-size or cache override changes the memory /validate estimates.
  const validateCall = runtime.slice(
    runtime.indexOf("await validateModel({"),
    runtime.indexOf("// Upgrade consent runs before"),
  );
  assert.match(
    validateCall,
    /loadLlamaExtraArgs !== undefined \? \{ llama_extra_args: loadLlamaExtraArgs \?\? \[\] \}/,
  );
});

test("a catalogue read from the previous binary is discarded", () => {
  const flagsApi = readSrc("features/model-picker/api/llama-flags.ts").replace(/\s+/g, " ");
  assert.match(flagsApi, /catalogGeneration \+= 1;/);
  assert.match(flagsApi, /inFlightCatalog = null;/);
  assert.match(
    flagsApi,
    /if \(generation !== catalogGeneration\) \{ .*return null; \}/,
  );
});

test("a catalogued flag left without its value is refused", () => {
  // The catalogue is the only place a flag's arity is known; the backend cannot ask the binary.
  assert.ok(levels("--rope-scaling").includes("error"));
  assert.ok(levels("--top-k 20 --numa").includes("error"));
  assert.deepEqual(
    diagnoseExtraArgs("--numa", CATALOG).map((d) => d.message),
    ["--numa needs a value after it."],
  );
  assert.deepEqual(diagnoseExtraArgs("--numa distribute", CATALOG), []);
  assert.ok(!levels("--verbose").includes("error"));
  // llama.cpp looks the whole token up, so "--numa=distribute" is invalid (b10342, b10360).
  assert.deepEqual(
    diagnoseExtraArgs("--numa=distribute", CATALOG).map((d) => d.message),
    ['llama-server does not read "--numa=value". Write --numa and its value as two arguments.'],
  );
});

test("an unverified flag keeps the benefit of the doubt at the end", () => {
  const unverified: LlamaFlagCatalog = {
    flags: {},
    managed: new Set<string>(),
    switches: new Set<string>(),
    maxBytes: 0,
    windowsCommandBudget: 0,
    defaultParallelSlots: 0,
    parallelSlotsClamped: false,
    probeOk: false,
  };
  assert.deepEqual(diagnoseExtraArgs("--rope-scaling", unverified), []);
  assert.deepEqual(diagnoseExtraArgs("--rope-scaling", null), []);
  assert.deepEqual(levels("--tempp"), ["warning"]);
  assert.ok(!levels("--tempp").includes("error"));
});

test("a two-value flag left short is refused whatever the catalogue says", () => {
  const unverified: LlamaFlagCatalog = {
    flags: {},
    managed: new Set<string>(),
    switches: new Set<string>(),
    maxBytes: 0,
    windowsCommandBudget: 0,
    defaultParallelSlots: 0,
    parallelSlotsClamped: false,
    probeOk: false,
  };
  assert.equal(levels("--control-vector-layer-range 1", unverified)[0], "error");
  assert.equal(levels("--control-vector-layer-range", unverified)[0], "error");
  assert.deepEqual(
    diagnoseExtraArgs("--control-vector-layer-range 1 10", unverified),
    [],
  );
  // llama.cpp has no attached spelling for this option to be half of.
  for (const text of [
    "--control-vector-layer-range=1",
    "--control-vector-layer-range=1 10",
  ]) {
    assert.ok(
      diagnoseExtraArgs(text, unverified).some(
        (d) => d.level === "error" && d.message.includes("does not read"),
      ),
      text,
    );
  }
});

test("Manual GPU memory reports the offload flags it removes", () => {
  // Manual mode strips offload flags (strip_shadowing_flags); only the layer count is translated.
  const manual = (input: string) =>
    diagnoseExtraArgs(input, CATALOG, { manualGpuMemory: true }).map((d) => d.message);
  assert.ok(
    manual("--n-cpu-moe 10")[0].includes("will be removed"),
    manual("--n-cpu-moe 10")[0],
  );
  assert.ok(manual("-ncmoe 10")[0].includes("-ncmoe will be removed"));
  assert.ok(manual("--fit on")[0].includes("will be removed"));
  assert.ok(!manual("--n-cpu-moe 10").includes("error"));
  assert.equal(
    diagnoseExtraArgs("--n-cpu-moe 10", CATALOG, { manualGpuMemory: true }).every(
      (d) => d.level !== "error",
    ),
    true,
  );
  assert.ok(
    diagnoseExtraArgs("--n-cpu-moe 10", CATALOG, {})
      .map((d) => d.message)
      .join(" ")
      .includes("wins"),
  );
  assert.ok(
    diagnoseExtraArgs("-ngl 20", CATALOG, { manualGpuMemory: true })
      .map((d) => d.message)
      .join(" ")
      .includes("wins"),
  );
});

test("a pass-through batch below the floor is refused", () => {
  // The loader raises --batch-size to max(slots, 2), but a pass-through -b is appended after it
  // and wins, so llama-server aborts on GGML_ASSERT.
  const at = (input: string, batchFloor: number) =>
    diagnoseExtraArgs(input, CATALOG, { batchFloor });
  assert.ok(at("-b 1", 2).some((d) => d.level === "error"));
  assert.ok(at("--batch-size 0", 2).some((d) => d.level === "error"));
  assert.ok(at("--batch-size 4", 8).some((d) => d.level === "error"));
  assert.match(
    at("--batch-size 4", 8).filter((d) => d.level === "error")[0].message,
    /8 parallel slot/,
  );
  const errors = (input: string, batchFloor: number) =>
    at(input, batchFloor).filter((d) => d.level === "error");
  assert.deepEqual(errors("--batch-size 8", 8), []);
  assert.deepEqual(errors("-b 2", 2), []);
  assert.deepEqual(errors("-ub 1", 2), []);
  assert.deepEqual(errors("-b 2", 1), []);
});

test("Model Memory reports the flags its settings remove", () => {
  // apply_model_memory_policy runs before extras reach the command line.
  const keep = (input: string) =>
    diagnoseExtraArgs(input, CATALOG, { keepResident: true }).map(
      (d) => d.message,
    );
  const noReserve = (input: string) =>
    diagnoseExtraArgs(input, CATALOG, { noRamReserve: true }).map(
      (d) => d.message,
    );
  assert.match(keep("--mlock")[0], /will be removed/);
  assert.match(keep("--load-mode mmap")[0], /Keep model in GPU memory/);
  assert.match(noReserve("--no-mmap")[0], /Don't reserve system RAM/);
  assert.equal(
    noReserve("--direct-io").some((message) => /will be removed/.test(message)),
    false,
  );
  assert.match(
    diagnoseExtraArgs("--mlock", CATALOG, {
      keepResident: true,
      noRamReserve: true,
    })[0].message,
    /Don't reserve system RAM/,
  );
  assert.equal(
    diagnoseExtraArgs("--mlock", CATALOG, {}).some((d) =>
      /will be removed/.test(d.message),
    ),
    false,
  );
});

test("llama.cpp's underscore spelling is not read as an attached value", () => {
  // _flag_name folds --ctx_size to --ctx-size, and the binary takes both spellings.
  assert.deepEqual(
    diagnoseExtraArgs("--numa distribute", CATALOG).filter(
      (d) => d.level === "error",
    ),
    [],
  );
  assert.deepEqual(
    diagnoseExtraArgs("--rope_scaling yarn", CATALOG).filter(
      (d) => d.level === "error",
    ),
    [],
  );
  assert.ok(
    diagnoseExtraArgs("--rope_scaling yarn --numa", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
  assert.ok(
    diagnoseExtraArgs("--n_parallel 8", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("a flag that interrupts another's value is refused", () => {
  // --numa is left without its value; an end-of-input check misses it once the next flag lands.
  const messages = (input: string) =>
    diagnoseExtraArgs(input, CATALOG)
      .filter((d) => d.level === "error")
      .map((d) => d.message);
  assert.deepEqual(messages("--numa --verbose"), [
    "--numa needs a value after it.",
  ]);
  assert.deepEqual(messages("--numa --numa distribute"), [
    "--numa needs a value after it.",
  ]);
  assert.deepEqual(messages("--verbose --numa distribute"), []);
  assert.deepEqual(messages("--tempp --numa distribute"), []);
});

test("the batch floor follows the server-wide slot default", () => {
  // With Slots blank the launch uses the server-wide --parallel (4 in run.py).
  const withDefault = { ...CATALOG, defaultParallelSlots: 4 };
  assert.ok(
    diagnoseExtraArgs("-b 2", withDefault, { batchFloor: 4 }).some(
      (d) => d.level === "error",
    ),
  );
  assert.deepEqual(
    diagnoseExtraArgs("-b 4", withDefault, { batchFloor: 4 }).filter(
      (d) => d.level === "error",
    ),
    [],
  );
});

test("the sanitizer drops what the validator refuses on shape", () => {
  // This mirror trims by size only, so it must know drop_managed_flags' rules.
  const managed = new Set<string>();
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--top-k", "20", "stray"], managed),
    ["--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["stray", "--top-k", "20"], managed),
    ["--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(
      ["--top-k", "20", "--control-vector-layer-range", "1"],
      managed,
    ),
    ["--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(
      ["--top-k", "20", "--control-vector-layer-range=1", "x".repeat(40000)],
      managed,
    ),
    ["--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--control-vector-layer-range=1", "10"], managed),
    [],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--top-k=20", "--numa", "distribute"], managed),
    ["--numa", "distribute"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--ctx-size", "abc", "--top-k", "20"], managed),
    ["--top-k", "20"],
  );
  assert.deepEqual(sanitizeStoredExtraArgs(["--cache-type-k"], managed), []);
  for (const list of [
    ["--ctx-size", "0"],
    ["-ngl", "-1"],
    ["--cache-type-k", "q8_0"],
    ["--numa", "distribute"],
    ["--ctx_size", "4096"],
  ]) {
    assert.deepEqual(sanitizeStoredExtraArgs(list, managed), list);
  }
});

test("a scaled sidecar may take its scale as a second token", () => {
  // The scale is optional: current llama.cpp writes FNAME:SCALE, older builds FNAME SCALE.
  const scaled: LlamaFlagCatalog = {
    ...CATALOG,
    flags: {
      ...CATALOG.flags,
      "--lora-scaled": "path with scaling",
      "--control-vector-scaled": "control vector with scaling",
    },
  };
  const errors = (input: string) =>
    diagnoseExtraArgs(input, scaled).filter((d) => d.level === "error");
  assert.deepEqual(errors("--lora-scaled /a.gguf 0.5"), []);
  assert.deepEqual(errors("--lora-scaled /a.gguf:0.5"), []);
  assert.deepEqual(errors("--control-vector-scaled /v.gguf 0.8 --top-k 20"), []);
  assert.deepEqual(errors("--lora-scaled /a.gguf"), []);
  assert.equal(errors("--lora-scaled /a.gguf 0.5 stray").length, 1);
  const managed = new Set<string>();
  for (const list of [
    ["--lora-scaled", "/a.gguf", "0.5"],
    ["--lora-scaled", "/a.gguf:0.5"],
    ["--control-vector-scaled", "/v.gguf", "0.8", "--top-k", "20"],
  ]) {
    assert.deepEqual(sanitizeStoredExtraArgs(list, managed), list);
  }
});

test("a trimmed value never leaves its flag behind, whatever the spelling", () => {
  // The flag whose value was shed goes too, underscore spelling ("--grammar_file") included.
  const managed = new Set<string>();
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--numa", "distribute", "--grammar_file", "x".repeat(40000)], managed),
    ["--numa", "distribute"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--numa", "distribute", "--grammar-file", "x".repeat(40000)], managed),
    ["--numa", "distribute"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--numa", "distribute", "x".repeat(40000)], managed),
    ["--numa", "distribute"],
  );
});

test("the attached spelling is refused wherever it is judged", () => {
  // llama.cpp folds only underscores, so "--top-k=20" is an invalid argument.
  const managed = new Set<string>();
  for (const text of ["--top-k=20", "--rope-scaling=yarn", "--flash-attn=on"]) {
    assert.ok(
      diagnoseExtraArgs(text, CATALOG).some(
        (d) => d.level === "error" && d.message.includes("does not read"),
      ),
      text,
    );
  }
  assert.ok(
    diagnoseExtraArgs("--parallel=8", CATALOG).every(
      (d) => !d.message.includes("does not read"),
    ),
  );
  assert.ok(
    diagnoseExtraArgs("--override-kv a=int:2", CATALOG).every(
      (d) => d.level !== "error",
    ),
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--override-kv", "a=int:2"], managed),
    ["--override-kv", "a=int:2"],
  );
});

test("the managed answer is invalidated with the catalogue", () => {
  const flagsApi = readSrc("features/model-picker/api/llama-flags.ts").replace(/\s+/g, " ");
  // defaultParallelSlots depends on the build, so a llama.cpp update must clear the cache.
  assert.match(flagsApi, /cachedManaged = null; inFlightManaged = null;/);
  assert.match(flagsApi, /defaultParallelSlots: number;/);
  assert.match(flagsApi, /parallelSlotsClamped: boolean;/);
});

test("a flag quoted with stray spaces is refused, not silently sent", () => {
  // parseExtraArgs keeps quoted whitespace; llama.cpp looks the whole "--top-k " token up.
  assert.ok(
    diagnoseExtraArgs("'--top-k ' 20", CATALOG).some(
      (d) => d.level === "error" && d.message.includes("Remove the spaces"),
    ),
  );
  assert.ok(
    diagnoseExtraArgs("--grammar 'root ::= [0-9] '", CATALOG).every(
      (d) => d.level !== "error",
    ),
  );
  const managed = new Set<string>();
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--top-k ", "20", "--numa", "distribute"], managed),
    ["--numa", "distribute"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--verbose ", "--numa", "distribute"], managed),
    ["--numa", "distribute"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--grammar", "root ::= x "], managed),
    ["--grammar", "root ::= x "],
  );
});

test("a quoted value that begins with a hyphen is a value, not a flag", () => {
  // llama.cpp takes the next argv element as the value whatever it starts with.
  for (const text of ["--chat-template '- hello'", '--chat-template "- hello"']) {
    assert.ok(
      diagnoseExtraArgs(text, CATALOG).every((d) => d.level !== "error"),
      text,
    );
  }
  assert.ok(
    diagnoseExtraArgs("--grammar '-x'", CATALOG).every(
      (d) => !d.message.includes("-x"),
    ),
  );
  assert.ok(
    diagnoseExtraArgs('"--top-k" 20', CATALOG).every((d) => d.level !== "error"),
  );
  assert.ok(
    diagnoseExtraArgs("'--numa'", CATALOG).some(
      (d) => d.level === "error" && d.message.includes("needs a value"),
    ),
  );
  const orphan = diagnoseExtraArgs("'- hello'", CATALOG);
  assert.ok(orphan.every((d) => d.level !== "error"));
  assert.ok(orphan.some((d) => d.level === "warning"));
  const parsed = parseExtraArgs("--chat-template '- hello'");
  assert.deepEqual(parsed.tokens, ["--chat-template", "- hello"]);
  assert.deepEqual([...parsed.quotedIndices], [1]);
});

test("a managed answer from the previous binary is never published", () => {
  const flagsApi = readSrc("features/model-picker/api/llama-flags.ts").replace(/\s+/g, " ");
  // A managed request in flight across a llama.cpp swap must not republish stale data.
  assert.match(
    flagsApi,
    /const generation = catalogGeneration; inFlightManaged \?\?=/,
  );
  assert.match(
    flagsApi,
    /if \(generation !== catalogGeneration\) \{ .*return null; \} cachedManaged = managed;/,
  );
  assert.match(
    flagsApi,
    /if \(generation === catalogGeneration\) \{ inFlightManaged = null; \}/,
  );
});

// validate_extra_args parses -ts in every gpu memory mode.

const _tsError = (input: string): string | null =>
  diagnoseExtraArgs(input, CATALOG).find((d) => d.level === "error")?.message ??
  null;

test("a tensor split the backend takes raises nothing here", () => {
  for (const good of [
    "-ts 2.2,1",
    "--tensor-split 3,1",
    // llama.cpp splits on [,/]+, so the slash form is the same instruction.
    "--tensor-split 3/1",
    "-ts 0,1",
    "-ts 0.75,0.25",
  ]) {
    assert.equal(_tsError(good), null, good);
    assert.ok(extraArgsAreLoadable(diagnoseExtraArgs(good, CATALOG)), good);
  }
});

test("a bare tensor split is refused rather than left to llama-server", () => {
  assert.equal(_tsError("-ts"), "-ts needs a value after it.");
  assert.equal(_tsError("-ts --top-k 20"), "-ts needs a value after it.");
});

test("a tensor split that is not a list of numbers is refused", () => {
  // std::stof throws on this, so llama-server would exit at startup.
  assert.match(_tsError("-ts abc") ?? "", /comma- or slash-separated list of numbers/);
});

test("a negative or non-finite share is refused", () => {
  for (const bad of ["-ts 1,-1", "--tensor-split nan,1", "--tensor-split inf,1"]) {
    assert.equal(
      _tsError(bad),
      `${bad.split(" ")[0]} entries must be finite and non-negative.`,
      bad,
    );
  }
});

test("a split that totals nothing is refused", () => {
  // llama.cpp normalizes by the total, so an all-zero list divides by zero.
  assert.match(_tsError("-ts 0,0") ?? "", /must have a positive total/);
});

test("every tensor split occurrence is judged, not only the last", () => {
  // llama.cpp reads the last occurrence; this side strictly reports a bad first one too.
  assert.match(_tsError("-ts 3,1 -ts abc") ?? "", /list of numbers/);
  assert.match(_tsError("-ts abc --tensor-split 3,1") ?? "", /list of numbers/);
});

test("a stored tensor split the backend now refuses is repaired, not shed", () => {
  assert.deepEqual(
    sanitizeStoredExtraArgs(["-ts", "abc", "--top-k", "20"], CATALOG.managed),
    ["--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["-ts", "2.2,1", "--top-k", "20"], CATALOG.managed),
    ["-ts", "2.2,1", "--top-k", "20"],
  );
});

test("the ratio mirror reads exactly what Python's float() reads", () => {
  // Number() accepts 0x/0b/0o and refuses PEP 515 grouping, unlike Python float().
  for (const bad of ["0x10,1", "0b10,1", "0o17,1", "1__0,1", "_1,1", "1_,1", "1e,1"]) {
    assert.match(_tsError(`-ts ${bad}`) ?? "", /list of numbers/, bad);
  }
  for (const good of ["1_0,1", "1_000.5,1", "1.,1", ".5,1", "1e1_0,1", "+1.5,1"]) {
    assert.equal(_tsError(`-ts ${good}`), null, good);
  }
});

test("a share llama.cpp's float array cannot hold is refused", () => {
  // std::stof throws out_of_range above FLT_MAX.
  assert.match(_tsError("-ts 1e39,1") ?? "", /32-bit float/);
  assert.equal(_tsError("-ts 3.4e38,1"), null);
  assert.match(_tsError("-ts 3e38,3e38") ?? "", /adds up past/);
});

test("a share that underflows std::stof is refused too", () => {
  // libstdc++ throws out_of_range on any subnormal result, so the floor is FLT_MIN.
  for (const bad of ["1e-50,1", "1e-45,1", "1e-40,1", "1e-38,1"]) {
    assert.match(_tsError(`-ts ${bad}`) ?? "", /0 or at least/, bad);
  }
  // 1.1754943508222874e-38 rounds up to FLT_MIN but is emitted at six digits as a subnormal.
  for (const good of ["0,1", "1.2e-38,1", "1e-30,1"]) {
    assert.equal(_tsError(`-ts ${good}`), null, good);
  }
});

test("the mirror judges the share the launcher will write, and totals it in float32", () => {
  // Six-digit emission can move a value out of range, and float64 sums differ from float32.
  // gpuLayers: 49, because only a manual load with a resolved count >= 0 rewrites the ratio.
  const manual = (input: string) =>
    diagnoseExtraArgs(input, CATALOG, {
      manualGpuMemory: true,
      gpuLayers: 49,
    }).find((d) => d.level === "error")?.message ?? null;
  assert.match(manual("-ts 1.1754943508222874e-38,1") ?? "", /0 or at least/);
  assert.equal(manual("-ts 1.2e-38,1"), null);
  // Real libstdc++ sums the emitted text to 3.40282e+38, which fits.
  assert.equal(
    manual("-ts 2.0829609943909916e38,7.170581961838338e37,6.028042758104631e37"),
    null,
  );
  assert.match(manual("-ts 3e38,3e38") ?? "", /adds up past/);
});

test("the ratio rounding follows the mode, and each share rounds before it is added", () => {
  const err = (input: string, manualGpuMemory: boolean) =>
    diagnoseExtraArgs(input, CATALOG, { manualGpuMemory, gpuLayers: 49 }).find(
      (d) => d.level === "error",
    )?.message ?? null;
  // Only the manual promotion rewrites the text at six significant digits.
  assert.equal(err("-ts 1.1754943508222874e-38,1", false), null);
  assert.match(err("-ts 1.1754943508222874e-38,1", true) ?? "", /0 or at least/);
  // Each share is a float before it joins the total, as `sum += std::stof(token)` does.
  for (const manual of [false, true]) {
    assert.match(
      err("-ts 3.17817e38,1.54601e37,7.00525e36", manual) ?? "",
      /adds up past/,
    );
  }
});

test("only a manual load that will rewrite the split judges the rewritten text", () => {
  // At Auto layers the launcher drops both copies, so the six-digit rendering never happens.
  const at = (input: string, ctx: object) =>
    diagnoseExtraArgs(input, CATALOG, ctx).find((d) => d.level === "error")
      ?.message ?? null;
  const v = "-ts 1.1754943508222874e-38,1";
  assert.match(at(v, { manualGpuMemory: true, gpuLayers: 49 }) ?? "", /0 or at least/);
  assert.equal(at(v, { manualGpuMemory: true, gpuLayers: -1 }), null);
  assert.equal(at(v, { manualGpuMemory: false, gpuLayers: 49 }), null);
  assert.equal(
    at(`-ngl -1 ${v}`, { manualGpuMemory: true, gpuLayers: 49 }),
    null,
  );
  assert.match(
    at(`-ngl 49 ${v}`, { manualGpuMemory: true, gpuLayers: -1 }) ?? "",
    /0 or at least/,
  );
});
