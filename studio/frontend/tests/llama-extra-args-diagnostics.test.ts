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
  parseExtraArgs,
  sanitizeStoredExtraArgs,
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
  // What this build says takes no value, so "--verbose foo" reads as the typo
  // llama-server calls it.
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
  // --api-key is not a row in this panel, so pointing at one would be a lie.
  const text = messages("--api-key secret");
  assert.match(text, /managed by Unsloth/);
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
  // The failure mode this exists to prevent: an unverifiable build would mark every
  // correct flag as a typo, which is worse than saying nothing.
  const unverified: LlamaFlagCatalog = {
    flags: {},
    managed: CATALOG.managed,
    // Nothing was read from the binary, so nothing is known to be a switch either.
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
  const withPick = diagnoseExtraArgs("--device CUDA0", CATALOG, { gpuSelectionActive: true });
  assert.equal(withPick[0].level, "warning");
  assert.match(withPick[0].message, /--device will be removed/);
  assert.match(withPick[0].message, /GPU selection/);
  // With no GPU picked the flag is the user's own and nothing is stripped.
  assert.deepEqual(diagnoseExtraArgs("--device CUDA0", CATALOG, {}), []);
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

test("a value-taking flag with nothing after it is an error too", () => {
  // _last_flag_value raises for these groups from inside validate_extra_args, so a
  // shadowing note on its own left Load enabled for a request that 400s.
  for (const input of ["--cache-type-k=", "--top-k 20 -sm", "--split-mode="]) {
    const out = diagnoseExtraArgs(input, CATALOG);
    assert.ok(
      out.some((d) => d.level === "error" && d.message.includes("needs a value")),
      `${input}: ${JSON.stringify(out)}`,
    );
  }
  // A value that is there says nothing about presence.
  assert.ok(
    !diagnoseExtraArgs("--cache-type-k q8_0", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("the stored sanitizer removes everything this build would refuse", () => {
  // Not only denied flags: the bounds, control characters and unpaired surrogates
  // are all new refusals that a list saved by the previous release can trip, and
  // hydration turns that list into an explicit request. Each of these mirrors a
  // case pinned against drop_managed_flags in the backend suite.
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
  // A poisoned value takes its flag with it, or the flag is left expecting one and
  // eats the next token instead.
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
  // And a poisoned flag takes its value, or the value is left as a bare positional
  // that llama-server reads as a model path.
  assert.deepEqual(
    sanitizeStoredExtraArgs([`--grammar${control}`, "root", "--top-k", "20"], managed),
    ["--top-k", "20"],
  );
  // The bounds, shed from the tail exactly as the backend sheds them.
  assert.equal(
    sanitizeStoredExtraArgs(new Array(300).fill("--verbose"), managed).length,
    256,
  );
  // A clean list is untouched.
  const clean = ["--numa", "distribute"];
  assert.deepEqual(sanitizeStoredExtraArgs(clean, managed), clean);
});

test("the sanitizer trims to the HOST's bounds, not the constants", () => {
  // A Windows server takes 24 KiB, not 32, and holds a quoted-command budget on top
  // of it. Trimming to the wider constant leaves a list /load answers 400 on, which
  // is the one outcome hydrating a stored override is supposed to prevent.
  const managed = new Set<string>();
  const stored = ["--grammar", "x".repeat(30000), "--top-k", "20"];
  assert.deepEqual(sanitizeStoredExtraArgs(stored, managed), stored);
  assert.deepEqual(
    sanitizeStoredExtraArgs(stored, managed, { maxBytes: 24 * 1024 }),
    [],
  );
  // The quoted length, not the byte count: a token needing quotes doubles the
  // backslash runs before its quotes, so this passes the byte bound and fails the
  // command-line one.
  const quoted = ["--grammar", `${"\\".repeat(10)}" `.repeat(400)];
  assert.deepEqual(sanitizeStoredExtraArgs(quoted, managed), quoted);
  assert.deepEqual(
    sanitizeStoredExtraArgs(quoted, managed, {
      maxBytes: 24 * 1024,
      windowsCommandBudget: 8192,
    }),
    [],
  );
  // Zero means "not known", not "nothing fits": an older server answers the
  // catalogue without either field.
  assert.deepEqual(
    sanitizeStoredExtraArgs(stored, managed, {
      maxBytes: 0,
      windowsCommandBudget: 0,
    }),
    stored,
  );
});

test("a two-value flag is shed whole when the bounds bite", () => {
  // Mirrors drop_managed_flags: dropping END alone leaves START looking like an
  // ordinary value, and llama-server refuses the option outright.
  const managed = new Set<string>();
  assert.deepEqual(
    sanitizeStoredExtraArgs(
      ["--top-k", "20", "--control-vector-layer-range", "1", "x".repeat(40000)],
      managed,
    ),
    ["--top-k", "20"],
  );
  // Intact, it survives.
  const whole = ["--control-vector-layer-range", "1", "10"];
  assert.deepEqual(sanitizeStoredExtraArgs(whole, managed), whole);
});

test("a whole surrogate pair is a character, not a fault", () => {
  // The class-based check matched both units of every emoji, so a chat template or
  // grammar carrying one was dropped on hydration even though Python encodes it
  // without complaint. Only half a pair is refused.
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
  // And the editor says the same about what is typed.
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
  // _has_control_characters allows tab and newline on purpose: a grammar, a JSON
  // schema or a chat template is routinely multi-line, and quoting one into a
  // single argv token is what the box is for.
  const grammar = "--grammar 'root ::= [0-9]+\n  | \"x\"'";
  assert.ok(
    !diagnoseExtraArgs(grammar, CATALOG).some((d) => d.level === "error"),
    JSON.stringify(diagnoseExtraArgs(grammar, CATALOG)),
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--grammar", "a\nb\tc", "--top-k", "20"], new Set<string>()),
    ["--grammar", "a\nb\tc", "--top-k", "20"],
  );
  // An escape sequence is still refused.
  assert.ok(
    diagnoseExtraArgs(`--grammar ${String.fromCharCode(0x1b)}x`, CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("a value with no flag in front of it is an error", () => {
  // validate_extra_args refuses it, and llama-server answers "invalid argument"
  // and refuses to start, so without this the box looks fine and the load 400s.
  const out = diagnoseExtraArgs("--top-k 20 /models/other.gguf", CATALOG);
  assert.ok(
    out.some((d) => d.level === "error" && d.message.includes("belongs to no flag")),
    JSON.stringify(out),
  );
  // The two-value option is not one of those.
  assert.ok(
    !diagnoseExtraArgs("--control-vector-layer-range 1 10", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
  // Nor is an ordinary value.
  assert.ok(
    !diagnoseExtraArgs("--numa distribute", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("a value after a switch is the typo llama-server calls it", () => {
  // -v, --verbose, --log-verbose is declared with no value in this build's help, and
  // "--verbose foo" exits with "error: invalid argument: foo".
  const out = diagnoseExtraArgs("--verbose foo", CATALOG);
  assert.ok(
    out.some((d) => d.level === "error" && d.message.includes("belongs to no flag")),
    JSON.stringify(out),
  );
  // A flag that does take one is untouched, and an unverified build says nothing:
  // only what the catalogue actually declares is acted on.
  assert.ok(
    !diagnoseExtraArgs("--numa distribute", CATALOG).some(
      (d) => d.level === "error",
    ),
  );
});

test("the size limits are the host's, not this file's", () => {
  // Windows caps extras lower, because the whole command line shares one 32767
  // character budget, and the editor has to draw the same line the load does.
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
  // The same input is fine where the host allows 32 KiB.
  assert.ok(!diagnoseExtraArgs(big, CATALOG).some((d) => d.level === "error"));
});

test("what the quoting makes of an argument counts on Windows", () => {
  // list2cmdline doubles a backslash run before a quote, so bytes alone do not say
  // whether the launch fits. Same rule as the backend's own check.
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
  // Resolved by the backend, whose folding rules are the ones the load applies.
  assert.match(
    body,
    /fetchLoadModelOverride\(loadId, configId, target\.ggufVariant, keys\)/,
  );
  // Through the resolver, not a literal lookup: the backend folds identities and
  // falls back from repo:QUANT to the bare repo before it reads a row.
  // The candidate keys still travel, as the fallback for a backend that predates
  // the resolving parameter.
  assert.match(body, /modelOverrideKey\(loadId, target\.ggufVariant\)/);
  // And through the denylist first: hydrating makes the stored list an explicit
  // request, which /load validates strictly rather than dropping a newly denied
  // flag the way the carry-over paths do.
  assert.match(body, /sanitizeStoredExtraArgs\( resolvedArgs\.tokens,/);
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
    /setExtraArgsLoadable\(true\); setExtraArgsHydrating\(target\.isGguf && !isDiffusion\); \}, \[configId, target\.ggufVariant, target\.isGguf, isDiffusion\]\)/,
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
    /isGguf && !targetIsDiffusion && loadLlamaExtraArgs !== undefined \? \{ llama_extra_args: loadLlamaExtraArgs \?\? \[\] \} : \{\}/,
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
  // And the short deadline is on the CATALOGUE, not on the gate: the first read of
  // it runs --help on a cold binary, and releasing Load on that timer would let a
  // click through while the stored arguments were already in hand.
  // And it waits on the DENYLIST, which needs no binary, rather than on the
  // catalogue behind a cold --help.
  assert.match(body, /loadManagedLlamaFlags\(\)/);
  assert.doesNotMatch(body, /loadLlamaFlagCatalog\(\)[^;]*Promise\.all/);
});

test("a model with no such field does not wait for one", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // The row and the load payload are both GGUF-only, so a Transformers or MLX model
  // must not have Load held shut while two requests it will never use settle.
  // A diffusion GGUF is GGUF-shaped but runs through the diffusion shim, which
  // appends no llama-server flags, so it must not wait either.
  assert.match(
    body,
    /if \(!target\.isGguf \|\| resolvedIsDiffusion\) \{ .*setExtraArgsHydrating\(false\); return; \}/,
  );
  assert.match(body, /useState\( \(\) => target\.isGguf && !isDiffusion, \)/);
});

test("a diffusion classification retires the argument objection", () => {
  const panel = pageSource.slice(
    pageSource.indexOf("export function ModelConfigPage("),
  );
  const body = panel.replace(/\s+/g, " ");
  // withoutUnsupportedDiffusionSettings strips the arguments from what loads while
  // the row keeps its objection (it deliberately has no cleanup), so Load would
  // stay disabled over arguments the request no longer carries.
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
  // A cached GGUF outside the active HF cache loads by its snapshot path while
  // configId is the repo id, and the auto-switch loader reads the path-qualified
  // key first, so an override left there is the one API loads apply.
  assert.match(body, /modelOverrideKey\(loadId, target\.ggufVariant\), modelOverrideKey\(configId, target\.ggufVariant\), loadId,/);
  // Including the filename-label key an early build wrote for a loose .gguf.
  assert.match(body, /fileVariant \? \[`\$\{loadId\}:\$\{fileVariant\}`\] : \[\]/);
});

test("a rollback restores the previous model with its arguments", () => {
  const runtime = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/hooks/use-chat-model-runtime.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  ).replace(/\s+/g, " ");
  // By the time this runs the TARGET is resident, so an omitted field inherits
  // across models, which the route refuses, and the previous model would come back
  // without the arguments it had been running.
  assert.match(
    runtime,
    /stateBeforeUnload\.loadedLlamaExtraArgs != null \? \{ llama_extra_args: stateBeforeUnload\.loadedLlamaExtraArgs \}/,
  );
  // And the snapshot is kept on every successful load, not only an explicit one,
  // taken from the server's own echo first: a reload that omits the field but sets
  // max_seq_length has its inherited --ctx-size stripped before launch, and the
  // status refresh that would notice runs while the load is still in flight.
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
  // With Advanced collapsed the row never mounts, so nothing objects to a stored
  // list this build refuses (the overrides route only validates its shape), and
  // Load would be live for a request that comes back 400.
  //
  // Judged on the list hydration ADOPTS, not on the row's: a row carrying no
  // arguments leaves the local ones standing, and reading the verdict off the empty
  // server list called them loadable. That one lands even with Advanced expanded,
  // where the row has already refused the list and republishes only on a change of
  // its own verdict, so nothing puts the objection back.
  assert.match(
    body,
    /const hydratedArgs = serverConfig\?\.llamaExtraArgs \?\? stored;/,
  );
  assert.match(
    body,
    /const hydratedIsLoadable = hydratedArgs\.length === 0 \? true : extraArgsAreLoadable\( diagnoseExtraArgs\( formatExtraArgs\(hydratedArgs\)/,
  );
  // But not over an edit made while the request was out: the row is judging that
  // text, and replacing its verdict re-enabled Load for invalid input.
  assert.match(
    body,
    /if \(configRef\.current\.llamaExtraArgs !== undefined\) \{ .*return; \} setExtraArgsLoadable\(hydratedIsLoadable\)/,
  );
});

test("the runtime preflight is sized with the arguments the load sends", () => {
  const runtime = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/hooks/use-chat-model-runtime.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  ).replace(/\s+/g, " ");
  // A --ctx-size or cache override changes the memory /validate estimates. During
  // training an approval it did not size for means unloading the resident model and
  // having /load refuse the target, which is a rollback the user never asked for.
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
  const flagsApi = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/api/llama-flags.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  ).replace(/\s+/g, " ");
  // A switch that completes mid-request must not have the old build's flags written
  // back, and the next caller must not be handed that same promise.
  assert.match(flagsApi, /catalogGeneration \+= 1;/);
  assert.match(flagsApi, /inFlightCatalog = null;/);
  assert.match(
    flagsApi,
    /if \(generation !== catalogGeneration\) \{ .*return null; \}/,
  );
});

test("a catalogued flag left without its value is refused", () => {
  // The catalogue is the only place a flag's arity is known: the backend validator
  // cannot ask the binary, so a text ending in "--rope-scaling" used to leave Load
  // enabled and llama-server then exited during startup, which is a failed load
  // instead of a red line under the box.
  assert.ok(levels("--rope-scaling").includes("error"));
  assert.ok(levels("--top-k 20 --numa").includes("error"));
  assert.deepEqual(
    diagnoseExtraArgs("--numa", CATALOG).map((d) => d.message),
    ["--numa needs a value after it."],
  );
  // Given its value, it is fine, and so is a switch on its own.
  assert.deepEqual(diagnoseExtraArgs("--numa distribute", CATALOG), []);
  // A switch on its own owes nothing (this fixture's help lists it as a switch
  // without describing it, hence the unrelated unknown-flag warning).
  assert.ok(!levels("--verbose").includes("error"));
  // The attached spelling is refused instead: llama.cpp looks the whole token up in
  // its option map, so "--numa=distribute" is an argument it has never heard of
  // (measured on b10342 and b10360, "error: invalid argument").
  assert.deepEqual(
    diagnoseExtraArgs("--numa=distribute", CATALOG).map((d) => d.message),
    ['llama-server does not read "--numa=value". Write --numa and its value as two arguments.'],
  );
});

test("an unverified flag keeps the benefit of the doubt at the end", () => {
  // A build this Studio could not probe, or a flag newer than the help it read:
  // calling either a missing value would disable Load over a launch that works.
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
  // Catalogued build, flag it has never heard of: warned about as unknown, not
  // refused for a value it may not even take.
  assert.deepEqual(levels("--tempp"), ["warning"]);
  assert.ok(!levels("--tempp").includes("error"));
});

test("a two-value flag left short is refused whatever the catalogue says", () => {
  // Its arity is known without a probe, and it is the one the backend validator
  // checks itself, so the two ends agree.
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
  // The attached spelling is refused as such, rather than read as one of the two:
  // llama.cpp has no such spelling to be half of.
  // The attached spelling is refused as such, rather than read as one of the two:
  // llama.cpp has no such spelling for this to be half of, whether an END follows
  // it or not.
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
  // /load calls strip_shadowing_flags(strip_offload=True) in Manual mode. The layer
  // count survives that, because the route translates it into the first-class field
  // first; nothing does the same for the MoE count or the fitter, so saying they win
  // was false and the model quietly ran the control's value instead.
  const manual = (input: string) =>
    diagnoseExtraArgs(input, CATALOG, { manualGpuMemory: true }).map((d) => d.message);
  assert.ok(
    manual("--n-cpu-moe 10")[0].includes("will be removed"),
    manual("--n-cpu-moe 10")[0],
  );
  assert.ok(manual("-ncmoe 10")[0].includes("-ncmoe will be removed"));
  assert.ok(manual("--fit on")[0].includes("will be removed"));
  // Nothing is refused: the load still runs, just without them.
  assert.ok(!manual("--n-cpu-moe 10").includes("error"));
  assert.equal(
    diagnoseExtraArgs("--n-cpu-moe 10", CATALOG, { manualGpuMemory: true }).every(
      (d) => d.level !== "error",
    ),
    true,
  );
  // In Default mode they are passed, and the note about who wins is the true one.
  assert.ok(
    diagnoseExtraArgs("--n-cpu-moe 10", CATALOG, {})
      .map((d) => d.message)
      .join(" ")
      .includes("wins"),
  );
  // The layer count is translated, not dropped, so it still reads as winning.
  assert.ok(
    diagnoseExtraArgs("-ngl 20", CATALOG, { manualGpuMemory: true })
      .map((d) => d.message)
      .join(" ")
      .includes("wins"),
  );
});

test("a pass-through batch below the floor is refused", () => {
  // The loader raises the --batch-size it emits itself (max(slots, 2), measured
  // upstream: b1 aborts at any slot count, b4/p8 aborts, b8/p8 loads), but a
  // pass-through -b is appended after it and wins, so the load starts and
  // llama-server aborts on GGML_ASSERT instead.
  const at = (input: string, batchFloor: number) =>
    diagnoseExtraArgs(input, CATALOG, { batchFloor });
  assert.ok(at("-b 1", 2).some((d) => d.level === "error"));
  assert.ok(at("--batch-size 0", 2).some((d) => d.level === "error"));
  assert.ok(at("--batch-size 4", 8).some((d) => d.level === "error"));
  assert.match(
    at("--batch-size 4", 8).filter((d) => d.level === "error")[0].message,
    /8 parallel slot/,
  );
  // At or above the floor it passes (the note about shadowing the control stays),
  // and the micro-batch is not policed here.
  const errors = (input: string, batchFloor: number) =>
    at(input, batchFloor).filter((d) => d.level === "error");
  assert.deepEqual(errors("--batch-size 8", 8), []);
  assert.deepEqual(errors("-b 2", 2), []);
  assert.deepEqual(errors("-ub 1", 2), []);
  // A blank slot count leaves only the hard floor of 2, the same limit the batch
  // control itself asserts.
  assert.deepEqual(errors("-b 2", 1), []);
});

test("Model Memory reports the flags its settings remove", () => {
  // apply_model_memory_policy runs before the extras reach the command line, so an
  // --mlock typed here was shown, saved, and never passed.
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
  // No-reserve leaves the loaders that hold no full host copy alone.
  assert.equal(
    noReserve("--direct-io").some((message) => /will be removed/.test(message)),
    false,
  );
  // With both on, no-reserve is the one that runs, and it names itself.
  assert.match(
    diagnoseExtraArgs("--mlock", CATALOG, {
      keepResident: true,
      noRamReserve: true,
    })[0].message,
    /Don't reserve system RAM/,
  );
  // With neither, nothing is stripped and a hand-typed flag still applies.
  assert.equal(
    diagnoseExtraArgs("--mlock", CATALOG, {}).some((d) =>
      /will be removed/.test(d.message),
    ),
    false,
  );
});

test("llama.cpp's underscore spelling is not read as an attached value", () => {
  // _flag_name folds --ctx_size to --ctx-size, and the binary takes both (measured
  // on b10360: `llama-server --ctx_size 4096 --help` prints its help). Deciding
  // attachment by comparing the folded name against the raw token called the 4096 a
  // bare value and disabled Load for a spelling that works.
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
  // Still attached when it really is: an "=" form, or a short with its value glued
  // on. --n_parallel folds onto the managed --n-parallel and stays refused.
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
  // "--numa --verbose" leaves --numa without the value this build's help says it
  // takes, and llama-server exits during startup. The end-of-input check could not
  // see it: the obligation was overwritten by the next flag first.
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
  // A switch owes nothing, and an unverified flag keeps the benefit of the doubt.
  assert.deepEqual(messages("--verbose --numa distribute"), []);
  assert.deepEqual(messages("--tempp --numa distribute"), []);
});

test("the batch floor follows the server-wide slot default", () => {
  // With Slots blank the launch serves the server-wide --parallel (4 in run.py),
  // so -b 2 aborts even though it clears the hard floor of 2. The catalogue
  // publishes that number because the browser cannot see it.
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
  // The upgrade case this exists for: a list saved by an older build, hydrated into
  // an EXPLICIT request that /load validates strictly. drop_managed_flags repairs
  // these on the server by re-validating after every cut it makes; this mirror trims
  // by size alone, so it has to know the same rules.
  const managed = new Set<string>();
  // A token belonging to no flag: llama-server would read it as the model path.
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--top-k", "20", "stray"], managed),
    ["--top-k", "20"],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["stray", "--top-k", "20"], managed),
    ["--top-k", "20"],
  );
  // A two-value option left half-written, in both spellings.
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
  // The attached spelling goes whether it is whole or not: the backend refuses it
  // outright, so hydrating one into an explicit request would 400 the load. Only
  // that token, the way drop_managed_flags sheds it, so the rest still loads.
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--control-vector-layer-range=1", "10"], managed),
    [],
  );
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--top-k=20", "--numa", "distribute"], managed),
    ["--numa", "distribute"],
  );
  // A value the backend's own parser refuses takes its flag with it, and only it:
  // the server sheds its whole tail instead, which costs whatever followed.
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--ctx-size", "abc", "--top-k", "20"], managed),
    ["--top-k", "20"],
  );
  assert.deepEqual(sanitizeStoredExtraArgs(["--cache-type-k"], managed), []);
  // Valid values are untouched, including the ones whose minimum is negative.
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
  // Today's llama.cpp writes it into the value ("--lora-scaled FNAME:SCALE") and
  // older builds took it separately ("--lora-scaled FNAME SCALE"); the launcher
  // reads both in _sidecar_weight_files. So the second token is allowed and never
  // required: demanding it would refuse the current syntax, and refusing it broke a
  // list that loaded before the positional check existed.
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
  // A third bare token still has no owner.
  assert.equal(errors("--lora-scaled /a.gguf 0.5 stray").length, 1);
  // And the sanitizer keeps the pair rather than reading the scale as ownerless.
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
  // The bounds are shed from the tail, and the flag whose value has just gone must
  // go with it: an orphan is a flag llama-server then rejects for want of a value,
  // after the switch has already unloaded the resident model. The check used to
  // compare the NORMALIZED name against the raw token, so llama.cpp's underscore
  // spelling never matched and "--grammar_file" was left standing.
  const managed = new Set<string>();
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--numa", "distribute", "--grammar_file", "x".repeat(40000)], managed),
    ["--numa", "distribute"],
  );
  // The hyphenated one behaved already, and still does.
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--numa", "distribute", "--grammar-file", "x".repeat(40000)], managed),
    ["--numa", "distribute"],
  );
  // A bare value at the tail takes nothing with it: it belongs to no flag, and the
  // token before it is a value of its own.
  assert.deepEqual(
    sanitizeStoredExtraArgs(["--numa", "distribute", "x".repeat(40000)], managed),
    ["--numa", "distribute"],
  );
});

test("the attached spelling is refused wherever it is judged", () => {
  // llama.cpp looks the whole token up in its option map and folds only underscores,
  // so "--top-k=20" is an argument it has never heard of: measured on b10342 and
  // b10360 as "error: invalid argument: --top-k=20". Accepting it left Load enabled
  // for a switch that unloads the running model and then fails to start the next.
  const managed = new Set<string>();
  // Not --ctx-size: a control owns it, and that message names the control instead.
  for (const text of ["--top-k=20", "--rope-scaling=yarn", "--flash-attn=on"]) {
    assert.ok(
      diagnoseExtraArgs(text, CATALOG).some(
        (d) => d.level === "error" && d.message.includes("does not read"),
      ),
      text,
    );
  }
  // A managed flag keeps the message that names the control owning it.
  assert.ok(
    diagnoseExtraArgs("--parallel=8", CATALOG).every(
      (d) => !d.message.includes("does not read"),
    ),
  );
  // And an "=" inside a VALUE is the value's own syntax, not an attached one. This
  // fixture's help does not list --override-kv, hence the unrelated warning.
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
  const flagsApi = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/api/llama-flags.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  ).replace(/\s+/g, " ");
  // Its denylist is Unsloth's own, but it carries defaultParallelSlots beside it and
  // that is the EFFECTIVE count: a build without --kv-unified serves one slot however
  // many are configured. Updating llama.cpp from the banner left a tab that had
  // already fetched it sizing the hidden hydration check's batch floor from the
  // previous backend, so "--batch-size 2" passed on a build now serving four slots.
  assert.match(flagsApi, /cachedManaged = null; inFlightManaged = null;/);
  // The dynamic limits are what make it stale, so they have to be in that answer.
  assert.match(flagsApi, /defaultParallelSlots: number;/);
  // parallelSlotsClamped goes stale the same way and for the same reason: it is
  // read off the same probe, so a cached answer can outlive the build it describes.
  assert.match(flagsApi, /parallelSlotsClamped: boolean;/);
});

test("a flag quoted with stray spaces is refused, not silently sent", () => {
  // parseExtraArgs keeps what was quoted, so "'--top-k ' 20" is the token
  // "--top-k " with the space still on it. Every check here trims before it looks
  // the name up, so it read as the supported --top-k and Load stayed enabled, while
  // llama.cpp looks the WHOLE token up and answers "error: invalid argument:
  // --top-k" (measured on b10342), naming a flag that looks correct in the log.
  assert.ok(
    diagnoseExtraArgs("'--top-k ' 20", CATALOG).some(
      (d) => d.level === "error" && d.message.includes("Remove the spaces"),
    ),
  );
  // A VALUE may legitimately end in whitespace: a grammar or a chat template does.
  assert.ok(
    diagnoseExtraArgs("--grammar 'root ::= [0-9] '", CATALOG).every(
      (d) => d.level !== "error",
    ),
  );
  // And the stored sanitizer sheds it with its value, rather than hydrating a list
  // that would 400 or start a launch that fails.
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
  // parseExtraArgs takes the quotes off, and "- hello" is flag-shaped, so the row
  // called --chat-template's value missing and disabled Load over a list the backend
  // accepts and llama.cpp reads correctly: it takes the next argv element for a
  // value-taking option without looking at what it starts with.
  // A control owns --chat-template, so the note about who wins stays; what must not
  // be here is an error saying the value is missing.
  for (const text of ["--chat-template '- hello'", '--chat-template "- hello"']) {
    assert.ok(
      diagnoseExtraArgs(text, CATALOG).every((d) => d.level !== "error"),
      text,
    );
  }
  // The quoted token is not reported as an unknown flag either.
  assert.ok(
    diagnoseExtraArgs("--grammar '-x'", CATALOG).every(
      (d) => !d.message.includes("-x"),
    ),
  );
  // Position matters as much as the quotes: quoting a FLAG out of habit still reads
  // as a flag, or a list that runs would be refused.
  assert.ok(
    diagnoseExtraArgs('"--top-k" 20', CATALOG).every((d) => d.level !== "error"),
  );
  assert.ok(
    diagnoseExtraArgs("'--numa'", CATALOG).some(
      (d) => d.level === "error" && d.message.includes("needs a value"),
    ),
  );
  // With no option in front of it there is nothing for it to be the value OF, so it
  // is judged as written: flag-shaped, unknown to this build, warned about and still
  // passed. The same answer the backend gives, which reads it as a flag too.
  const orphan = diagnoseExtraArgs("'- hello'", CATALOG);
  assert.ok(orphan.every((d) => d.level !== "error"));
  assert.ok(orphan.some((d) => d.level === "warning"));
  // The tokeniser records which tokens were quoted; the token list itself is
  // unchanged, since argv has no room for that distinction.
  const parsed = parseExtraArgs("--chat-template '- hello'");
  assert.deepEqual(parsed.tokens, ["--chat-template", "- hello"]);
  assert.deepEqual([...parsed.quotedIndices], [1]);
});

test("a managed answer from the previous binary is never published", () => {
  const flagsApi = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/api/llama-flags.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  ).replace(/\s+/g, " ");
  // Clearing the cache is not enough on its own: a managed request already on the
  // wire when llama.cpp is replaced would resolve afterwards and put the old
  // build's defaultParallelSlots back, where it would stay for the session. The
  // full catalogue has read its generation before the request and checked it
  // before publishing since the start; this path now does the same, including the
  // finally, which used to clear a newer request's in-flight promise.
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
