// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  EXTRA_ARGS_MAX_TOKENS,
  extraArgFlagName,
  extraArgFlags,
  formatExtraArgs,
  parseExtraArgs,
} from "../src/features/model-picker/model-config/llama-extra-args.ts";

// The wire format is one argv token per entry, so this split decides what the child
// process actually receives. A token boundary in the wrong place turns one flag's
// value into another flag.

test("a plain command splits on whitespace", () => {
  assert.deepEqual(parseExtraArgs("--top-k 20 --seed 42").tokens, [
    "--top-k",
    "20",
    "--seed",
    "42",
  ]);
});

test("runs of whitespace collapse and the edges are ignored", () => {
  assert.deepEqual(parseExtraArgs("   --top-k    20  ").tokens, [
    "--top-k",
    "20",
  ]);
  assert.deepEqual(parseExtraArgs("").tokens, []);
  assert.deepEqual(parseExtraArgs("   ").tokens, []);
});

test("newlines separate, so one flag per line reads as one command", () => {
  assert.deepEqual(
    parseExtraArgs("--top-k 20\n--seed 42\r\n--numa distribute").tokens,
    ["--top-k", "20", "--seed", "42", "--numa", "distribute"],
  );
});

test("a quoted value keeps its spaces in one token", () => {
  // The reason quoting exists here at all: a chat template or a grammar is one
  // argv entry containing spaces.
  assert.deepEqual(parseExtraArgs(`--chat-template "a b c"`).tokens, [
    "--chat-template",
    "a b c",
  ]);
  assert.deepEqual(parseExtraArgs(`--chat-template 'a b c'`).tokens, [
    "--chat-template",
    "a b c",
  ]);
});

test("quotes can open mid-token and more than once", () => {
  assert.deepEqual(parseExtraArgs(`--opt=a" "b`).tokens, ["--opt=a b"]);
  assert.deepEqual(parseExtraArgs(`'a'"b"c`).tokens, ["abc"]);
});

test("an empty quoted string is still a token", () => {
  // --grammar '' is a real thing to pass; dropping it would silently change the
  // command rather than fail.
  assert.deepEqual(parseExtraArgs(`--grammar ''`).tokens, ["--grammar", ""]);
});

test("an unterminated quote is reported, not swallowed", () => {
  const parsed = parseExtraArgs(`--chat-template "a b`);
  assert.equal(parsed.unterminatedQuote, '"');
  // The tokens so far are still returned, so the row can show what it did read.
  assert.deepEqual(parsed.tokens, ["--chat-template", "a b"]);
  assert.equal(parseExtraArgs("--top-k 20").unterminatedQuote, null);
});

test("a backslash escapes the next character outside quotes", () => {
  assert.deepEqual(parseExtraArgs("--path a\\ b").tokens, ["--path", "a b"]);
  assert.deepEqual(parseExtraArgs('--x \\"quoted\\"').tokens, [
    "--x",
    '"quoted"',
  ]);
});

test("single quotes take everything literally", () => {
  // What makes a regex-bearing grammar survive: inside single quotes a backslash
  // is a backslash, as in a shell.
  assert.deepEqual(parseExtraArgs(`--grammar 'root ::= [\\d]+'`).tokens, [
    "--grammar",
    "root ::= [\\d]+",
  ]);
});

test("double quotes escape only what a shell escapes", () => {
  assert.deepEqual(parseExtraArgs(`"a\\"b"`).tokens, ['a"b']);
  // A backslash before an ordinary character stays, so a Windows path survives.
  assert.deepEqual(parseExtraArgs(`"C:\\Users\\model"`).tokens, [
    "C:\\Users\\model",
  ]);
});

test("a trailing backslash-newline continues the line", () => {
  // POSIX 2.2.1: an unquoted backslash before a newline is a line continuation and
  // both characters go. This is the one place the split deliberately differs from
  // Python's shlex, which is a lexer with no continuation rule and leaves a literal
  // newline inside the token. The box is multi-line and people paste wrapped
  // commands into it, so the shell reading is the one that matches the intent.
  assert.deepEqual(parseExtraArgs("--top-k \\\n20").tokens, ["--top-k", "20"]);
  // Inside single quotes it is literal, as in a shell.
  assert.deepEqual(parseExtraArgs("--x 'a\\\nb'").tokens, ["--x", "a\\\nb"]);
});

test("a trailing lone backslash is kept, not treated as an error", () => {
  // The other deliberate difference from shlex, which raises here. A text field is
  // half-typed most of the time, so refusing the whole box mid-keystroke is worse
  // than carrying the character.
  assert.deepEqual(parseExtraArgs("--x \\").tokens, ["--x", "\\"]);
  assert.equal(parseExtraArgs("--x \\").unterminatedQuote, null);
});

test("an unquoted Windows path loses its separators, as in a shell", () => {
  // Not a bug to fix in the splitter: an unquoted backslash escapes the next
  // character in every POSIX shell, and changing that would break every escape the
  // hint tells people to use. It IS a trap on Windows, which is why the row's hint
  // names backslashes and not just spaces.
  assert.deepEqual(
    parseExtraArgs("--chat-template-file C:\\a\\b.jinja").tokens,
    ["--chat-template-file", "C:ab.jinja"],
  );
  // Quoted, it survives whole, and that is what the hint asks for.
  assert.deepEqual(
    parseExtraArgs('--chat-template-file "C:\\a\\b.jinja"').tokens,
    ["--chat-template-file", "C:\\a\\b.jinja"],
  );
  assert.deepEqual(
    parseExtraArgs("--chat-template-file 'C:\\a\\b.jinja'").tokens,
    ["--chat-template-file", "C:\\a\\b.jinja"],
  );
});

test("shell metacharacters are literal, because nothing here runs a shell", () => {
  // The child is spawned from a list, so pretending otherwise would invent a
  // meaning the backend does not implement.
  assert.deepEqual(parseExtraArgs("--x a;b|c>d").tokens, ["--x", "a;b|c>d"]);
  assert.deepEqual(parseExtraArgs("--x $HOME").tokens, ["--x", "$HOME"]);
  assert.deepEqual(parseExtraArgs("--x *.gguf").tokens, ["--x", "*.gguf"]);
});

// --- round-tripping ---------------------------------------------------------
// The stored value is a token list, so the box is re-rendered from tokens every
// time the panel reopens. Anything that does not round-trip accumulates escaping.

const ROUND_TRIP: string[][] = [
  ["--top-k", "20"],
  ["--chat-template", "a b c"],
  ["--grammar", "root ::= [\\d]+"],
  ["--x", ""],
  ["--x", "it's"],
  ["--x", 'say "hi"'],
  ["--x", "both'and\""],
  ["--x", "C:\\Users\\model"],
  ["--x", "a;b|c"],
  ["--x", "$HOME"],
  ["--x", "tab\there"],
];

for (const tokens of ROUND_TRIP) {
  test(`round-trips ${JSON.stringify(tokens)}`, () => {
    assert.deepEqual(parseExtraArgs(formatExtraArgs(tokens)).tokens, tokens);
  });
}

test("formatting leaves ordinary tokens unquoted", () => {
  // Or the box would fill with quotes the user never typed.
  assert.equal(formatExtraArgs(["--top-k", "20"]), "--top-k 20");
  assert.equal(formatExtraArgs([]), "");
  assert.equal(formatExtraArgs(null), "");
  assert.equal(formatExtraArgs(undefined), "");
});

// --- flag names -------------------------------------------------------------
// Mirrors _flag_name in llama_server_args.py. Where these disagree, the UI accepts
// an argument the load then refuses, or warns about one that would have worked.

test("a flag name is peeled from its value", () => {
  assert.equal(extraArgFlagName("--top-k"), "--top-k");
  assert.equal(extraArgFlagName("--top-k=20"), "--top-k");
  assert.equal(extraArgFlagName("-fa"), "-fa");
});

test("long-option underscores normalise like llama.cpp", () => {
  assert.equal(extraArgFlagName("--top_k"), "--top-k");
  assert.equal(extraArgFlagName("--n_parallel=4"), "--n-parallel");
});

test("a value is not a flag", () => {
  assert.equal(extraArgFlagName("20"), null);
  assert.equal(extraArgFlagName("-1"), null);
  assert.equal(extraArgFlagName("-0.5"), null);
  assert.equal(extraArgFlagName("-"), null);
  assert.equal(extraArgFlagName("--"), null);
});

test("an attached -np value still names -np", () => {
  // Otherwise a denied flag slips through glued to its value.
  assert.equal(extraArgFlagName("-np8"), "-np");
  assert.equal(extraArgFlagName("-np-1"), "-np");
  assert.equal(extraArgFlagName("-np"), "-np");
});

test("whitespace padding does not hide a flag", () => {
  assert.equal(extraArgFlagName(" --parallel"), "--parallel");
  assert.equal(extraArgFlagName("\t-np "), "-np");
});

test("flags are collected in order without duplicates", () => {
  assert.deepEqual(
    extraArgFlags(["--top-k", "20", "--seed", "42", "--top-k", "30"]),
    ["--top-k", "--seed"],
  );
  assert.deepEqual(extraArgFlags(["20", "42"]), []);
});

test("the token cap mirrors the backend", () => {
  // The backend refuses past this, so the editor has to warn at the same number
  // rather than let the load fail.
  assert.equal(EXTRA_ARGS_MAX_TOKENS, 256);
});
