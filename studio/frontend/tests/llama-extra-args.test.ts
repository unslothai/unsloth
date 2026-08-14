// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { activeLlamaArgumentsHydrationMatches } from "../src/features/model-picker/model-config/active-arguments-hydration.ts";
import {
  type LlamaServerArgument,
  completeLlamaExtraArgs,
  diagnoseLlamaExtraArgs,
  formatLlamaExtraArgs,
  llamaExtraArgRows,
  llamaExtraArgsCatalogBlocksPersistence,
  llamaExtraArgsPayload,
  parseLlamaExtraArgs,
} from "../src/features/model-picker/model-config/llama-extra-args.ts";

const arg = (
  name: string,
  aliases: string[] = [],
  patch: Partial<LlamaServerArgument> = {},
): LlamaServerArgument => ({
  name,
  aliases,
  value_hint: null,
  choices: [],
  description: name,
  default_value: null,
  env_var: null,
  group: "common params",
  policy_category: "Compute/placement",
  value_arity: 0,
  deprecated: false,
  managed_by_studio: false,
  overlaps_studio_control: false,
  ...patch,
});

const catalog = [
  arg("--fit", ["-fit"], {
    value_hint: "<on|off>",
    choices: ["on", "off"],
    value_arity: 1,
  }),
  arg("--override-tensor", ["-ot"], {
    value_hint: "<pattern>=<buffer>",
    value_arity: 1,
  }),
  arg("--cpu-range", ["-Cr"], {
    value_hint: "<lo-hi>",
    value_arity: 1,
  }),
  arg("--host", [], {
    policy_category: "Routing/listening",
    managed_by_studio: true,
    value_arity: 1,
  }),
];

test("argv editing preserves syntax, arity, aliases, and completion intent", () => {
  const tokens = ["--fit", "off", "C:\\models\\a.gguf", "two words"];
  assert.deepEqual(
    parseLlamaExtraArgs(formatLlamaExtraArgs(tokens)).tokens,
    tokens,
  );
  const rows = llamaExtraArgRows(["-Cr0-7", "--fit"], catalog);
  assert.deepEqual(
    rows.map(({ flag, value, separator }) => ({ flag, value, separator })),
    [
      { flag: "-Cr", value: "0-7", separator: "attached" },
      { flag: "--fit", value: undefined, separator: "none" },
    ],
  );

  const [short] = completeLlamaExtraArgs("-ot", 3, catalog);
  assert.deepEqual(
    [short.label, short.insertText, short.argument.name],
    ["-ot", "-ot", "--override-tensor"],
  );
  assert.deepEqual(
    completeLlamaExtraArgs("--fit o", 7, catalog).map(({ label }) => label),
    ["on", "off"],
  );
});

test("diagnostics enforce managed, malformed, control, and catalog boundaries", () => {
  const messages = diagnoseLlamaExtraArgs(
    "--host private --fit --future value",
    catalog,
  );
  assert.ok(messages.some(({ kind }) => kind === "managed"));
  assert.ok(messages.some(({ kind }) => kind === "missing-value"));
  assert.ok(messages.some(({ kind }) => kind === "unknown"));
  for (const separator of ["\r", "\n", "\u0085", "\u2028", "\u2029"]) {
    assert.ok(
      diagnoseLlamaExtraArgs(`--fit${separator}off`, catalog).some(
        ({ kind, severity }) => kind === "limit" && severity === "error",
      ),
    );
  }
  assert.equal(
    llamaExtraArgsCatalogBlocksPersistence(["--fit", "off"], false, false),
    true,
  );
});

test("request payload keeps omitted, clear, and replacement distinct", () => {
  assert.deepEqual(llamaExtraArgsPayload(undefined), {});
  assert.deepEqual(llamaExtraArgsPayload([]), { llama_extra_args: [] });
  assert.deepEqual(llamaExtraArgsPayload(["--fit", "off"]), {
    llama_extra_args: ["--fit", "off"],
  });
});

test("hydration and source contracts reject stale or shared argument state", () => {
  const response = {
    model_identifier: "unsloth/Args-GGUF",
    gguf_variant: "Q4_K_M",
    runtime_revision: "runtime-2",
    llama_extra_args: [],
  };
  const identity = {
    effectiveLoadIdentifier: "unsloth/Args-GGUF",
    ggufVariant: "q4_k_m",
    runtimeRevision: "runtime-2",
  };
  assert.equal(
    activeLlamaArgumentsHydrationMatches(response, identity, identity),
    true,
  );
  assert.equal(
    activeLlamaArgumentsHydrationMatches(response, identity, {
      ...identity,
      runtimeRevision: "runtime-3",
    }),
    false,
  );
  const editor = readFileSync(
    new URL(
      "../src/features/model-picker/components/llama-extra-args-editor.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(editor, /avoidCollisions=\{false\}/);
  assert.match(editor, /side=\{completionSide\}/);
  assert.match(editor, /LLAMA_ARG_\* env vars are ignored\./);
});
