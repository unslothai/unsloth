// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { activeLlamaArgumentsHydrationMatches } from "../src/features/model-picker/model-config/active-arguments-hydration.ts";
import {
  type LlamaServerArgument,
  areLlamaExtraArgsWithinLimits,
  completeLlamaExtraArgs,
  diagnoseLlamaExtraArgs,
  formatLlamaExtraArgs,
  llamaExtraArgRows,
  llamaExtraArgsCatalogBlocksPersistence,
  llamaExtraArgsPayload,
  parseLlamaExtraArgs,
} from "../src/features/model-picker/model-config/llama-extra-args.ts";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
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
  arg("--override-tensor", ["-ot"], { value_arity: 1 }),
  arg("--cpu-range", ["-Cr"], { value_arity: 1 }),
  arg("--host", [], {
    policy_category: "Routing/listening",
    managed_by_studio: true,
    value_arity: 1,
  }),
];
test("argv editing preserves syntax, arity, aliases, and completion intent", () => {
  const tokens = ["--fit", "off", "C:\\models\\a.gguf", "two words"];
  assert.deepEqual(parseLlamaExtraArgs(formatLlamaExtraArgs(tokens)).tokens, tokens);
  assert.equal(parseLlamaExtraArgs('""').error?.message, "Arguments cannot be empty.");
  assert.deepEqual(
    llamaExtraArgRows(["-Cr0-7", "--fit"], catalog).map(({ flag, value, separator }) => ({
      flag,
      value,
      separator,
    })),
    [
      { flag: "-Cr", value: "0-7", separator: "attached" },
      { flag: "--fit", value: undefined, separator: "none" },
    ],
  );
  const [short] = completeLlamaExtraArgs("-ot", 3, catalog);
  assert.deepEqual([short.label, short.insertText, short.argument.name], [
    "-ot",
    "-ot",
    "--override-tensor",
  ]);
  assert.deepEqual(
    completeLlamaExtraArgs("--fit o", 7, catalog).map(({ label }) => label),
    ["on", "off"],
  );
});
test("diagnostics enforce managed, malformed, control, and catalog boundaries", () => {
  const kinds = diagnoseLlamaExtraArgs("--host private --fit --future value", catalog).map(
    ({ kind }) => kind,
  );
  for (const expected of ["managed", "missing-value", "unknown"]) {
    assert.ok(kinds.includes(expected as (typeof kinds)[number]));
  }
  for (const separator of [
    "\r", "\n", "\u007f", "\u0080", "\u0085", "\u009f", "\ud800", "\udfff", "\u2028", "\u2029",
  ]) {
    assert.ok(
      diagnoseLlamaExtraArgs(`--fit${separator}off`, catalog).some(
        ({ kind, severity }) => kind === "limit" && severity === "error",
      ),
    );
  }
  assert.equal(areLlamaExtraArgsWithinLimits(["--fit", "on\tfast", "😀"]), true);
  for (const malformed of ['--fit " off "', "--"]) {
    assert.ok(
      diagnoseLlamaExtraArgs(malformed, catalog).some(
        ({ kind, severity }) => kind === "syntax" && severity === "error",
      ),
    );
  }
  assert.equal(llamaExtraArgsCatalogBlocksPersistence(["--fit", "off"], false, false), true);
});
test("request payload keeps omitted, clear, and replacement distinct", () => {
  assert.deepEqual(llamaExtraArgsPayload(undefined), {});
  assert.deepEqual(llamaExtraArgsPayload([]), { llama_extra_args: [] });
  assert.deepEqual(llamaExtraArgsPayload(["--fit", "off"]), {
    llama_extra_args: ["--fit", "off"],
  });
});

test("hydration and lifecycle contracts isolate and validate arguments", () => {
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
  assert.equal(activeLlamaArgumentsHydrationMatches(response, identity, identity), true);
  assert.equal(
    activeLlamaArgumentsHydrationMatches(response, identity, {
      ...identity,
      runtimeRevision: "runtime-3",
    }),
    false,
  );
  const editor = source("../src/features/model-picker/components/llama-extra-args-editor.tsx");
  for (const contract of [/avoidCollisions=\{false\}/, /side=\{completionSide\}/, /LLAMA_ARG_\* env vars are ignored\./]) {
    assert.match(editor, contract);
  }
  const runtime = source("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const validated = runtime.indexOf("const validation = await validateModel");
  const persisted = runtime.indexOf("selection.onValidated?.()", validated);
  const teardown = runtime.indexOf("await unloadModel", validated);
  assert.ok(validated >= 0 && validated < persisted && persisted < teardown);
  assert.match(
    source("../src/features/model-picker/components/model-config-page.tsx"),
    /if \(effectivePersistenceOnly\) \{\s+onRun\([\s\S]+persistSettings\(\)/,
  );
  assert.match(source("../src/features/hub/hub-page.tsx"), /current === target \? null : current/);
  const chatPage = source("../src/features/chat/chat-page.tsx");
  assert.match(chatPage, /latestPendingHubSelectionRef\.current = selection/);
  assert.match(chatPage, /stageOrLoad\(\{ \.\.\.selection, isDownloaded: true \}\)/);
});
