// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";
import {
  type LlamaServerArgument,
  applyLlamaExtraArgsCompletion,
  completeLlamaExtraArgs,
  diagnoseLlamaExtraArgs,
  formatLlamaExtraArgs,
  llamaExtraArgRows,
  llamaExtraArgsCatalogBlocksPersistence,
  llamaExtraArgsPayload,
  llamaServerArgumentGroupLabel,
  llamaServerArgumentTakesValue,
  llamaServerDiagnosticCatalog,
  moveLlamaExtraArgsSelection,
  parseLlamaExtraArgs,
  replaceLlamaExtraArgRow,
  replaceLlamaExtraArgRowFlag,
} from "../src/features/model-picker/model-config/llama-extra-args.ts";

test("payloads preserve absent, explicit-empty, and replacement semantics", () => {
  assert.deepEqual(llamaExtraArgsPayload(undefined), {});
  assert.deepEqual(llamaExtraArgsPayload([]), { llama_extra_args: [] });
  assert.deepEqual(llamaExtraArgsPayload(["--fit-target", "1024"]), {
    llama_extra_args: ["--fit-target", "1024"],
  });
});

test("non-empty arguments cannot persist without an authoritative catalog", () => {
  for (const sensitive of ["--model", "--api-key"]) {
    assert.equal(
      llamaExtraArgsCatalogBlocksPersistence(
        [sensitive, "secret"],
        false,
        false,
      ),
      true,
    );
    assert.equal(
      llamaExtraArgsCatalogBlocksPersistence(
        [sensitive, "secret"],
        true,
        false,
      ),
      true,
    );
  }
  assert.equal(llamaExtraArgsCatalogBlocksPersistence([], false, false), false);
  assert.equal(
    llamaExtraArgsCatalogBlocksPersistence(["--top-k", "40"], true, true),
    false,
  );
});

test("keyboard selection wraps in both directions", () => {
  assert.equal(moveLlamaExtraArgsSelection(0, "previous", 3), 2);
  assert.equal(moveLlamaExtraArgsSelection(2, "next", 3), 0);
  assert.equal(moveLlamaExtraArgsSelection(0, "next", 0), 0);
});

const catalog: LlamaServerArgument[] = [
  {
    name: "--fit",
    aliases: ["-fit"],
    value_hint: "<on|off>",
    choices: ["on", "off"],
    description: "Fit the model to available memory.",
    default_value: "on",
    env_var: "LLAMA_ARG_FIT",
    group: "model",
    policy_category: "Unclassified",
    value_arity: 1,
    deprecated: false,
    managed_by_studio: false,
    overlaps_studio_control: true,
  },
  {
    name: "--fit-ctx",
    aliases: [],
    value_hint: "<tokens>",
    choices: [],
    description: "Context target for fitting.",
    default_value: null,
    env_var: null,
    group: "model",
    policy_category: "Unclassified",
    value_arity: 1,
    deprecated: false,
    managed_by_studio: false,
    overlaps_studio_control: false,
  },
  {
    name: "--host",
    aliases: [],
    value_hint: "<address>",
    choices: [],
    description: "Bind address.",
    default_value: "127.0.0.1",
    env_var: null,
    group: "server",
    policy_category: "Routing/listening",
    value_arity: 1,
    deprecated: false,
    managed_by_studio: true,
    overlaps_studio_control: false,
  },
  {
    name: "--old-flag",
    aliases: [],
    value_hint: null,
    choices: [],
    description: "Old behavior.",
    default_value: null,
    env_var: null,
    group: "common",
    policy_category: "Unclassified",
    value_arity: 0,
    deprecated: true,
    managed_by_studio: false,
    overlaps_studio_control: false,
  },
];

test("flat tokens become editable argument rows without losing separators", () => {
  const tokens = ["--fit=off", "--fit-ctx", "1024", "--old-flag"];
  const rows = llamaExtraArgRows(tokens, catalog);

  assert.deepEqual(
    rows.map(({ flag, value, separator }) => ({ flag, value, separator })),
    [
      { flag: "--fit", value: "off", separator: "equals" },
      { flag: "--fit-ctx", value: "1024", separator: "separate" },
      { flag: "--old-flag", value: undefined, separator: "none" },
    ],
  );
  assert.deepEqual(replaceLlamaExtraArgRow(tokens, rows[0], "on"), [
    "--fit=on",
    "--fit-ctx",
    "1024",
    "--old-flag",
  ]);
  assert.deepEqual(replaceLlamaExtraArgRow(tokens, rows[1], "2048"), [
    "--fit=off",
    "--fit-ctx",
    "2048",
    "--old-flag",
  ]);
  assert.deepEqual(
    replaceLlamaExtraArgRowFlag(tokens, rows[0], "--fit-target"),
    ["--fit-target=off", "--fit-ctx", "1024", "--old-flag"],
  );
  assert.deepEqual(replaceLlamaExtraArgRowFlag(tokens, rows[1], "--ctx-size"), [
    "--fit=off",
    "--ctx-size",
    "1024",
    "--old-flag",
  ]);
});

test("catalog-unknown arguments keep an editable value", () => {
  const paired = llamaExtraArgRows(["daw", "w"], catalog);
  assert.deepEqual(
    paired.map(({ flag, value, valueExpected }) => ({
      flag,
      value,
      valueExpected,
    })),
    [{ flag: "daw", value: "w", valueExpected: true }],
  );

  const withoutValue = llamaExtraArgRows(["--future-flag"], catalog)[0];
  assert.equal(withoutValue.valueExpected, true);
  assert.equal(withoutValue.value, undefined);
  assert.deepEqual(
    replaceLlamaExtraArgRow(["--future-flag"], withoutValue, "42"),
    ["--future-flag", "42"],
  );
});

test("structured multi-word hints keep override-tensor value-aware", () => {
  const overrideTensor: LlamaServerArgument = {
    name: "--override-tensor",
    aliases: ["-ot"],
    value_hint: "<tensor name pattern>=<buffer type>,...",
    choices: [],
    description: "Override tensor buffer type.",
    default_value: null,
    env_var: "LLAMA_ARG_OVERRIDE_TENSOR",
    group: "common",
    policy_category: "Unclassified",
    value_arity: 1,
    deprecated: false,
    managed_by_studio: false,
    overlaps_studio_control: true,
  };

  assert.equal(llamaServerArgumentTakesValue(overrideTensor), true);
  const row = llamaExtraArgRows(["--override-tensor"], [overrideTensor])[0];
  assert.equal(row.valueExpected, true);
  assert.equal(row.value, undefined);
});

const arityCatalog: LlamaServerArgument[] = [
  ["--cpu-range", "-Cr", "<lo-hi>"],
  ["--cpu-range-batch", "-Crb", "<lo-hi>"],
  ["--override-kv", "", "<key=type:value>"],
  ["--logit-bias", "-l", "<token+bias>"],
  ["--spec-draft-cpu-range", "-Crd", "<lo-hi>"],
  ["--ctx-size", "-c", "<tokens>"],
  ["--gpu-layers", "-ngl", "<layers>"],
].map(([name, alias, valueHint]) => ({
  ...catalog[1],
  name,
  aliases: alias ? [alias] : [],
  value_hint: valueHint,
  value_arity: 1,
}));

test("backend value arity drives completion and missing-value diagnostics", () => {
  for (const argument of arityCatalog.slice(0, 5)) {
    assert.equal(llamaServerArgumentTakesValue(argument), true);
    const diagnostics = diagnoseLlamaExtraArgs(argument.name, arityCatalog);
    assert.ok(
      diagnostics.some(
        (item) => item.kind === "missing-value" && item.severity === "error",
      ),
      argument.name,
    );
  }

  const optionalHint = {
    ...catalog[0],
    name: "--optional-switch",
    choices: ["on", "off"],
    value_hint: "[on|off]",
    value_arity: 0,
  };
  assert.equal(llamaServerArgumentTakesValue(optionalHint), false);
  assert.equal(
    diagnoseLlamaExtraArgs("--optional-switch", [optionalHint]).some(
      (item) => item.kind === "missing-value",
    ),
    false,
  );

  const pairArgument = {
    ...catalog[1],
    name: "--value-pair",
    value_hint: "<first> <second>",
    value_arity: 2,
  };
  assert.ok(
    diagnoseLlamaExtraArgs("--value-pair first", [pairArgument]).some(
      (item) => item.kind === "missing-value" && item.severity === "error",
    ),
  );
  assert.equal(
    diagnoseLlamaExtraArgs("--value-pair first second", [pairArgument]).some(
      (item) => item.kind === "missing-value",
    ),
    false,
  );
  assert.deepEqual(
    completeLlamaExtraArgs("--value-pair first se", 21, [pairArgument]),
    [],
    "the second free-form value must not fall through to flag completion",
  );
});

test("every catalog short alias accepts a canonical attached value", () => {
  const tokens = [
    "-Cr0-7",
    "-Crb2-5",
    "-l42+1.5",
    "-Crd1-3",
    "-c4096",
    "-ngl99",
  ];
  const rows = llamaExtraArgRows(tokens, arityCatalog);
  assert.deepEqual(
    rows.map(({ argument, flag, value, separator }) => ({
      canonical: argument?.name,
      flag,
      value,
      separator,
    })),
    [
      {
        canonical: "--cpu-range",
        flag: "-Cr",
        value: "0-7",
        separator: "attached",
      },
      {
        canonical: "--cpu-range-batch",
        flag: "-Crb",
        value: "2-5",
        separator: "attached",
      },
      {
        canonical: "--logit-bias",
        flag: "-l",
        value: "42+1.5",
        separator: "attached",
      },
      {
        canonical: "--spec-draft-cpu-range",
        flag: "-Crd",
        value: "1-3",
        separator: "attached",
      },
      {
        canonical: "--ctx-size",
        flag: "-c",
        value: "4096",
        separator: "attached",
      },
      {
        canonical: "--gpu-layers",
        flag: "-ngl",
        value: "99",
        separator: "attached",
      },
    ],
  );
  assert.equal(
    diagnoseLlamaExtraArgs(tokens.join(" "), arityCatalog).some((item) =>
      ["unknown", "missing-value"].includes(item.kind),
    ),
    false,
  );
  assert.deepEqual(replaceLlamaExtraArgRow(tokens, rows[0], "4-11"), [
    "-Cr4-11",
    ...tokens.slice(1),
  ]);
});

test("attached aliases prefer exact spellings and then the longest prefix", () => {
  const ambiguousCatalog: LlamaServerArgument[] = [
    {
      ...catalog[1],
      name: "--memory",
      aliases: ["-m"],
      value_arity: 1,
    },
    {
      ...catalog[3],
      name: "--memory-lock",
      aliases: ["-mlock"],
      deprecated: false,
      value_arity: 0,
    },
    {
      ...catalog[1],
      name: "--memory-gpu",
      aliases: ["-mg"],
      value_arity: 1,
    },
  ];

  const rows = llamaExtraArgRows(["-mlock", "-mg0"], ambiguousCatalog);
  assert.deepEqual(
    rows.map(({ argument, flag, value, separator }) => ({
      canonical: argument?.name,
      flag,
      value,
      separator,
    })),
    [
      {
        canonical: "--memory-lock",
        flag: "-mlock",
        value: undefined,
        separator: "none",
      },
      {
        canonical: "--memory-gpu",
        flag: "-mg",
        value: "0",
        separator: "attached",
      },
    ],
  );
});

test("multiline quoting, escaping, and equals tokens parse without a shell", () => {
  const parsed = parseLlamaExtraArgs(
    '--fit=off\n--model-draft "C:\\\\Models\\\\draft model.gguf" path\\ with\\ spaces',
  );
  assert.equal(parsed.error, null);
  assert.deepEqual(parsed.tokens, [
    "--fit=off",
    "--model-draft",
    "C:\\Models\\draft model.gguf",
    "path with spaces",
  ]);
});

test("bare Windows and POSIX paths retain literal backslashes and slashes", () => {
  assert.deepEqual(
    parseLlamaExtraArgs("C:\\models\\draft.gguf /opt/models/draft.gguf").tokens,
    ["C:\\models\\draft.gguf", "/opt/models/draft.gguf"],
  );
});

test("formatter round-trips empty, quoted, Windows, and POSIX tokens", () => {
  const tokens = [
    "",
    "--flag=value",
    "two words",
    "C:\\Models\\a.gguf",
    "/tmp/a.gguf",
  ];
  const formatted = formatLlamaExtraArgs(tokens);
  assert.deepEqual(parseLlamaExtraArgs(formatted).tokens, tokens);
});

test("incomplete quote and trailing escape are blocking syntax diagnostics", () => {
  assert.equal(
    parseLlamaExtraArgs('--fit "on').error?.message,
    "Unterminated double quote.",
  );
  assert.equal(
    diagnoseLlamaExtraArgs("--fit on\\", catalog)[0]?.severity,
    "error",
  );
});

test("line separators are blocking while horizontal tabs remain accepted", () => {
  for (const separator of ["\r", "\n", "\u0085", "\u2028", "\u2029"]) {
    const diagnostics = diagnoseLlamaExtraArgs(`--fit${separator}off`, catalog);
    assert.ok(
      diagnostics.some(
        (diagnostic) =>
          diagnostic.kind === "limit" && diagnostic.severity === "error",
      ),
    );
  }
  assert.equal(
    diagnoseLlamaExtraArgs("--fit\toff", catalog).some(
      (diagnostic) => diagnostic.kind === "limit",
    ),
    false,
  );
});

test("fi and --fi rank installed fit flags before fuzzy matches", () => {
  for (const input of ["fi", "--fi"]) {
    const results = completeLlamaExtraArgs(input, input.length, catalog);
    assert.deepEqual(
      results.slice(0, 2).map((result) => result.label),
      ["--fit", "--fit-ctx"],
    );
  }
});

test("flag completion returns at most eight results", () => {
  const many = Array.from({ length: 12 }, (_, index) => ({
    ...catalog[0],
    name: `--flag-${index}`,
    aliases: [],
  }));
  assert.equal(completeLlamaExtraArgs("fl", 2, many).length, 8);
});

test("autocomplete never offers flags Studio will reject", () => {
  assert.deepEqual(completeLlamaExtraArgs("--ho", 4, catalog), []);
});

test("known choices complete after a flag and after equals", () => {
  const separated = completeLlamaExtraArgs("--fit o", 7, catalog);
  assert.deepEqual(
    separated.map((result) => result.label),
    ["on", "off"],
  );
  const attached = completeLlamaExtraArgs("--fit=o", 7, catalog);
  assert.deepEqual(
    attached.map((result) => result.label),
    ["on", "off"],
  );
});

test("free-form values never fall through to unrelated flag completion", () => {
  const numericFlag: LlamaServerArgument = {
    ...catalog[3],
    name: "--gpt-oss-20b-default",
    aliases: [],
    deprecated: false,
  };
  const withNumericFlag = [...catalog, numericFlag];

  assert.deepEqual(
    completeLlamaExtraArgs(
      "--fit-ctx 20",
      "--fit-ctx 20".length,
      withNumericFlag,
    ),
    [],
  );
  assert.deepEqual(
    completeLlamaExtraArgs(
      "--fit-ctx=20",
      "--fit-ctx=20".length,
      withNumericFlag,
    ),
    [],
  );
});

test("completion replacement preserves surrounding text and returns the caret", () => {
  const [completion] = completeLlamaExtraArgs("--fit o --old-flag", 7, catalog);
  const applied = applyLlamaExtraArgsCompletion(
    "--fit o --old-flag",
    completion,
  );
  assert.equal(applied.text, "--fit on --old-flag");
  assert.equal(applied.caret, "--fit on ".length);
});

test("diagnostics block managed flags and warn for expert-accessible problems", () => {
  const diagnostics = diagnoseLlamaExtraArgs(
    "--host 0.0.0.0 --fit maybe --fit off --old-flag --unknown",
    catalog,
  );
  assert.ok(
    diagnostics.some(
      (item) => item.kind === "managed" && item.severity === "error",
    ),
  );
  assert.ok(diagnostics.some((item) => item.kind === "overlap"));
  assert.ok(diagnostics.some((item) => item.kind === "invalid-choice"));
  assert.ok(diagnostics.some((item) => item.kind === "duplicate"));
  assert.ok(diagnostics.some((item) => item.kind === "deprecated"));
  assert.ok(diagnostics.some((item) => item.kind === "unknown"));
});

test("catalog-unavailable fallback still parses and enforces local limits", () => {
  assert.deepEqual(diagnoseLlamaExtraArgs("--future value", null), []);
  const tooMany = Array.from({ length: 257 }, () => "x").join(" ");
  assert.ok(
    diagnoseLlamaExtraArgs(tooMany, null).some((item) => item.kind === "limit"),
  );
  assert.ok(
    diagnoseLlamaExtraArgs("a\0b", null).some((item) => item.kind === "limit"),
  );
});

const managedPolicyGroups = [
  ["-np", "--parallel", "--n-parallel"],
  ["-m", "--model"],
  ["-a", "--alias"],
  ["-hf", "-hfr", "--hf-repo"],
  ["-hff", "--hf-file"],
  ["-hft", "--hf-token"],
  ["-mm", "--mmproj"],
  ["--host"],
  ["--port"],
  ["--path"],
  ["--api-prefix"],
  ["--api-key"],
  ["--api-key-file"],
  ["--ssl-key-file"],
  ["--ssl-cert-file"],
  ["--webui", "--no-webui"],
  ["--ui", "--no-ui"],
  ["--ui-config", "--webui-config"],
  ["--ui-config-file", "--webui-config-file"],
  [
    "--ui-mcp-proxy",
    "--no-ui-mcp-proxy",
    "--webui-mcp-proxy",
    "--no-webui-mcp-proxy",
  ],
  ["-h", "--help", "--usage"],
  ["--version"],
  ["--list-devices"],
  ["-cl", "--cache-list"],
  ["--completion-bash"],
  ["--tools"],
  ["-ag", "--agent", "-no-ag", "--no-agent"],
  ["--tools-runtime"],
  ["--mcp-servers-config"],
  ["--mcp-servers-json"],
  ["--cors-origins"],
  ["--cors-headers"],
  ["--cors-methods"],
  ["--cors-credentials", "--no-cors-credentials"],
  ["--media-path"],
  ["--log-file"],
  ["--log-disable"],
  ["--slot-save-path"],
];

test("backend managed policy remains usable without installed help", () => {
  const policyCatalog = llamaServerDiagnosticCatalog({
    arguments: [],
    managed_flag_groups: managedPolicyGroups,
    managed_flags: managedPolicyGroups.flat(),
  });
  const diagnostics = diagnoseLlamaExtraArgs(
    "-np 8 --api_key=secret --webui-config=file.json --help",
    policyCatalog,
  );
  assert.equal(diagnostics.filter((item) => item.kind === "managed").length, 4);
});

test("partial help keeps managed errors but does not invent unknown warnings", () => {
  const policyCatalog = llamaServerDiagnosticCatalog({
    arguments: [],
    managed_flag_groups: managedPolicyGroups,
    managed_flags: managedPolicyGroups.flat(),
  });
  const diagnostics = diagnoseLlamaExtraArgs(
    "--future-flag value --host 0.0.0.0",
    policyCatalog,
    false,
  );
  assert.equal(diagnostics.filter((item) => item.kind === "managed").length, 1);
  assert.equal(diagnostics.filter((item) => item.kind === "unknown").length, 0);
});

test("catalog entries without a backend category display as Unclassified", () => {
  assert.equal(
    llamaServerArgumentGroupLabel({
      group: "model",
      policy_category: "Unclassified",
    }),
    "Unclassified",
  );
  assert.equal(
    llamaServerArgumentGroupLabel({
      group: "server",
      policy_category: "Routing/listening",
    }),
    "Routing/listening",
  );
});

test("overlap copy says custom arguments win", () => {
  const warning = diagnoseLlamaExtraArgs("--fit off", catalog).find(
    (item) => item.kind === "overlap",
  );
  assert.equal(warning?.message, "--fit overrides the matching Run Setting.");
});
