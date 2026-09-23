// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { MAX_RUN_CONFIG_URL_LENGTH } from "../src/features/model-picker/sharing/link-address.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { createRunConfigLink, parseRunConfigLink } = await import(
  "./helpers/sharing-links.ts"
);
const { sharedExtraArgsError, validSharedExtraArgs } = await import(
  "../src/features/model-picker/sharing/extra-args.ts"
);
const { diagnoseExtraArgs, extraArgsAreLoadable, formatExtraArgs } =
  await import("../src/features/model-picker/model-config/llama-extra-args.ts");
const { SHARED_CONFIG_KEYS, mergeSharedRunConfig } = await import(
  "../src/features/model-picker/sharing/fields.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const prefixes = ["unsloth://run?v=1&", "http://localhost:8888/chat#run?v=1&"];
const query = (key: string, value: unknown) =>
  new URLSearchParams({
    nParallel: "2",
    [key]: JSON.stringify(value),
  }).toString();

function rejects(query: string) {
  for (const prefix of prefixes) {
    const result = parseRunConfigLink(`${prefix}${query}`);
    assert.equal(result.kind, "invalid", `${prefix}${query}`);
    assert.equal(Object.hasOwn(result, "value"), false);
  }
}

const fullConfig: Omit<
  typeof DEFAULT_PER_MODEL_CONFIG,
  "chatTemplateOverride"
> = {
  customContextLength: 32768,
  maxSeqLength: 32768,
  kvCacheDtype: "q8_0",
  mlxKvBits: 4,
  speculativeType: "dspark",
  specDraftNMax: 8,
  specDraftCacheDtype: "q4_0",
  nParallel: 4,
  reasoningBudget: 0,
  reasoningBudgetMessage: "",
  nBatch: 2048,
  nUbatch: 512,
  loadMode: "mmap+mlock",
  ctxCheckpoints: 0,
  cacheRam: -1,
  tensorParallel: true,
  disableVision: false,
  llamaExtraArgs: [
    "--rope-scaling",
    "yarn",
    "--yarn-orig-ctx",
    "32768",
    "--flash-attn",
    "on",
    "--no-warmup",
  ],
  gpuMemoryMode: "manual",
  gpuLayers: 0,
  nCpuMoe: 0,
  selectedGpuIds: [1, 0],
  selectedGpuIndexKind: "physical",
};

test("native schemes and hosts accept case variations without relaxing address validation", () => {
  const value = { model: "owner/model", config: { nParallel: 3 } };
  const link = createRunConfigLink(value);
  for (const address of ["unsloth://run", "UNSLOTH://RUN", "UnSlOtH://RuN"]) {
    const url = link.replace("unsloth://run", address);
    assert.deepEqual(parseRunConfigLink(url), { kind: "valid", value });
    assert.equal(parseRunConfigLink(`${url}#ignored`).kind, "invalid");
    assert.deepEqual(
      parseRunConfigLink(
        `${url}&reasoningBudgetMessage=${"a".repeat(MAX_RUN_CONFIG_URL_LENGTH)}`,
      ),
      { kind: "invalid", error: "This run configuration link is too long." },
    );
    assert.equal(
      parseRunConfigLink(`${address.slice(0, -3)}hub?model=owner/model`).kind,
      "unrelated",
    );
  }
});

test("both link formats require exactly one supported version", () => {
  for (const prefix of ["unsloth://run", "https://example.com/chat#run"]) {
    for (const query of ["", "?", "?model=owner/model", "?nParallel=3"]) {
      assert.deepEqual(parseRunConfigLink(`${prefix}${query}`), {
        kind: "invalid",
        error: "This run configuration link is missing its version.",
      });
    }
    for (const query of ["v=", "v=0", "v=2", "v=01", "v=1&v=1", "v=1&%76=1"]) {
      assert.equal(parseRunConfigLink(`${prefix}?${query}`).kind, "invalid");
    }
    assert.deepEqual(parseRunConfigLink(`${prefix}?nParallel=3&v=1`), {
      kind: "valid",
      value: { config: { nParallel: 3 } },
    });
  }
});

test("every field and the complete payload round-trip in browser and desktop links", () => {
  assert.deepEqual(
    Object.keys(fullConfig).sort(),
    [...SHARED_CONFIG_KEYS].sort(),
  );
  const values = [
    ...SHARED_CONFIG_KEYS.map((key) => ({
      config: { [key]: fullConfig[key] },
    })),
    {
      model: "unsloth/Model-GGUF",
      ggufVariant: "Q4_K_M/model-00001-of-00002.gguf",
      isGguf: true,
      config: fullConfig,
    },
  ];
  for (const value of values) {
    for (const base of [
      undefined,
      "http://localhost:8888/hub?token=private#old",
      "https://studio.example/chat",
    ]) {
      const url = createRunConfigLink(value, base);
      assert.deepEqual(parseRunConfigLink(url), { kind: "valid", value });
      assert.ok(!url.includes("token=private"));
    }
  }
});

test("an empty link and independently omitted model identity fields are valid", () => {
  for (const url of [
    "unsloth://run?v=1",
    "unsloth://run/?v=1",
    "http://localhost:8888/chat#run?v=1",
    "https://example.com/chat#run?v=1",
  ]) {
    assert.deepEqual(parseRunConfigLink(url), {
      kind: "valid",
      value: { config: {} },
    });
  }
  for (const identity of [
    {},
    { model: "unsloth/Model-GGUF" },
    { model: "own_er/mod_el" },
    { ggufVariant: "Q4_K_M" },
    { isGguf: false },
  ]) {
    const value = { ...identity, config: {} };
    assert.deepEqual(parseRunConfigLink(createRunConfigLink(value)), {
      kind: "valid",
      value,
    });
  }
});

test("invalid input never produces a partial configuration", () => {
  const invalid = [
    "customContextLength=4096&maxSeqLength=8192",
    "nParallel=2&nParallel=3",
    "model=owner/model&model=other/model",
    "model=_owner/model",
    "model=owner_/model",
    "model=owner/_model",
    "model=owner/model_",
    "__proto__=true",
    "constructor=true",
    "hfToken=secret",
    "nParallel=2&unknown=true",
    "nParallel=0",
    "nParallel=65",
    "nParallel=2.5",
    "nParallel=%222%22",
    "nParallel=NaN",
    "tensorParallel=1",
    "tensorParallel=null",
    "disableVision=%22false%22",
    "customContextLength=127",
    "maxSeqLength=1048577",
    "reasoningBudget=-2",
    "nBatch=0",
    "nUbatch=65537",
    "ctxCheckpoints=257",
    "cacheRam=-2",
    "kvCacheDtype=%22unsupported%22",
    "speculativeType=%22unknown%22",
    "mlxKvBits=1",
    "selectedGpuIds=[1,1]",
    "selectedGpuIds=[]",
    "selectedGpuIds=[-1]",
    "selectedGpuIds=[0.5]",
    "llamaExtraArgs={}",
    "llamaExtraArgs=[3]",
    "llamaExtraArgs=[%22\\u0000%22]",
    "llamaExtraArgs=[%22\\ud800%22]",
    "llamaExtraArgs=[%22\\r%22]",
    "reasoningBudgetMessage=%FF",
    "reasoningBudgetMessage=%GG",
    "reasoningBudgetMessage=%",
    "model=/tmp/model",
    "model=C:%5Cmodels%5Cmodel",
    "model=owner/..",
    "model=owner/repo.git",
    "ggufVariant=../model.gguf",
    "ggufVariant=C:%5Cmodel.gguf",
    "ggufVariant=%00",
    "isGguf=false&ggufVariant=Q4_K_M",
  ];
  for (const value of invalid) rejects(value);
  for (const url of [
    "unsloth://run/extra?v=1",
    "unsloth://run?v=1&model=owner/model#ignored",
    "unsloth://user@run?v=1",
    "unsloth://run:80?v=1",
  ]) {
    assert.equal(parseRunConfigLink(url).kind, "invalid", url);
  }
});

test("unrelated links are left to their existing handlers", () => {
  for (const url of [
    "invalid",
    "unsloth://open_from_hf?model=owner/model",
    "https://example.com/chat?model=owner/model",
    "https://example.com/chat#running",
    "https://example.com/a b",
    `https://example.com/chat?unrelated=${"a".repeat(MAX_RUN_CONFIG_URL_LENGTH)}`,
    "javascript:alert(1)",
  ]) {
    assert.equal(parseRunConfigLink(url).kind, "unrelated");
  }
});

test("links and field payloads have bounded sizes", () => {
  assert.equal(
    parseRunConfigLink(
      `unsloth://run?reasoningBudgetMessage=${"a".repeat(MAX_RUN_CONFIG_URL_LENGTH)}`,
    ).kind,
    "invalid",
  );
  assert.throws(() =>
    createRunConfigLink({
      config: { reasoningBudgetMessage: "🦥".repeat(2049) },
    }),
  );
  assert.throws(() =>
    createRunConfigLink({ config: { llamaExtraArgs: Array(257).fill("x") } }),
  );
  assert.throws(() =>
    createRunConfigLink({ config: {} }, "file:///tmp/index.html"),
  );
  assert.throws(() =>
    createRunConfigLink({ config: {} }, "https://user:secret@example.com/"),
  );
});

test("invalid field values use readable labels for malformed and decoded input", () => {
  for (const prefix of prefixes) {
    for (const value of ["unsupported", '"unsupported"', '"unterminated']) {
      assert.deepEqual(
        parseRunConfigLink(
          `${prefix}kvCacheDtype=${encodeURIComponent(value)}`,
        ),
        {
          kind: "invalid",
          error: "The setting “KV cache type” is invalid.",
        },
      );
    }
    for (const value of ["", '""']) {
      const params = new URLSearchParams({
        reasoningBudgetMessage: value,
      });
      assert.deepEqual(parseRunConfigLink(`${prefix}${params}`), {
        kind: "valid",
        value: {
          config: { reasoningBudgetMessage: "" },
        },
      });
    }
  }
  rejects("reasoningBudgetMessage=null");
});

const sensitiveFlags = [
  "--model",
  "-m",
  "--model-url",
  "--hf-repo",
  "--hf-token",
  "--model-draft",
  "--spec-draft-model",
  "--spec-draft-hf",
  "--mmproj",
  "--mmproj-url",
  "--lora",
  "--lora-scaled",
  "--control-vector",
  "--control-vector-scaled",
  "--rpc",
  "--rpc-server",
  "--host",
  "--port",
  "--api-key",
  "--api-key-file",
  "--ssl-key-file",
  "--ssl-cert-file",
  "--path",
  "--media-path",
  "--log-file",
  "--logdir",
  "--log-disable",
  "--slot-save-path",
  "--prompt-cache",
  "--prompt-cache-all",
  "--file",
  "-f",
  "--grammar-file",
  "--grammar",
  "--json-schema",
  "--json-schema-file",
  "--chat-template",
  "--chat-template-file",
  "--chat-template-kwargs",
  "--jinja",
  "--tools",
  "--agent",
  "-ag",
  "--tools-runtime",
  "--mcp-servers-json",
  "--mcp-servers-config",
  "--ui-mcp-proxy",
  "--webui-config-file",
  "--cors-origins",
  "--cors-credentials",
  "--models-dir",
  "--models-preset",
  "--models-autoload",
  "--device",
  "--override-tensor",
  "--props",
  "--slots",
  "--metrics",
  "--help",
  "--completion-bash",
  "--new-unknown-option",
];

test("shared arguments reject file, network, tool, template and unknown capabilities", () => {
  for (const flag of sensitiveFlags) {
    for (const spelling of [
      flag,
      flag.replaceAll("-", "_"),
      `${flag}=x`,
      `${flag} x`,
    ]) {
      for (const tokens of [
        [spelling],
        [spelling, "x"],
        ["--threads", "4", spelling, "x"],
      ]) {
        rejects(query("llamaExtraArgs", tokens));
      }
    }
  }
});

test("encoded flag spellings cannot evade the decoded argument allowlist", () => {
  for (const flag of sensitiveFlags) {
    const json = JSON.stringify([flag, "x"]);
    const variants = [
      encodeURIComponent(json),
      encodeURIComponent(json).replaceAll("-", "%2d"),
      [...json]
        .map((char) => `%${char.charCodeAt(0).toString(16).padStart(2, "0")}`)
        .join(""),
      encodeURIComponent(JSON.stringify([encodeURIComponent(flag), "x"])),
      encodeURIComponent(json.replaceAll("-", "\\u002d")),
      encodeURIComponent(encodeURIComponent(json)),
    ];
    for (const variant of variants) {
      rejects(`nParallel=2&llamaExtraArgs=${variant}`);
    }
  }
});

test("generation enforces the same restrictions as receiving a link", () => {
  for (const value of [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ]) {
    assert.throws(() => createRunConfigLink({ config: { nBatch: value } }));
  }
  for (const flag of sensitiveFlags) {
    assert.throws(() =>
      createRunConfigLink({ config: { llamaExtraArgs: [flag, "x"] } }),
    );
  }
  assert.throws(() =>
    createRunConfigLink({ ggufVariant: "C:/model.gguf", config: {} }),
  );
});

test("only exact argument boundaries and bounded inference values are accepted", () => {
  const invalid = [
    ["--threads=4"],
    ["--threads", "4.0"],
    ["--threads"],
    ["--threads="],
    ["--threads", ""],
    ["--threads", "4", "orphan"],
    ["--no-warmup", "x"],
    ["--no-warmup=true"],
    ["--threads=4", "5"],
    ["--threads", "4", "--threads", "5"],
    [" --threads", "4"],
    ["--threads ", "4"],
    ["--threads\n", "4"],
    ["--no-warmup", " --no-context-shift"],
    ["--no-context-shift", "--no-warmup\n"],
    ["--no_warmup", "\t--no_context_shift"],
    ["--threads", "4\n"],
    ["--threads", " 4"],
    ["--threads", "4\t"],
    ["--threads", "-2"],
    ["--seed", "-2"],
    ["-c", "-1"],
    ["-c", "127"],
    ["-t", "4", "--threads", "5"],
    ["--gpu-layers", "4", "--n-gpu-layers", "5"],
    ["--ctx-size", "4096", "-c", "0"],
    ["--threads-batch", "2", "-tb", "4"],
    ["--threads", "1025"],
    ["--threads", "4.5"],
    ["--threads", "0x4"],
    ["--threads", "+4"],
    ["--threads", "04"],
    ["--threads", "4e0"],
    ["--threads", "NaN"],
    ["--threads", "Infinity"],
    ["--threads", "４"],
    ["--threads", "4\u0000--agent"],
    ["--threads", "4\u202e"],
    ["--threads", "4\u0085"],
    ["--threads", "4\ud800"],
    ["--threads", "4; touch injected"],
    ["--threads", "$(touch injected)"],
    ["--threads", "`touch injected`"],
    ["--threads", "%COMSPEC% /c whoami"],
    ["--threads", "& calc.exe"],
    ["--threads", "--agent"],
    ["--threads=4 --agent"],
    ["--threads", "4\n--agent"],
    ["--threads", "4\r\n--agent"],
    ["--threads", '4&llamaExtraArgs=["--agent"]'],
    ["--"],
    ["@arguments.txt"],
    ["-t4"],
    ["--flash-attn", "on --agent"],
    ["--cache-type-k", "q8_0;whoami"],
    ["--tensor-split", "0,0"],
    ["--tensor-split", "1,,2"],
    ["--tensor-split", "1,2\n"],
    ["--tensor-split", "1,RPC:evil:5000"],
    ["--tensor-split", Array(257).fill("1").join(",")],
    ["-ts", "0,0"],
    ["-ts", "1,RPC:evil:5000"],
    ["-ts", "1,1", "--tensor-split", "3,1"],
    ["-s", "-2"],
    ["-s", "4294967296"],
    ["-s", "1", "--seed", "2"],
    ["--temperature", "0.7;whoami"],
    ["--temperature", "101"],
    ["--temperature", "0.7", "--temp", "0.8"],
  ];
  for (const args of invalid) rejects(query("llamaExtraArgs", args));
  const valid = [
    [],
    ["--threads", "0"],
    ["-t", "-1"],
    ["-tb", "-1"],
    ["-c", "0", "--seed", "-1"],
    ["--ctx-size", "0", "--yarn-orig-ctx", "0"],
    ["-t", "4", "--threads-batch", "8"],
    ["--threads", "4"],
    ["-t", "4"],
    ["--no-warmup", "--no-context-shift"],
    ["--gpu-layers", "-1"],
    ["--top-p", "0.95"],
    ["--flash-attn", "auto"],
    ["--rope-scaling", "yarn", "--yarn-orig-ctx", "32768"],
    ["--tensor-split", "3,1.5,0"],
    ["-ts", "3,1.5,0"],
    ["-s", "-1"],
    ["-s", "4294967295"],
    ["--temperature", "0.7"],
    ["-ts", "1,1", "-s", "42", "--temperature", "0.7"],
    ["--threads_batch", "4"],
    ["--tensor_split", "1,1", "--cache_type_k", "q8_0"],
  ];
  for (const args of valid) {
    const value = { config: { llamaExtraArgs: args } };
    assert.deepEqual(parseRunConfigLink(createRunConfigLink(value)), {
      kind: "valid",
      value,
    });
  }
});

test("sharing diagnostics identify the offending option without exposing its value", () => {
  assert.equal(sharedExtraArgsError(["-ts", "1,1"]), null);
  assert.match(
    sharedExtraArgsError(["--threads"]) ?? "",
    /--threads requires a value/,
  );
  assert.match(
    sharedExtraArgsError(["--threads", "1025"]) ?? "",
    /--threads has an invalid or unsupported value/,
  );
  assert.match(
    sharedExtraArgsError(["-ts", "1,1", "--tensor-split", "2,1"]) ?? "",
    /--tensor-split is specified more than once/,
  );
  assert.match(
    sharedExtraArgsError(["--threads=4"]) ?? "",
    /Write --threads and its value as two arguments/,
  );
  assert.match(
    sharedExtraArgsError(["--no-warmup=true"]) ?? "",
    /--no-warmup does not take a value/,
  );
  for (const flag of sensitiveFlags) {
    for (const payload of [
      "private-api-key",
      "C:\\private\\adapter.gguf",
      "/home/user/private.gguf",
      '<img src="/attack" onerror="alert(1)">',
      "\u202e--threads\n4",
    ]) {
      for (const tokens of [[flag, payload], [`${flag}=${payload}`]]) {
        assert.equal(
          sharedExtraArgsError(["--threads", "4", ...tokens]),
          `${flag} is not supported in shared links.`,
        );
      }
      assert.equal(
        sharedExtraArgsError(["--threads", payload]),
        "--threads has an invalid or unsupported value.",
      );
    }
  }
  for (const token of [
    " --threads",
    "--threads\n",
    "\t--no_context_shift",
    "--metrics\u202e",
    '<img src="/attack">',
    `--${"x".repeat(1000)}`,
    "C:\\private\\adapter.gguf",
  ]) {
    assert.equal(
      sharedExtraArgsError([token]),
      "Extra arguments contain an unexpected token.",
    );
  }
});

test("shared arguments use upstream integer diagnostics offline", () => {
  for (const args of [
    ["--batch-size", "1"],
    ["-b", "1"],
    ["--batch_size", "1"],
    ["--ctx-size", "128.5"],
    ["-c", "128.0"],
    ["--gpu-layers", "4.5"],
    ["--n_gpu_layers", "4.0"],
    ["-ngl", "4.5"],
    ["-ncmoe", "1.5"],
    ["--ubatch_size", "2.5"],
  ]) {
    const diagnostics = diagnoseExtraArgs(formatExtraArgs(args), null);
    assert.equal(extraArgsAreLoadable(diagnostics), false);
    const upstreamError = diagnostics.find((item) => item.level === "error");
    assert.equal(sharedExtraArgsError(args), upstreamError?.message);
    rejects(query("llamaExtraArgs", args));
  }
});

test("upstream normalization preserves argv without bypassing sharing restrictions", () => {
  for (const args of [
    ["--threads_batch", "4", "-tb", "8"],
    ["--tensor_split", "1,1", "-ts", "2,1"],
    ["--n_gpu_layers", "4", "--gpu-layers", "8"],
    ["--no_warmup", "--no-warmup"],
    ["--cache_type_k", "q8_0", "-ctk", "f16"],
  ]) {
    rejects(query("llamaExtraArgs", args));
  }
  for (const args of [
    ["--new-unknown-option", "x"],
    ["--lora", "/private/adapter.gguf"],
    ["--grammar_file", "C:\\private\\grammar.txt"],
    ["--chat_template", "{{ messages }}"],
  ]) {
    assert.equal(
      extraArgsAreLoadable(diagnoseExtraArgs(formatExtraArgs(args), null)),
      true,
    );
    assert.equal(validSharedExtraArgs(args), false);
    rejects(query("llamaExtraArgs", args));
  }
  const sparse = new Array<string>(3);
  sparse[0] = "--threads";
  sparse[2] = "4";
  for (const args of [Array(1), sparse, [undefined], [1]]) {
    assert.equal(validSharedExtraArgs(args), false);
  }
});

test("prototype and nested configuration keys are rejected without side effects", () => {
  for (const key of [
    "__proto__",
    "constructor",
    "prototype",
    "toString",
    "valueOf",
    "config",
    "config.nParallel",
    "config[nParallel]",
    "__proto__[polluted]",
    "llamaExtraArgs[]",
    "llamaExtraArgs[0]",
    "trust_remote_code",
    "hfToken",
    "nativePathToken",
    "command",
    "redirect",
    "url",
    "script",
    "onload",
  ])
    rejects(
      new URLSearchParams({
        nParallel: "2",
        [key]: '{"polluted":true}',
      }).toString(),
    );
  for (const value of [
    { __proto__: null },
    { constructor: [] },
    [["--threads", "4"]],
    [null],
    [4],
  ]) {
    rejects(query("llamaExtraArgs", value));
  }
  rejects(query("nParallel", { valueOf: 2 }));
  rejects(query("selectedGpuIds", { 0: 1, length: 1 }));
  const patch = JSON.parse(
    '{"__proto__":{"polluted":true},"command":"whoami","nParallel":2}',
  );
  const merged = mergeSharedRunConfig(DEFAULT_PER_MODEL_CONFIG, patch, true);
  assert.equal(Object.getPrototypeOf(merged), Object.prototype);
  assert.equal(Object.hasOwn(merged, "__proto__"), false);
  assert.equal(Object.hasOwn(merged, "command"), false);
  assert.equal(Object.hasOwn(Object.prototype, "polluted"), false);
  assert.equal(merged.nParallel, 2);
});

test("parameter smuggling and duplicate encoded keys fail as a whole", () => {
  for (const value of [
    "nParallel=2&%6eParallel=3",
    "nParallel=2&n%50arallel=3",
    "llamaExtraArgs=[]&llamaExtraArgs=%5B%22--agent%22%5D",
    "nParallel=2&%5f%5fproto%5f%5f=true",
    "nParallel=2&%255f%255fproto%255f%255f=true",
    "nParallel=2%26llamaExtraArgs%3D%5B%22--agent%22%5D",
    "nParallel=2;llamaExtraArgs=[]",
    "nParallel=2&nParallel%00=3",
    "nParallel=2&nParallel+=3",
    "nParallel=2&nParallel%0a=3",
    "nParallel=2&%256eParallel=3",
    "nParallel=2&=3",
    "nParallel=2&NParallel=3",
    "llamaExtraArgs=%25255B%252522--agent%252522%25255D",
  ])
    rejects(value);
});

test("malformed encodings, noncanonical addresses and URL control normalization fail closed", () => {
  for (const sequence of [
    "%",
    "%0",
    "%GG",
    "%C0%AF",
    "%ED%A0%80",
    "%E0%A4",
    "%F4%90%80%80",
    "%FF",
  ]) {
    rejects(`reasoningBudgetMessage=${sequence}`);
  }
  for (const url of [
    "unsloth://run/a/../?v=1&nParallel=2",
    "unsloth://run/%2e/?v=1&nParallel=2",
    "unsloth://run/%2e%2e/?v=1&nParallel=2",
    "unsloth://user@run?v=1&nParallel=2",
    "unsloth://run:1?v=1&nParallel=2",
    "unsloth://run?v=1&nParallel=2#anything",
    " unsloth://run?v=1&nParallel=2",
    "unsloth://run?v=1&nParallel=2\n",
    "un\nsloth://run?v=1&nParallel=2",
    "unsloth://run?v=1&nParallel=\t2",
    "https://user:pass@localhost/chat#run?v=1&nParallel=2",
  ])
    assert.equal(parseRunConfigLink(url).kind, "invalid", url);
  for (const protocol of [
    "javascript:",
    "data:text/html,",
    "file:///",
    "vbscript:",
  ]) {
    assert.notEqual(
      parseRunConfigLink(`${protocol}#run?v=1&nParallel=2`).kind,
      "valid",
    );
  }
});

test("model and variant identities cannot supply paths, devices, traversal or URLs", () => {
  const variants = [
    "/etc/passwd",
    "../model.gguf",
    "weights/../model.gguf",
    "weights/.. /model.gguf",
    "weights/./model.gguf",
    "weights//model.gguf",
    "C:/model.gguf",
    "C:model.gguf",
    "C:\\model.gguf",
    "\\\\server\\share\\model.gguf",
    "//server/share/model.gguf",
    "model.gguf:stream",
    "CON",
    "nul.gguf",
    "AUX/model.gguf",
    "COM1.gguf",
    "LPT9",
    "weights./model.gguf",
    "weights /model.gguf",
    "Q4_K_M\n",
    "Q4_K_M\u202e",
    "weights/%2e%2e/model.gguf",
    "weights/%252e%252e/model.gguf",
    "https://attacker.invalid/model.gguf",
    "$(whoami).gguf",
    "`whoami`.gguf",
    "model*.gguf",
    "model?.gguf",
    "[model].gguf",
    "model.gguf|whoami",
    "..\u2215model.gguf",
    "..\uff0fmodel.gguf",
    "..\uff3cmodel.gguf",
    "folder\n/model.gguf",
    "model.gguf\u0000",
    "a".repeat(256),
  ];
  for (const ggufVariant of variants) {
    rejects(new URLSearchParams({ ggufVariant }).toString());
  }
  for (const model of [
    "C:/model",
    "../model",
    "owner/../model",
    "owner\\model",
    "/etc/passwd",
    "https://attacker.invalid/model",
    "owner/model\n",
    "owner\n/model",
    "owner%2fmodel",
    "owner/model.git",
    "owner/model;whoami",
    "owner/model%00",
  ])
    rejects(new URLSearchParams({ model }).toString());
});

test("Windows device aliases with spaces before an extension are rejected", () => {
  for (const device of ["CON", "nul", "PrN", "AUX", "COM1", "LPT9"]) {
    for (const directory of ["", "weights/"]) {
      for (const spaces of [" ", "  "]) {
        const ggufVariant = `${directory}${device}${spaces}.gguf`;
        rejects(new URLSearchParams({ ggufVariant }).toString());
        assert.throws(() => createRunConfigLink({ ggufVariant, config: {} }));
      }
    }
  }
  for (const ggufVariant of ["my model.gguf", "weights/model .gguf"]) {
    const value = { ggufVariant, config: {} };
    assert.deepEqual(parseRunConfigLink(createRunConfigLink(value)), {
      kind: "valid",
      value,
    });
  }
});

test("templates cannot be shared or clear a recipient's template", () => {
  const recipient = {
    ...DEFAULT_PER_MODEL_CONFIG,
    chatTemplateOverride: "Recipient template",
  };
  for (const template of [
    null,
    "",
    "{{ messages[0]['content'] }}",
    "{{ cycler.__init__.__globals__.os.popen('id').read() }}",
    "{% for x in range(1000000000) %}x{% endfor %}",
    "{{ 'x' * 1000000000 }}",
    "{% include '/etc/passwd' %}",
    "hello",
    "null",
    " ",
  ]) {
    rejects(query("chatTemplateOverride", template));
    rejects(query("llamaExtraArgs", ["--chat-template", template]));
    const config = { nParallel: 2, chatTemplateOverride: template };
    for (const address of [undefined, "http://localhost:8888/chat"]) {
      const link = createRunConfigLink({ config }, address);
      assert.equal(link.includes("chatTemplateOverride"), false);
      assert.deepEqual(parseRunConfigLink(link), {
        kind: "valid",
        value: { config: { nParallel: 2 } },
      });
    }
    assert.equal(
      mergeSharedRunConfig(recipient, config, true).chatTemplateOverride,
      recipient.chatTemplateOverride,
    );
  }
  rejects("chatTemplateOverride=");
});

test("shared reasoning messages reject invisible separators and format controls", () => {
  for (const character of [
    "\u200b",
    "\u2060",
    "\u2061",
    "\u2062",
    "\u2063",
    "\u2064",
    "\ufeff",
  ]) {
    const message = `Review${character}this`;
    rejects(`reasoningBudgetMessage=${encodeURIComponent(message)}`);
    for (const base of [undefined, "http://localhost:8888/chat"]) {
      assert.throws(() =>
        createRunConfigLink(
          { config: { reasoningBudgetMessage: message } },
          base,
        ),
      );
    }
  }
});

test("custom reasoning messages cannot enter links as plain text, HTML or encoded instructions", () => {
  for (const message of [
    '<img src=x onerror="globalThis.injected=true">',
    "</textarea><script>alert(1)</script>",
    "$(touch injected) & %COMSPEC% `whoami`",
    "%253Cscript%253E",
    'hello &llamaExtraArgs=["--agent"]#run?command=x + 100%',
    "回答 🦥\nline two",
    "می\u200cخواهم",
    "क्\u200dष",
    "👩\u200d💻",
  ]) {
    const value = { config: { reasoningBudgetMessage: message } };
    for (const base of [undefined, "http://localhost:8888/chat"]) {
      assert.throws(
        () => createRunConfigLink(value, base),
        /Custom reasoning messages/,
      );
      for (const text of [message, JSON.stringify(message)]) {
        const prefix = base ? `${base}#run?` : "unsloth://run?";
        assert.equal(
          parseRunConfigLink(
            `${prefix}v=1&reasoningBudgetMessage=${encodeURIComponent(text)}`,
          ).kind,
          "invalid",
        );
      }
    }
  }
});

test("bounded mutations of a permitted argv value cannot introduce syntax", () => {
  const syntax = [
    '"',
    "'",
    "`",
    "$",
    "&",
    ";",
    "|",
    "<",
    ">",
    "=",
    "\\",
    "/",
    "%",
    "@",
    "#",
    "\n",
    "\r",
    "\t",
    "\u0000",
    "\u202e",
    "\u2066",
  ];
  for (const character of syntax) {
    for (const value of [
      `${character}4`,
      `4${character}`,
      `4${character}--agent`,
    ]) {
      assert.equal(
        validSharedExtraArgs(["--threads", value]),
        false,
        JSON.stringify(value),
      );
    }
  }
});

test("10,000 deterministic adversarial inputs reject against independent expectations", () => {
  let seed = 0x5eed;
  const next = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed;
  };
  const prototypes = Object.getOwnPropertyDescriptors(Object.prototype);
  const hostile = [";", "\\", "/", "\n", "\u0000", "\u202e", "$", "`", " "];
  const mutations = [
    () =>
      query("llamaExtraArgs", [
        "--threads",
        `${1 + (next() % 1024)}${hostile[next() % hostile.length]}`,
      ]),
    () =>
      query("llamaExtraArgs", [
        sensitiveFlags[next() % sensitiveFlags.length],
        String(next()),
      ]),
    () => new URLSearchParams({ nParallel: String(65 + next()) }).toString(),
    () => new URLSearchParams({ model: `owner/../model${next()}` }).toString(),
    () =>
      new URLSearchParams({ ggufVariant: `../model${next()}.gguf` }).toString(),
    () => `nParallel=2&%6eParallel=${1 + (next() % 64)}`,
    () => query("__proto__", { polluted: next() }),
    () =>
      query("llamaExtraArgs", [
        "-t",
        "4",
        "--threads",
        String(1 + (next() % 1024)),
      ]),
    () =>
      new URLSearchParams({
        customContextLength: "4096",
        maxSeqLength: String(4097 + (next() % 1000)),
      }).toString(),
    () => query("tensorParallel", { value: next() }),
  ];
  for (let index = 0; index < 10_000; index += 1) {
    rejects(mutations[index % mutations.length]());
  }
  assert.deepEqual(
    Object.getOwnPropertyDescriptors(Object.prototype),
    prototypes,
  );
});
