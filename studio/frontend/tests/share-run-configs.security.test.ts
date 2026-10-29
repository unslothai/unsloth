// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { DeepLinkHandler as Handler } from "../src/features/deep-links/deep-link-handler.tsx";
import { createDeepLinkIntentGate } from "../src/features/deep-links/deep-link-intent.ts";
import { parseUnslothDeepLink } from "../src/features/deep-links/parse-deep-link.ts";
import { MAX_RUN_CONFIG_URL_LENGTH } from "../src/features/model-picker/sharing/inbox.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

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
const { receiverHarness, settle } = await import(
  "./helpers/sharing-receiver.ts"
);

const prefixes = ["unsloth://run?v=1&", "http://localhost:8888/chat#run?v=1&"];
const browser = "http://localhost:8888/chat";
const long = "a".repeat(MAX_RUN_CONFIG_URL_LENGTH);
const query = (key: string, value: unknown) =>
  new URLSearchParams({
    nParallel: "2",
    [key]: JSON.stringify(value),
  }).toString();
const params = (key: string, value: string) =>
  new URLSearchParams({ [key]: value }).toString();
const args = (list: unknown) => query("llamaExtraArgs", list);
const valid = (config: object) => ({ kind: "valid", value: { config } });

function rejects(query: string) {
  for (const prefix of prefixes) {
    const result = parseRunConfigLink(`${prefix}${query}`);
    assert.equal(result.kind, "invalid", `${prefix}${query}`);
    assert.equal(Object.hasOwn(result, "value"), false);
  }
}

function roundTrips(
  value: Parameters<typeof createRunConfigLink>[0],
  base?: string,
) {
  const url = createRunConfigLink(value, base);
  assert.deepEqual(parseRunConfigLink(url), { kind: "valid", value });
  return url;
}

const fullConfig: Omit<
  typeof DEFAULT_PER_MODEL_CONFIG,
  "chatTemplateOverride" | "tensorSplit" | "maxSeqLength" | "mlxKvQuant"
> = {
  customContextLength: 32768,
  kvCacheDtype: "q8_0",
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

test("link addresses: case-insensitive native hosts, one exact version, empty and unrelated links", () => {
  const value = { model: "owner/model", config: { nParallel: 3 } };
  for (const address of ["unsloth://run", "UnSlOtH://RuN"]) {
    const url = createRunConfigLink(value).replace("unsloth://run", address);
    assert.deepEqual(parseRunConfigLink(url), { kind: "valid", value });
    assert.equal(parseRunConfigLink(`${url}#ignored`).kind, "invalid");
    assert.deepEqual(
      parseRunConfigLink(`${url}&reasoningBudgetMessage=${long}`),
      { kind: "invalid", error: "This run configuration link is too long." },
    );
    assert.equal(
      parseRunConfigLink(`${address.slice(0, -3)}hub?model=owner/model`).kind,
      "unrelated",
    );
  }
  for (const prefix of ["unsloth://run", "https://example.com/chat#run"]) {
    assert.deepEqual(parseRunConfigLink(`${prefix}?nParallel=3`), {
      kind: "invalid",
      error: "This run configuration link is missing its version.",
    });
    for (const query of ["v=2", "v=1&%76=1"]) {
      assert.equal(parseRunConfigLink(`${prefix}?${query}`).kind, "invalid");
    }
    assert.deepEqual(
      parseRunConfigLink(`${prefix}?nParallel=3&v=1`),
      valid({ nParallel: 3 }),
    );
  }
  for (const url of ["unsloth://run/?v=1", `${browser}#run?v=1`]) {
    assert.deepEqual(parseRunConfigLink(url), valid({}));
  }
  for (const url of ["invalid", `https://example.com/chat?unrelated=${long}`]) {
    assert.equal(parseRunConfigLink(url).kind, "unrelated");
  }
});

test("every field, the complete payload and safe identities round-trip in browser and desktop links", () => {
  assert.deepEqual(
    Object.keys(fullConfig).sort(),
    [...SHARED_CONFIG_KEYS].sort(),
  );
  for (const value of [
    ...SHARED_CONFIG_KEYS.map((key) => ({
      config: { [key]: fullConfig[key] },
    })),
    {
      model: "unsloth/Model-GGUF",
      ggufVariant: "Q4_K_M/model-00001-of-00002.gguf",
      config: fullConfig,
    },
  ]) {
    for (const base of [
      undefined,
      "http://localhost:8888/hub?token=private#old",
      "https://unsloth.example/chat",
    ]) {
      assert.ok(!roundTrips(value, base).includes("token=private"));
    }
  }
  roundTrips({ ggufVariant: "weights/model .gguf", config: {} });
});

test("GPU splits stay local, and importing a different GPU selection or automatic placement clears them", () => {
  const config = { tensorSplit: [3, 1], nParallel: 2 };
  assert.deepEqual(
    parseRunConfigLink(createRunConfigLink({ config })),
    valid({ nParallel: 2 }),
  );
  const current = {
    ...DEFAULT_PER_MODEL_CONFIG,
    ...fullConfig,
    selectedGpuIds: [0, 1],
    tensorSplit: [3, 1],
  };
  const split = (patch: object) =>
    mergeSharedRunConfig(current, patch).tensorSplit;
  for (const patch of [
    { selectedGpuIds: [1, 0] },
    { selectedGpuIndexKind: null },
    { gpuMemoryMode: "auto" },
    { gpuLayers: -1 },
  ]) {
    assert.equal(split(patch), null, JSON.stringify(patch));
  }
  for (const patch of [
    { nParallel: 2 },
    { selectedGpuIds: [0, 1] },
    { selectedGpuIds: undefined },
  ]) {
    assert.deepEqual(split(patch), [3, 1], JSON.stringify(patch));
  }
  assert.deepEqual(current.tensorSplit, [3, 1]);
});

test("invalid, smuggled, malformed or path-like input never produces a partial configuration", () => {
  const queries = [
    "customContextLength=4096&maxSeqLength=8192",
    "nParallel=2&nParallel=3",
    "hfToken=secret",
    "nParallel=0",
    "nParallel=2.5",
    "nParallel=%222%22",
    "tensorParallel=null",
    "selectedGpuIds=[1,1]",
    "selectedGpuIds=[]",
    "llamaExtraArgs=[3]",
    "reasoningBudgetMessage=null",
    "chatTemplateOverride=",
    "nParallel=2&%6eParallel=3",
    "nParallel=2&=3",
    "reasoningBudgetMessage=%",
    "reasoningBudgetMessage=%C0%AF",
    query("__proto__", { polluted: true }),
    args([["--threads", "4"]]),
    query("nParallel", { valueOf: 2 }),
    ...[
      "/etc/passwd",
      "weights/../model.gguf",
      "weights./model.gguf",
      "nul  .gguf",
      "Q4_K_M\n",
      "a".repeat(256),
    ].map((variant) => params("ggufVariant", variant)),
    ...["_owner/model", "owner/repo.git", "https://attacker.invalid/model"].map(
      (model) => params("model", model),
    ),
  ];
  for (const query of queries) rejects(query);
  for (const url of [
    "unsloth://run/extra?v=1",
    "unsloth://user@run?v=1",
    "unsloth://run:80?v=1",
    "unsloth://run?v=1&nParallel=\t2",
    "https://user:pass@localhost/chat#run?v=1",
  ]) {
    assert.equal(parseRunConfigLink(url).kind, "invalid", url);
  }
  const merged = mergeSharedRunConfig(
    DEFAULT_PER_MODEL_CONFIG,
    JSON.parse('{"__proto__":{"polluted":true},"command":"x","nParallel":2}'),
  );
  assert.equal(Object.getPrototypeOf(merged), Object.prototype);
  assert.equal(Object.hasOwn(merged, "command"), false);
  assert.equal(Object.hasOwn(Object.prototype, "polluted"), false);
  assert.equal(merged.nParallel, 2);
});

test("generation enforces bounded sizes and the same restrictions as receiving a link", () => {
  for (const config of [
    { nBatch: Number.NaN },
    { llamaExtraArgs: ["--host", "x"] },
    { reasoningBudgetMessage: "\u{1F9A5}".repeat(2049) },
  ]) {
    assert.throws(() => createRunConfigLink({ config }));
  }
  for (const ggufVariant of ["C:/model.gguf", "COM1 .gguf"]) {
    assert.throws(() => createRunConfigLink({ ggufVariant, config: {} }));
  }
  for (const base of ["file:///index.html", "https://user:pw@example.com/"]) {
    assert.throws(() => createRunConfigLink({ config: {} }, base));
  }
});

test("invalid field values use readable labels for malformed and decoded input", () => {
  const error = "The setting “KV cache type” is invalid.";
  for (const prefix of prefixes) {
    for (const value of ["unsupported", "%22unterminated"]) {
      assert.deepEqual(parseRunConfigLink(`${prefix}kvCacheDtype=${value}`), {
        kind: "invalid",
        error,
      });
    }
    for (const value of ["", "%22%22"]) {
      assert.deepEqual(
        parseRunConfigLink(`${prefix}reasoningBudgetMessage=${value}`),
        valid({ reasoningBudgetMessage: "" }),
      );
    }
  }
});

test("sensitive and unknown flags are rejected in every spelling and encoding", () => {
  for (const flag of sensitiveFlags) {
    for (const spelling of [
      flag,
      flag.replaceAll("-", "_"),
      `${flag}=x`,
      `${flag} x`,
    ]) {
      rejects(args([spelling, "x"]));
      rejects(args(["--threads", "4", spelling]));
    }
    const json = JSON.stringify([flag, "x"]);
    for (const variant of [
      encodeURIComponent(json).replaceAll("-", "%2d"),
      encodeURIComponent(JSON.stringify([encodeURIComponent(flag), "x"])),
      encodeURIComponent(json.replaceAll("-", "\\u002d")),
      encodeURIComponent(encodeURIComponent(json)),
    ]) {
      rejects(`nParallel=2&llamaExtraArgs=${variant}`);
    }
  }
});

test("only exact argument boundaries and bounded inference values are accepted", () => {
  for (const argv of [
    ["--threads", "4.0"],
    ["--threads", "4", "orphan"],
    ["--no-warmup", "x"],
    ["--threads", "-2"],
    ["--threads", "0x4"],
    ["-c", "127"],
    ["--flash-attn", "on --agent"],
    ["--cache-type-k", "q8_0;whoami"],
    ["--tensor-split", "0,0"],
    ["--tensor-split", Array(257).fill("1").join(",")],
    ["--temperature", "101"],
    ["-t4"],
  ]) {
    rejects(args(argv));
  }
  for (const argv of [
    [],
    ["-t", "-1", "-c", "0", "--seed", "-1", "--top-p", "0.95"],
    ["--no-warmup", "--flash-attn", "auto", "-ts", "3,1.5,0", "-ngl", "-1"],
    ["-s", "4294967295", "--threads_batch", "4", "--cache_type_k", "q8_0"],
  ]) {
    roundTrips({ config: { llamaExtraArgs: argv } });
  }
});

test("sharing diagnostics identify the offending option without exposing its value", () => {
  const unexpected = "Extra arguments contain an unexpected token.";
  for (const [argv, message] of [
    [["-ts", "1,1"], null],
    [["--threads"], "--threads requires a value."],
    [["--threads", "1025"], "--threads has an invalid or unsupported value."],
    [
      ["-ts", "1,1", "--tensor-split", "2,1"],
      "--tensor-split is specified more than once.",
    ],
    [["--threads=4"], "Write --threads and its value as two arguments."],
    [["--no-warmup=true"], "--no-warmup does not take a value."],
    [["x"], unexpected],
    [[" --threads"], unexpected],
    [[`--${"x".repeat(1000)}`], unexpected],
  ] as const) {
    assert.equal(sharedExtraArgsError([...argv]), message);
  }
  for (const flag of sensitiveFlags) {
    assert.equal(
      sharedExtraArgsError(["--threads", "4", `${flag}=C:\\key`]),
      `${flag} is not supported in shared links.`,
    );
  }
});

test("shared arguments use upstream diagnostics offline without letting normalization bypass the allowlist", () => {
  for (const argv of [
    ["--batch-size", "1"],
    ["-ncmoe", "1.5"],
  ]) {
    const diagnostics = diagnoseExtraArgs(formatExtraArgs(argv), null);
    assert.equal(extraArgsAreLoadable(diagnostics), false);
    const upstreamError = diagnostics.find((item) => item.level === "error");
    assert.equal(sharedExtraArgsError(argv), upstreamError?.message);
    rejects(args(argv));
  }
  const loadable = ["--chat_template", "{{ x }}"];
  assert.equal(
    extraArgsAreLoadable(diagnoseExtraArgs(formatExtraArgs(loadable), null)),
    true,
  );
  const sparse = new Array<string>(3);
  sparse[0] = "--threads";
  sparse[2] = "4";
  for (const argv of [loadable, sparse, [1], "--threads 4"]) {
    assert.equal(validSharedExtraArgs(argv), false);
  }
});

test("templates and custom reasoning messages cannot enter links or clear a recipient's own", () => {
  const recipient = {
    ...DEFAULT_PER_MODEL_CONFIG,
    chatTemplateOverride: "Mine",
  };
  for (const template of [null, "{{ x }}"]) {
    rejects(query("chatTemplateOverride", template));
    rejects(args(["--chat-template", template]));
    const config = { nParallel: 2, chatTemplateOverride: template };
    for (const base of [undefined, browser]) {
      assert.deepEqual(
        parseRunConfigLink(createRunConfigLink({ config }, base)),
        valid({ nParallel: 2 }),
      );
    }
    assert.equal(
      mergeSharedRunConfig(recipient, config).chatTemplateOverride,
      "Mine",
    );
  }
  for (const message of ["<script>alert(1)</script>", "Review\u200bthis"]) {
    for (const base of [undefined, browser]) {
      assert.throws(
        () =>
          createRunConfigLink(
            { config: { reasoningBudgetMessage: message } },
            base,
          ),
        /Custom reasoning messages/,
      );
      const prefix = base ? `${base}#run?` : "unsloth://run?";
      for (const text of [message, JSON.stringify(message)]) {
        const encoded = encodeURIComponent(text);
        assert.equal(
          parseRunConfigLink(`${prefix}v=1&reasoningBudgetMessage=${encoded}`)
            .kind,
          "invalid",
        );
      }
    }
  }
});

const hub = "unsloth://open_from_hf?model=owner/model";
const run = "unsloth://run?v=1&model=owner/model&nParallel=3";
const invalid =
  "unsloth://run?v=1&llamaExtraArgs=%5B%22--host%22%2C%220.0.0.0%22%5D";

function desktopSession() {
  const values = new Map<string, string>();
  const storage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
  };
  Object.assign(globalThis, { sessionStorage: storage });
  return storage;
}

function harness(sharedLinks = true) {
  const { inbox, errors, receiver } = receiverHarness({ desktop: true });
  const navigations: {
    to: string;
    search: { model: string; intent: number };
  }[] = [];
  const commands: string[] = [];
  const counts = { subscriptions: 0, unsubscriptions: 0 };
  let listener: ((urls: string[]) => void) | undefined;
  let cleanup: (() => void) | undefined;
  let releaseStartup!: (urls: string[] | null) => void;
  const startup = new Promise<string[] | null>((resolve) => {
    releaseStartup = resolve;
  });
  const { DeepLinkHandler } = loadWithStubs<{
    DeepLinkHandler: typeof Handler;
  }>(
    new URL(
      "../src/features/deep-links/deep-link-handler.tsx",
      import.meta.url,
    ),
    {
      "@/lib/api-base": { isTauri: true },
      "@tanstack/react-router": {
        useNavigate: () => async (to: (typeof navigations)[number]) =>
          navigations.push(to),
      },
      react: { useEffect: (effect: () => () => void) => (cleanup = effect()) },
      "./deep-link-intent": { createDeepLinkIntentGate },
      "./parse-deep-link": { parseUnslothDeepLink },
      "@tauri-apps/api/core": {
        invoke: async (command: string) => commands.push(command),
      },
      "@tauri-apps/plugin-deep-link": {
        getCurrent: () => startup,
        onOpenUrl: async (callback: typeof listener) => {
          counts.subscriptions += 1;
          listener = callback;
          return () => (counts.unsubscriptions += 1);
        },
      },
    },
  );
  DeepLinkHandler(
    sharedLinks ? { onOpenUrls: receiver.receiveSharedRunConfigUrls } : {},
  );
  return {
    inbox,
    navigations,
    commands,
    errors,
    counts,
    releaseStartup,
    emit: (urls: string[]) => {
      assert.ok(listener);
      listener(urls);
    },
    cleanup: () => cleanup?.(),
  };
}

async function started(startup: string[] | null, sharedLinks = true) {
  const app = harness(sharedLinks);
  await settle();
  app.releaseStartup(startup);
  await settle();
  return app;
}

for (const [delivery, url] of [
  ["startup", run],
  ["live", invalid],
] as const) {
  test(`a ${delivery} run link is not replayed by a later desktop document but a new event is: ${url}`, async () => {
    desktopSession();
    const accepted = (app: ReturnType<typeof harness>) => {
      assert.deepEqual(app.commands, ["reveal_main_window"]);
      assert.equal(app.errors.length, url === invalid ? 1 : 0);
      assert.equal(app.inbox.getSnapshot() !== null, url === run);
      app.cleanup();
    };
    const before = await started(delivery === "startup" ? [url] : null);
    if (delivery === "live") before.emit([url]);
    await settle();
    accepted(before);
    const after = await started([hub, url]);
    assert.equal(after.inbox.getSnapshot(), null);
    assert.deepEqual(
      [after.navigations, after.commands, after.errors],
      [[], [], []],
    );
    after.emit([url]);
    await settle();
    accepted(after);
  });
}

test("a different startup link, a fresh desktop session and blocked session storage still open run links", async (t) => {
  desktopSession();
  (await started([run])).cleanup();
  for (const [reset, nParallel] of [
    [false, 4],
    [true, 3],
    ["blocked", 3],
  ] as const) {
    if (reset) {
      const storage = desktopSession();
      if (reset === "blocked") {
        for (const method of ["getItem", "setItem"] as const) {
          t.mock.method(storage, method, () => {
            throw new Error("Storage blocked");
          });
        }
      }
    }
    const app = await started([run.replace("=3", `=${nParallel}`)]);
    assert.equal(app.inbox.getSnapshot()?.value.config.nParallel, nParallel);
    assert.deepEqual(app.commands, ["reveal_main_window"]);
    assert.deepEqual(app.errors, []);
    app.cleanup();
  }
});

for (const sharedLinks of [false, true]) {
  test(`Hub links keep routing and deduplication with sharing ${sharedLinks ? "enabled; a rejected run link retires the duplicate without resetting its sequence" : "absent"}`, async () => {
    const app = await started([hub], sharedLinks);
    for (const url of [hub, "https://example.invalid/unrelated", hub]) {
      app.emit([url]);
    }
    await settle();
    assert.deepEqual(
      app.navigations.map(({ to, search }) => [to, search.model]),
      [["/hub", "owner/model"]],
    );
    assert.equal(app.inbox.getSnapshot(), null);
    assert.deepEqual(app.commands, ["reveal_main_window"]);
    if (sharedLinks) {
      app.emit([invalid]);
      await settle();
      app.emit([hub]);
      await settle();
      assert.equal(app.navigations.length, 2);
      assert.equal(
        app.navigations[1].search.intent,
        app.navigations[0].search.intent + 1,
      );
      assert.equal(app.errors.length, 1);
    }
    app.cleanup();
    assert.deepEqual(app.counts, { subscriptions: 1, unsubscriptions: 1 });
  });
}

test("the newest recognized intent wins in mixed native batches, and web fragments never replace a pending native intent", async () => {
  const app = await started(null);
  app.emit([hub, run]);
  await settle();
  assert.equal(app.navigations.length, 0);
  const pending = app.inbox.getSnapshot();
  assert.equal(pending?.value.config.nParallel, 3);
  for (const url of [
    "https://example.invalid/chat#run?v=1&model=owner/other&nParallel=4",
    "http://localhost/chat#run?v=1&unknown=true",
  ]) {
    app.emit([url]);
    assert.equal(app.inbox.getSnapshot(), pending);
  }
  app.emit([run, hub]);
  assert.equal(app.navigations.length, 1);
  assert.equal(app.inbox.getSnapshot(), null);
  app.emit([run]);
  await settle();
  app.emit([hub, "https://example.invalid/chat#run?v=1&nParallel=4"]);
  assert.equal(app.inbox.getSnapshot(), null);
  assert.equal(app.navigations.length, 2);
  app.emit([hub, invalid]);
  await settle();
  assert.equal(app.navigations.length, 2);
  assert.equal(app.errors.length, 1);
  app.cleanup();
});

test("live shared links supersede delayed desktop startup URLs and disposal ignores later events", async () => {
  const app = harness();
  await settle();
  app.emit([run]);
  await settle();
  const pending = app.inbox.getSnapshot();
  app.releaseStartup([hub]);
  await settle();
  app.cleanup();
  app.emit([hub]);
  app.emit([invalid]);
  await settle();
  assert.equal(app.inbox.getSnapshot(), pending);
  assert.equal(app.navigations.length, 0);
  assert.deepEqual(app.errors, []);
  assert.deepEqual(app.counts, { subscriptions: 1, unsubscriptions: 1 });
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
