// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import type { GgufVariantDetail } from "../src/features/hub/inventory/api.ts";
import type * as CachedTarget from "../src/features/model-picker/sharing/cached-target.ts";
import type * as ImportConfig from "../src/features/model-picker/sharing/import-config.ts";
import type { RunConfigImport } from "../src/features/model-picker/sharing/inbox.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const identity = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);
const variantsRequest = await import(
  "../src/features/chat/api/gguf-variants-request.ts"
);
const abortSignals = await import("../src/features/hub/lib/abort-signals.ts");
const freshness = await import(
  "../src/features/hub/inventory/inventory-freshness.ts"
);
const hubIds = await import("../src/features/hub/lib/model-identity.ts");
const { hubTokenHeader } = await import(
  "../src/features/hub/lib/hub-token-header.ts"
);
const { buildLocalInventoryRows } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { modelConfigTarget } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);
const { wantsDownloadManagerStaging } = await import(
  "../src/features/chat/utils/model-download-staging.ts"
);
const {
  isRunConfigModelInput,
  isRunConfigVariantUnresolved,
  resolveRunConfigTarget,
} = await import("./helpers/sharing-target.ts");

type Target = NonNullable<ReturnType<typeof resolveRunConfigTarget>>;
type Resolve = typeof resolveRunConfigTarget;
const model = "owner/Model-GGUF";
const noGguf =
  "This model has no GGUF files. Shared run settings apply only to GGUF models.";
const empty = { models: [], loras: [], activeNativePathToken: null };
const selection = {
  ...{ ...empty, params: { checkpoint: "" } },
  ...{ activeGgufVariant: null, activeLoadId: null, loadedIsGguf: null },
};
const quant: GgufVariantDetail = {
  quant: "Q4_K_M",
  filename: "model-Q4_K_M.gguf",
  size_bytes: 1024,
  downloaded: true,
};
const q8 = { ...quant, quant: "Q8_0", filename: "model-Q8_0.gguf" };
const remote = { ...quant, downloaded: false };
const partial = { ...quant, partial: true };
const pick = (
  id?: string,
  value: Parameters<Resolve>[0] = { config: {} },
  from: Parameters<Resolve>[1] = selection,
) => {
  const resolved = resolveRunConfigTarget(value, from, id);
  assert.ok(resolved);
  return resolved;
};
const target = pick(undefined, { model, ggufVariant: "Q4_K_M", config: {} });
const withVariant = (ggufVariant?: string) => ({
  ...target,
  meta: { ...target.meta, ggufVariant },
});
const row = (load_id?: string, extra = {}) => ({
  repo_id: model,
  size_bytes: 1024,
  ...(load_id && { load_id }),
  ...extra,
});
const localPath = "/models/local.gguf";
const localRow = (source: string, extra = {}) => ({
  ...{ id: localPath, path: localPath, model_id: model, source },
  ...{ model_format: "gguf", display_name: "Local", ...extra },
});
const meta = (r: Target, ...keys: (keyof Target["meta"])[]) =>
  keys.map((key) => r.meta[key]);
const handoff = (r: Target) => {
  const next = modelConfigTarget(r.id, r.meta);
  return [next.id, next.ggufVariant, next.meta.ggufFilename];
};
const staged = (r: Target) =>
  wantsDownloadManagerStaging({ id: r.id, ...r.meta });
const unresolved = (r: Target) =>
  isRunConfigVariantUnresolved(modelConfigTarget(r.id, r.meta));
const noInventory = ["cachedGguf", "localModels"];
const defaults = {
  cachedGguf: [] as object[],
  localModels: [] as object[],
  variants: [quant] as GgufVariantDetail[],
  hubVariants: [remote] as GgufVariantDetail[],
  defaultVariant: undefined as string | null | undefined,
  hubMetadataAvailable: true,
  listings: undefined as Record<string, GgufVariantDetail[]> | undefined,
  sourceErrors: [] as string[],
  listingErrors: [] as string[],
  ...{ delayMs: 0, inventoryDelayMs: 0, status: 200 },
  ...{ checkLocalPath: false, resolvedLocally: false, hubError: false },
  localResponse: undefined as unknown,
};

function harness(options: Partial<typeof defaults> = {}) {
  const o = { ...defaults, ...options };
  const scans: string[] = [];
  const requests: string[][] = [];
  const hubRequests: unknown[][] = [];
  const delay = (ms: number) =>
    ms && new Promise((resolve) => setTimeout(resolve, ms));
  const module = loadWithStubs<typeof CachedTarget>(
    new URL(
      "../src/features/model-picker/sharing/cached-target.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async (url: string) => {
          await delay(o.delayMs);
          const query = new URL(url, "http://localhost").searchParams;
          const [repo, ...rest] = ["repo_id", "local_path", "offline"].map(
            (key) => `${query.get(key)}`,
          );
          requests.push([repo, ...rest]);
          if (o.listingErrors.includes(repo)) throw new Error("unavailable");
          const listed = o.listings ? (o.listings[repo] ?? []) : o.variants;
          const body =
            "localResponse" in options
              ? o.localResponse
              : {
                  variants: listed,
                  default_variant: listed[0]?.quant,
                  resolved_locally: o.resolvedLocally,
                };
          return new Response(JSON.stringify(body), { status: o.status });
        },
      },
      "@/features/hub": {
        ...{ ...freshness, ...abortSignals, ...hubIds, hubTokenHeader },
        buildLocalInventoryRows,
        listGgufVariants: async (
          repoId: string,
          hfToken?: string,
          init?: { signal?: AbortSignal },
        ) => {
          hubRequests.push([repoId, hfToken, init?.signal]);
          await delay(o.delayMs);
          if (o.hubError) throw new Error("Hub listing unavailable");
          return {
            variants: o.hubVariants,
            default_variant:
              "defaultVariant" in options
                ? o.defaultVariant
                : o.hubVariants[0]?.quant,
            dependencies_resolved: o.hubMetadataAvailable,
          };
        },
        useDeviceInventoryStore: {
          getState: () => {
            const idle = { ready: false, loading: false, refreshedAt: null };
            return { cachedGguf: idle, localModels: idle };
          },
        },
        fetchInventorySource: async (source: "cachedGguf" | "localModels") => {
          scans.push(source);
          await delay(o.inventoryDelayMs);
          if (o.sourceErrors.includes(source)) throw new Error("unavailable");
          return o[source];
        },
      },
      "@/features/chat": variantsRequest,
      "../model-config/model-identity": identity,
    },
  );
  return {
    Resolution: module.RunConfigResolutionError,
    ...{ scans, requests, hubRequests },
    resolve: (input: Target = target, signal = new AbortController().signal) =>
      module.resolveCachedRunConfigTarget(input, {
        inventoryVersion: 0,
        signal,
        checkLocalPath: o.checkLocalPath,
      }),
  };
}

test("local paths resolve offline, before any inventory scan or Hub lookup", async () => {
  for (const [id, checkLocalPath, variants] of [
    ["models/my-native-model", true, []],
    ["models/my-quantized-model", true, [quant]],
    ["models/model.gguf", true, [quant]],
    ["C:\\Models\\quantized-qwen", false, [quant]],
    ["checkpoint-500", false, []],
  ] as [string, boolean, GgufVariantDetail[]][]) {
    assert.equal(isRunConfigModelInput(id), true);
    const app = harness({
      ...{ sourceErrors: noInventory, hubError: true, variants },
      ...{ checkLocalPath, resolvedLocally: checkLocalPath },
    });
    const input = pick(id);
    if (variants.length === 0) {
      await assert.rejects(app.resolve(input), {
        constructor: app.Resolution,
        message: noGguf,
      });
    } else {
      const resolved = await app.resolve(input);
      const local = checkLocalPath ? `./${id}` : id;
      const file = id.endsWith(".gguf");
      assert.deepEqual(
        [resolved.id, ...meta(resolved, "source", "isGguf", "isDownloaded")],
        [local, "local", true, file ? undefined : true],
      );
      assert.deepEqual(
        handoff(resolved),
        file ? [local, null, undefined] : [local, quant.quant, quant.filename],
      );
      assert.equal(staged(resolved), false);
    }
    assert.deepEqual(
      [app.scans, app.hubRequests, app.requests],
      [[], [], [[id, id, "true"]]],
    );
  }
});

test("local files and Ollama references need no lookup; failed local checks never guess", async () => {
  const id = "/Users/test/Models/model.gguf";
  const file = harness({ sourceErrors: noInventory });
  const value = { config: {}, ggufVariant: "Q4_K_M" };
  const resolved = await file.resolve(pick(id, value));
  assert.deepEqual(
    [resolved.id, ...meta(resolved, "isGguf", "ggufVariant", "source")],
    [id, true, undefined, "local"],
  );
  assert.equal(staged(resolved), false);
  assert.deepEqual([file.scans, file.requests, file.hubRequests], [[], [], []]);
  for (const [id, checkLocalPath] of [
    ["models/qwen", true],
    ["/models/qwen", false],
  ] as const) {
    const app = harness({ checkLocalPath, status: 503 });
    await assert.rejects(app.resolve(pick(id)), /Could not check cached GGUF/);
    assert.deepEqual([app.scans, app.hubRequests], [[], []]);
  }
});

test("recipient-selected Hub IDs keep cached, uncached and non-GGUF outcomes after the local check", async () => {
  for (const cached of [false, true]) {
    const app = harness({
      checkLocalPath: true,
      cachedGguf: cached ? [row("/cache/model")] : [],
    });
    const value = { config: {}, ggufVariant: quant.quant };
    const resolved = await app.resolve(pick(model, value));
    assert.deepEqual(
      [resolved.id, ...meta(resolved, "source"), staged(resolved)],
      [model, "hub", !cached],
    );
    assert.deepEqual([app.scans, app.hubRequests], [noInventory, []]);
  }
  const input = pick("owner/native");
  const none = { checkLocalPath: true, variants: [], hubVariants: [] };
  const app = harness(none);
  await assert.rejects(app.resolve(input), {
    constructor: app.Resolution,
    message: noGguf,
  });
  assert.equal(app.hubRequests.length, 1);
  const offline = harness({ ...none, hubMetadataAvailable: false });
  assert.equal(await offline.resolve(input), input);
});

test("cached and active GGUF copies bypass staging and keep their exact load path", async () => {
  const loadId = "/cache/models--owner--Model-GGUF/snapshots/pinned";
  const app = harness({
    cachedGguf: [row(loadId, { repo_id: model.toLowerCase() })],
  });
  const resolved = await app.resolve();
  assert.equal(staged(resolved), false);
  assert.deepEqual(handoff(resolved), [loadId, quant.quant, quant.filename]);
  assert.deepEqual(
    [app.requests, app.hubRequests],
    [[[loadId, loadId, "true"]], []],
  );
  for (const [checkpoint, activeLoadId] of [
    [loadId, null],
    [model, "/secondary/pinned"],
  ] as const) {
    for (const explicit of [undefined, model]) {
      const loaded = {
        ...{ ...empty, params: { checkpoint }, activeLoadId },
        ...{ loadedIsGguf: true, activeGgufVariant: "Q4_K_M" },
      };
      const idle = harness({ sourceErrors: noInventory });
      const value = { model: explicit, config: {} };
      const kept = await idle.resolve(pick(undefined, value, loaded));
      assert.equal(handoff(kept)[0], activeLoadId ?? checkpoint);
      assert.equal(staged(kept), false);
      assert.deepEqual([idle.scans, idle.requests], [[], []]);
      const moved = { ...loaded, activeNativePathToken: "local-only" };
      const other = pick(undefined, { ...value, ggufVariant: "Q8_0" }, moved);
      const inherited = meta(
        other,
        "loadId",
        "nativePathToken",
        "isDownloaded",
      );
      assert.deepEqual(inherited, [undefined, undefined, undefined]);
    }
  }
});

test("inventory candidates are checked one by one for a complete copy", async () => {
  const cachedGguf = [row("/cache/missing"), row("/cache/available")];
  const listings = { "/cache/missing": [], "/cache/available": [quant] };
  const skipped = await harness({ cachedGguf, listings }).resolve();
  assert.deepEqual(meta(skipped, "loadId", "isDownloaded"), [
    "/cache/available",
    true,
  ]);
  const listingErrors = ["/cache/missing"];
  const failed = await harness({ cachedGguf, listingErrors }).resolve();
  assert.equal(failed.meta.loadId, "/cache/available");
  const legacy = localRow("hf_cache", { model_format: undefined });
  const local = await harness({ localModels: [legacy] }).resolve();
  assert.deepEqual(meta(local, "loadId", "isDownloaded"), [localPath, true]);
  for (const source of noInventory) {
    const app = harness({
      sourceErrors: [source],
      cachedGguf: [row("/cache/pinned")],
      localModels: [
        localRow("hf_cache", { id: model, load_id: "/local", path: "/local" }),
      ],
    });
    assert.equal(
      (await app.resolve()).meta.loadId,
      source === "localModels" ? "/cache/pinned" : "/local",
    );
  }
});

test("unrelated, incomplete, non-cache or unmatched inventory keeps the Hub target for staging", async () => {
  for (const variants of [[q8], [remote], [partial], []]) {
    const app = harness({ cachedGguf: [row("/cache/pinned")], variants });
    const resolved = await app.resolve();
    assert.equal(resolved, target);
    assert.equal(staged(resolved), true);
  }
  for (const options of [
    { cachedGguf: [row(undefined, { repo_id: "other/model" })] },
    { cachedGguf: [row(undefined, { partial: true })] },
    { localModels: [localRow("custom", { id: model, load_id: "/local" })] },
    { sourceErrors: noInventory },
  ]) {
    const app = harness(options);
    assert.equal(await app.resolve(), target);
    assert.deepEqual(
      [app.scans, app.requests, app.hubRequests],
      [noInventory, [], []],
    );
  }
  assert.equal(target.meta.isDownloaded, undefined);
});

test("Hub lookups resolve the exact file or the default, or keep an unloadable review", async () => {
  const chosen = {
    ...remote,
    quant: "chosen/Q4_K_M",
    filename: "chosen/m.gguf",
  };
  const input = withVariant(chosen.filename);
  const offline = await harness({ hubError: true }).resolve(input);
  assert.equal(offline, input);
  assert.equal(unresolved(offline), true);
  const hub = await harness({ hubVariants: [chosen] }).resolve(input);
  assert.deepEqual(handoff(hub), [model, chosen.quant, chosen.filename]);
  assert.equal(unresolved(hub), false);
  const hubVariants = [remote, { ...q8, downloaded: false }];
  const fallback = harness({ hubVariants, defaultVariant: "Q8_0" });
  const picked = await fallback.resolve(withVariant(undefined));
  assert.deepEqual(
    meta(picked, "ggufVariant", "ggufFilename", "isDownloaded"),
    ["Q8_0", q8.filename, false],
  );
  for (const variant of [chosen.filename, undefined]) {
    const missing = harness({ hubVariants: [q8], defaultVariant: "Q4_K_M" });
    await assert.rejects(missing.resolve(withVariant(variant)), {
      constructor: missing.Resolution,
      message:
        "The shared GGUF variant is unavailable for this model. Ask the sender for an updated link.",
    });
  }
});

test("cancellation cannot produce a handoff, even mid lookup", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  for (const hubError of [false, true]) {
    const slow = harness({ delayMs: 1_000, hubError });
    const abort = new AbortController();
    const pending = slow.resolve(withVariant(quant.filename), abort.signal);
    await new Promise<void>((resolve) => setImmediate(resolve));
    assert.equal(slow.hubRequests[0][2], abort.signal);
    abort.abort();
    t.mock.timers.tick(1_000);
    await assert.rejects(pending, { name: "AbortError" });
  }
  const budgets = harness({
    ...{ delayMs: 20_000, inventoryDelayMs: 20_000 },
    cachedGguf: [row("/cache/missing"), row("/cache/available")],
    listings: { "/cache/missing": [], "/cache/available": [quant] },
  });
  const pending = budgets.resolve();
  for (let step = 0; step < 3; step += 1) {
    t.mock.timers.tick(20_000);
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  assert.equal((await pending).meta.loadId, "/cache/available");
});

test("malformed variant responses cannot mark a model downloaded or redirect a local path to the Hub", async () => {
  const bad = (field: object) => ({ variants: [{ ...quant, ...field }] });
  for (const response of [
    null,
    {},
    { variants: {} },
    { variants: [null] },
    ...[{ filename: 1 }, { filename: "" }, { quant: null }].map(bad),
    ...[{ quant: "" }, { size_bytes: -1 }, { size_bytes: "1024" }].map(bad),
    ...[{ downloaded: "true" }, { partial: "false" }].map(bad),
    { variants: [quant], default_variant: {} },
    { variants: [quant], resolved_locally: "true" },
    { variants: [quant], dependencies_resolved: "false" },
  ]) {
    const local = harness({ localResponse: response, checkLocalPath: true });
    await assert.rejects(
      local.resolve(pick("./models/native")),
      /Invalid GGUF variants response/,
    );
    assert.deepEqual([local.hubRequests, local.scans], [[], []]);
  }
});

const drafts = await import(
  "../src/features/model-picker/model-config/model-config-draft.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const fields = await import("../src/features/model-picker/sharing/fields.ts");
const inboxModule = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
const { parseRunConfigLink } = await import("./helpers/sharing-links.ts");
type Config = typeof DEFAULT_PER_MODEL_CONFIG;
let sequence = 0;
const settle = () => Promise.resolve();
const n3 = { nParallel: 3 };

function importHarness(
  t: TestContext,
  patch: Partial<Config> = n3,
  ggufVariant: string | null = "Q4_K_M",
  value: { model?: string; ggufVariant?: string } = {},
) {
  const id = `owner/model-${++sequence}`;
  const key = drafts.modelConfigDraftKey(id, ggufVariant);
  const inbox = inboxModule.createRunConfigInbox();
  const imports: unknown[][] = [];
  const toasts: string[] = [];
  const releaseDraft = drafts.retainModelConfigDraft(key);
  t.after(releaseDraft);
  const args = { maxSeqLength: 4096, customContextLength: 4096 };
  const warm = ["--no-warmup"];
  const config = { ...DEFAULT_PER_MODEL_CONFIG, ...args, llamaExtraArgs: warm };
  drafts.primeModelConfigDraft(key, { config, remembered: true }, "none");
  const meta = { source: "hub" as const, isLora: false, isGguf: true };
  inbox.submit({
    id: "import",
    draftKey: key,
    value: { ...value, config: patch },
    target: { id, meta: { ...meta, ggufVariant: ggufVariant ?? undefined } },
  });
  const { scheduleRunConfigImport: schedule } = loadWithStubs<
    typeof ImportConfig
  >(
    new URL(
      "../src/features/model-picker/sharing/import-config.ts",
      import.meta.url,
    ),
    {
      "@/lib/toast": {
        toast: {
          error: () => toasts.push("error"),
          success: () => toasts.push("success"),
        },
      },
      "../model-config/model-config-draft": drafts,
      "./fields": fields,
      "./inbox": { ...inboxModule, runConfigInbox: inbox },
    },
  );
  const options = {
    ...{ key, canImport: true, ready: true, hydrated: true },
    pending: inbox.getSnapshot(),
    onImport: ({ changes, model, ggufVariant }: RunConfigImport) =>
      imports.push([changes, model, ggufVariant]),
  };
  const draft = () => drafts.readModelConfigDraft(key);
  const edited = () => drafts.isModelConfigDraftEdited(key);
  const count = (kind: string) => toasts.filter((k) => k === kind).length;
  return {
    ...{ inbox, key, imports, options, draft, edited, count },
    ...{ releaseDraft, schedule },
    async run(overrides: Partial<typeof options> = {}) {
      const cancel = schedule({ ...options, ...overrides });
      await settle();
      return cancel;
    },
  };
}

test("imports report changes, the linked model and only variants retained by target resolution", async (t) => {
  const m = "owner/model";
  // [settings, link variant, resolved variant, link model, reported variant; "none" = no report]
  for (const [patch, requested, resolved, model, reported] of [
    [{}, "Q8_0", "Q8_0", undefined, "Q8_0"],
    [{}, "model-Q8_0.gguf", "Q8_0", undefined, "Q8_0"],
    [{}, undefined, "Q4_K_M", undefined, "none"],
    [{}, undefined, "Q4_K_M", m, undefined],
    [n3, undefined, "Q8_0", undefined, undefined],
    [n3, "Q8_0", null, undefined, undefined],
    [n3, "Q8_0", "Q8_0", m, "Q8_0"],
  ] as const) {
    const value = { model, ggufVariant: requested };
    const app = importHarness(t, patch, resolved, value);
    const before = app.draft();
    await app.run();
    assert.equal(app.inbox.getSnapshot(), null);
    const expected = reported === "none" ? [] : [[patch, model, reported]];
    assert.deepEqual(app.imports, expected);
    const settingsOnly = Object.keys(patch).length === 0;
    assert.equal(app.draft() === before, settingsOnly);
    assert.equal(app.edited(), !settingsOnly);
    assert.equal(app.count("success"), settingsOnly ? 0 : 1);
  }
});

test("a shared context imports without touching the pinned sequence length", async (t) => {
  for (const customContextLength of [8192, null]) {
    const link = `unsloth://run?v=1&customContextLength=${customContextLength}`;
    const parsed = parseRunConfigLink(link);
    assert.ok(parsed.kind === "valid");
    const app = importHarness(t, parsed.value.config);
    await app.run();
    const config = app.draft()?.config;
    assert.equal(config?.customContextLength, customContextLength);
    assert.equal(config?.maxSeqLength, 4096);
    assert.deepEqual(app.imports[0][0], { customContextLength });
  }
});

test("Strict Mode cleanup leaves the request for the surviving editor and applies once", async (t) => {
  const app = importHarness(t);
  const firstHost = app.inbox.retainEditor(app.key);
  app.schedule(app.options)?.();
  firstHost();
  t.after(app.inbox.retainEditor(app.key));
  app.schedule(app.options);
  await app.run();
  assert.equal(app.draft()?.config.nParallel, 3);
  assert.deepEqual(app.imports, [[n3, undefined, undefined]]);
  assert.deepEqual([app.count("success"), app.edited()], [1, true]);
});

test("imports wait without consuming until the editor, draft and request line up", async (t) => {
  for (const [reason, change] of [
    ["sidebar", { canImport: false }],
    ["hydrating", { ready: false }],
    ["wrong draft", { key: "other" }],
    ["no request", { pending: null }],
    ["no draft", {}],
  ] as const) {
    const app = importHarness(t);
    if (reason === "no draft") app.releaseDraft();
    assert.equal(app.schedule({ ...app.options, ...change }), undefined);
    await settle();
    assert.equal(app.inbox.getSnapshot(), app.options.pending, reason);
    assert.deepEqual(app.imports, [], reason);
  }
});

test("a queued import cannot overwrite an unmount, edit, newer link or released draft", async (t) => {
  for (const reason of ["unmount", "edit", "new link", "draft released"]) {
    const app = importHarness(t);
    const cancel = app.schedule(app.options);
    if (reason === "unmount") cancel?.();
    if (reason === "edit") {
      app.inbox.clear("import");
      drafts.patchModelConfigDraft(app.key, (c) => ({ ...c, nParallel: 7 }));
    }
    if (reason === "new link") {
      const value = { config: { nParallel: 8 } };
      app.inbox.submit({ id: "new", draftKey: app.key, value });
    }
    if (reason === "draft released") app.releaseDraft();
    await settle();
    assert.deepEqual([app.imports, app.count("success")], [[], 0], reason);
    if (reason === "edit") assert.equal(app.draft()?.config.nParallel, 7);
    if (reason === "new link") assert.equal(app.inbox.getSnapshot()?.id, "new");
    if (reason === "draft released") assert.equal(app.draft(), undefined);
  }
});

test("failed hydration reports once, never cancels on close, and a reopened link imports once", async (t) => {
  const app = importHarness(t);
  const cancellations: string[] = [];
  const onCancel = (request: { id: string }) => cancellations.push(request.id);
  const release = app.inbox.retainEditor(app.key, onCancel);
  t.after(release);
  const before = app.draft();
  app.schedule({ ...app.options, hydrated: false });
  await app.run({ hydrated: false });
  assert.deepEqual([app.draft(), app.edited()], [before, false]);
  assert.deepEqual([app.inbox.getSnapshot(), app.count("error")], [null, 1]);
  release();
  await app.run();
  assert.deepEqual(
    [cancellations, app.imports, app.count("success")],
    [[], [], 0],
  );
});

test("imported extra arguments replace a raw edit", async (t) => {
  const edit = { text: "unfinished '", source: "--no-warmup" };
  const llamaExtraArgs = ["--rope-scaling", "yarn"];
  const unchanged = { tensorParallel: DEFAULT_PER_MODEL_CONFIG.tensorParallel };
  const patch = { customContextLength: 8192, llamaExtraArgs, ...unchanged };
  const app = importHarness(t, patch);
  drafts.setExtraArgsEditForDraft(app.key, edit);
  await app.run();
  const changes = { customContextLength: 8192, llamaExtraArgs };
  assert.deepEqual(app.imports[0][0], changes);
  assert.equal(drafts.readExtraArgsEditForDraft(app.key), undefined);
  assert.equal(app.draft()?.remember, true);
});
