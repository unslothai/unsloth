// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  GgufVariantDetail,
  LocalModelInfo,
} from "../src/features/hub/inventory/api.ts";
import type * as CachedTarget from "../src/features/model-picker/sharing/cached-target.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const modelIdentity = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);
const variantsRequest = await import(
  "../src/features/chat/api/gguf-variants-request.ts"
);
const abortSignals = await import("../src/features/hub/lib/abort-signals.ts");
const inventoryFreshness = await import(
  "../src/features/hub/inventory/inventory-freshness.ts"
);
const { hubTokenHeader } = await import(
  "../src/features/hub/lib/hub-token-header.ts"
);
const { buildLocalInventoryRows } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { ggufVariantsMatch, residentModelIdMatches } = await import(
  "../src/features/hub/lib/model-identity.ts"
);
const { isRunConfigVariantUnresolved, resolveRunConfigTarget } = await import(
  "./helpers/sharing-target.ts"
);
const { modelConfigTarget } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);
const { wantsDownloadManagerStaging } = await import(
  "../src/features/chat/utils/model-download-staging.ts"
);

const model = "owner/Model-GGUF";
const selection = {
  params: { checkpoint: "" },
  activeGgufVariant: null,
  activeNativePathToken: null,
  activeLoadId: null,
  loadedIsGguf: null,
  models: [],
  loras: [],
};
const initialTarget = resolveRunConfigTarget(
  { model, ggufVariant: "Q4_K_M", config: {} },
  selection,
);
assert.ok(initialTarget);
const target = initialTarget;
const quant: GgufVariantDetail = {
  quant: "Q4_K_M",
  filename: "model-Q4_K_M.gguf",
  size_bytes: 1024,
  downloaded: true,
};

function harness({
  cachedGguf = [],
  cachedModels = [],
  localModels = [],
  variants = [quant],
  hubVariants = [{ ...quant, downloaded: false }],
  defaultVariant = hubVariants[0]?.quant ?? null,
  hubMetadataAvailable = true,
  hubError = false,
  listingsByRepo,
  status = 200,
  inventoryError = false,
  sourceErrors = [],
  listingErrors = [],
  inventoryDelayMs = 0,
  variantDelayMs = 0,
}: {
  cachedGguf?: CachedGgufRepo[];
  cachedModels?: CachedModelRepo[];
  localModels?: LocalModelInfo[];
  variants?: GgufVariantDetail[];
  hubVariants?: GgufVariantDetail[];
  defaultVariant?: string | null;
  hubMetadataAvailable?: boolean;
  hubError?: boolean;
  listingsByRepo?: Record<string, GgufVariantDetail[]>;
  status?: number;
  inventoryError?: boolean;
  sourceErrors?: string[];
  listingErrors?: string[];
  inventoryDelayMs?: number;
  variantDelayMs?: number;
} = {}) {
  const scans: string[] = [];
  const requests: URL[] = [];
  const hubRequests: {
    repoId: string;
    hfToken?: string;
    signal?: AbortSignal;
  }[] = [];
  const cachedTarget = loadWithStubs<typeof CachedTarget>(
    new URL(
      "../src/features/model-picker/sharing/cached-target.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async (url: string) => {
          if (variantDelayMs)
            await new Promise((resolve) => setTimeout(resolve, variantDelayMs));
          const request = new URL(url, "http://localhost");
          requests.push(request);
          if (listingErrors.includes(request.searchParams.get("repo_id") ?? ""))
            throw new Error("Listing unavailable");
          const listed = listingsByRepo
            ? (listingsByRepo[request.searchParams.get("repo_id") ?? ""] ?? [])
            : variants;
          return new Response(
            JSON.stringify({
              variants: listed,
              default_variant: listed[0]?.quant,
            }),
            { status },
          );
        },
      },
      "@/features/hub": {
        ...inventoryFreshness,
        ...abortSignals,
        buildLocalInventoryRows,
        ggufVariantsMatch,
        hubTokenHeader,
        listGgufVariants: async (
          repoId: string,
          hfToken?: string,
          options?: { signal?: AbortSignal },
        ) => {
          hubRequests.push({ repoId, hfToken, signal: options?.signal });
          if (variantDelayMs)
            await new Promise((resolve) => setTimeout(resolve, variantDelayMs));
          if (hubError) throw new Error("Hub listing unavailable");
          return {
            variants: hubVariants,
            default_variant: defaultVariant,
            dependencies_resolved: hubMetadataAvailable,
          };
        },
        residentModelIdMatches,
        useDeviceInventoryStore: {
          getState: () =>
            Object.fromEntries(
              ["cachedGguf", "cachedModels", "localModels"].map((source) => [
                source,
                { ready: false, loading: false, refreshedAt: null },
              ]),
            ),
        },
        fetchInventorySource: async (
          source: "cachedGguf" | "cachedModels" | "localModels",
        ) => {
          scans.push(source);
          if (inventoryDelayMs)
            await new Promise((resolve) =>
              setTimeout(resolve, inventoryDelayMs),
            );
          if (inventoryError || sourceErrors.includes(source))
            throw new Error("Inventory unavailable");
          return { cachedGguf, cachedModels, localModels }[source];
        },
      },
      "@/features/chat": variantsRequest,
      "../model-config/model-identity": modelIdentity,
    },
  );
  return {
    RunConfigResolutionError: cachedTarget.RunConfigResolutionError,
    scans,
    requests,
    hubRequests,
    resolve: (
      input = target,
      signal = new AbortController().signal,
      hfToken?: string,
    ) =>
      cachedTarget.resolveCachedRunConfigTarget(input, {
        inventoryVersion: 0,
        signal,
        hfToken,
      }),
  };
}

test("recipient-selected local files and Ollama references need no inventory or network lookup", async () => {
  for (const [id, isGguf] of [
    ["/Users/test/Models/model.gguf", true],
    ["C:\\Models\\model.gguf", true],
    ["\\\\server\\models\\model.gguf", true],
    ["/mnt/c/Models/model.gguf", true],
    ["ollama-manifest:registry.ollama.ai/library/llama3/latest", true],
  ] as const) {
    const app = harness({ inventoryError: true });
    const target = resolveRunConfigTarget(
      { config: { nParallel: 3 }, isGguf: false, ggufVariant: "Q4_K_M" },
      selection,
      id,
    );
    assert.ok(target);
    const resolved = await app.resolve(target);
    assert.equal(resolved.id, id);
    assert.equal(resolved.meta.isGguf, isGguf);
    assert.equal(resolved.meta.ggufVariant, undefined);
    assert.equal(resolved.meta.source, "local");
    assert.equal(wantsDownloadManagerStaging({ id, ...resolved.meta }), false);
    assert.deepEqual(app.scans, []);
    assert.deepEqual(app.requests, []);
    assert.deepEqual(app.hubRequests, []);
  }
});

for (const id of [
  "/models/quantized-qwen",
  "/Users/test/Models/quantized-qwen",
  "C:\\Models\\quantized-qwen",
  "\\\\server\\models\\quantized-qwen",
  "/mnt/c/Models/quantized-qwen",
]) {
  test(`settings-only imports discover local GGUF folders without Hub access: ${id}`, async () => {
    const app = harness({ inventoryError: true, hubError: true });
    const input = resolveRunConfigTarget(
      {
        config: {
          nParallel: 3,
          reasoningBudget: 512,
          llamaExtraArgs: ["--no-warmup"],
        },
      },
      selection,
      id,
    );
    assert.ok(input);
    const resolved = await app.resolve(input);
    assert.equal(resolved.id, id);
    assert.equal(resolved.meta.isGguf, true);
    assert.equal(resolved.meta.ggufVariant, quant.quant);
    assert.equal(resolved.meta.ggufFilename, quant.filename);
    assert.equal(resolved.meta.isDownloaded, true);
    assert.equal(modelConfigTarget(resolved.id, resolved.meta).isGguf, true);
    assert.equal(wantsDownloadManagerStaging({ id, ...resolved.meta }), false);
    assert.equal(app.requests.length, 1);
    assert.equal(app.requests[0].searchParams.get("repo_id"), id);
    assert.equal(app.requests[0].searchParams.get("local_path"), id);
    assert.equal(app.requests[0].searchParams.get("offline"), "true");
    assert.deepEqual(app.scans, []);
    assert.deepEqual(app.hubRequests, []);
  });
}

test("local native folders keep native loading even with a GGUF name or shared format hint", async () => {
  for (const isGguf of [undefined, true, false]) {
    const app = harness({ variants: [] });
    const input = resolveRunConfigTarget(
      { config: {}, isGguf },
      selection,
      "/models/native-GGUF",
    );
    assert.ok(input);
    const resolved = await app.resolve(input);
    assert.equal(resolved.meta.isGguf, false);
    assert.equal(resolved.meta.ggufVariant, undefined);
    assert.deepEqual(app.hubRequests, []);
  }
});

test("local discovery rejects failed requests instead of guessing the model format", async () => {
  const app = harness({ status: 503 });
  const input = resolveRunConfigTarget(
    { config: {} },
    selection,
    "/models/qwen",
  );
  assert.ok(input);
  await assert.rejects(
    app.resolve(input),
    /Could not check cached GGUF variants/,
  );
  assert.deepEqual(app.hubRequests, []);
});

test("local GGUF discovery preserves explicit filenames and refuses missing or partial variants", async () => {
  for (const ggufVariant of [quant.filename, "Q8_0"]) {
    for (const partial of [false, true]) {
      const app = harness({ variants: [{ ...quant, partial }] });
      const input = resolveRunConfigTarget(
        { config: {}, ggufVariant },
        selection,
        "/models/qwen",
      );
      assert.ok(input);
      if (ggufVariant === quant.filename && !partial) {
        assert.equal((await app.resolve(input)).meta.ggufVariant, quant.quant);
      } else {
        await assert.rejects(app.resolve(input), app.RunConfigResolutionError);
      }
      assert.deepEqual(app.hubRequests, []);
    }
  }
});

for (const loadId of [
  "/secondary/cache/models--owner--Model-GGUF/snapshots/pinned",
  "/Users/test/Library/Caches/models--owner--Model-GGUF/snapshots/pinned",
  "C:\\Models\\models--owner--Model-GGUF\\snapshots\\pinned",
  "\\\\server\\Models\\models--owner--Model-GGUF\\snapshots\\pinned",
  "/mnt/d/Models/models--owner--Model-GGUF/snapshots/pinned",
]) {
  test(`cached GGUF handoffs bypass staging and retain the exact load path: ${loadId}`, async () => {
    const app = harness({
      cachedGguf: [
        { repo_id: model.toLowerCase(), load_id: loadId, size_bytes: 1024 },
      ],
    });
    const resolved = await app.resolve();
    assert.equal(
      wantsDownloadManagerStaging({ id: resolved.id, ...resolved.meta }),
      false,
    );
    assert.equal(modelConfigTarget(resolved.id, resolved.meta).id, loadId);
    assert.equal(resolved.meta.ggufVariant, "Q4_K_M");
    assert.equal(resolved.meta.ggufFilename, quant.filename);
    assert.equal(app.requests.length, 1);
    assert.equal(app.requests[0].searchParams.get("offline"), "true");
    assert.equal(app.requests[0].searchParams.get("repo_id"), loadId);
    assert.equal(app.requests[0].searchParams.get("local_path"), loadId);
    assert.equal(target.meta.isDownloaded, undefined);
    assert.deepEqual(app.hubRequests, []);
  });

  test(`status-discovered models retain their checkpoint path without a separate load ID: ${loadId}`, async () => {
    for (const isGguf of [false, true]) {
      for (const explicitModel of [undefined, model]) {
        const active = resolveRunConfigTarget(
          { model: explicitModel, config: { nParallel: 3 } },
          {
            ...selection,
            params: { checkpoint: loadId },
            loadedIsGguf: isGguf,
            activeGgufVariant: isGguf ? "Q4_K_M" : null,
          },
        );
        assert.ok(active);
        const app = harness({ inventoryError: true });
        const resolved = await app.resolve(active);
        assert.equal(modelConfigTarget(resolved.id, resolved.meta).id, loadId);
        assert.equal(
          wantsDownloadManagerStaging({ id: resolved.id, ...resolved.meta }),
          false,
        );
        assert.deepEqual(app.scans, []);
        assert.deepEqual(app.requests, []);
      }
    }
  });
}

test("a missing snapshot cannot borrow a downloaded variant from another copy", async () => {
  const app = harness({
    cachedGguf: [
      { repo_id: model, load_id: "/cache/missing", size_bytes: 1024 },
      { repo_id: model, load_id: "/cache/available", size_bytes: 1024 },
    ],
    listingsByRepo: {
      [model]: [quant],
      "/cache/missing": [],
      "/cache/available": [quant],
    },
  });
  const resolved = await app.resolve();
  assert.equal(resolved.meta.loadId, "/cache/available");
  assert.equal(resolved.meta.isDownloaded, true);
});

test("a complete native snapshot bypasses staging without listing GGUF variants", async () => {
  const app = harness({
    cachedModels: [
      { repo_id: "owner/native", load_id: "/native/pinned", size_bytes: 1024 },
    ],
  });
  const native = resolveRunConfigTarget(
    { model: "owner/native", isGguf: false, config: {} },
    selection,
  );
  assert.ok(native);
  const resolved = await app.resolve(native);
  assert.equal(
    wantsDownloadManagerStaging({ id: resolved.id, ...resolved.meta }),
    false,
  );
  assert.equal(
    modelConfigTarget(resolved.id, resolved.meta).id,
    "/native/pinned",
  );
  assert.deepEqual(app.requests, []);
});

test("a matching HF cache model in local inventory uses the locally supplied load path", async () => {
  const app = harness({
    localModels: [
      {
        id: "/models/local.gguf",
        path: "/models/local.gguf",
        model_id: model,
        model_format: "gguf",
        source: "hf_cache",
        display_name: "Local",
      },
    ],
  });
  const resolved = await app.resolve();
  assert.equal(resolved.meta.loadId, "/models/local.gguf");
  assert.equal(resolved.meta.isDownloaded, true);
});

test("legacy local GGUF metadata cannot satisfy a native-format import", async () => {
  const app = harness({
    localModels: [
      {
        id: "/models/local.gguf",
        path: "/models/local.gguf",
        model_id: model,
        source: "hf_cache",
        display_name: "Local",
      },
    ],
  });
  const native = {
    ...target,
    meta: { ...target.meta, isGguf: false, ggufVariant: undefined },
  };
  assert.equal(await app.resolve(native), native);
  assert.equal((await app.resolve()).meta.isDownloaded, true);
});

for (const variants of [
  [{ ...quant, quant: "Q8_0", filename: "model-Q8_0.gguf" }],
  [
    {
      ...quant,
      quant: "sibling/Q4_K_M",
      filename: "sibling/model-Q4_K_M.gguf",
    },
  ],
  [{ ...quant, downloaded: false }],
  [{ ...quant, partial: true }],
  [],
]) {
  test(`an unavailable exact GGUF variant cannot bypass staging: ${JSON.stringify(variants)}`, async () => {
    const app = harness({
      cachedGguf: [
        { repo_id: model, load_id: "/cache/pinned", size_bytes: 1024 },
      ],
      variants,
    });
    const resolved = await app.resolve();
    assert.equal(resolved, target);
    assert.equal(
      wantsDownloadManagerStaging({ id: resolved.id, ...resolved.meta }),
      true,
    );
  });
}

test("an exact GGUF filename is checked without substituting a same-quant sibling", async () => {
  const filenameTarget = {
    ...target,
    meta: { ...target.meta, ggufVariant: "chosen/model-Q4_K_M.gguf" },
  };
  const cachedGguf = [
    { repo_id: model, load_id: "/cache/pinned", size_bytes: 1024 },
  ];
  await assert.rejects(
    harness({ cachedGguf }).resolve(filenameTarget),
    /The shared GGUF variant is unavailable/,
  );
  const app = harness({
    cachedGguf,
    variants: [{ ...quant, filename: filenameTarget.meta.ggufVariant }],
  });
  const resolved = await app.resolve(filenameTarget);
  assert.equal(resolved.meta.isDownloaded, true);
  assert.equal(resolved.meta.ggufVariant, quant.quant);
  assert.equal(resolved.meta.ggufFilename, filenameTarget.meta.ggufVariant);
  const handoff = modelConfigTarget(resolved.id, resolved.meta);
  assert.equal(handoff.id, "/cache/pinned");
  assert.equal(handoff.ggufVariant, quant.quant);
});

for (const identity of [
  { quant: "Q4_K_M", filename: "model-Q4_K_M.gguf" },
  { quant: "chosen/Q4_K_M", filename: "chosen/model-Q4_K_M.gguf" },
  {
    quant: "chosen/model-Q4_K_M",
    filename: "chosen/model-Q4_K_M-00001-of-00002.gguf",
  },
]) {
  test(`filename links hand off the listing's exact canonical identity: ${identity.quant}`, async () => {
    const app = harness({
      cachedGguf: [
        { repo_id: model, load_id: "C:\\Models\\snapshot", size_bytes: 1024 },
      ],
      variants: [{ ...quant, ...identity }],
    });
    const resolved = await app.resolve({
      ...target,
      meta: { ...target.meta, ggufVariant: identity.filename },
    });
    const handoff = modelConfigTarget(resolved.id, resolved.meta);
    assert.equal(handoff.ggufVariant, identity.quant);
    assert.equal(handoff.meta.ggufFilename, identity.filename);
    assert.equal(handoff.id, "C:\\Models\\snapshot");
    assert.equal(
      wantsDownloadManagerStaging({ id: resolved.id, ...resolved.meta }),
      false,
    );
  });
}

test("unrelated and incomplete inventories do not mark a target downloaded", async () => {
  const app = harness({
    cachedGguf: [
      { repo_id: "other/model", size_bytes: 1024 },
      { repo_id: model, partial: true, size_bytes: 1024 },
    ],
  });
  assert.equal(await app.resolve(), target);
  assert.deepEqual(app.requests, []);
});

test("same-artifact imports preserve active paths without scanning inventory", async () => {
  for (const isGguf of [false, true]) {
    for (const explicitModel of [undefined, model]) {
      const active = resolveRunConfigTarget(
        { model: explicitModel, config: { nParallel: 3 } },
        {
          ...selection,
          params: { checkpoint: model },
          activeLoadId: "/secondary/pinned",
          loadedIsGguf: isGguf,
          activeGgufVariant: isGguf ? "Q4_K_M" : null,
        },
      );
      assert.ok(active);
      const app = harness({ inventoryError: true });
      const resolved = await app.resolve(active);
      assert.equal(
        modelConfigTarget(resolved.id, resolved.meta).id,
        "/secondary/pinned",
      );
      assert.equal(
        wantsDownloadManagerStaging({ id: resolved.id, ...resolved.meta }),
        false,
      );
      assert.deepEqual(app.scans, []);
      assert.deepEqual(app.requests, []);
    }
  }
});

test("a variant change cannot inherit the active path or native file token", () => {
  const other = resolveRunConfigTarget(
    { ggufVariant: "Q8_0", config: {} },
    {
      ...selection,
      params: { checkpoint: model },
      loadedIsGguf: true,
      activeGgufVariant: "Q4_K_M",
      activeLoadId: "/secondary/pinned",
      activeNativePathToken: "local-only",
    },
  );
  assert.equal(other?.meta.loadId, undefined);
  assert.equal(other?.meta.nativePathToken, undefined);
  assert.equal(other?.meta.isDownloaded, undefined);
});

test("failed local variant listings preserve canonical Hub targets for review", async () => {
  const app = harness({
    cachedGguf: [{ repo_id: model, size_bytes: 1024 }],
    status: 503,
  });
  assert.equal(await app.resolve(), target);
  assert.deepEqual(app.hubRequests, []);
});

test("cancellation cannot produce a handoff", async () => {
  const controller = new AbortController();
  controller.abort();
  const app = harness({ cachedGguf: [{ repo_id: model, size_bytes: 1024 }] });
  await assert.rejects(app.resolve(target, controller.signal), {
    name: "AbortError",
  });
  assert.deepEqual(app.requests, []);
  assert.deepEqual(app.hubRequests, []);
});

for (const source of ["lmstudio", "custom", "ollama"] as const) {
  test(`${source} models cannot be adopted into a Hub repo's settings identity`, async () => {
    const app = harness({
      localModels: [
        {
          id: model,
          model_id: model,
          load_id: "/models/local.gguf",
          path: "/models/local.gguf",
          source,
          model_format: "gguf",
          display_name: "Local",
        },
      ],
    });
    assert.equal(await app.resolve(), target);
    assert.deepEqual(app.requests, []);
  });
}

test("an unrelated inventory failure cannot hide a complete cached copy", async () => {
  for (const source of ["localModels", "cachedGguf"]) {
    const app = harness({
      sourceErrors: [source],
      cachedGguf: [
        { repo_id: model, load_id: "/cache/pinned", size_bytes: 1024 },
      ],
      localModels: [
        {
          id: model,
          model_id: model,
          load_id: "/local/pinned",
          path: "/local/pinned",
          source: "hf_cache",
          model_format: "gguf",
          display_name: "Local",
        },
      ],
    });
    assert.equal(
      (await app.resolve()).meta.loadId,
      source === "localModels" ? "/cache/pinned" : "/local/pinned",
    );
  }
});

test("failed variant listings cannot hide a later complete snapshot", async () => {
  const app = harness({
    cachedGguf: ["/cache/failed", "/cache/available"].map((loadId) => ({
      repo_id: model,
      load_id: loadId,
      size_bytes: 1024,
    })),
    listingErrors: ["/cache/failed"],
  });
  assert.equal((await app.resolve()).meta.loadId, "/cache/available");
});

test("failed inventory scans preserve native and canonical GGUF Hub targets for review", async () => {
  for (const sourceErrors of [
    ["localModels"],
    ["cachedModels", "cachedGguf"],
    ["localModels", "cachedModels", "cachedGguf"],
  ]) {
    for (const input of [
      target,
      {
        id: "owner/Native",
        meta: { ...target.meta, isGguf: false, ggufVariant: undefined },
      },
    ]) {
      const app = harness({ sourceErrors });
      assert.equal(await app.resolve(input), input);
      assert.equal(input.meta.isDownloaded, undefined);
      assert.deepEqual(app.requests, []);
      assert.deepEqual(app.hubRequests, []);
    }
  }
});

for (const identity of [
  { quant: "Q4_K_M", filename: "model-Q4_K_M.gguf" },
  { quant: "chosen/Q4_K_M", filename: "chosen/model-Q4_K_M.gguf" },
  {
    quant: "chosen/model-Q4_K_M",
    filename: "chosen/model-Q4_K_M-00001-of-00002.gguf",
  },
]) {
  test(`uncached filename links resolve the exact download identity: ${identity.quant}`, async () => {
    const app = harness({
      hubVariants: [{ ...quant, ...identity, downloaded: false }],
    });
    const signal = new AbortController().signal;
    const resolved = await app.resolve(
      {
        ...target,
        meta: { ...target.meta, ggufVariant: identity.filename },
      },
      signal,
      "recipient-token",
    );
    const handoff = modelConfigTarget(resolved.id, resolved.meta);
    assert.equal(handoff.id, model);
    assert.equal(handoff.ggufVariant, identity.quant);
    assert.equal(handoff.meta.ggufFilename, identity.filename);
    assert.equal(resolved.meta.isDownloaded, false);
    assert.equal(
      wantsDownloadManagerStaging({ id: resolved.id, ...resolved.meta }),
      true,
    );
    assert.deepEqual(app.hubRequests, [
      { repoId: model, hfToken: "recipient-token", signal },
    ]);
  });
}

test("GGUF links without a variant use the listing's default before download staging", async () => {
  const app = harness({
    hubVariants: [
      { ...quant, downloaded: false },
      {
        ...quant,
        quant: "Q8_0",
        filename: "model-Q8_0.gguf",
        downloaded: false,
      },
    ],
    defaultVariant: "Q8_0",
  });
  const resolved = await app.resolve({
    ...target,
    meta: { ...target.meta, ggufVariant: undefined },
  });
  assert.equal(resolved.meta.ggufVariant, "Q8_0");
  assert.equal(resolved.meta.ggufFilename, "model-Q8_0.gguf");
  assert.equal(resolved.meta.isDownloaded, false);
});

test("cached default GGUF variants do not need a Hub lookup", async () => {
  const app = harness({
    cachedGguf: [{ repo_id: model, size_bytes: 1024 }],
    hubError: true,
  });
  const resolved = await app.resolve({
    ...target,
    meta: { ...target.meta, ggufVariant: undefined },
  });
  assert.equal(resolved.meta.ggufVariant, quant.quant);
  assert.equal(resolved.meta.isDownloaded, true);
  assert.deepEqual(app.hubRequests, []);
});

test("failed inventory scans still allow GGUF filename resolution", async () => {
  for (const [variant, downloaded] of [
    [{ ...quant, downloaded: false }, false],
    [quant, true],
    [{ ...quant, partial: true }, false],
  ] as const) {
    const app = harness({ inventoryError: true, hubVariants: [variant] });
    const resolved = await app.resolve({
      ...target,
      meta: { ...target.meta, ggufVariant: quant.filename },
    });
    assert.equal(resolved.meta.ggufVariant, quant.quant);
    assert.equal(resolved.meta.isDownloaded, downloaded);
  }
});

test("failed GGUF lookups preserve offline review without allowing an unresolved load", async () => {
  for (const ggufVariant of [quant.filename, undefined]) {
    const input = { ...target, meta: { ...target.meta, ggufVariant } };
    const offline = harness({ hubError: true });
    const resolved = await offline.resolve(input);
    assert.equal(resolved, input);
    assert.equal(
      isRunConfigVariantUnresolved(
        modelConfigTarget(resolved.id, resolved.meta),
      ),
      true,
    );

    const available = await harness().resolve(resolved);
    assert.equal(available.meta.ggufVariant, quant.quant);
    assert.equal(available.meta.ggufFilename, quant.filename);
    assert.equal(
      isRunConfigVariantUnresolved(
        modelConfigTarget(available.id, available.meta),
      ),
      false,
    );
  }
});

test("variant resolution only blocks uncached Hub GGUFs without a canonical variant", () => {
  for (const source of ["hub", "local"] as const) {
    for (const isGguf of [false, true]) {
      for (const isDownloaded of [false, true]) {
        for (const ggufVariant of [
          undefined,
          "model.GGUF",
          "Q4_K_M",
          "chosen/Q4_K_M",
        ]) {
          const pick = modelConfigTarget(model, {
            source,
            isLora: false,
            isGguf,
            isDownloaded,
            ggufVariant,
          });
          assert.equal(
            isRunConfigVariantUnresolved(pick),
            source === "hub" &&
              isGguf &&
              !isDownloaded &&
              (ggufVariant === undefined || ggufVariant === "model.GGUF"),
          );
        }
      }
    }
  }
});

test("cache-only listings cannot reject shared GGUF filenames or missing defaults", async () => {
  for (const ggufVariant of ["model-Q8_0.gguf", undefined]) {
    const input = { ...target, meta: { ...target.meta, ggufVariant } };
    const app = harness({
      cachedGguf: [{ repo_id: model, size_bytes: 1024 }],
      variants: [{ ...quant, downloaded: false, partial: true }],
      hubVariants: [{ ...quant, downloaded: false, partial: true }],
      defaultVariant: null,
      hubMetadataAvailable: false,
    });
    const resolved = await app.resolve(input);
    assert.equal(resolved, input);
    assert.equal(
      isRunConfigVariantUnresolved(
        modelConfigTarget(resolved.id, resolved.meta),
      ),
      true,
    );
    assert.equal(app.hubRequests.length, 1);
  }
});

test("known unavailable GGUF filenames and missing defaults report a resolution error", async () => {
  for (const ggufVariant of [quant.filename, undefined]) {
    const input = { ...target, meta: { ...target.meta, ggufVariant } };
    const missing = harness({ hubVariants: [] });
    await assert.rejects(missing.resolve(input), {
      constructor: missing.RunConfigResolutionError,
      message:
        "The shared GGUF variant is unavailable for this model. Ask the sender for an updated link.",
    });
  }
  await assert.rejects(
    harness({ defaultVariant: null }).resolve({
      ...target,
      meta: { ...target.meta, ggufVariant: undefined },
    }),
    /The shared GGUF variant is unavailable/,
  );
});

for (const hubError of [false, true]) {
  test(`cancelling a Hub variant lookup preserves the abort: failure=${hubError}`, async (t) => {
    t.mock.timers.enable({ apis: ["setTimeout"] });
    const app = harness({ variantDelayMs: 1_000, hubError });
    const controller = new AbortController();
    const pending = app.resolve(
      { ...target, meta: { ...target.meta, ggufVariant: quant.filename } },
      controller.signal,
    );
    await new Promise<void>((resolve) => setImmediate(resolve));
    assert.equal(app.hubRequests[0].signal, controller.signal);
    controller.abort();
    t.mock.timers.tick(1_000);
    await assert.rejects(pending, { name: "AbortError" });
  });
}

test("inventory scans and each variant listing have independent timeout budgets", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const app = harness({
    inventoryDelayMs: 20_000,
    variantDelayMs: 20_000,
    cachedGguf: ["/cache/missing", "/cache/available"].map((load_id) => ({
      repo_id: model,
      load_id,
      size_bytes: 1024,
    })),
    listingsByRepo: { "/cache/missing": [], "/cache/available": [quant] },
  });
  const pending = app.resolve();
  for (let step = 0; step < 3; step += 1) {
    t.mock.timers.tick(20_000);
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  assert.equal((await pending).meta.loadId, "/cache/available");
});
