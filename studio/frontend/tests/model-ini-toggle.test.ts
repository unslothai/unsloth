// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Off must leave every stored record, signature and load payload as before the field existed.

import assert from "node:assert/strict";
import test from "node:test";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  DEFAULT_PER_MODEL_CONFIG,
  normalizePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);
const configSignature = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);
const perModelConfigModule = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const gpuTensorSplit = await import("../src/hooks/gpu-tensor-split.ts");
let runtimeState: Record<string, unknown> = {};
const { perModelConfigsEqual, currentRuntimePerModelConfig } = loadWithStubs<{
  perModelConfigsEqual: (a: PerModelConfig, b: PerModelConfig) => boolean;
  currentRuntimePerModelConfig: () => PerModelConfig;
}>(
  new URL(
    "../src/features/model-picker/model-config/apply-per-model-config.ts",
    import.meta.url,
  ),
  {
    "@/features/chat/stores/chat-runtime-store": {
      useChatRuntimeStore: { getState: () => runtimeState },
      normalizeSpeculativeType: (value: unknown) => value ?? null,
      readPersistedSpeculativeType: () => "auto",
    },
    "@/features/chat/presets/preset-policy": {},
    "./config-signature": configSignature,
    "./per-model-config": perModelConfigModule,
    "@/hooks/gpu-tensor-split": gpuTensorSplit,
  },
);
const { residentRuntimeMatchesConfig } = await import(
  "../src/features/chat/lib/resident-config-match.ts"
);
const {
  formatModelIniSettings,
  modelIniLocationLabel,
  shouldShowModelIniRow,
  structuredKvCacheDtypeAfterLoad,
  withoutModelIniOffloadFlags,
} = await import("../src/features/model-picker/model-config/model-ini.ts");

const STORAGE_KEY = "unsloth_model_configs";

function config(overrides: Partial<PerModelConfig> = {}): PerModelConfig {
  return { ...DEFAULT_PER_MODEL_CONFIG, ...overrides };
}

function storedRecord(): Record<string, unknown> {
  const raw = store.get(STORAGE_KEY);
  assert.ok(raw, "nothing was persisted");
  const map = JSON.parse(raw) as Record<string, Record<string, unknown>>;
  const keys = Object.keys(map);
  assert.equal(keys.length, 1);
  return map[keys[0]];
}

test("off by default, and an unset record carries no key", () => {
  assert.equal(DEFAULT_PER_MODEL_CONFIG.useModelIni, undefined);
  assert.equal(
    "useModelIni" in normalizePerModelConfig(DEFAULT_PER_MODEL_CONFIG),
    false,
  );
  for (const bad of ["true", 1, null, {}, false]) {
    assert.equal(
      "useModelIni" in
        normalizePerModelConfig({
          ...DEFAULT_PER_MODEL_CONFIG,
          useModelIni: bad,
        }),
      false,
      `for ${JSON.stringify(bad)}`,
    );
  }
});

test("the switch round-trips through save and load, stamped with the newest schema", () => {
  store.clear();
  savePerModelConfig(
    "unsloth/Qwen3-GGUF",
    "Q4_K_M",
    config({ useModelIni: true }),
  );
  const record = storedRecord();
  assert.equal(record.useModelIni, true);
  assert.equal(record.version, 12);
  assert.equal(
    resolveInitialConfig("unsloth/Qwen3-GGUF", "Q4_K_M").config.useModelIni,
    true,
  );
});

test("a record without it keeps the version it had before the field", () => {
  store.clear();
  savePerModelConfig(
    "unsloth/Qwen3-GGUF",
    "Q4_K_M",
    config({ tensorSplit: [1, 1], selectedGpuIds: [0, 1] }),
  );
  assert.equal(storedRecord().version, 10);
  assert.equal("useModelIni" in storedRecord(), false);
});

test("turning it off drops the entry back to default", () => {
  store.clear();
  savePerModelConfig(
    "unsloth/Qwen3-GGUF",
    "Q4_K_M",
    config({ useModelIni: true }),
  );
  savePerModelConfig(
    "unsloth/Qwen3-GGUF",
    "Q4_K_M",
    config({ useModelIni: false }),
  );
  const raw = store.get(STORAGE_KEY);
  const map = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
  assert.equal(Object.keys(map).length, 0);
});

test("toggling changes the signature and equality, unset stays as before", () => {
  const off = config();
  const on = config({ useModelIni: true });
  assert.notEqual(loadedConfigSignature(on), loadedConfigSignature(off));
  assert.equal(
    loadedConfigSignature(config({ useModelIni: false })),
    loadedConfigSignature(off),
  );
  assert.equal(loadedConfigSignature(off).endsWith("|ini"), false);
  assert.equal(perModelConfigsEqual(on, off), false);
  assert.equal(perModelConfigsEqual(config({ useModelIni: false }), off), true);
});

test("the running config snapshot carries the switch only when on", () => {
  runtimeState = { params: {}, useModelIni: true };
  assert.equal(currentRuntimePerModelConfig().useModelIni, true);
  runtimeState = { params: {}, useModelIni: false };
  assert.equal("useModelIni" in currentRuntimePerModelConfig(), false);
});

test("a resident server matches only when it ran with the file exactly when asked", () => {
  const standing = {
    speculativeType: "auto",
    gpuMemoryMode: "auto" as const,
    gpuLayers: -1,
    nCpuMoe: 0,
    reconcileGpuIds: (ids: number[] | null) => ids,
    resolveContextLength: (ctx: number | null) => ctx ?? 0,
    parallelSlots: 1,
    splitRatio: null,
    normalizeSpeculative: (v: string | null | undefined) =>
      v == null ? null : String(v),
  };
  const off = config();
  const on = config({ useModelIni: true });
  assert.equal(residentRuntimeMatchesConfig({}, off, standing), true);
  assert.equal(
    residentRuntimeMatchesConfig({ model_ini_applied: false }, off, standing),
    true,
  );
  assert.equal(
    residentRuntimeMatchesConfig({ model_ini_applied: true }, off, standing),
    false,
  );
  assert.equal(residentRuntimeMatchesConfig({}, on, standing), false);
  assert.equal(
    residentRuntimeMatchesConfig({ model_ini_applied: true }, on, standing),
    true,
  );
  assert.equal(
    residentRuntimeMatchesConfig(
      { model_ini_applied: true, model_ini_cache_type: true, cache_type_kv: "q8_0" },
      on,
      standing,
    ),
    true,
  );
  assert.equal(
    residentRuntimeMatchesConfig(
      { model_ini_applied: true, model_ini_cache_type: false, cache_type_kv: "q8_0" },
      on,
      standing,
    ),
    false,
  );
  assert.equal(
    residentRuntimeMatchesConfig({ model_ini_applied: false, cache_type_kv: "q8_0" }, off, standing),
    false,
  );
});

test("a cache type the INI set is not adopted as the structured setting; a temperature-only INI's echo is", () => {
  assert.equal(structuredKvCacheDtypeAfterLoad("q8_0", null, true), null);
  assert.equal(structuredKvCacheDtypeAfterLoad("q8_0", "f16", true), "f16");
  assert.equal(structuredKvCacheDtypeAfterLoad("q8_0", null, false), "q8_0");
  assert.equal(structuredKvCacheDtypeAfterLoad("q8_0", null, undefined), "q8_0");
  assert.equal(structuredKvCacheDtypeAfterLoad(undefined, undefined, false), null);
  const sites: [string, RegExp][] = [
    [
      "features/chat/hooks/use-chat-model-runtime.ts",
      /const loadedKv = structuredKvCacheDtypeAfterLoad\(\s*loadResponse\.cache_type_kv,\s*loadKvCacheDtype,\s*loadResponse\.model_ini_cache_type,\s*\)/,
    ],
    [
      "features/chat/shared-composer.tsx",
      /kvCacheDtype: structuredKvCacheDtypeAfterLoad\(\s*resp\.cache_type_kv,\s*ownConfig\.kvCacheDtype,\s*resp\.model_ini_cache_type,\s*\)/,
    ],
  ];
  for (const [path, pattern] of sites) {
    assert.match(readSrc(path), pattern, path);
  }
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const adopted =
    adapter.match(
      /structuredKvCacheDtypeAfterLoad\(\s*loadResp\.cache_type_kv,\s*config\.kvCacheDtype,\s*loadResp\.model_ini_cache_type,\s*\)/g,
    ) ?? [];
  assert.equal(adopted.length, 4);
  assert.match(
    readSrc("features/chat/lib/apply-inference-status-to-store.ts"),
    /status\.cache_type_kv !== undefined &&\s*status\.model_ini_cache_type !== true &&/,
  );
});

test("server override hydration keeps the browser's choice", () => {
  // model-overrides.ts imports the auth barrel, which this runner cannot load; the field is browser-only.
  const src = readSrc("features/model-picker/api/model-overrides.ts");
  assert.match(src, /useModelIni: local\.useModelIni,/);
});

test("the row shows only for a found file on a llama-server GGUF", () => {
  assert.equal(shouldShowModelIniRow({ found: true }, true, false), true);
  assert.equal(shouldShowModelIniRow({ found: false }, true, false), false);
  assert.equal(shouldShowModelIniRow(null, true, false), false);
  assert.equal(shouldShowModelIniRow({ found: true }, false, false), false);
  assert.equal(shouldShowModelIniRow({ found: true }, true, true), false);
});

test("a switch left on for a file that is gone keeps its row, so it can be turned off", () => {
  assert.equal(shouldShowModelIniRow({ found: false }, true, false, true), true);
  assert.equal(shouldShowModelIniRow(null, true, false, true), true);
  assert.equal(shouldShowModelIniRow(undefined, true, false, true), true);
  assert.equal(shouldShowModelIniRow({ found: false }, false, false, true), false);
  assert.equal(shouldShowModelIniRow({ found: false }, true, true, true), false);
  const page = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(page, /unsloth\.ini was not found beside this model/);
});

test("a file-picked GGUF's INI lookup carries a native lease", () => {
  const api = readSrc("features/chat/api/chat-api.ts");
  const fetcher = api.slice(api.indexOf("export async function fetchModelIni"));
  assert.match(
    fetcher.slice(0, 1200),
    /consumeNativePathToken\(\s*options\.nativePathToken,\s*"validate-model",?\s*\)[\s\S]*params\.set\("native_path_lease", nativePathLease\)/,
  );
  assert.match(fetcher.slice(0, 1200), /params\.set\("offline", "true"\)/);
  const page = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(page, /const modelIniKey = [^;]*nativePathToken \?\? ""/);
  assert.match(page, /fetchModelIni\(target\.id, target\.ggufVariant, \{[^}]*nativePathToken,/);
});

test("the description names the location and the settings", () => {
  assert.equal(
    modelIniLocationLabel({
      filename: "unsloth.ini",
      location: "variant_folder",
    }),
    "unsloth.ini in this quant's folder",
  );
  assert.equal(
    modelIniLocationLabel({ filename: "unsloth.ini", location: "repo_root" }),
    "unsloth.ini in the repo root",
  );
  assert.equal(
    formatModelIniSettings(
      [
        "--ctx-size",
        "56000",
        "--fit",
        "off",
        "--no-mmap",
        "--gpu-layers",
        "-1",
        "--temp=1",
      ],
      1,
    ),
    "ctx-size=56000, fit=off, no-mmap, gpu-layers=-1, temp=1, parallel=1",
  );
  assert.equal(formatModelIniSettings([], null), "");
});

test("every load path sends use_model_ini only when on", () => {
  const sources = [
    readSrc("features/chat/hooks/use-chat-model-runtime.ts"),
    readSrc("features/chat/shared-composer.tsx"),
    readSrc("features/chat/api/chat-adapter.ts"),
  ];
  for (const src of sources) {
    const sends = src.match(/use_model_ini: [^,}\s]+/g) ?? [];
    assert.ok(sends.length > 0);
    for (const send of sends) {
      assert.equal(send, "use_model_ini: true", "never an unconditional false");
    }
    const spreads =
      src.match(/\? (?:\/\/[^\n]*\n\s*)?\{ use_model_ini: true \}\s*: \{\}/g) ??
      [];
    assert.equal(spreads.length, sends.length);
  }
});

test("the config page renders the switch behind shouldShowModelIniRow with the exact label", () => {
  const page = readSrc(
    "features/model-picker/components/model-config-page.tsx",
  );
  assert.match(
    page,
    /shouldShowModelIniRow\(\s*modelIni,\s*true,\s*isDiffusion,\s*config\.useModelIni === true,?\s*\) && \(\s*<ModelIniRow/,
  );
  assert.match(page, />Use \.ini file \(optional\)</);
});

test("an enabled INI is priced in the memory estimate", () => {
  const page = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(page, /const iniInEstimate =\s*config\.useModelIni === true && modelIni\?\.found \? modelIni : null;/);
  assert.match(page, /nParallel: iniInEstimate\?\.n_parallel \?\? runtimeConfig\.nParallel/);
  assert.match(
    page,
    /\.\.\.\(runtimeGpuMemoryMode === "manual"\s*\? withoutModelIniOffloadFlags\(iniInEstimate\.args, runtimeConfig\.gpuLayers\)\s*: iniInEstimate\.args\),\s*\.\.\.\(runtimeConfig\.llamaExtraArgs \?\? \[\]\),/,
  );
  assert.match(page, /except Extra Arguments,\s*which still win/);
});

test("Manual GPU memory prices the INI without the placement flags the load drops", () => {
  const ini = [
    "--ctx-size",
    "4096",
    "--gpu-layers",
    "-1",
    "--fit",
    "off",
    "--n-cpu-moe",
    "30",
    "--cpu-moe",
    "--tensor-split",
    "1,1",
    "-ngl=99",
    "--cache-type-k",
    "q8_0",
    "--temp",
    "0.42",
  ];
  for (const layers of [0, 24]) {
    assert.deepEqual(withoutModelIniOffloadFlags(ini, layers), [
      "--ctx-size",
      "4096",
      "--cache-type-k",
      "q8_0",
      "--temp",
      "0.42",
    ]);
  }
  for (const layers of [-1, null, undefined]) {
    assert.deepEqual(withoutModelIniOffloadFlags(ini, layers), [
      "--ctx-size",
      "4096",
      "--tensor-split",
      "1,1",
      "--cache-type-k",
      "q8_0",
      "--temp",
      "0.42",
    ]);
  }
  assert.deepEqual(withoutModelIniOffloadFlags(["-ts=3,1"], -1), ["-ts=3,1"]);
  assert.deepEqual(withoutModelIniOffloadFlags(["-ts=3,1"], 8), []);
  assert.deepEqual(withoutModelIniOffloadFlags([], 0), []);
});
