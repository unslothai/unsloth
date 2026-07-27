# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted per-model GPU picks across the #7210 upgrade.

Two things must keep working on upgrade: ``unsloth_model_configs`` entries written before
``selectedGpuIndexKind`` / ``ggufMemoryMode`` existed, and the older ``unsloth_load_settings``
blob the migration imports. Both run against the REAL ``per-model-config.ts`` under node with a
localStorage shim, plus the real ``reconcilePersistedGpuIds`` body, so an untagged legacy pick
survives on the host it was saved from and is dropped under a different index namespace.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_SRC = _REPO_ROOT / "studio" / "frontend" / "src"
_PER_MODEL_CONFIG = _FRONTEND_SRC / "features/model-picker/model-config/per-model-config.ts"
_RUNTIME_STORE = _FRONTEND_SRC / "features/chat/stores/chat-runtime-store.ts"
_RUNTIME_HOOK = _FRONTEND_SRC / "features/chat/hooks/use-chat-model-runtime.ts"
_HUB_IDENTITY = _FRONTEND_SRC / "features/hub/lib/model-identity.ts"
_MODEL_IDENTITY = _FRONTEND_SRC / "features/model-picker/model-config/model-identity.ts"
_APPLY_CONFIG = _FRONTEND_SRC / "features/model-picker/model-config/apply-per-model-config.ts"


def _require_node():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    for path in (_PER_MODEL_CONFIG, _RUNTIME_STORE, _RUNTIME_HOOK, _HUB_IDENTITY):
        if not path.exists():
            pytest.skip(f"studio frontend source not present: {path}")
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 30,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _extract_function(source: str, name: str) -> str:
    """Slice an exported function body out of a module too heavy to import."""
    start = source.index(f"export function {name}(")
    index = source.index("{", source.index(")", start))
    depth = 0
    while True:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    return source[start : index + 1].replace("export function", "function", 1)


def _write_harness(workdir: Path, script: str) -> None:
    """Write the node harness into a pytest tmp dir (never the repo tree)."""
    workdir.mkdir(parents = True, exist_ok = True)
    (workdir / "register.mjs").write_text(
        "import { register } from 'node:module';\nregister('./loader.mjs', import.meta.url);\n"
    )
    # Resolve the app's "@/" alias and Vite's extensionless relative imports.
    (workdir / "loader.mjs").write_text(
        "const HUB = %s;\n"
        "export function resolve(specifier, context, next) {\n"
        "  if (specifier === '@/features/hub') return next(HUB, context);\n"
        "  if (specifier.startsWith('.') && !specifier.endsWith('.ts')) {\n"
        "    return next(specifier + '.ts', context);\n"
        "  }\n"
        "  return next(specifier, context);\n"
        "}\n" % json.dumps(_HUB_IDENTITY.as_uri())
    )
    (workdir / "run.mts").write_text(script)


def _run(workdir: Path, script: str) -> dict:
    _require_node()
    _write_harness(workdir, script)
    result = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--import=./register.mjs",
            "--no-warnings",
            "run.mts",
        ],
        cwd = str(workdir),
        capture_output = True,
        text = True,
        timeout = 120,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


_STORAGE_SHIM = """
const store = new Map<string, string>();
(globalThis as any).window = globalThis;
(globalThis as any).localStorage = {
  getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
  setItem: (k: string, v: string) => { store.set(k, String(v)); },
  removeItem: (k: string) => { store.delete(k); },
  clear: () => store.clear(),
};
"""


def test_legacy_config_without_namespace_tag_still_applies(tmp_path):
    """A pick saved before the tag existed must load, not be discarded."""
    script = (
        _STORAGE_SHIM
        + """
const identity = await import(__IDENTITY__);
// A v1 entry written by the previous release: GPU knobs, no namespace tag and
// no host-memory mode, because neither field existed yet.
localStorage.setItem(
  "unsloth_model_configs",
  JSON.stringify({
    [identity.modelStorageKey("unsloth/model-GGUF", "Q4_K_M")]: {
      version: 1,
      customContextLength: 8192,
      maxSeqLength: null,
      kvCacheDtype: "q8_0",
      speculativeType: null,
      specDraftNMax: null,
      tensorParallel: false,
      chatTemplateOverride: null,
      gpuMemoryMode: "manual",
      gpuLayers: 20,
      nCpuMoe: 2,
      selectedGpuIds: [0, 1],
    },
  }),
);

const mod = await import(%s);
const { config, remembered } = mod.resolveInitialConfig("unsloth/model-GGUF", "Q4_K_M");
console.log("RESULT " + JSON.stringify({
  remembered,
  customContextLength: config.customContextLength,
  kvCacheDtype: config.kvCacheDtype,
  gpuMemoryMode: config.gpuMemoryMode,
  gpuLayers: config.gpuLayers,
  nCpuMoe: config.nCpuMoe,
  selectedGpuIds: config.selectedGpuIds,
  hasIndexKind: Object.hasOwn(config, "selectedGpuIndexKind"),
  hasMemoryMode: Object.hasOwn(config, "ggufMemoryMode"),
  isDefault: mod.isDefaultConfig(config),
}));
"""
        % json.dumps(_PER_MODEL_CONFIG.as_uri())
    ).replace("__IDENTITY__", json.dumps(_MODEL_IDENTITY.as_uri()))
    result = _run(tmp_path, script)
    assert result["remembered"] is True
    # Every pre-existing field survives untouched.
    assert result["customContextLength"] == 8192
    assert result["kvCacheDtype"] == "q8_0"
    assert result["gpuMemoryMode"] == "manual"
    assert result["gpuLayers"] == 20
    assert result["nCpuMoe"] == 2
    assert result["selectedGpuIds"] == [0, 1]
    # The new fields stay ABSENT, not null: absent is what reconcile reads as legacy physical.
    assert result["hasIndexKind"] is False
    assert result["hasMemoryMode"] is False
    assert result["isDefault"] is False


def test_old_install_load_settings_migration_keeps_gpu_pick(tmp_path):
    """The pre-per-model-config blob an old install left behind still imports."""
    script = (
        _STORAGE_SHIM
        + """
// unsloth_load_settings is the storage key the previous generation used.
localStorage.setItem(
  "unsloth_load_settings",
  JSON.stringify({
    "unsloth/model-GGUF::Q4_K_M": {
      contextLength: 4096,
      kvCacheDtype: "q8_0",
      tensorParallel: true,
      gpuMemoryMode: "manual",
      gpuLayers: 12,
      nCpuMoe: 1,
      selectedGpuIds: [1],
    },
  }),
);

const mod = await import(%s);
const { config, remembered } = mod.resolveInitialConfig("unsloth/model-GGUF", "Q4_K_M");
console.log("RESULT " + JSON.stringify({
  remembered,
  customContextLength: config.customContextLength,
  tensorParallel: config.tensorParallel,
  gpuMemoryMode: config.gpuMemoryMode,
  gpuLayers: config.gpuLayers,
  selectedGpuIds: config.selectedGpuIds,
  hasIndexKind: Object.hasOwn(config, "selectedGpuIndexKind"),
  migrationFlag: localStorage.getItem("unsloth_model_configs_migrated"),
}));
"""
        % json.dumps(_PER_MODEL_CONFIG.as_uri())
    )
    result = _run(tmp_path, script)
    assert result["remembered"] is True
    assert result["customContextLength"] == 4096
    assert result["tensorParallel"] is True
    assert result["gpuMemoryMode"] == "manual"
    assert result["gpuLayers"] == 12
    assert result["selectedGpuIds"] == [1]
    assert result["hasIndexKind"] is False
    assert result["migrationFlag"] == "1"


def test_reconcile_matrix_over_host_gpu_namespaces(tmp_path):
    """The saved namespace decides whether a pick may be reused on this host."""
    body = _extract_function(_RUNTIME_STORE.read_text(), "reconcilePersistedGpuIds")
    assert "cachedPinnableGpuIndexKind" in body, "extracted the wrong function"
    script = (
        """
type GpuIndexKind = "physical" | "vulkan";
let CURRENT_KIND: GpuIndexKind | null | undefined;
let PINNABLE: number[] | null;
function cachedPinnableGpuIndexKind() { return CURRENT_KIND; }
function cachedPinnableGpuIndices() { return PINNABLE; }

%s

type Case = {
  name: string;
  kind: GpuIndexKind | null | undefined;
  pinnable: number[] | null;
  ids: number[] | null;
  saved?: GpuIndexKind | null;
  tagged: boolean;
};

const cases: Case[] = [
  // Legacy blob (no tag) on the multi-GPU CUDA host it was saved from.
  { name: "legacy_on_physical", kind: "physical", pinnable: [0, 1], ids: [0, 1],
    saved: undefined, tagged: true },
  // Same legacy blob carried to a Vulkan-only build: the ids mean other cards.
  { name: "legacy_on_vulkan", kind: "vulkan", pinnable: [0, 1], ids: [0, 1],
    saved: undefined, tagged: true },
  { name: "vulkan_on_vulkan", kind: "vulkan", pinnable: [0, 1], ids: [0, 1],
    saved: "vulkan", tagged: true },
  { name: "vulkan_on_physical", kind: "physical", pinnable: [0, 1], ids: [0, 1],
    saved: "vulkan", tagged: true },
  // Single GPU: the picker is hidden, so cachedPinnableGpuIndices reports [].
  { name: "single_gpu_host", kind: null, pinnable: [], ids: [0],
    saved: "physical", tagged: false },
  // No GPU at all: same shape as single-GPU, nothing is pinnable.
  { name: "no_gpu_host", kind: null, pinnable: [], ids: [0],
    saved: undefined, tagged: false },
  // Cold /api/system cache: cannot validate, so keep the pick (backend guards).
  { name: "cold_cache", kind: undefined, pinnable: null, ids: [0, 1],
    saved: "physical", tagged: true },
  // A pick partly out of range on this host keeps only the surviving ids.
  { name: "narrowed_pick", kind: "physical", pinnable: [0, 1], ids: [1, 5],
    saved: "physical", tagged: true },
  // No pick at all (every pre-#7164 config) passes straight through.
  { name: "no_pick", kind: "physical", pinnable: [0, 1], ids: null,
    saved: undefined, tagged: true },
];

const out: Record<string, number[] | null> = {};
for (const c of cases) {
  CURRENT_KIND = c.kind;
  PINNABLE = c.pinnable;
  out[c.name] = c.tagged
    ? reconcilePersistedGpuIds(c.ids, c.saved)
    : reconcilePersistedGpuIds(c.ids);
}
console.log("RESULT " + JSON.stringify(out));
"""
        % body
    )
    result = _run(tmp_path, script)
    assert result["legacy_on_physical"] == [0, 1]
    assert result["legacy_on_vulkan"] is None
    assert result["vulkan_on_vulkan"] == [0, 1]
    assert result["vulkan_on_physical"] is None
    assert result["single_gpu_host"] is None
    assert result["no_gpu_host"] is None
    assert result["cold_cache"] == [0, 1]
    assert result["narrowed_pick"] == [1]
    assert result["no_pick"] is None


def test_staged_load_config_forwards_the_saved_namespace():
    """Run-settings Load stages a persisted config; it must carry the tag, or a remembered CUDA
    pick is re-read as a ggml Vulkan ordinal and llama-server is pinned to another card."""
    hook = _RUNTIME_HOOK.read_text()
    staged = re.findall(
        r"reconcilePersistedGpuIds\(\s*\n\s*pendingLoadConfig\.selectedGpuIds,"
        r"\s*\n\s*pendingLoadConfig\.selectedGpuIndexKind,",
        hook,
    )
    # Both the click-time snapshot and the cross-model reset re-derive the pick.
    assert len(staged) == 2, "staged pendingLoadConfig pick must forward the namespace tag"
    assert "reconcilePersistedGpuIds(pendingLoadConfig.selectedGpuIds)" not in hook

    # Every other persisted-config reader already forwards it.
    for relative in (
        "features/model-picker/model-config/apply-per-model-config.ts",
        "features/chat/api/chat-adapter.ts",
        "features/chat/shared-composer.tsx",
    ):
        source = (_FRONTEND_SRC / relative).read_text()
        assert "selectedGpuIndexKind," in source, relative


def test_cold_cache_resave_keeps_the_stored_gpu_namespace(tmp_path):
    """Saving before /api/system resolves must not untag a Vulkan pick: writing it back untagged
    makes the next reconcile read it as legacy physical and discard it on its own host."""
    snapshot = _extract_function(_APPLY_CONFIG.read_text(), "currentRuntimePerModelConfig")
    assert "cachedPinnableGpuIndexKind" in snapshot, "extracted the wrong function"
    reconcile = _extract_function(_RUNTIME_STORE.read_text(), "reconcilePersistedGpuIds")
    script = (
        _STORAGE_SHIM
        + """
type GpuIndexKind = "physical" | "vulkan";
let CURRENT_KIND: GpuIndexKind | null | undefined;
let PINNABLE: number[] | null;
function cachedPinnableGpuIndexKind() { return CURRENT_KIND; }
function cachedPinnableGpuIndices() { return PINNABLE; }

// Runtime store stub: only the fields the snapshot reads.
const STATE: any = {
  customContextLength: 8192,
  params: { maxSeqLength: 4096 },
  kvCacheDtype: "q8_0",
  speculativeType: null,
  specDraftNMax: null,
  tensorParallel: false,
  chatTemplateOverride: null,
  gpuMemoryMode: "manual",
  gpuLayers: 20,
  nCpuMoe: 0,
  selectedGpuIds: [0, 1],
  ggufMemoryMode: null,
};
const useChatRuntimeStore = { getState: () => STATE };
function cleanTemplate(v: string | null | undefined): string | null {
  return v?.trim() ? v : null;
}
function normalizeSpeculativeType(v: any) { return v ?? null; }

const mod = await import(%(config)s);
const identity = await import(%(identity)s);
const { normalizeMaxSeqLength } = mod;

%(snapshot)s

%(reconcile)s

const MODEL = "unsloth/model-GGUF";
const VARIANT = "Q4_K_M";
const KEY = identity.modelStorageKey(MODEL, VARIANT);
const readEntry = () =>
  JSON.parse(localStorage.getItem("unsloth_model_configs") ?? "{}")[KEY] ?? null;

// Session 1: the cache is warm, the picker is usable, the pick is tagged.
CURRENT_KIND = "vulkan";
PINNABLE = [0, 1];
mod.savePerModelConfig(MODEL, VARIANT, currentRuntimePerModelConfig());
const firstKind = readEntry()?.selectedGpuIndexKind ?? null;

// Session 2: a fresh page load. The pick is restored while /api/system is
// still in flight (reconcile keeps it, it cannot validate yet), then the user
// edits an unrelated field and saves again.
CURRENT_KIND = undefined;
PINNABLE = null;
STATE.customContextLength = 16384;
const snapshotKindIsUndefined =
  currentRuntimePerModelConfig().selectedGpuIndexKind === undefined;
mod.savePerModelConfig(MODEL, VARIANT, currentRuntimePerModelConfig());
const entry = readEntry();

// The cache resolves: still the same Vulkan host.
CURRENT_KIND = "vulkan";
PINNABLE = [0, 1];
const { config: reread } = mod.resolveInitialConfig(MODEL, VARIANT);
const restored = reconcilePersistedGpuIds(
  reread.selectedGpuIds,
  reread.selectedGpuIndexKind,
);

console.log("RESULT " + JSON.stringify({
  firstKind,
  snapshotKindIsUndefined,
  storedKind: entry?.selectedGpuIndexKind ?? null,
  storedIds: entry?.selectedGpuIds ?? null,
  storedContext: entry?.customContextLength ?? null,
  restored,
}));
"""
        % {
            "config": json.dumps(_PER_MODEL_CONFIG.as_uri()),
            "identity": json.dumps(_MODEL_IDENTITY.as_uri()),
            "snapshot": snapshot,
            "reconcile": reconcile,
        }
    )
    result = _run(tmp_path, script)
    assert result["firstKind"] == "vulkan"
    # The snapshot itself still cannot know the namespace; that is the input.
    assert result["snapshotKindIsUndefined"] is True
    # The unrelated edit is saved, and the namespace already on record survives.
    assert result["storedContext"] == 16384
    assert result["storedIds"] == [0, 1]
    assert result["storedKind"] == "vulkan"
    # So the pick is still usable on the host it was made on.
    assert result["restored"] == [0, 1]


def test_staged_diffusion_name_check_matches_the_backend(tmp_path):
    """The staged config page must classify DiffusionGemma like /validate does.

    ``_classify_diffusion_gguf`` has no header before the download, so it strips non-alphanumerics
    and looks for the family name. The frontend gates Host Memory on the same rule; disagreement
    would offer a mode /load 400s.
    """
    identities = [
        "unsloth/DiffusionGemma-2B-GGUF",
        "unsloth/diffusion_gemma-2b-GGUF",
        "/models/DiffusionGemma.Q4_K_M.gguf",
        "unsloth/gemma-3-4b-it-GGUF",
        "unsloth/Qwen3-8B-GGUF",
        "",
    ]
    script = """
const identity = await import(%(identity)s);
const out: Record<string, boolean> = {};
for (const name of %(names)s) {
  out[name] = identity.looksLikeDiffusionGemma(name);
}
console.log("RESULT " + JSON.stringify(out));
""" % {
        "identity": json.dumps(_MODEL_IDENTITY.as_uri()),
        "names": json.dumps(identities),
    }
    result = _run(tmp_path, script)
    for name in identities:
        # The backend rule, transcribed from routes/inference.py.
        expected = "diffusiongemma" in re.sub(r"[^a-z0-9]+", "", name.lower())
        assert result[name] is expected, name
