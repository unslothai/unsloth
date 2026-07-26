# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted per-model GPU picks across the #7210 upgrade.

Two things must keep working for a user upgrading an existing install:

  * ``unsloth_model_configs`` entries written before ``selectedGpuIndexKind`` /
    ``ggufMemoryMode`` existed, and
  * the even older ``unsloth_load_settings`` blob the migration imports.

Both are exercised against the REAL ``per-model-config.ts`` under node, with a
localStorage shim, plus the real ``reconcilePersistedGpuIds`` body so an
untagged legacy pick is proved to survive on the host it was saved from and to
be dropped on a host using a different GPU index namespace.
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
        "import { register } from 'node:module';\n"
        "register('./loader.mjs', import.meta.url);\n"
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
    # The new fields stay ABSENT rather than being invented as null: absent is
    # what the reconcile reads as "legacy physical pick".
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
    script = """
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
""" % body
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
    """Run-settings Load stages a persisted config; it must carry the tag.

    Without it a remembered physical CUDA pick is re-read as a ggml Vulkan
    ordinal and llama-server is pinned to a different card (silently).
    """
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
