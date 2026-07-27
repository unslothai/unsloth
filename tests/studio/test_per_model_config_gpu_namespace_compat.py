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
_GPU_INFO = _FRONTEND_SRC / "hooks/use-gpu-info.ts"
_CHAT_API = _FRONTEND_SRC / "features/chat/api/chat-api.ts"
_ACTIVE_MODEL_CONFIG = _FRONTEND_SRC / "features/model-picker/hooks/use-active-model-config.ts"


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


def _extract_module_function(source: str, name: str) -> str:
    """Same slice for a module-private helper (no ``export`` keyword)."""
    return _extract_function(
        source.replace(f"\nfunction {name}(", f"\nexport function {name}(", 1), name
    )


def _write_harness(
    workdir: Path,
    script: str,
    modules: dict[str, str] | None = None,
    stubs: dict[str, str] | None = None,
) -> None:
    """Write the node harness into a pytest tmp dir (never the repo tree).

    ``modules`` maps a bare/aliased specifier onto the file the loader should serve for
    it, so a module under test can be imported for real while its heavy neighbours (the
    feature barrels, React) are served from ``stubs`` written alongside the script.
    """
    workdir.mkdir(parents = True, exist_ok = True)
    (workdir / "register.mjs").write_text(
        "import { register } from 'node:module';\nregister('./loader.mjs', import.meta.url);\n"
    )
    for name, source in (stubs or {}).items():
        (workdir / name).write_text(source)
    mapping = {"@/features/hub": _HUB_IDENTITY.as_uri()}
    for specifier, target in (modules or {}).items():
        mapping[specifier] = target if "://" in target else (workdir / target).as_uri()
    # Resolve the app's "@/" alias and Vite's extensionless relative imports.
    (workdir / "loader.mjs").write_text(
        "const MAP = %s;\n"
        "const SRC = %s;\n"
        "export function resolve(specifier, context, next) {\n"
        "  if (Object.hasOwn(MAP, specifier)) return next(MAP[specifier], context);\n"
        "  if (specifier.startsWith('@/')) {\n"
        "    const rest = specifier.slice(2);\n"
        "    return next(SRC + rest + (rest.includes('.') ? '' : '.ts'), context);\n"
        "  }\n"
        "  if (specifier.startsWith('.') && !specifier.endsWith('.ts')) {\n"
        "    return next(specifier + '.ts', context);\n"
        "  }\n"
        "  return next(specifier, context);\n"
        "}\n" % (json.dumps(mapping), json.dumps(_FRONTEND_SRC.as_uri() + "/"))
    )
    (workdir / "run.mts").write_text(script)


def _run(
    workdir: Path,
    script: str,
    modules: dict[str, str] | None = None,
    stubs: dict[str, str] | None = None,
) -> dict:
    _require_node()
    _write_harness(workdir, script, modules, stubs)
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


def test_loaded_header_classification_beats_the_diffusion_name(tmp_path):
    """Once a model is loaded, its header classification decides, not its name.

    The backend already resolves this the same way: ``_preflight_is_diffusion`` is tri-state
    (None until a header is really read) so a name hint cannot override a header. The config
    page has to agree, or an ordinary GGUF carrying ``DiffusionGemma`` in its id loses the Host
    Memory control and the Vulkan GPU picker that ``/load`` accepts for it, while a real
    DiffusionGemma whose id says nothing keeps offering controls ``/load`` rejects.
    """
    # (loaded classification, model id, expected gate).
    cases = [
        # Loaded and the header says ordinary: the name must not re-hide the controls.
        [False, "unsloth/DiffusionGemma-derivative-GGUF", False],
        [False, "unsloth/diffusion_gemma-distill-GGUF", False],
        # Loaded and the header says diffusion: gate it whatever the id looks like.
        [True, "unsloth/DiffusionGemma-2B-GGUF", True],
        [True, "unsloth/gemma-3-4b-it-GGUF", True],
        # Nothing loaded: no header has been read on either side, so the name stands in
        # for exactly the rule the backend falls back to.
        [None, "unsloth/DiffusionGemma-2B-GGUF", True],
        [None, "/models/DiffusionGemma.Q4_K_M.gguf", True],
        [None, "unsloth/DiffusionGemma-derivative-GGUF", True],
        [None, "unsloth/gemma-3-4b-it-GGUF", False],
    ]
    script = """
const identity = await import(%(identity)s);
const out: boolean[] = [];
for (const [loaded, id] of %(cases)s as [boolean | null, string, boolean][]) {
  out.push(identity.resolveIsDiffusion(loaded, id));
}
console.log("RESULT " + JSON.stringify(out));
""" % {
        "identity": json.dumps(_MODEL_IDENTITY.as_uri()),
        "cases": json.dumps(cases),
    }
    result = _run(tmp_path, script)
    assert result == [case[2] for case in cases]


def test_hidden_host_memory_mode_never_reaches_the_load(tmp_path):
    """A remembered host-memory mode a diffusion gate hides must not be sent, only kept.

    ``_reject_diffusion_memory_mode`` 400s an explicit ``gguf_memory_mode``, and the control that
    would clear it is hidden, so sending it would fail every load with no way out. Dropping it from
    STORAGE instead would throw away a setting that is valid again as soon as the same id loads
    with a header saying it is not diffusion, which is the staged case the name check cannot tell
    apart. So the load payload loses the mode and the saved config keeps it.
    """
    body = _extract_function(_APPLY_CONFIG.read_text(), "diffusionSafeLoadConfig")
    script = (
        body
        + """
const remembered = {
  customContextLength: 8192,
  maxSeqLength: null,
  kvCacheDtype: null,
  speculativeType: "auto",
  specDraftNMax: null,
  tensorParallel: false,
  chatTemplateOverride: null,
  gpuMemoryMode: "manual",
  gpuLayers: 20,
  nCpuMoe: 0,
  selectedGpuIds: null,
  ggufMemoryMode: "pinned",
} as any;
const gated = diffusionSafeLoadConfig(remembered, true);
const ungated = diffusionSafeLoadConfig(remembered, false);
const noMode = { ...remembered, ggufMemoryMode: undefined };
console.log("RESULT " + JSON.stringify({
  sentMode: gated.ggufMemoryMode ?? null,
  // Everything else the load needs has to survive the strip untouched.
  sentContext: gated.customContextLength,
  sentGpuMemoryMode: gated.gpuMemoryMode,
  sentGpuLayers: gated.gpuLayers,
  // The caller's object is what gets saved, so the strip must not mutate it.
  rememberedMode: remembered.ggufMemoryMode,
  // A non-diffusion model keeps its mode, and neither case copies needlessly.
  ungatedMode: ungated.ggufMemoryMode ?? null,
  ungatedIsSame: ungated === remembered,
  unsetIsSame: diffusionSafeLoadConfig(noMode, true) === noMode,
}));
"""
    )
    result = _run(tmp_path, script)
    assert result["sentMode"] is None
    assert result["sentContext"] == 8192
    assert result["sentGpuMemoryMode"] == "manual"
    assert result["sentGpuLayers"] == 20
    assert result["rememberedMode"] == "pinned"
    assert result["ungatedMode"] == "pinned"
    assert result["ungatedIsSame"] is True
    assert result["unsetIsSame"] is True


def test_gguf_pin_candidates_read_their_own_channel(tmp_path):
    """/api/system publishes the llama.cpp pin space separately from the PyTorch devices.

    The GGUF picker must follow ``gguf_gpu_devices`` when the backend sends it, while the
    aggregate every other feature sizes against (Hub fit filtering, training model sizing,
    the VRAM monitor) keeps summing ``devices``. Older backends omit the new key, so the
    picker has to fall back to ``devices`` there. Runs the REAL selectors under node.
    """
    source = _GPU_INFO.read_text()
    _start = source.index("const DEFAULT_GPU")
    default_gpu = source[_start : source.index("\n};", _start) + 3]
    script = (
        default_gpu
        + "\n"
        + _extract_module_function(source, "toGpuInfo")
        + "\n"
        + _extract_module_function(source, "toGpuDevices")
        + """
const torchDevices = [
  { index: 0, index_kind: "physical", name: "GPU0", memory_total_gb: 8 },
  { index: 1, index_kind: "physical", name: "GPU1", memory_total_gb: 8 },
];
// A shared-memory iGPU only ggml can see, four times the real accelerator budget.
const vulkanDevices = [
  { index: 0, index_kind: "vulkan", name: "iGPU", memory_total_gb: 64 },
  { index: 1, index_kind: "vulkan", name: "dGPU", memory_total_gb: 8 },
];
const base = { cpu: {}, memory: {} };
const forcedVulkan = { ...base, gpu: {
  available: true, backend: "cuda", devices: torchDevices,
  gguf_gpu_devices: vulkanDevices, gguf_gpu_ids_supported: true,
} };
const oldBackend = { ...base, gpu: {
  available: true, backend: "cuda", devices: torchDevices, gguf_gpu_ids_supported: true,
} };
const probeFailed = { ...base, gpu: {
  available: true, backend: "cuda", devices: torchDevices,
  gguf_gpu_devices: [], gguf_gpu_ids_supported: false,
} };
console.log("RESULT " + JSON.stringify({
  forced_pin_kinds: toGpuDevices(forcedVulkan).map((d) => d.indexKind),
  forced_pin_names: toGpuDevices(forcedVulkan).map((d) => d.name),
  forced_sizing_gb: toGpuInfo(forcedVulkan).memoryTotalGb,
  old_pin_kinds: toGpuDevices(oldBackend).map((d) => d.indexKind),
  old_sizing_gb: toGpuInfo(oldBackend).memoryTotalGb,
  probe_failed_pinnable: toGpuDevices(probeFailed).map((d) => d.pinnable),
  probe_failed_names: toGpuDevices(probeFailed).map((d) => d.name),
}));
"""
    )
    result = _run(tmp_path, script)

    # The picker pins in ggml's ordinal space...
    assert result["forced_pin_kinds"] == ["vulkan", "vulkan"]
    assert result["forced_pin_names"] == ["iGPU", "dGPU"]
    # ...while everything sized against the training backend keeps the PyTorch total.
    assert result["forced_sizing_gb"] == 16

    # An older backend sends no pin channel: fall back to devices, unchanged.
    assert result["old_pin_kinds"] == ["physical", "physical"]
    assert result["old_sizing_gb"] == 16

    # Vulkan probe unreachable: the picker is hidden, the devices still list.
    assert result["probe_failed_pinnable"] == [False, False]
    assert result["probe_failed_names"] == ["GPU0", "GPU1"]


_AUTH_CAPTURE_STUB = """
export const calls: { url: string; body: any }[] = [];
export async function authFetch(url: string, init?: any) {
  calls.push({ url, body: init?.body ? JSON.parse(init.body) : null });
  return {
    ok: true,
    status: 200,
    json: async () => ({
      status: "ok", model: "m", display_name: "m",
      is_vision: false, is_lora: false, valid: true, message: "",
    }),
  } as any;
}
"""

_HF_AUTH_STUB = """
export async function prepareHfTokenForUse(token: string | null | undefined) {
  return { proceed: true, token: token ?? null };
}
"""

_NATIVE_INTENTS_STUB = """
export async function consumeNativePathToken() {
  return { nativePathLease: null };
}
"""


def test_host_memory_gate_covers_every_load_path(tmp_path):
    """A remembered diffusion host-memory mode must not reach /validate or /load, ever.

    ``_reject_diffusion_memory_mode`` 400s an explicit ``gguf_memory_mode`` on both endpoints,
    and the control that would clear it is hidden. Gating only the settings page left the picker
    row click on a remembered model, the startup auto-load, Hub Run and a compare pane's Send
    sending it anyway, so the gate lives in the two request builders every one of them calls.
    This drives the REAL ``chat-api.ts`` and inspects the bodies it posts.
    """
    # id, variant, mode, loaded classification (undefined = hint not supplied).
    cases = [
        # A staged pick: no header has been read, so the name stands in and the mode goes.
        ["unsloth/DiffusionGemma-2B-GGUF", "Q4_K_M", "pinned", None],
        # The name can also arrive on the variant alone.
        ["/models/model.gguf", "DiffusionGemma-Q4_K_M", "resident", None],
        # Loaded, and the header says ordinary: the id must not cost it the mode.
        ["unsloth/DiffusionGemma-derivative-GGUF", "Q4_K_M", "pinned", False],
        # Loaded, and the header says diffusion: gate it whatever the id looks like.
        ["unsloth/gemma-3-4b-it-GGUF", "Q4_K_M", "resident", True],
        # An ordinary model keeps its mode with no hint at all.
        ["unsloth/gemma-3-4b-it-GGUF", "Q4_K_M", "pinned", None],
    ]
    script = """
const api = await import("./chat-api-under-test.ts");
const auth = await import("./auth-stub.ts");

const out: any = { validate: [], load: [], leaked: [] };
for (const [id, variant, mode, loaded] of %(cases)s as any[]) {
  for (const kind of ["validate", "load"]) {
    auth.calls.length = 0;
    const payload: any = {
      model_path: id,
      gguf_variant: variant,
      hf_token: null,
      max_seq_length: 0,
      load_in_4bit: true,
      is_lora: false,
      gguf_memory_mode: mode,
    };
    // null is a real hint ("a header was read and it said ordinary"); the absent
    // case has to stay absent, which is what every un-updated caller sends.
    if (loaded !== null) payload.loadedIsDiffusion = loaded;
    if (kind === "validate") await api.validateModel(payload);
    else await api.loadModel(payload);
    const body = auth.calls[0].body;
    out[kind].push(body.gguf_memory_mode ?? null);
    if (Object.hasOwn(body, "loadedIsDiffusion")) out.leaked.push(id);
  }
}

// A payload with no mode at all must not gain one, on either endpoint.
auth.calls.length = 0;
await api.loadModel({
  model_path: "unsloth/DiffusionGemma-2B-GGUF", gguf_variant: "Q4_K_M",
  hf_token: null, max_seq_length: 0, load_in_4bit: true, is_lora: false,
} as any);
out.loadOmitsAbsentMode = !Object.hasOwn(auth.calls[0].body, "gguf_memory_mode");

console.log("RESULT " + JSON.stringify(out));
""" % {"cases": json.dumps([[c[0], c[1], c[2], c[3]] for c in cases])}
    result = _run(
        tmp_path,
        script,
        modules = {
            "@/features/auth": "auth-stub.ts",
            "@/features/hf-auth": "hf-auth-stub.ts",
            "@/features/native-intents/api": "native-intents-stub.ts",
        },
        stubs = {
            "chat-api-under-test.ts": f"export * from {json.dumps(_CHAT_API.as_uri())};\n",
            "auth-stub.ts": _AUTH_CAPTURE_STUB,
            "hf-auth-stub.ts": _HF_AUTH_STUB,
            "native-intents-stub.ts": _NATIVE_INTENTS_STUB,
        },
    )
    expected = [None, None, "pinned", None, "pinned"]
    # Both endpoints reject it, so both have to strip it, identically.
    assert result["validate"] == expected
    assert result["load"] == expected
    # The classification is a client-side hint; the backend never sees the field.
    assert result["leaked"] == []
    assert result["loadOmitsAbsentMode"] is True


# Enough of React to drive a real hook: state cells keyed by call order, effects
# flushed after the render that scheduled them, memo recompute on dep identity.
_REACT_STUB = """
type Cell = { seeded?: boolean; value?: any; deps?: any[] };
const cells: Cell[] = [];
let cursor = 0;
let queued: (() => void)[] = [];

function cell(): Cell {
  const existing = cells[cursor] ?? (cells[cursor] = {});
  cursor += 1;
  return existing;
}

function depsChanged(slot: Cell, deps: any[] | undefined): boolean {
  if (!slot.seeded || deps === undefined || slot.deps === undefined) return true;
  return (
    slot.deps.length !== deps.length ||
    deps.some((d, i) => !Object.is(d, slot.deps![i]))
  );
}

export function useState<T>(init: T | (() => T)): [T, (next: any) => void] {
  const slot = cell();
  if (!slot.seeded) {
    slot.seeded = true;
    slot.value = typeof init === "function" ? (init as () => T)() : init;
  }
  const set = (next: any) => {
    slot.value = typeof next === "function" ? next(slot.value) : next;
  };
  return [slot.value as T, set];
}

export function useEffect(fn: () => void, deps?: any[]): void {
  const slot = cell();
  const changed = depsChanged(slot, deps);
  slot.seeded = true;
  slot.deps = deps;
  if (changed) queued.push(fn);
}

export function useMemo<T>(fn: () => T, deps: any[]): T {
  const slot = cell();
  if (depsChanged(slot, deps)) slot.value = fn();
  slot.seeded = true;
  slot.deps = deps;
  return slot.value as T;
}

/** One render pass, then the effects it committed. */
export function render<T>(component: () => T): T {
  cursor = 0;
  const out = component();
  const effects = queued;
  queued = [];
  for (const effect of effects) effect();
  return out;
}
"""

# Only the two exports use-active-model-config.ts pulls from the chat barrel.
_CHAT_BARREL_STUB = """
export const state: any = {
  params: { checkpoint: null, maxSeqLength: 4096 },
  activeGgufVariant: null,
  ggufContextLength: null,
  customContextLength: null,
  kvCacheDtype: null,
  speculativeType: null,
  specDraftNMax: null,
  tensorParallel: false,
  chatTemplateOverride: null,
  gpuMemoryMode: "manual",
  gpuLayers: 20,
  nCpuMoe: 0,
  selectedGpuIds: null,
  ggufMemoryMode: null,
};
export function isExternalModelId(id: string | null | undefined): boolean {
  return typeof id === "string" && id.startsWith("external:");
}
export function useChatRuntimeStore(selector: (s: any) => any) {
  return selector(state);
}
"""

# /api/system answers only once the script releases it, so the first render really
# does run against a cold cache.
_SYSTEM_GATE_STUB = """
const DEVICE = (index: number, kind: string, name: string) => ({
  index, index_kind: kind, name, memory_total_gb: 8, vram_free_gb: 8,
});
const SYSTEM = {
  gpu: {
    available: true,
    backend: "vulkan",
    gguf_gpu_ids_supported: true,
    devices: [DEVICE(0, "physical", "GPU0"), DEVICE(1, "physical", "GPU1")],
    gguf_gpu_devices: [DEVICE(0, "vulkan", "iGPU"), DEVICE(1, "vulkan", "dGPU")],
  },
};
let release: (value: any) => void = () => {};
const gate = new Promise((resolve) => { release = resolve; });
export function releaseSystem() { release(SYSTEM); }
export async function authFetch(_url: string) {
  const data = await gate;
  return { ok: true, status: 200, json: async () => data } as any;
}
"""


def test_first_gpu_save_waits_for_the_vulkan_namespace(tmp_path):
    """A first save must not untag a Vulkan pick just because /api/system was still in flight.

    ``selectedGpuIds`` is seeded from /api/inference/status, which never warms the device cache,
    and the memoised snapshot has no other dependency that changes when /api/system lands. Frozen
    untagged, and with no earlier entry for ``keepStoredGpuIndexKind`` to recover a namespace
    from, the first saved entry reads back as a legacy physical pick and the reconcile drops ids
    that were always valid on this host. Drives the REAL hook and the REAL per-model store.
    """
    reconcile = _extract_function(_RUNTIME_STORE.read_text(), "reconcilePersistedGpuIds")
    script = (
        _STORAGE_SHIM
        + """
const react = await import("./react-stub.ts");
const chat = await import("./chat-barrel-stub.ts");
const system = await import("./system-stub.ts");
const gpu = await import(%(gpu)s);
const hook = await import(%(hook)s);
const configs = await import(%(config)s);
const identity = await import(%(identity)s);

const { cachedPinnableGpuIndexKind, cachedPinnableGpuIndices } = gpu;
%(reconcile)s

const MODEL = "unsloth/model-GGUF";
const VARIANT = "Q4_K_M";
const KEY = identity.modelStorageKey(MODEL, VARIANT);
const readEntry = () =>
  JSON.parse(localStorage.getItem("unsloth_model_configs") ?? "{}")[KEY] ?? null;

// A model is already loaded with a GPU pick when the page mounts: /status seeds
// the store, /api/system has not answered, and nothing here warms it.
chat.state.params.checkpoint = MODEL;
chat.state.activeGgufVariant = VARIANT;
chat.state.ggufContextLength = 8192;
chat.state.customContextLength = 8192;
chat.state.selectedGpuIds = [0, 1];

const cold = react.render(() => hook.useActiveModelConfig());
const coldKind = String(cold.config.selectedGpuIndexKind);
const coldCache = String(cachedPinnableGpuIndexKind());

// /api/system lands. No store field the memo watches has changed since.
system.releaseSystem();
await new Promise((resolve) => setTimeout(resolve, 0));
const warm = react.render(() => hook.useActiveModelConfig());
const warmKind = String(warm.config.selectedGpuIndexKind);

// The user hits Remember for the first time: what the snapshot carries is what
// this model's only stored entry gets.
configs.savePerModelConfig(MODEL, VARIANT, warm.config);
const stored = readEntry();
const reread = configs.resolveInitialConfig(MODEL, VARIANT).config;

console.log("RESULT " + JSON.stringify({
  coldKind,
  coldCache,
  warmKind,
  storedKind: stored?.selectedGpuIndexKind ?? null,
  storedIds: stored?.selectedGpuIds ?? null,
  restored: reconcilePersistedGpuIds(reread.selectedGpuIds, reread.selectedGpuIndexKind),
}));
"""
        % {
            "gpu": json.dumps(_GPU_INFO.as_uri()),
            "hook": json.dumps(_ACTIVE_MODEL_CONFIG.as_uri()),
            "config": json.dumps(_PER_MODEL_CONFIG.as_uri()),
            "identity": json.dumps(_MODEL_IDENTITY.as_uri()),
            "reconcile": reconcile,
        }
    )
    result = _run(
        tmp_path,
        script,
        modules = {
            "react": "react-stub.ts",
            "@/features/chat": "chat-barrel-stub.ts",
            "@/features/auth": "system-stub.ts",
        },
        stubs = {
            "react-stub.ts": _REACT_STUB,
            "chat-barrel-stub.ts": _CHAT_BARREL_STUB,
            "system-stub.ts": _SYSTEM_GATE_STUB,
        },
    )
    # Cold, the namespace is genuinely unknown, and the snapshot says so rather than guessing.
    assert result["coldCache"] == "undefined"
    assert result["coldKind"] == "undefined"
    # Once the cache answers, the snapshot must follow it instead of freezing.
    assert result["warmKind"] == "vulkan"
    # So this model's first stored entry carries the namespace...
    assert result["storedKind"] == "vulkan"
    assert result["storedIds"] == [0, 1]
    # ...and the pick still applies on the host it was made on.
    assert result["restored"] == [0, 1]
