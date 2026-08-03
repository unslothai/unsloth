# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed auto-load of a cached model must not become a Hub download.

Runs the real ``autoLoadSmallestModel`` from chat-adapter.ts under node with the module boundary
stubbed, so these assert behaviour, not source text. The sweep's catches are parameterless, so a
cached repo whose load rejected fell through to fetching an unrelated default model.
"""

import json
import os
import shutil
import subprocess
import re
import tempfile
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def _source_path(relative_path: str) -> Path:
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


ADAPTER = _source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")
TEMP = WORKDIR / "temp" / "chat_autoload_failure_gate"
DEFAULT_MODEL = "unsloth/Qwen3.5-4B-MTP-GGUF"
GEMMA_REPO = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"

# Stubs for everything autoLoadSmallestModel imports; each scenario supplies the cache inventory
# and how /validate and /load answer per model_path.
PREAMBLE = """
type LastLocalModelKind = "gguf" | "model";
type GgufVariantDetail = {
  quant?: string | null;
  filename?: string | null;
  downloaded?: boolean;
  size_bytes: number;
};
type ChatModelSummary = Record<string, unknown>;

export type Scenario = {
  ggufRepos: any[];
  modelRepos: any[];
  variants: Record<string, any>;
  lastLoaded: any;
  validate: (payload: any) => any;
  load: (payload: any) => any;
};

export const EVENTS: any[] = [];
let SCENARIO: Scenario;
export function setScenario(scenario: Scenario) {
  SCENARIO = scenario;
  EVENTS.length = 0;
  STORE = makeStore();
}

const GPU_LAYERS_AUTO = -1;

function makeStore(): any {
  const state: any = {
    hfToken: null,
    params: { maxSeqLength: 4096, checkpoint: "" },
    activeGgufVariant: null,
    activePresetSource: null,
    gpuMemoryMode: "auto",
    selectedGpuIds: null,
    models: [],
    setCheckpoint: () => {},
    setModelRequiresTrustRemoteCode: () => {},
    setParams: (p: any) => { state.params = p; },
    setModels: (m: any[]) => { state.models = m; },
  };
  return state;
}
let STORE: any = makeStore();

// Imported by the sliced region from ../hooks/use-chat-model-runtime, so the
// harness has to supply it or the call is a bare ReferenceError -- which the
// retry loop below catches and scores as a failed load, making a healthy model
// look broken. Mirrors the real upsert closely enough for these scenarios: the
// tests assert on which models are loaded, not on the capability fields.
// Both sit behind `candidate.kind === "gguf" && (selectedGpuIds != null ||
// tensorParallel)`, which no scenario here sets, so they are unreached today --
// but they are the same landmine as syncModelCapabilities was, waiting for the
// first scenario that turns a GPU option on. Stubbed so the harness is complete
// rather than complete-by-luck.
async function prepareHfTokenForUse(token: any) {
  return { proceed: true, token };
}
async function fetchGgufStagedMetadata(_req: any) {
  // camelCase, matching chat-api.ts: the call site reads `.isDiffusion`, so a
  // snake_case key here would be silently undefined.
  return {
    contextLength: null,
    layerCount: null,
    moeLayerCount: null,
    isDiffusion: false,
    diffusionUnknown: false,
  };
}

function syncModelCapabilities(modelId: string, resp: any) {
  const store = useChatRuntimeStore.getState();
  const models = store.models;
  const synced = {
    isVision: Boolean(resp.is_vision),
    isGguf: Boolean(resp.is_gguf),
    isAudio: Boolean(resp.is_audio),
    audioType: resp.audio_type ?? null,
    hasAudioInput: Boolean(resp.has_audio_input),
  };
  const idx = models.findIndex((m: any) => m.id === modelId);
  if (idx === -1) {
    store.setModels([
      ...models,
      { id: modelId, name: resp.display_name || modelId, isLora: Boolean(resp.is_lora), ...synced },
    ]);
  } else {
    const next = [...models];
    next[idx] = { ...next[idx], ...synced };
    store.setModels(next);
  }
}
const useChatRuntimeStore = {
  getState: () => STORE,
  setState: (_p: any) => {},
};

function createLoadingToastIcon() { return null; }
const toast: any = Object.assign(
  (_msg: string, _opts?: any) => "toast-id",
  {
    message: (msg: string, opts?: any) => {
      EVENTS.push({ kind: "toast.message", msg, description: opts?.description });
      return "toast-id";
    },
    success: (msg: string) => EVENTS.push({ kind: "toast.success", msg }),
    error: (msg: string, opts?: any) =>
      EVENTS.push({ kind: "toast.error", msg, description: opts?.description }),
    dismiss: () => EVENTS.push({ kind: "toast.dismiss" }),
    info: (msg: string) => EVENTS.push({ kind: "toast.info", msg }),
  },
);

async function tryAdoptServerActiveModel() { return false; }
function resolveSpeculativeSettingsForLoad() {
  return { speculativeType: null, specDraftNMax: 0 };
}
function readLastLocalModelLoad() { return SCENARIO.lastLoaded; }
function recordLastLocalModelLoad(_x: any) {}
function resolveInitialConfig(_id: string, _variant: any) {
  return { config: {
    customContextLength: null, maxSeqLength: null, gpuMemoryMode: null,
    gpuLayers: null, nCpuMoe: null, selectedGpuIds: undefined,
    speculativeType: null, specDraftNMax: null, chatTemplateOverride: null,
    kvCacheDtype: null, tensorParallel: false,
  } };
}
function resolveLoadMaxSeqLength(args: any) { return args.maxSeqLength ?? 0; }
function resolveFitMaxSeqLength(..._a: any[]) { return 0; }
function resolveManualAutoCtxPin(..._a: any[]) { return null; }
async function ensureGpuDeviceCache() {}
function reconcilePersistedGpuIds(ids: any) { return ids; }
function saveSpeculativeType(_x: any) {}
function persistGpuMemoryModeOnLoad(..._a: any[]) {}
function reasoningCapsFromLoad(_x: any) { return {}; }
function resolveToolsEnabledOnLoad(_x: any) { return {}; }
function loadedGpuMemoryFields(_x: any) { return {}; }
function resolveLoadedSpeculativeSettings(_x: any) { return {}; }
function isMultimodalResponse(_x: any) { return false; }

async function listCachedGguf() { return SCENARIO.ggufRepos as any; }
async function listCachedModels() { return SCENARIO.modelRepos as any; }
async function listGgufVariants(repoId: string, _b?: any, _c?: any) {
  const entry = SCENARIO.variants[repoId];
  if (entry === "throw") throw new Error("variant listing failed");
  return entry ?? { variants: [] };
}
async function validateModel(payload: any) {
  const result = SCENARIO.validate(payload);
  if (result instanceof Error) throw result;
  return result;
}
async function loadModel(payload: any) {
  const result = SCENARIO.load(payload);
  EVENTS.push({
    kind: "loadModel",
    model_path: payload.model_path,
    gguf_variant: payload.gguf_variant ?? null,
    rejected: result instanceof Error,
  });
  if (result instanceof Error) throw result;
  return result;
}
"""

SCENARIO_HELPERS = """
    const GEMMA = {
      repo_id: "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
      load_id: "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
      cache_path:
        "/home/john-doe/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF",
      size_bytes: 15800000000,
    };
    const GEMMA_VARIANTS = {
      variants: [{
        quant: "UD-Q4_K_XL",
        filename: "UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        downloaded: true,
        size_bytes: 15800000000,
      }],
    };
    const OOM =
      "Failed to load model: llama-server was stopped by the operating system " +
      "(signal 9), most likely out of memory.";
    const VALIDATE_OK = () => ({
      requires_trust_remote_code: false,
      requires_security_review: false,
      requires_transformers_upgrade: false,
    });
    const LOADED = (payload) => ({
      model: payload.model_path,
      is_gguf: true,
      context_length: 32768,
    });
    const scenario = (over) => ({
      ggufRepos: [],
      modelRepos: [],
      variants: {},
      lastLoaded: null,
      validate: VALIDATE_OK,
      load: LOADED,
      ...over,
    });
"""


def _require_node():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not ADAPTER.exists():
        pytest.skip("studio chat sources not present")
    try:
        result = subprocess.run(
            ["node", "--experimental-strip-types", "--version"],
            capture_output = True,
            text = True,
            # A cold Windows runner is slow to start node; an impatient probe would fail the gate.
            timeout = 60,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("node could not be started")
    if result.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _build_harness(run_dir: Path):
    """Slice autoLoadSmallestModel and its helpers verbatim out of the adapter."""
    lines = ADAPTER.read_text(encoding = "utf-8").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("const MAX_AUTO_LOAD_ATTEMPTS")),
        None,
    )
    end = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("export function createOpenAIStreamAdapter")
        ),
        None,
    )
    assert (
        start is not None and end is not None and start < end
    ), "could not locate the auto-load region in chat-adapter.ts"
    body = "\n".join(lines[start:end])
    assert "async function autoLoadSmallestModel" in body
    # Anything the adapter imports and the sliced region uses has to exist in the
    # preamble. Otherwise it is a bare ReferenceError at runtime, the retry loop
    # catches it and scores it as a failed load, and the scenario fails as a
    # wrong-model assertion that says nothing about the real cause. That is what
    # #7699 did by adding a syncModelCapabilities call here.
    imported = set()
    for match in re.finditer(r"^import\s+\{([^}]*)\}\s+from", "\n".join(lines), re.M):
        for spec in match.group(1).split(","):
            spec = spec.strip()
            # `import { type Foo }` is erased at runtime, so it can never be a
            # ReferenceError and must not be demanded of the harness.
            if not spec or spec.startswith("type "):
                continue
            name = spec.split(" as ")[-1].strip()
            if name:
                imported.add(name)
    # Any mention, not just a call. useChatRuntimeStore, toast and GPU_LAYERS_AUTO
    # are all used in this region without a following paren, and each would be the
    # same ReferenceError; they pass today only because the preamble happens to
    # define them. Comments are stripped first so a name discussed in prose does
    # not count as a use.
    code = re.sub(r"/\*.*?\*/", "", body, flags = re.S)
    code = re.sub(r"//[^\n]*", "", code)
    missing = sorted(
        name for name in imported
        if re.search(rf"\b{re.escape(name)}\b", code)
        and not re.search(
            rf"^(?:export\s+)?(?:async\s+)?"
            rf"(?:function\s+|const\s+|let\s+|var\s+|class\s+){re.escape(name)}\b",
            PREAMBLE,
            re.M,
        )
    )
    assert not missing, (
        f"the sliced region uses {missing}, imported by chat-adapter.ts but absent "
        "from this harness. Add a stub to PREAMBLE, or these scenarios will fail as "
        "wrong-model assertions rather than saying what is actually missing."
    )
    (run_dir / "harness.ts").write_text(
        "// @ts-nocheck\n" + PREAMBLE + "\n" + body + "\nexport { autoLoadSmallestModel };\n",
        encoding = "utf-8",
    )


def _run(scenario_expr: str) -> dict:
    _require_node()
    # Its own directory per invocation: a shared file lets one runner read another's rewrite.
    TEMP.mkdir(parents = True, exist_ok = True)
    run_dir = Path(tempfile.mkdtemp(prefix = "run", dir = TEMP))
    _build_harness(run_dir)
    script = (
        textwrap.dedent(
            """
        // @ts-nocheck
        import { autoLoadSmallestModel, setScenario, EVENTS } from "./harness.ts";
        """
        )
        + SCENARIO_HELPERS
        + textwrap.dedent(
            f"""
        setScenario({scenario_expr});
        const result = await autoLoadSmallestModel();
        console.log(JSON.stringify({{ result, events: EVENTS }}));
        """
        )
    )
    (run_dir / "run.mts").write_text(script, encoding = "utf-8")
    completed = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(run_dir),
        capture_output = True,
        text = True,
        timeout = 60,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert completed.returncode == 0, f"stderr: {completed.stderr}\nstdout: {completed.stdout}"
    last = [line for line in completed.stdout.strip().splitlines() if line.strip()][-1]
    return json.loads(last)


def _loaded_paths(out: dict) -> list[str]:
    return [event["model_path"] for event in out["events"] if event["kind"] == "loadModel"]


def _toasts(out: dict, kind: str) -> list[dict]:
    return [event for event in out["events"] if event["kind"] == kind]


def test_failed_cached_load_does_not_download_the_default_model():
    """The reported case: the only cached repo enumerates fine but its load OOMs, so auto-load stops
    there rather than fetch a model the user never asked for."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: (p) => p.model_path === GEMMA.repo_id ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == ["unsloth/gemma-4-26B-A4B-it-qat-GGUF"]
    assert DEFAULT_MODEL not in _loaded_paths(out)
    assert out["result"]["loaded"] is False
    assert _toasts(out, "toast.success") == []
    assert not any(
        "Downloading a small model" in event["msg"] for event in _toasts(out, "toast.message")
    )


def test_failed_cached_load_surfaces_the_backend_reason():
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: () => new Error(OOM) })"
    )

    [error] = _toasts(out, "toast.error")
    assert error["msg"] == "Could not load unsloth/gemma-4-26B-A4B-it-qat-GGUF (UD-Q4_K_XL)"
    assert "out of memory" in error["description"]


def test_load_rejection_without_a_message_still_names_the_model():
    """Old backends and non-Error throws carry no detail; still name the model, not the default."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: () => new Error('') })"
    )

    assert DEFAULT_MODEL not in _loaded_paths(out)
    [error] = _toasts(out, "toast.error")
    assert "unsloth/gemma-4-26B-A4B-it-qat-GGUF (UD-Q4_K_XL)" in error["msg"]
    assert error["description"]


def test_empty_device_still_downloads_the_default_model():
    """Nothing cached means nothing failed, so the download path is untouched."""
    out = _run("scenario({})")

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.error") == []
    assert [event["msg"] for event in _toasts(out, "toast.success")] == [
        "Loaded Qwen3.5-4B-MTP (UD-Q4_K_XL)"
    ]


def test_enumeration_failure_still_downloads_the_default_model():
    """A repo whose variants cannot be listed never reached /load, so it keeps falling through."""
    out = _run("scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: 'throw' } })")

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True


def test_consent_gated_candidate_still_downloads_the_default_model():
    """trust_remote_code / security review block before the load: a deferral, not a failure."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " validate: (p) => p.model_path === GEMMA.repo_id"
        " ? { requires_trust_remote_code: true, requires_security_review: false,"
        " requires_transformers_upgrade: false } : VALIDATE_OK() })"
    )

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True


def test_attempt_cap_still_gates_the_default_download():
    """Four broken repos: the sweep tries smaller candidates, stops at three, never reaches the Hub."""
    out = _run(
        "scenario({ ggufRepos: [1, 2, 3, 4].map((i) => ({ ...GEMMA, repo_id: `r${i}`,"
        " load_id: `r${i}`, size_bytes: i })),"
        " variants: Object.fromEntries([1, 2, 3, 4].map((i) => [`r${i}`, GEMMA_VARIANTS])),"
        " load: () => new Error(OOM) })"
    )

    assert _loaded_paths(out) == ["r1", "r2", "r3"]
    assert DEFAULT_MODEL not in _loaded_paths(out)


def test_a_later_cached_model_can_still_load_after_an_earlier_failure():
    """One broken repo must not veto a working one; the failure toast is only for an empty sweep."""
    out = _run(
        "scenario({ ggufRepos: [1, 2].map((i) => ({ ...GEMMA, repo_id: `r${i}`,"
        " load_id: `r${i}`, size_bytes: i })),"
        " variants: { r1: GEMMA_VARIANTS, r2: GEMMA_VARIANTS },"
        " load: (p) => p.model_path === 'r1' ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == ["r1", "r2"]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.error") == []


def test_reported_failure_is_flagged_so_callers_drop_the_generic_advice():
    """Both send paths toast a generic "No model loaded" on loaded: false, burying the detailed one."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: () => new Error(OOM) })"
    )

    assert out["result"]["loaded"] is False
    assert out["result"]["loadFailureReported"] is True


def test_empty_device_does_not_flag_a_reported_failure():
    """Nothing was attempted, so the callers must keep their generic advice."""
    out = _run("scenario({ ggufRepos: [], models: [] })")

    assert out["result"].get("loadFailureReported") is not True


def test_a_resume_only_cached_row_is_skipped_so_the_default_still_downloads():
    """An interrupted download leaves a partial / can_chat=false row for resume and delete; sweeping
    it anyway would let its rejection suppress the default download."""
    out = _run(
        "scenario({ modelRepos: [{ repo_id: 'org/half', load_id: 'org/half',"
        " size_bytes: 1, partial: true, capabilities: { can_chat: false } }],"
        " load: (p) => p.model_path === 'org/half'"
        " ? new Error('config.json not found') : LOADED(p) })"
    )

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.error") == []


def test_a_can_chat_false_cached_row_is_skipped_on_its_own():
    """The two fields are set independently, so can_chat alone must be enough to skip a row."""
    out = _run(
        "scenario({ ggufRepos: [{ ...GEMMA, capabilities: { can_chat: false } }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: (p) => p.model_path === GEMMA.repo_id ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True


def test_the_last_used_model_is_skipped_when_its_row_went_partial():
    """The last-used shortcut reads the same rows, so a cancelled update must not spend an attempt."""
    out = _run(
        "scenario({ ggufRepos: [{ ...GEMMA, partial: true }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " lastLoaded: { id: GEMMA.repo_id, kind: 'gguf', ggufVariant: 'UD-Q4_K_XL' },"
        " load: (p) => p.model_path === GEMMA.repo_id ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True


def test_a_complete_cached_row_is_still_attempted():
    """Guard on the filter: a healthy row is still swept, and an older backend omitting the fields
    keeps its behaviour."""
    out = _run(
        "scenario({ ggufRepos: [{ ...GEMMA, partial: false, capabilities: { can_chat: true } }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })"
    )

    assert _loaded_paths(out) == [GEMMA_REPO]
    assert out["result"]["loaded"] is True


def test_a_rejected_validation_does_not_download_the_default_model():
    """A cached candidate can be refused by /validate rather than /load. The sweep's catches are
    bare, so an unrecorded rejection reads as an empty device and fetches an unasked-for model."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " validate: (p) => p.model_path === GEMMA.repo_id"
        " ? new Error('snapshot no longer exists') : ({}) })"
    )

    assert DEFAULT_MODEL not in _loaded_paths(out)
    assert out["result"]["loaded"] is False
    assert out["result"]["loadFailureReported"] is True
    [error] = _toasts(out, "toast.error")
    assert error["msg"] == "Could not load unsloth/gemma-4-26B-A4B-it-qat-GGUF (UD-Q4_K_XL)"
    assert "snapshot no longer exists" in error["description"]


def test_a_rejected_validation_still_lets_a_later_cached_model_load():
    """Control for the test above: recording must not end the sweep; only a declined dialog does."""
    out = _run(
        "scenario({ ggufRepos: [1, 2].map((i) => ({ ...GEMMA, repo_id: `r${i}`,"
        " load_id: `r${i}`, size_bytes: i })),"
        " variants: { r1: GEMMA_VARIANTS, r2: GEMMA_VARIANTS },"
        " validate: (p) => p.model_path === 'r1'"
        " ? new Error('snapshot no longer exists') : ({}) })"
    )

    assert _loaded_paths(out) == ["r2"]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.error") == []
