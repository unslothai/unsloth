import json
import os
import shutil
import subprocess
import tempfile
import uuid
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def _source_path(relative_path: str) -> Path:
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


PRESET_POLICY = _source_path("studio/frontend/src/features/chat/presets/preset-policy.ts")
RUNTIME_TYPES = _source_path("studio/frontend/src/features/chat/types/runtime.ts")
TEMP = WORKDIR / "temp" / "chat_preset_builtin_invariants"


def _require_node():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not PRESET_POLICY.exists() or not RUNTIME_TYPES.exists():
        pytest.skip("studio chat sources not present")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if result.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _write_atomic(path: Path, text: str):
    """Write through a unique temp file and os.replace.

    register.mjs and loader.mjs are shared by every _run, and write_text truncates
    before it writes, so rewriting one while another worker's node process is
    importing it can hand that process an empty or partial module. Contents are
    constant, so the rename leaves every reader a whole file.
    """
    fd, tmp = tempfile.mkstemp(dir = str(path.parent), prefix = path.name, suffix = ".tmp")
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _ensure_harness():
    TEMP.mkdir(parents = True, exist_ok = True)
    _write_atomic(
        TEMP / "register.mjs",
        "import { register } from 'node:module';\nregister('./loader.mjs', import.meta.url);\n",
    )
    _write_atomic(
        TEMP / "loader.mjs",
        "export function resolve(specifier, context, next) {\n"
        "  if (specifier.endsWith('/types/runtime')) return next(specifier + '.ts', context);\n"
        "  return next(specifier, context);\n"
        "}\n",
    )


def _run(script: str):
    _require_node()
    _ensure_harness()
    # Unique per call: a shared run.mts let two xdist workers interleave write and exec, so one ran the other's script
    # (5-6 of these 9 failed on every -n 4 run).
    # A unique name, not a per-call dir like _node_harness.py uses: these scripts reach the sources by a path relative
    # to TEMP, so an extra level breaks every import.
    script_path = TEMP / f"run_{uuid.uuid4().hex}.mts"
    script_path.write_text(script, encoding = "utf-8")
    env = dict(os.environ, NODE_NO_WARNINGS = "1")
    result = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--import=./register.mjs",
            "--no-warnings",
            script_path.name,
        ],
        cwd = str(TEMP),
        capture_output = True,
        text = True,
        timeout = 30,
        env = env,
    )
    script_path.unlink(missing_ok = True)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    last = [line for line in result.stdout.strip().splitlines() if line.strip()][-1]
    return json.loads(last)


def _policy_path():
    return os.path.relpath(PRESET_POLICY, TEMP).replace("\\", "/")


def _runtime_path():
    return os.path.relpath(RUNTIME_TYPES, TEMP).replace("\\", "/")


def test_default_builtin_matches_default_inference_params():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ BUILTIN_PRESETS, isSamePresetConfig }} from "{_policy_path()}";
            import {{ DEFAULT_INFERENCE_PARAMS }} from "{_runtime_path()}";
            const def = BUILTIN_PRESETS.find((p) => p.name === "Default");
            console.log(JSON.stringify({{
                found: !!def,
                matches: def ? isSamePresetConfig(def.params, DEFAULT_INFERENCE_PARAMS) : null,
            }}));
            """
        )
    )
    assert out["found"] is True
    assert out["matches"] is True


def test_is_same_preset_config_detects_temperature_edit():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ BUILTIN_PRESETS, isSamePresetConfig }} from "{_policy_path()}";
            const def = BUILTIN_PRESETS.find((p) => p.name === "Default");
            const edited = {{ ...def.params, temperature: def.params.temperature + 0.1 }};
            console.log(JSON.stringify({{ same: isSamePresetConfig(def.params, edited) }}));
            """
        )
    )
    assert out["same"] is False


def test_is_same_preset_config_detects_system_prompt_edit():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ BUILTIN_PRESETS, isSamePresetConfig }} from "{_policy_path()}";
            const def = BUILTIN_PRESETS.find((p) => p.name === "Default");
            const edited = {{ ...def.params, systemPrompt: "you are a pirate" }};
            console.log(JSON.stringify({{ same: isSamePresetConfig(def.params, edited) }}));
            """
        )
    )
    assert out["same"] is False


def test_is_same_preset_config_ignores_checkpoint_difference():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ BUILTIN_PRESETS, isSamePresetConfig }} from "{_policy_path()}";
            const def = BUILTIN_PRESETS.find((p) => p.name === "Default");
            const withCheckpoint = {{ ...def.params, checkpoint: "meta-llama/Llama-3-8B" }};
            console.log(JSON.stringify({{ same: isSamePresetConfig(def.params, withCheckpoint) }}));
            """
        )
    )
    assert out["same"] is True


def test_is_same_preset_config_ignores_model_owned_fields():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ BUILTIN_PRESETS, isSamePresetConfig }} from "{_policy_path()}";
            const def = BUILTIN_PRESETS.find((p) => p.name === "Default");
            const edited = {{
                ...def.params,
                maxSeqLength: def.params.maxSeqLength + 1024,
                trustRemoteCode: !def.params.trustRemoteCode,
            }};
            console.log(JSON.stringify({{ same: isSamePresetConfig(def.params, edited) }}));
            """
        )
    )
    assert out["same"] is True


def test_preset_owned_config_key_ignores_model_owned_fields():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ BUILTIN_PRESETS, getPresetOwnedConfigKey }} from "{_policy_path()}";
            const def = BUILTIN_PRESETS.find((p) => p.name === "Default");
            const edited = {{
                ...def.params,
                checkpoint: "foo/bar",
                maxSeqLength: def.params.maxSeqLength + 1024,
                trustRemoteCode: !def.params.trustRemoteCode,
            }};
            console.log(JSON.stringify({{
                same: getPresetOwnedConfigKey(def.params) === getPresetOwnedConfigKey(edited),
            }}));
            """
        )
    )
    assert out["same"] is True


def test_to_preset_params_strips_model_owned_fields():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ toPresetParams }} from "{_policy_path()}";
            const sanitized = toPresetParams({{
                temperature: 0.9,
                topP: 0.8,
                topK: 40,
                minP: 0.05,
                repetitionPenalty: 1.1,
                presencePenalty: 0.4,
                maxSeqLength: 16384,
                maxTokens: 2048,
                systemPrompt: "hello",
                checkpoint: "foo/bar",
                trustRemoteCode: true,
            }});
            console.log(JSON.stringify({{
                checkpoint: sanitized.checkpoint,
                trustRemoteCode: sanitized.trustRemoteCode,
                maxSeqLength: sanitized.maxSeqLength,
                maxTokens: sanitized.maxTokens,
                systemPrompt: sanitized.systemPrompt,
            }}));
            """
        )
    )
    assert out["checkpoint"] == ""
    assert out["trustRemoteCode"] is False
    assert out["maxSeqLength"] == 4096
    assert out["maxTokens"] == 2048
    assert out["systemPrompt"] == "hello"


def test_apply_preset_params_preserves_model_owned_fields():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ applyPresetParams }} from "{_policy_path()}";
            const samplingPreset = {{
                temperature: 1.5,
                topP: 1,
                topK: 0,
                minP: 0.1,
                repetitionPenalty: 1,
                presencePenalty: 0,
                maxSeqLength: 4096,
                maxTokens: 2048,
                systemPrompt: "",
                checkpoint: "",
                trustRemoteCode: false,
            }};
            const applied = applyPresetParams(
                {{
                    temperature: 0.6,
                    topP: 0.95,
                    topK: 20,
                    minP: 0.01,
                    repetitionPenalty: 1.0,
                    presencePenalty: 0.0,
                    maxSeqLength: 16384,
                    maxTokens: 8192,
                    systemPrompt: "keep me?",
                    checkpoint: "foo/bar",
                    trustRemoteCode: true,
                }},
                samplingPreset,
            );
            console.log(JSON.stringify({{
                checkpoint: applied.checkpoint,
                trustRemoteCode: applied.trustRemoteCode,
                maxSeqLength: applied.maxSeqLength,
                temperature: applied.temperature,
                topK: applied.topK,
            }}));
            """
        )
    )
    assert out["checkpoint"] == "foo/bar"
    assert out["trustRemoteCode"] is True
    assert out["maxSeqLength"] == 16384
    assert out["temperature"] == 1.5
    assert out["topK"] == 0


def test_default_is_only_builtin_preset():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ BUILTIN_PRESETS }} from "{_policy_path()}";
            console.log(JSON.stringify({{
                names: BUILTIN_PRESETS.map((p) => p.name),
            }}));
            """
        )
    )
    assert out["names"] == ["Default"]
