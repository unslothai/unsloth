import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]
PRESET_POLICY = (
    WORKDIR / "unsloth_repo/studio/frontend/src/features/chat/presets/preset-policy.ts"
)
RUNTIME_TYPES = (
    WORKDIR / "unsloth_repo/studio/frontend/src/features/chat/types/runtime.ts"
)
TEMP = WORKDIR / "temp" / "chat_preset_builtin_invariants"


def _require_node():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not PRESET_POLICY.exists() or not RUNTIME_TYPES.exists():
        pytest.skip("studio chat sources not present")


def _ensure_harness():
    TEMP.mkdir(parents = True, exist_ok = True)
    (TEMP / "register.mjs").write_text(
        "import { register } from 'node:module';\n"
        "register('./loader.mjs', import.meta.url);\n"
    )
    (TEMP / "loader.mjs").write_text(
        "export function resolve(specifier, context, next) {\n"
        "  if (specifier.endsWith('/types/runtime')) return next(specifier + '.ts', context);\n"
        "  return next(specifier, context);\n"
        "}\n"
    )


def _run(script: str):
    _require_node()
    _ensure_harness()
    script_path = TEMP / "run.mts"
    script_path.write_text(script)
    env = dict(os.environ, NODE_NO_WARNINGS = "1")
    result = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--import=./register.mjs",
            "--no-warnings",
            "run.mts",
        ],
        cwd = str(TEMP),
        capture_output = True,
        text = True,
        timeout = 30,
        env = env,
    )
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


def test_creative_and_precise_builtins_differ_from_default():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ BUILTIN_PRESETS, isSamePresetConfig }} from "{_policy_path()}";
            const def = BUILTIN_PRESETS.find((p) => p.name === "Default");
            const creative = BUILTIN_PRESETS.find((p) => p.name === "Creative");
            const precise = BUILTIN_PRESETS.find((p) => p.name === "Precise");
            console.log(JSON.stringify({{
                creativeDiffers: !isSamePresetConfig(def.params, creative.params),
                preciseDiffers: !isSamePresetConfig(def.params, precise.params),
            }}));
            """
        )
    )
    assert out["creativeDiffers"] is True
    assert out["preciseDiffers"] is True
