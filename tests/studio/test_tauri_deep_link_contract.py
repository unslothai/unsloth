# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Desktop deep-link configuration, routing, and validation contracts."""

import json
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = pytest.importorskip("tomli")


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend"
TAURI = REPO / "studio/src-tauri"
PARSER = FRONTEND / "src/features/deep-links/parse-deep-link.ts"
INTENT_GATE = FRONTEND / "src/features/deep-links/deep-link-intent.ts"
GGUF_FILENAME = FRONTEND / "src/features/hub/lib/gguf-filename.ts"


def test_unsloth_deep_link_parser_guardrails(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")

    (tmp_path / "parse-deep-link.ts").write_text(
        PARSER.read_text(encoding = "utf-8"), encoding = "utf-8"
    )

    (tmp_path / "gguf-filename.ts").write_text(
        GGUF_FILENAME.read_text(encoding = "utf-8"), encoding = "utf-8"
    )

    (tmp_path / "deep-link-intent.ts").write_text(
        INTENT_GATE.read_text(encoding = "utf-8"), encoding = "utf-8"
    )
    script = textwrap.dedent("""
        import assert from "node:assert/strict";
        import { parseUnslothDeepLink } from "./parse-deep-link.ts";

        import { createDeepLinkIntentGate } from "./deep-link-intent.ts";
        import {
          ggufFilenamesMatch,
          ggufSelectionOverrideMatchesIntent,
        } from "./gguf-filename.ts";

        const valid = new Map([
          [
            "unsloth://open_from_hf?model=unsloth/Laguna-S-2.1-GGUF",
            { model: "unsloth/Laguna-S-2.1-GGUF" },
          ],
          [
            "unsloth://open_from_hf/?model=org/repo_name",
            { model: "org/repo_name" },
          ],
          [
            "unsloth://open_from_hf?model=org%2Frepo",
            { model: "org/repo" },
          ],
          [
            "unsloth://open_from_hf?model=unsloth/Laguna-S-2.1-GGUF&file=Laguna-S-2.1-UD-IQ3_XXS.gguf",
            {
              model: "unsloth/Laguna-S-2.1-GGUF",
              file: "Laguna-S-2.1-UD-IQ3_XXS.gguf",
            },
          ],
          [
            "unsloth://open_from_hf?file=weights%2Fmodel-Q4_K_M.gguf&model=org/repo",
            { model: "org/repo", file: "weights/model-Q4_K_M.gguf" },
          ],
          [
            `unsloth://open_from_hf?model=${"a".repeat(96)}/${"b".repeat(96)}`,
            { model: `${"a".repeat(96)}/${"b".repeat(96)}` },
          ],
        ]);
        for (const [url, intent] of valid) {
          assert.deepEqual(parseUnslothDeepLink(url), intent, url);
        }

        assert.equal(
          ggufFilenamesMatch(
            "weights/model-Q4_K_M-00002-of-00002.gguf",
            "weights/model-Q4_K_M-00001-of-00002.gguf",
          ),
          true,
        );
        assert.equal(
          ggufFilenamesMatch("model-Q4_K_M.GGUF", "model-q4_k_m.gguf"),
          true,
        );
        assert.equal(ggufFilenamesMatch("mmproj-F16.gguf", "model-F16.gguf"), false);

        assert.equal(ggufSelectionOverrideMatchesIntent("a.gguf", 2, "a.gguf", 2), true);
        assert.equal(ggufSelectionOverrideMatchesIntent("a.gguf", 2, "a.gguf", 1), false);

        let now = 1_000;
        const acceptIntent = createDeepLinkIntentGate(2_000, () => now);
        assert.equal(acceptIntent("org/repo", "a.gguf"), 1);
        assert.equal(acceptIntent("org/repo", "a.gguf"), null);
        assert.equal(acceptIntent("org/repo", "b.gguf"), 2);
        now = 3_000;
        assert.equal(acceptIntent("org/repo", "b.gguf"), 3);


        const invalid = [
          "",
          "https://open_from_hf?model=org/repo",
          "UNSLOTH://open_from_hf?model=org/repo",
          "unsloth://OPEN_FROM_HF?model=org/repo",
          "unsloth://open_from_hf/path?model=org/repo",
          "unsloth://open_from_hf/%2e%2e?model=org/repo",
          "unsloth://user@open_from_hf?model=org/repo",
          "unsloth://open_from_hf:42?model=org/repo",
          "unsloth://open_from_hf?model=org/repo#fragment",
          "unsloth://open_from_hf?model=org/repo&download=true",

          "unsloth://open_from_hf?model=org/repo&file=model.gguf&file=other.gguf",
          "unsloth://open_from_hf?model=org/repo&file=",
          "unsloth://open_from_hf?model=org/repo&file=../model.gguf",
          "unsloth://open_from_hf?model=org/repo&file=%2Fmodel.gguf",
          "unsloth://open_from_hf?model=org/repo&file=model.safetensors",
          "unsloth://open_from_hf?model=org/repo&model=other/repo",
          "unsloth://open_from_hf?model=repo",
          "unsloth://open_from_hf?model=org/repo/extra",
          "unsloth://open_from_hf?model=-org/repo",
          "unsloth://open_from_hf?model=org/repo.",

          "unsloth://open_from_hf?model=org/repo.git",
          "unsloth://open_from_hf?model=org/repo--name",
          "unsloth://open_from_hf?model=org/repo..name",
        ];
        for (const url of invalid) {
          assert.equal(parseUnslothDeepLink(url), null, url);
        }
    """)
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "--input-type=module"],
        input = script,
        cwd = tmp_path,
        capture_output = True,
        text = True,
        timeout = 30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"


def test_tauri_registers_only_the_unsloth_scheme() -> None:
    cargo = tomllib.loads((TAURI / "Cargo.toml").read_text(encoding = "utf-8"))
    dependencies = cargo["dependencies"]
    assert "tauri-plugin-deep-link" in dependencies
    single_instance = dependencies["tauri-plugin-single-instance"]
    assert isinstance(single_instance, dict)
    assert "deep-link" in single_instance.get("features", [])

    config = json.loads((TAURI / "tauri.conf.json").read_text(encoding = "utf-8"))
    assert config["plugins"]["deep-link"]["desktop"]["schemes"] == ["unsloth"]

    capabilities = json.loads((TAURI / "capabilities/default.json").read_text(encoding = "utf-8"))
    assert "deep-link:default" in capabilities["permissions"]
    assert "core:window:allow-unminimize" in capabilities["permissions"]

    main = (TAURI / "src/main.rs").read_text(encoding = "utf-8")
    assert main.index("tauri_plugin_single_instance::init") < main.index(
        "tauri_plugin_deep_link::init()"
    )
    assert "DeepLinkExt" in main
    assert "if let Err(error) = app.deep_link().register_all()" in main
    assert 'warn!("Failed to register deep-link handlers: {error}")' in main
    assert 'target_os = "linux"' in main
    desktop_template = TAURI / "linux/unsloth.desktop"
    assert config["bundle"]["linux"]["deb"]["desktopTemplate"] == "./linux/unsloth.desktop"
    desktop = desktop_template.read_text(encoding = "utf-8")
    assert "Exec={{exec}} %u" in desktop
    assert "MimeType=x-scheme-handler/unsloth;" in desktop
