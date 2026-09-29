# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model text stays intact when it carries non-ASCII.

``open()`` and ``Path.read_text()`` fall back to ``locale.getencoding()`` when
no ``encoding`` is passed. On Windows that is the ANSI codepage, not UTF-8, so
a chat template or model config holding ``ä ö ü → 世`` mojibakes or raises
``UnicodeDecodeError``. These files are UTF-8, so the reads must say so.

Each fixture writes raw UTF-8 (``ensure_ascii = False``), matching what
Hugging Face actually ships, rather than ASCII ``\\uXXXX`` escapes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent


def test_config_json_round_trips_non_ascii(tmp_path: Path) -> None:
    from utils import transformers_version

    name = "Modell für Grüße 世界"
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "_name_or_path": name}, ensure_ascii = False),
        encoding = "utf-8",
    )
    transformers_version._config_json_cache.clear()

    cfg = transformers_version._load_config_json(str(tmp_path))

    assert cfg is not None
    assert cfg["_name_or_path"] == name


def test_tokenizer_config_round_trips_non_ascii_chat_template(tmp_path: Path) -> None:
    """Chat templates commonly hold ``→`` and smart quotes, which cp1252 mangles."""
    from utils import transformers_version

    template = "{{ '→ Grüße 世界' }}"
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"tokenizer_class": "TokenizersBackend", "chat_template": template},
            ensure_ascii = False,
        ),
        encoding = "utf-8",
    )
    transformers_version._tokenizer_class_cache.clear()

    assert transformers_version._check_tokenizer_config_needs_v5(str(tmp_path)) is True


def test_config_json_survives_a_utf8_bom(tmp_path: Path) -> None:
    """Notepad wrote "UTF-8 with BOM" by default for years, so hand-edited
    configs on Windows carry one. Plain utf-8 keeps the BOM and json.load then
    fails on it; utf-8-sig strips it and is identical otherwise."""
    from utils import transformers_version

    name = "Grüße 世界"
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "_name_or_path": name}, ensure_ascii = False),
        encoding = "utf-8-sig",
    )
    transformers_version._config_json_cache.clear()

    cfg = transformers_version._load_config_json(str(tmp_path))

    assert cfg is not None
    assert cfg["_name_or_path"] == name


def test_remote_code_scan_reads_non_ascii_sources(tmp_path: Path) -> None:
    """A German Windows profile also puts umlauts in the model sources scanned."""
    from utils.security import remote_code_scan

    source = "# Grüße über Öl\nVALUE = '世界'\n"
    # newline = "" pins the bytes on disk, so Windows line end translation cannot make the
    # read back differ by \r. open() because Path.write_text() only grew newline in 3.10.
    with open(
        tmp_path / "modeling_custom.py",
        "w",
        encoding = "utf-8",
        newline = "",
    ) as handle:
        handle.write(source)

    files = remote_code_scan.repo_remote_code_files(str(tmp_path))

    assert files["modeling_custom.py"] == source


def test_model_config_reads_do_not_rely_on_the_locale_encoding(tmp_path: Path) -> None:
    """The reads above pass anywhere the locale is already UTF-8, which hides
    the Windows bug on Linux and macOS. ``-X warn_default_encoding`` makes
    CPython flag any text I/O that falls back to the locale, so this fails on
    every platform if an ``encoding`` argument goes missing again."""
    # The readers swallow exceptions, so record the warnings instead of raising.
    script = textwrap.dedent(
        f"""
        import sys, warnings
        sys.path.insert(0, {str(BACKEND_ROOT)!r})
        from utils import transformers_version

        target = {str(tmp_path)!r}
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            transformers_version._config_json_cache.clear()
            transformers_version._tokenizer_class_cache.clear()
            assert transformers_version._load_config_json(target) is not None
            assert transformers_version._check_tokenizer_config_needs_v5(target) is True

        missing = [str(w.message) for w in caught if w.category is EncodingWarning]
        if missing:
            sys.exit("text I/O fell back to the locale encoding: " + "; ".join(missing))
        """
    )
    for name, payload in (
        ("config.json", {"model_type": "llama", "_name_or_path": "Grüße"}),
        ("tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"}),
    ):
        (tmp_path / name).write_text(json.dumps(payload, ensure_ascii = False), encoding = "utf-8")

    result = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-c", script],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 120,
    )

    assert result.returncode == 0, result.stderr


def test_utf8_child_env_round_trips_non_ascii(tmp_path: Path) -> None:
    """A Python child encodes stdout with its locale unless told otherwise, so
    reading its pipe as utf-8 needs the child told to emit utf-8."""
    from utils.child_stdio import utf8_child_env

    payload = "Grüße über Öl → 世界"
    child = tmp_path / "child.py"
    child.write_text("import sys\nsys.stdout.write(" + repr(payload) + ")\n", encoding = "utf-8")

    env = utf8_child_env()
    assert env["PYTHONIOENCODING"] == "utf-8"

    proc = subprocess.run(
        [sys.executable, str(child)],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = env,
        timeout = 120,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == payload


def test_python_children_are_told_to_emit_utf8() -> None:
    """Any child we decode as utf-8 must also be told to write utf-8, or a
    cp1252 console silently mangles what it prints."""
    import ast

    offenders: list[str] = []
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        parts = path.relative_to(BACKEND_ROOT).parts
        if any(p in ("tests", "node_modules", "plugins", "__pycache__") for p in parts):
            continue
        source = path.read_text(encoding = "utf-8")
        for node in ast.walk(ast.parse(source, filename = str(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in ("run", "Popen")):
                continue
            segment = ast.get_source_segment(source, node) or ""
            if "sys.executable" not in segment or 'encoding = "utf-8"' not in segment:
                continue
            if "utf8_child_env" in segment or "PYTHONIOENCODING" in segment:
                continue
            offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        "these spawn a Python child and decode it as utf-8 without setting the "
        "child's own stdio encoding; wrap env in utf8_child_env():\n  " + "\n  ".join(offenders)
    )
