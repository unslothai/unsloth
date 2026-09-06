# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
import pytest

from core.rag import config, folder_sync, parsers
import utils.native_path_leases as leases
from routes.rag import _save_native_path_upload

SECRET = b"n" * 32


@pytest.fixture(autouse = True)
def _lease_secret(monkeypatch):
    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV,
        base64.urlsafe_b64encode(SECRET).decode("ascii").rstrip("="),
    )
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)
    yield
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(
    path,
    *,
    operation = "attach",
    path_kind = "attachment",
    identity_options = None,
    nonce = None,
    secret = SECRET,
):
    st = os.stat(path)
    path_type = "directory" if os.path.isdir(path) else "file"
    now_ms = int(time.time() * 1000)
    identities = identity_options or ((st.st_dev, st.st_ino),)
    payload = {
        "version": 1,
        "operation": operation,
        "canonical_path": str(path),
        "path_kind": path_kind,
        "path_type": path_type,
        "source_kind": "drop",
        "token_id_hash": hashlib.sha256(b"path_token").hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": nonce or os.urandom(16).hex(),
        "display_label": os.path.basename(path),
        "size_bytes": st.st_size if path_type == "file" else None,
        "modified_ms": int(st.st_mtime_ns // 1_000_000) if path_type == "file" else None,
        "device_id": ":".join(format(identity[0], "x") for identity in identities),
        "file_id": ":".join(format(identity[1], "x") for identity in identities),
    }
    payload_b64 = _b64(json.dumps(payload).encode("utf-8"))
    signature = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64(signature)}"


def test_config_source_extensions_exist():
    assert ".js" in config.SOURCE_TEXT_EXTS
    assert ".ts" in config.SOURCE_TEXT_EXTS
    assert ".py" in config.SOURCE_TEXT_EXTS
    assert ".cs" in config.SOURCE_TEXT_EXTS
    assert ".php" in config.SOURCE_TEXT_EXTS
    assert ".json" in config.SOURCE_TEXT_EXTS
    assert ".yaml" in config.SOURCE_TEXT_EXTS
    assert ".rs" in config.SOURCE_TEXT_EXTS
    assert ".go" in config.SOURCE_TEXT_EXTS
    assert ".cpp" in config.SOURCE_TEXT_EXTS

    assert config.SOURCE_TEXT_EXTS.issubset(config.TEXT_EXTS)
    assert config.SOURCE_TEXT_EXTS.issubset(config.ALL_UPLOAD_EXTS)


def test_parsers_text_file_supports_source_code(tmp_path):
    source_files = {
        "app.js": "function add(a, b) { return a + b; }",
        "Program.cs": "using System;\nclass Program { static void Main() { Console.WriteLine(123); } }",
        "main.py": "def hello():\n    return 'unsloth'",
        "index.php": "<?php echo 'hello world'; ?>",
        "deploy.yaml": "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test",
        "config.json": '{"name": "unsloth", "version": "1.0.0"}',
        "service.ts": "export interface Service { run(): void; }",
    }

    for name, content in source_files.items():
        file_path = tmp_path / name
        file_path.write_text(content, encoding = "utf-8")
        pages = parsers.parse(str(file_path))
        assert len(pages) == 1
        assert pages[0].text == content, f"Failed parsing {name}"


def test_parsers_text_file_rejects_binary(tmp_path):
    binary_file = tmp_path / "fake.py"
    binary_file.write_bytes(b"import sys\n\x00\x01\x02binary")
    with pytest.raises(ValueError, match = "unsupported binary content"):
        parsers.parse(str(binary_file))


def test_parsers_text_file_strips_bom(tmp_path):
    bom_file = tmp_path / "script.py"
    bom_file.write_bytes(b"\xef\xbb\xbfdef test(): pass")
    pages = parsers.parse(str(bom_file))
    assert len(pages) == 1
    assert pages[0].text == "def test(): pass"


def test_folder_sync_scan_finds_source_code_and_ignores_dirs(tmp_path):
    # Setup mock workspace
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "app.js").write_text("console.log('test')", encoding = "utf-8")
    (src_dir / "service.ts").write_text("const x = 1;", encoding = "utf-8")
    (src_dir / "main.py").write_text("print('test')", encoding = "utf-8")
    (src_dir / "Program.cs").write_text("// C# code", encoding = "utf-8")
    (src_dir / "index.php").write_text("<?php phpinfo(); ?>", encoding = "utf-8")
    (src_dir / "config.json").write_text("{}", encoding = "utf-8")
    (src_dir / "deploy.yaml").write_text("k: v", encoding = "utf-8")
    (src_dir / "unsupported.exe").write_bytes(b"binary")

    # Ignored directories
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "dep.js").write_text("console.log('dep')", encoding = "utf-8")

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding = "utf-8")

    found_dict, _ = folder_sync._scan(str(tmp_path))
    found_keys = {k.replace("\\", "/") for k in found_dict.keys()}

    # Included files
    assert "src/app.js" in found_keys
    assert "src/service.ts" in found_keys
    assert "src/main.py" in found_keys
    assert "src/Program.cs" in found_keys
    assert "src/index.php" in found_keys
    assert "src/config.json" in found_keys
    assert "src/deploy.yaml" in found_keys

    # Excluded files
    assert "src/unsupported.exe" not in found_keys
    assert "node_modules/dep.js" not in found_keys
    assert ".git/config" not in found_keys


def test_routes_accept_source_file_native_drop(rag_home, tmp_path):
    source = tmp_path / "index.php"
    source.write_text("<?php echo 'drop works'; ?>", encoding = "utf-8")

    stored_path, filename = _save_native_path_upload(_sign(source))
    assert filename == "index.php"
    assert os.path.isfile(stored_path)
    with open(stored_path, encoding = "utf-8") as f:
        assert f.read() == "<?php echo 'drop works'; ?>"


def test_start_ingestion_validates_extensions(rag_home, tmp_path):
    from core.rag import ingestion, store
    scope = store.project_scope("proj-1")

    # Unsupported file type raises ValueError
    bad_path = tmp_path / "binary.exe"
    bad_path.write_bytes(b"MZ...")
    with pytest.raises(ValueError, match = "unsupported file type: .exe"):
        ingestion.start_ingestion(scope, "proj-1", None, "binary.exe", str(bad_path))

    # Supported source code file passes extension validation
    path = tmp_path / "service.ts"
    path.write_text("export interface User { id: string; }", encoding = "utf-8")
    try:
        ingestion.start_ingestion(scope, "proj-1", None, "service.ts", str(path))
    except ValueError as exc:
        pytest.fail(f"start_ingestion incorrectly rejected source file extension: {exc}")
    except Exception:
        # Passed extension check; DB error is expected when sqlite-vec native lib is uninitialized in mock pytest
        pass
