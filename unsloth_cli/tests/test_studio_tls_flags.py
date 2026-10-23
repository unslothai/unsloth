# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio --ssl-certfile/--ssl-keyfile`: a bad pair fails before the password gate, a good one is forwarded."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


class _ExecCaptured(Exception):
    pass


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "host.example.ts.net")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes = 5))
        .not_valid_after(now + datetime.timedelta(days = 1))
        .sign(key, hashes.SHA256())
    )
    crt, key_path = tmp_path / "host.crt", tmp_path / "host.key"
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return crt, key_path


def _invoke(monkeypatch, args):
    import typer as _typer
    from unsloth_cli.commands import studio as studio_mod

    calls = {"exec": [], "auth": []}
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(studio_mod, "_ensure_studio_env_exported", lambda: None)
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: Path("/fake/venv/bin/python"))
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: Path("/fake/frontend/dist"))
    for name in (
        "_apply_supplied_password_before_launch",
        "_enforce_password_change_before_exposure",
    ):
        monkeypatch.setattr(studio_mod, name, lambda *a, _n = name, **k: calls["auth"].append(_n))

    def fake_execvp(file, argv):
        calls["exec"].append(list(argv))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)
    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, args, catch_exceptions = True)
    return result, calls


@pytest.mark.parametrize(
    "case",
    ["cert_only", "key_only", "mismatched_key", "missing_file", "with_cloudflare", "with_secure"],
)
def test_bad_tls_launch_fails_before_touching_credentials(monkeypatch, tmp_path, case):
    crt, key = _pair(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _, other_key = _pair(other)
    args = ["-H", "0.0.0.0"] + {
        "cert_only": ["--ssl-certfile", str(crt)],
        "key_only": ["--ssl-keyfile", str(key)],
        "mismatched_key": ["--ssl-certfile", str(crt), "--ssl-keyfile", str(other_key)],
        "missing_file": ["--ssl-certfile", str(crt), "--ssl-keyfile", str(tmp_path / "absent.key")],
        "with_cloudflare": ["--cloudflare", "--ssl-certfile", str(crt), "--ssl-keyfile", str(key)],
        "with_secure": ["--secure", "--ssl-certfile", str(crt), "--ssl-keyfile", str(key)],
    }[case]
    result, calls = _invoke(monkeypatch, args)
    assert result.exit_code == 2, result.output
    assert calls == {"exec": [], "auth": []}


def test_valid_pair_is_forwarded_to_the_server(monkeypatch, tmp_path):
    crt, key = _pair(tmp_path)
    _, calls = _invoke(
        monkeypatch, ["-H", "0.0.0.0", "--ssl-certfile", str(crt), "--ssl-keyfile", str(key)]
    )
    assert len(calls["exec"]) == 1, calls
    argv = calls["exec"][0]
    assert argv[argv.index("--ssl-certfile") + 1] == str(crt)
    assert argv[argv.index("--ssl-keyfile") + 1] == str(key)
    assert calls["auth"] == [
        "_apply_supplied_password_before_launch",
        "_enforce_password_change_before_exposure",
    ]
