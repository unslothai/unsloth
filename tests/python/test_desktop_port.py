# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "studio" / "src-tauri" / "src"


def _function(source, name):
    start = source.index(f"fn {name}(")
    return source[start : source.index("\n}", start) + 2]


@pytest.fixture(scope = "module")
def desktop_port_binary(tmp_path_factory):
    rustc = shutil.which("rustc")
    assert rustc is not None, "rustc is required to exercise the Desktop launch code"
    scratch = tmp_path_factory.mktemp("desktop-port")
    process = (SRC / "process.rs").read_text(encoding = "utf-8")
    owner = (SRC / "desktop_backend_owner.rs").read_text(encoding = "utf-8")
    start = process.index("pub fn start_backend(")
    start = process.index("{", start) + 1
    prefix = process[start : process.index("let _runtime_launch_guard", start)]
    probe = _function(owner, "probe_owned_backend_state_with_timeout")
    owner_prefix = probe[
        probe.index("{") + 1 : probe.index("    probe_owned_backend_state_on_ports")
    ]
    module = SRC / "desktop_port.rs"
    modules = f'#[path = "{module.as_posix()}"] mod desktop_port;\n' if module.exists() else ""
    constants = "\n".join(re.findall(r"^const DESKTOP_PORT_.*;$", owner, re.MULTILINE))
    source = (
        modules
        + constants
        + "\n"
        + _function(process, "backend_args")
        + "\n"
        + _function(owner, "desktop_candidate_ports")
        + "\nfn launch_port(port: u16) -> Result<u16, String> {"
        + prefix
        + "Ok(port)}\n"
        + "struct Owner(u16); impl Owner { fn requested_port(&self) -> u16 { self.0 } }\n"
        + "fn owned_ports(port: Option<u16>) -> Vec<u16> { let owner = Owner(9123);"
        + owner_prefix
        + "ports}\n"
        + "fn main() { match launch_port(8888) { Ok(port) => { "
        + 'println!("{}", backend_args(port).join("|")); '
        + 'println!("{}", desktop_candidate_ports().map(|p| p.to_string()).collect::<Vec<_>>().join(",")); '
        + 'if std::env::args().nth(1).as_deref() == Some("owner") { '
        + 'println!("{:?}", owned_ports(None)); } '
        + '}, Err(error) => {eprintln!("{error}"); std::process::exit(2);} } }\n'
    )
    path = scratch / "main.rs"
    path.write_text(source, encoding = "utf-8")
    binary = scratch / ("desktop-port.exe" if os.name == "nt" else "desktop-port")
    result = subprocess.run(
        [rustc, "--edition=2021", str(path), "-o", str(binary)],
        text = True,
        capture_output = True,
    )
    assert result.returncode == 0, result.stderr
    return binary


def _run(binary, value):
    env = os.environ.copy()
    env.pop("UNSLOTH_DESKTOP_PORT", None)
    if value is not None:
        env["UNSLOTH_DESKTOP_PORT"] = value
    return subprocess.run([str(binary)], env = env, capture_output = True, text = True)


@pytest.mark.parametrize("value", [None, "", "  "])
def test_default_launch_and_discovery_unchanged(desktop_port_binary, value):
    result = _run(desktop_port_binary, value)
    assert result.returncode == 0, result.stderr
    argv, ports = result.stdout.strip().splitlines()
    assert argv == "studio|--api-only|-H|127.0.0.1|-p|8888"
    assert [int(port) for port in ports.split(",")] == list(range(8888, 8909))


def test_custom_port_reaches_child_argv_and_discovery_on_repeated_launch(desktop_port_binary):
    for _ in range(3):
        result = _run(desktop_port_binary, " 9123 ")
        assert result.returncode == 0, result.stderr
        argv, ports = result.stdout.strip().splitlines()
        assert argv == "studio|--api-only|-H|127.0.0.1|-p|9123"
        discovered = [int(port) for port in ports.split(",")]
        assert set(range(9123, 9144)).issubset(discovered)
        assert set(range(8888, 8909)).issubset(discovered)
        assert len(discovered) == len(set(discovered))


@pytest.mark.parametrize(
    "value", ["0", "65536", "-1", "localhost:9123", "6000", "5999", "10080", "65535"]
)
def test_invalid_override_does_not_launch(desktop_port_binary, value):
    result = _run(desktop_port_binary, value)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "UNSLOTH_DESKTOP_PORT" in result.stderr


def test_portless_owner_uses_its_recorded_request_after_env_is_removed(desktop_port_binary):
    env = os.environ.copy()
    env.pop("UNSLOTH_DESKTOP_PORT", None)
    result = subprocess.run(
        [str(desktop_port_binary), "owner"], env = env, capture_output = True, text = True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == str(list(range(9123, 9144)))
