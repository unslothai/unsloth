# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Optional installed Pillow compatibility lane, not full runtime qualification.

Set UNSLOTH_TEST_PILLOW_RUNTIME to the selected isolated environment directory or
its Scripts/python.exe. The environment must already contain Pillow, the current
companion wheel, pinned pefile and broker dependencies. An explicitly requested
lane fails on missing prerequisites; this test never installs or replaces them.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
FORMATS = ("PNG", "JPEG", "GIF", "BMP", "TIFF", "WEBP", "AVIF")

PAYLOAD = r"""
from PIL import Image
import io, json

source = Image.new('RGB', (12, 12), (32, 64, 128))
results = {}
for format in ('PNG', 'JPEG', 'GIF', 'BMP', 'TIFF', 'WEBP', 'AVIF'):
    data = io.BytesIO()
    source.save(data, format=format)
    assert data.tell() > 0, format
    data.seek(0)
    with Image.open(data) as decoded:
        decoded.load()
        assert decoded.size == (12, 12), (format, decoded.size)
        assert decoded.format == format, (format, decoded.format)
        results[format] = list(decoded.size)
print(json.dumps(results, sort_keys=True), flush=True)
"""

BROKER = r"""
import hashlib, importlib.metadata, json, os, pathlib, subprocess, sys
environment_before = dict(os.environ)
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
root = pathlib.Path(sys.argv[2])
from core.inference.os_sandbox import ToolLaunchPlan, spawn_prepared_launch
from core.inference.windows_sandbox.artifacts import DISTRIBUTION, VERSION, PEFILE_VERSION
from core.inference.windows_sandbox.launch import prepare_python_launch
from core.inference.windows_sandbox.profiles import PYTHON_PROFILE
from core.inference.windows_sandbox.protocol import LaunchBinding

assert sys.prefix != sys.base_prefix, 'Select a dedicated isolated runtime environment'
assert importlib.metadata.version(DISTRIBUTION) == VERSION, 'Current companion wheel required'
assert importlib.metadata.version('pefile') == PEFILE_VERSION, 'Pinned PE parser required'
pillow = importlib.metadata.distribution('Pillow')
pillow_root = pathlib.Path(pillow.locate_file('PIL'))
images = tuple(sorted(pillow_root.glob('_imaging*.pyd')))
assert images, 'Selected environment is missing native Pillow images'
image_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in images}

work = root / 'work'
script = work / 'tool.py'
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work),
    env={}, execution_kind='python')
prepared = prepare_python_launch(spec, root / 'cache', timeout=120)
owner = prepared.spawn_callback.__self__
identity = owner.identity
manifest = pathlib.Path(identity.manifest_path)
assert prepared.execution_record is None, 'Compatibility probe must not claim qualification'
try:
    process = spawn_prepared_launch(prepared, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
        encoding='utf-8', errors='replace', close_fds=True,
        creationflags=subprocess.CREATE_NO_WINDOW, cwd=prepared.workdir, env=prepared.env)
    binding = owner.startup_binding
    assert type(binding) is LaunchBinding, 'Native startup gate did not return its binding'
    assert binding.pid == process.pid
    assert binding.nonce == owner.nonce
    assert binding.profile_digest.hex() == PYTHON_PROFILE.digest
    assert binding.content_digest.hex() == owner.published.content_digest
    output = process.stdout.read()
    assert process.wait(timeout=15) == 0, output
    rows = json.loads(output)
    assert rows == {name: [12, 12] for name in ('PNG','JPEG','GIF','BMP','TIFF','WEBP','AVIF')}, rows
finally:
    prepared.cleanup()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
assert owner.closed and identity.cleaned and owner.identity is None
assert not manifest.exists(), 'Invocation identity journal survived cleanup'
assert not owner.handles and not owner.pins.handles and not owner.file_pins.handles
assert owner.access is None and owner.catalog is None and owner.process is None
assert not list((root / 'cache' / '.readers').iterdir()), 'Runtime read leases survived cleanup'
assert dict(os.environ) == environment_before, 'Broker environment was mutated'
assert image_hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in images}, 'Installed Pillow changed'
print(json.dumps({'formats': rows, 'pillow': pillow.version,
    'companion': VERSION, 'native_binding': True, 'cleanup_complete': True,
    'environment_unchanged': True, 'qualification': False}, sort_keys=True))
"""


def test_installed_pillow_codecs_pass_native_gate_and_cleanup(tmp_path):
    configured = os.environ.get("UNSLOTH_TEST_PILLOW_RUNTIME")
    if not configured:
        pytest.skip("Optional installed Pillow compatibility lane: set UNSLOTH_TEST_PILLOW_RUNTIME")
    assert sys.platform == "win32", "The explicitly requested Pillow LPAC lane requires Windows"
    selected = Path(configured)
    if selected.is_dir():
        selected = selected / "Scripts" / "python.exe"
    assert (
        selected.is_absolute() and selected.is_file()
    ), "Selected Pillow runtime executable is missing"
    work = tmp_path / "work"
    work.mkdir()
    (work / "tool.py").write_text(PAYLOAD, encoding = "utf-8")
    harness = tmp_path / "pillow-broker.py"
    harness.write_text(BROKER, encoding = "utf-8")
    environment_before = dict(os.environ)
    result = subprocess.run(
        [str(selected), "-I", str(harness), str(BACKEND), str(tmp_path)],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 240,
        check = False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["formats"] == {format: [12, 12] for format in FORMATS}
    assert evidence["native_binding"] and evidence["cleanup_complete"]
    assert evidence["environment_unchanged"] and evidence["qualification"] is False
    assert dict(os.environ) == environment_before
