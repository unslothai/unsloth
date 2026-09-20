# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The wheelhouse signer: what it signs, what it refuses, and what it must not break.

A .whl cannot be Authenticode signed, so the native binaries inside it are, and the wheel is
repacked. That makes RECORD the thing most likely to go quietly wrong -- signing changes every
signed member's bytes, and a stale RECORD ships a wheel that reads as corrupt to anything that
checks it. The signing call itself is stubbed here: this is about the packing, the refusals and
the RECORD, none of which need Azure.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNER_PY = REPO_ROOT / ".github" / "scripts" / "sign_wheel_binaries.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "woa-wheelhouse.yml"


@pytest.fixture(scope = "module")
def mod():
    spec = importlib.util.spec_from_file_location("_sign_wheel_binaries", SIGNER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pe(
    machine: int = 0xAA64,
    *,
    signed: bool = False,
    plus: bool = True,
) -> bytes:
    """The smallest byte string the reader will accept as a PE image.

    Built rather than vendored: a real .pyd in the tree would be a binary nobody can review,
    and every field this tool reads is one of the four below.
    """
    header_at = 0x80
    blob = bytearray(b"\0" * 0x400)
    blob[0:2] = b"MZ"
    struct.pack_into("<I", blob, 0x3C, header_at)
    blob[header_at : header_at + 4] = b"PE\0\0"
    struct.pack_into("<H", blob, header_at + 4, machine)
    magic_at = header_at + 24
    struct.pack_into("<H", blob, magic_at, 0x20B if plus else 0x10B)
    directory_at = magic_at + (112 if plus else 96) + 4 * 8
    if signed:
        struct.pack_into("<II", blob, directory_at, 0x300, 0x40)
    return bytes(blob)


def _record(names_and_blobs) -> str:
    lines = []
    for name, blob in names_and_blobs:
        digest = base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).rstrip(b"=").decode()
        lines.append(f"{name},sha256={digest},{len(blob)}")
    return "\n".join(lines) + "\n"


def _wheel(path: Path, members: dict) -> Path:
    """A wheel whose RECORD matches its contents, the way a real one arrives."""
    dist = "demo-1.0.dist-info"
    record_name = f"{dist}/RECORD"
    body = dict(members)
    body.setdefault(f"{dist}/WHEEL", b"Wheel-Version: 1.0\nTag: cp313-cp313-win_arm64\n")
    body.setdefault(f"{dist}/METADATA", b"Name: demo\nVersion: 1.0\n")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, blob in body.items():
            z.writestr(name, blob)
        z.writestr(record_name, _record(list(body.items())) + f"{record_name},,\n")
    return path


@pytest.fixture
def fake_signer(tmp_path):
    """Stands in for trusted-signing-cli: appends a certificate table to the file it is given.

    The real one talks to Azure. What this file's tests are about is everything around that
    call, so the stub does the one thing the caller checks for -- the file comes back signed.
    """
    script = tmp_path / "fake_signer.py"
    script.write_text(
        "import struct, sys\n"
        "from pathlib import Path\n"
        "p = Path(sys.argv[1]); d = bytearray(p.read_bytes())\n"
        "h = struct.unpack_from('<I', d, 0x3C)[0]\n"
        "magic_at = h + 24\n"
        "magic = struct.unpack_from('<H', d, magic_at)[0]\n"
        "at = magic_at + (112 if magic == 0x20b else 96) + 4 * 8\n"
        "struct.pack_into('<II', d, at, len(d), 0x40)\n"
        "d += b'CERTIFICATE'\n"
        "p.write_bytes(bytes(d))\n",
        encoding = "utf-8",
    )
    return [sys.executable, str(script)]


class TestWhatCountsAsAPeImage:
    def test_a_pe_is_recognised(self, mod):
        assert mod.pe_machine(_pe()) == 0xAA64

    @pytest.mark.parametrize(
        "blob, why",
        [
            (b"", "empty"),
            (b"not a pe at all, just text in a wheel", "plain data"),
            (b"MZ" + b"\0" * 8, "an MZ header too short to hold an offset"),
            (b"MZ" + b"\0" * 0x200, "an MZ header whose PE offset points at zeros"),
        ],
    )
    def test_everything_else_is_not(self, mod, blob, why):
        assert mod.pe_machine(blob) is None, why

    def test_an_unsigned_image_reads_as_unsigned(self, mod):
        assert mod.is_signed(_pe()) is False

    @pytest.mark.parametrize("plus", [True, False], ids = ["PE32+", "PE32"])
    def test_a_signed_image_reads_as_signed(self, mod, plus):
        """PE32 and PE32+ put the data directory at different offsets; a reader that assumed
        one would call every 32-bit signed DLL unsigned and re-sign somebody else's binary."""
        assert mod.is_signed(_pe(signed = True, plus = plus)) is True


class TestSigningAWheel:
    def test_every_unsigned_binary_is_signed(self, mod, tmp_path, fake_signer):
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/__init__.py": b"x = 1\n",
                "demo/_core.cp313-win_arm64.pyd": _pe(),
                "demo/helper.dll": _pe(),
            },
        )
        out, signed, skipped = mod.sign_wheel(wheel, tmp_path / "out", fake_signer)
        assert sorted(signed) == ["demo/_core.cp313-win_arm64.pyd", "demo/helper.dll"]
        assert skipped == []
        with zipfile.ZipFile(out) as z:
            for name in signed:
                assert mod.is_signed(z.read(name)), name

    def test_an_already_signed_binary_is_left_alone(self, mod, tmp_path, fake_signer):
        """delvewheel vendors Microsoft's redistributables into pyarrow already signed by
        Microsoft. Re-signing one replaces their signature with ours."""
        original = _pe(signed = True)
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/_core.cp313-win_arm64.pyd": _pe(),
                "demo.libs/msvcp140-abcdef.dll": original,
            },
        )
        out, signed, skipped = mod.sign_wheel(wheel, tmp_path / "out", fake_signer)
        assert signed == ["demo/_core.cp313-win_arm64.pyd"]
        assert skipped == ["demo.libs/msvcp140-abcdef.dll"]
        with zipfile.ZipFile(out) as z:
            assert z.read("demo.libs/msvcp140-abcdef.dll") == original, "byte-for-byte untouched"

    def test_the_wheel_keeps_its_name(self, mod, tmp_path, fake_signer):
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/_core.cp313-win_arm64.pyd": _pe(),
            },
        )
        out, _, _ = mod.sign_wheel(wheel, tmp_path / "out", fake_signer)
        assert out.name == wheel.name, "the index and every pin name the wheel by filename"

    def test_non_binary_members_are_carried_through_unchanged(self, mod, tmp_path, fake_signer):
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/__init__.py": b"x = 1\n",
                "demo/data.json": b'{"a": 1}',
                "demo/_core.cp313-win_arm64.pyd": _pe(),
            },
        )
        out, _, _ = mod.sign_wheel(wheel, tmp_path / "out", fake_signer)
        with zipfile.ZipFile(out) as z:
            assert z.read("demo/__init__.py") == b"x = 1\n"
            assert z.read("demo/data.json") == b'{"a": 1}'

    def test_member_order_is_preserved(self, mod, tmp_path, fake_signer):
        members = {
            "demo/__init__.py": b"x = 1\n",
            "demo/_core.cp313-win_arm64.pyd": _pe(),
            "demo/zzz.py": b"pass\n",
        }
        wheel = _wheel(tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl", members)
        with zipfile.ZipFile(wheel) as z:
            before = z.namelist()
        out, _, _ = mod.sign_wheel(wheel, tmp_path / "out", fake_signer)
        with zipfile.ZipFile(out) as z:
            assert z.namelist() == before, "a diff should show signatures and nothing else"


class TestTheRecordIsRegenerated:
    """Signing appends a certificate table, so every signed member's digest and length move.
    A wheel shipped with its original RECORD is one `pip check` away from reading as corrupt."""

    @staticmethod
    def _rows(wheel: Path) -> dict:
        with zipfile.ZipFile(wheel) as z:
            record = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
            rows = {}
            for line in z.read(record).decode().splitlines():
                if line.strip():
                    parts = line.split(",")
                    rows[parts[0]] = (parts[1], parts[2])
            return rows

    def test_every_member_digest_matches_the_bytes_shipped(self, mod, tmp_path, fake_signer):
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/__init__.py": b"x = 1\n",
                "demo/_core.cp313-win_arm64.pyd": _pe(),
            },
        )
        out, _, _ = mod.sign_wheel(wheel, tmp_path / "out", fake_signer)
        rows = self._rows(out)
        with zipfile.ZipFile(out) as z:
            for name in z.namelist():
                if name.endswith(".dist-info/RECORD"):
                    continue
                blob = z.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).rstrip(b"=")
                assert rows[name] == (f"sha256={expected.decode()}", str(len(blob))), name

    def test_the_signed_binary_digest_actually_changed(self, mod, tmp_path, fake_signer):
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/_core.cp313-win_arm64.pyd": _pe(),
            },
        )
        before = self._rows(wheel)["demo/_core.cp313-win_arm64.pyd"]
        out, _, _ = mod.sign_wheel(wheel, tmp_path / "out", fake_signer)
        after = self._rows(out)["demo/_core.cp313-win_arm64.pyd"]
        assert (
            before != after
        ), "the fixture did not really change the bytes, so this proves nothing"

    def test_record_names_itself_without_a_digest(self, mod, tmp_path, fake_signer):
        """PEP 376: RECORD cannot hash itself, and a tool that writes one anyway makes the
        file self-contradictory."""
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/_core.cp313-win_arm64.pyd": _pe(),
            },
        )
        out, _, _ = mod.sign_wheel(wheel, tmp_path / "out", fake_signer)
        assert self._rows(out)["demo-1.0.dist-info/RECORD"] == ("", "")


class TestItRefusesRatherThanShipSomethingWrong:
    def test_a_foreign_architecture_is_refused(self, mod, tmp_path, fake_signer):
        """0x8664 is x64. A win_arm64 filename says nothing about what was linked in, and
        signing one would publish a correctly signed wheel that cannot load."""
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/_core.cp313-win_arm64.pyd": _pe(machine = 0x8664),
            },
        )
        with pytest.raises(SystemExit, match = "0x8664"):
            mod.sign_wheel(wheel, tmp_path / "out", fake_signer)

    def test_a_foreign_architecture_is_allowed_when_asked(self, mod, tmp_path, fake_signer):
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_amd64.whl",
            {
                "demo/_core.cp313-win_amd64.pyd": _pe(machine = 0x8664),
            },
        )
        _, signed, _ = mod.sign_wheel(wheel, tmp_path / "out", fake_signer, require_machine = None)
        assert signed == ["demo/_core.cp313-win_amd64.pyd"]

    def test_a_wheel_with_no_binaries_is_refused(self, mod, tmp_path, fake_signer):
        """Nothing to sign means the build produced a pure-Python wheel where a native one was
        expected. Publishing it silently is how an unsigned wheelhouse comes back."""
        wheel = _wheel(tmp_path / "demo-1.0-py3-none-win_arm64.whl", {"demo/__init__.py": b"x\n"})
        with pytest.raises(SystemExit, match = "no PE images"):
            mod.sign_wheel(wheel, tmp_path / "out", fake_signer)

    def test_a_signer_that_fails_fails_the_run(self, mod, tmp_path):
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/_core.cp313-win_arm64.pyd": _pe(),
            },
        )
        failing = [sys.executable, "-c", "import sys; sys.exit(3)"]
        with pytest.raises(SystemExit, match = "signing failed"):
            mod.sign_wheel(wheel, tmp_path / "out", failing)

    def test_a_signer_that_silently_does_nothing_fails_the_run(self, mod, tmp_path):
        """Exit 0 with the file untouched is the dangerous case: it would publish an unsigned
        wheel while every log line says it was signed."""
        wheel = _wheel(
            tmp_path / "demo-1.0-cp313-cp313-win_arm64.whl",
            {
                "demo/_core.cp313-win_arm64.pyd": _pe(),
            },
        )
        noop = [sys.executable, "-c", "import sys"]
        with pytest.raises(SystemExit, match = "no certificate after signing"):
            mod.sign_wheel(wheel, tmp_path / "out", noop)


class TestTheWorkflowUsesTheProcedureTheDesktopReleaseUses:
    """One signing procedure, one definition. A second copy of the endpoint or the identity
    drifts, and the drift is invisible until a release is signed by the wrong certificate."""

    @pytest.fixture(scope = "class")
    def text(self):
        return WORKFLOW.read_text(encoding = "utf-8")

    def test_it_delegates_to_the_shared_signing_helper(self, text):
        assert "studio/src-tauri/windows/sign-with-trusted-signing.ps1" in text
        assert "codesigning.azure.net" not in text, "the endpoint belongs to the helper only"

    def test_the_signing_cli_is_digest_pinned(self, text):
        assert "TRUSTED_SIGNING_CLI_SHA256" in text
        desktop = (REPO_ROOT / ".github" / "workflows" / "release-desktop.yml").read_text(
            encoding = "utf-8"
        )
        import re

        pinned = re.findall(r'TRUSTED_SIGNING_CLI_SHA256:\s*"([0-9a-f]{64})"', text)
        assert pinned, "no digest pin"
        assert pinned[0] in desktop, "the two workflows pin different signing binaries"

    def test_the_secrets_are_gated_by_an_environment(self, text):
        assert "environment: release-signing" in text, (
            "without an environment there is nowhere to hang a required reviewer, so anyone "
            "who can dispatch this can sign under our identity"
        )

    def test_only_the_signing_job_reads_the_azure_secrets(self, text):
        import yaml
        data = yaml.safe_load(text)
        for name, job in data["jobs"].items():
            uses_azure = "AZURE_" in yaml.safe_dump(job)
            if uses_azure:
                assert name == "sign", f"{name} reads the signing secrets"
                assert job.get("environment") == "release-signing"

    def test_it_cannot_be_reached_from_a_pull_request(self, text):
        import yaml

        data = yaml.safe_load(text)
        triggers = data[True] if True in data else data["on"]
        assert set(triggers) == {
            "workflow_dispatch"
        }, "this job holds signing credentials and writes a release"

    def test_publishing_is_opt_in(self, text):
        import yaml

        data = yaml.safe_load(text)
        publish = data[True]["workflow_dispatch"]["inputs"]["publish"]
        assert publish["default"] is False, "a dispatch must not publish unless asked"
        assert "inputs.publish" in str(data["jobs"]["publish"]["if"])

    def test_write_permission_is_scoped_to_the_publish_job(self, text):
        import yaml

        data = yaml.safe_load(text)
        assert data["permissions"] == {"contents": "read"}, "the default must be read"
        assert data["jobs"]["publish"]["permissions"] == {"contents": "write"}
        for name, job in data["jobs"].items():
            if name != "publish":
                assert "permissions" not in job or job["permissions"].get("contents") != "write"

    def test_the_published_set_is_checked_before_signing(self, text):
        """A short matrix leg would otherwise publish a wheelhouse missing an interpreter,
        which reads downstream as 'no wheel for this Python' rather than as a failed build."""
        assert "expected 5 wheels" in text
