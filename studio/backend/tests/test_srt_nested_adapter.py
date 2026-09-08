# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Native diagnostic protocol controls; not container qualification or user admission."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from core.inference import srt_adapter


class VariantAdmissionTest(unittest.TestCase):
    def test_nested_protocol_is_linux_only_and_explicit(self):
        with patch.object(srt_adapter, "_request_for", side_effect = lambda *args, **kwargs: {}):
            with patch.object(srt_adapter.sys, "platform", "linux"):
                request = srt_adapter.request_for(
                    [sys.executable], ".", {}, 1, isolation_variant = "nested"
                )
                self.assertEqual(request["isolationVariant"], "nested")
                self.assertNotIn(
                    "isolationVariant", srt_adapter.request_for([sys.executable], ".", {}, 1)
                )
            for platform in ("darwin", "win32"):
                with (
                    patch.object(srt_adapter.sys, "platform", platform),
                    self.assertRaises(srt_adapter.SrtError),
                ):
                    srt_adapter.request_for(
                        [sys.executable], ".", {}, 1, isolation_variant = "nested"
                    )


@unittest.skipUnless(
    sys.platform == "linux" and os.environ.get("UNSLOTH_SRT_NATIVE_TESTS") == "1",
    "requires provisioned native Linux",
)
class NativeVariantTest(unittest.TestCase):
    def test_private_protocol_preserves_caps_files_and_network(self):
        with (
            tempfile.TemporaryDirectory(prefix = "studio-nested-adapter-") as directory,
            socket.socket() as listener,
        ):
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            sentinel = root / "owned-sentinel"
            sentinel.write_text("PRIVATE")
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            address = listener.getsockname()
            with socket.create_connection(address, timeout = 1):
                accepted, _ = listener.accept()
                accepted.close()
            code = f"""import pathlib,socket,sys
assert sys.executable == {sys.executable!r}
for line in pathlib.Path('/proc/self/status').read_text().splitlines():
 if line.startswith(('CapEff:','CapPrm:','CapBnd:')): assert int(line.split()[1],16)==0
try: pathlib.Path({str(sentinel)!r}).read_bytes()
except OSError: pass
else: raise AssertionError('unrelated read permitted')
try: socket.create_connection({address!r},timeout=.5)
except OSError: pass
else: raise AssertionError('host network permitted')
pathlib.Path('positive').write_text('ok')
print('VARIANT_DIAGNOSTIC_OK')
"""
            for variant in ("standard", "nested"):
                request = srt_adapter.request_for(
                    [sys.executable, "-I", "-S", "-c", code],
                    str(work),
                    {"PATH": os.defpath},
                    30,
                    operation = "probe",
                    isolation_variant = variant,
                )
                proc = srt_adapter.spawn(
                    request,
                    cwd = str(work),
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE,
                    preexec_fn = os.setsid,
                )
                try:
                    output, error = proc.communicate(timeout = 40)
                    self.assertEqual(proc.returncode, 0, error.decode(errors = "replace"))
                    srt_adapter.verify_success(proc)
                    self.assertEqual(output.strip(), b"VARIANT_DIAGNOSTIC_OK")
                    self.assertEqual((work / "positive").read_text(), "ok")
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout = 5)
                    srt_adapter.release_control(proc)


if __name__ == "__main__":
    unittest.main()
