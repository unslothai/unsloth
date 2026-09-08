# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Native adapter proof for large explicit read lists, without parent-directory grants."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from core.inference import srt_adapter


@unittest.skipUnless(
    sys.platform == "linux" and os.environ.get("UNSLOTH_SRT_NATIVE_TESTS") == "1",
    "requires prepared native Linux SRT runtime",
)
class ReadRootCountTest(unittest.TestCase):
    def test_selected_files_are_readable_but_their_sibling_is_not(self):
        with tempfile.TemporaryDirectory(prefix = "srt-root-count-") as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            libraries = root / "libraries"
            libraries.mkdir()
            selected = []
            for index in range(143):
                library = libraries / f"libfixture{index}.so"
                library.write_text("selected")
                selected.append(str(library))
            sentinel = libraries / "credential-shaped-sentinel"
            sentinel.write_text("benign-host-control")
            self.assertEqual(sentinel.read_text(), "benign-host-control")
            code = (
                "from pathlib import Path\n"
                f"assert all(Path(p).read_text() == 'selected' for p in {selected!r})\n"
                "try:\n"
                f" Path({str(sentinel)!r}).read_text()\n"
                "except OSError:\n pass\n"
                "else:\n raise AssertionError('sibling read escaped the selected roots')\n"
                "Path('positive').write_text('workdir-ok')\n"
                "print('LARGE_READ_ROOTS_NATIVE_OK')\n"
            )
            request = srt_adapter.request_for(
                [sys.executable, "-c", code], str(work), {"PATH": os.defpath}, 30,
                additional_read_roots = selected,
            )
            self.assertGreater(len(request["readRoots"]), 128)
            self.assertNotIn(str(libraries), request["readRoots"])
            self.assertLess(len(json.dumps(request).encode()), srt_adapter.MAX_REQUEST)
            proc = srt_adapter.spawn(
                request, cwd = str(work), stdout = subprocess.PIPE,
                stderr = subprocess.PIPE, text = True, preexec_fn = os.setsid,
            )
            try:
                stdout, stderr = proc.communicate(timeout = 40)
                self.assertEqual(proc.returncode, 0, stderr)
                srt_adapter.verify_success(proc)
                self.assertEqual(stdout.strip(), "LARGE_READ_ROOTS_NATIVE_OK")
                self.assertEqual((work / "positive").read_text(), "workdir-ok")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                srt_adapter.release_control(proc)


if __name__ == "__main__":
    unittest.main()
