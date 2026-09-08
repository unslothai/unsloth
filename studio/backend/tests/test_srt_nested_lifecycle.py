# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Native nested diagnostic lifecycle; no mocked executor or admission bypass claim."""

import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest

from core.inference import srt_adapter
from core.inference.tools import _build_safe_env, _sandbox_launcher_preexec


@unittest.skipUnless(
    sys.platform == "linux" and os.environ.get("UNSLOTH_SRT_NATIVE_TESTS") == "1",
    "requires provisioned native Linux",
)
class NestedLifecycleTest(unittest.TestCase):
    def launch(
        self,
        work,
        code,
        timeout = 10,
    ):
        request = srt_adapter.request_for(
            [sys.executable, "-I", "-S", "-c", code],
            str(work),
            _build_safe_env(str(work)),
            timeout,
            operation = "probe",
            isolation_variant = "nested",
        )
        proc = srt_adapter.spawn(
            request,
            cwd = str(work),
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            preexec_fn = _sandbox_launcher_preexec,
        )
        self.addCleanup(self.cleanup, proc)
        return proc

    @staticmethod
    def cleanup(proc):
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout = 5)
        srt_adapter.release_control(proc)

    def test_stream_and_failed_execution_have_distinct_receipts(self):
        with tempfile.TemporaryDirectory() as work:
            proc = self.launch(
                work, "import time; print('FIRST',flush=True); time.sleep(.3); print('LAST')"
            )
            self.assertTrue(select.select([proc.stdout], [], [], 10)[0])
            self.assertEqual(proc.stdout.readline(), b"FIRST\n")
            output, error = proc.communicate(timeout = 15)
            self.assertEqual(output, b"LAST\n", error)
            srt_adapter.verify_success(proc)
            failed = self.launch(work, "import sys; print('FAILED_ONCE'); sys.exit(7)")
            output, error = failed.communicate(timeout = 15)
            self.assertEqual(output, b"FAILED_ONCE\n", error)
            self.assertEqual(failed.returncode, 7)
            self.assertEqual(srt_adapter.completion_receipt(failed)["code"], 7)
            with self.assertRaises(srt_adapter.SrtError):
                srt_adapter.verify_success(failed)

    def test_timeout_and_cancel_clean_detached_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            for cancel in (False, True):
                with self.subTest(cancel = cancel):
                    work = Path(directory) / str(cancel)
                    work.mkdir()
                    child = "import pathlib,time; pathlib.Path('child-ready').write_text('ready'); time.sleep(4); pathlib.Path('escaped').write_text('escaped')"
                    code = (
                        "import subprocess,sys,time,pathlib; "
                        f"subprocess.Popen([sys.executable,'-I','-S','-c',{child!r}],start_new_session=True); "
                        "\nwhile not pathlib.Path('child-ready').exists(): time.sleep(.01)\n"
                        "print('CHILD_STARTED',flush=True); time.sleep(60)"
                    )
                    proc = self.launch(work, code, timeout = 10 if cancel else 2)
                    self.assertTrue(select.select([proc.stdout], [], [], 10)[0])
                    self.assertEqual(proc.stdout.readline(), b"CHILD_STARTED\n")
                    started = time.monotonic()
                    if cancel:
                        proc.terminate()
                    output, error = proc.communicate(timeout = 8)
                    self.assertLess(time.monotonic() - started, 8)
                    self.assertNotEqual(proc.returncode, 0, (output, error))
                    self.assertEqual(
                        srt_adapter.completion_receipt(proc)["reason"],
                        "cancelled" if cancel else "timeout",
                    )
                    time.sleep(4.5)
                    self.assertFalse(
                        (work / "escaped").exists(), "detached descendant survived shutdown"
                    )

            # The same detached child must be able to finish while its parent
            # stays alive; otherwise an absent file would not prove cleanup.
            control = Path(directory) / "control"
            control.mkdir()
            proc = self.launch(control, code.replace("time.sleep(60)", "time.sleep(5)"))
            proc.communicate(timeout = 15)
            srt_adapter.verify_success(proc)
            self.assertEqual((control / "escaped").read_text(), "escaped")


if __name__ == "__main__":
    unittest.main()
