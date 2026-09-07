# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A changed catalog must refuse the owned target before payload execution."""

import sys
import pytest

from test_launch import LAUNCH, run_harness, installed_runtime, runtime_wheel

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows catalog handoff")


def test_catalog_qualification_mismatch_reaps_target_before_payload(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
{LAUNCH}
owner.expected_catalog_binding = '00' * 32
try:
    spawn_prepared_launch(prepared, **kwargs)
except WindowsRuntimeError as error:
    assert 'binding' in str(error).lower(), str(error)
else:
    raise AssertionError('Changed catalog started a payload')
finally:
    prepared.cleanup()
assert owner.closed and owner.catalog is None and owner.process is None
assert not owner.handles and not owner.retained_processes and not owner.retained_raw
assert not owner.reservation.path.exists() and not launch._pending_cleanup
assert not (work/'payload-ran').exists()
assert not list((root/'cache'/'.readers').iterdir())
print('CHANGED_CATALOG_REFUSED_AND_REAPED')
""",
    )
    assert "CHANGED_CATALOG_REFUSED_AND_REAPED" in output
