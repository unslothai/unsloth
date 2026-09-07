# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Host-controlled fixed probe observations, not complete Windows qualification."""

import json
import sys

import pytest

from test_launch import run_harness, installed_runtime, runtime_wheel
from test_preparation import BACKEND
from core.inference.windows_sandbox import probe
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows probe enforcement")


def test_fixed_probe_host_controls_and_cleanup(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        """
from dataclasses import FrozenInstanceError
from core.inference.windows_sandbox import probe,launch,identity
owners = []
original = launch._PythonLaunch.__init__
def capture(owner,*args):
    original(owner,*args)
    owners.append(owner)
launch._PythonLaunch.__init__ = capture
try:
    result = probe.run_python_probe(sys.executable,root/'cache')
finally:
    launch._PythonLaunch.__init__ = original
    for owner in owners:
        owner.cleanup()
assert len(owners) == 1
owner = owners[0]
assert result.checks == probe.CORE_CHECKS + probe.HOST_CHECKS
assert result.version == tuple(sys.version_info[:3])
assert result.content_digest == owner.published.content_digest
assert result.runtime_digest == owner.published.core.digest
assert result.artifact_digest == owner.published.artifacts.digest
assert result.elapsed_seconds > 0
assert len(owner.probe_ports) == 4 and all(0 < port < 65536 for port in owner.probe_ports)
assert not hasattr(result,'qualified') and not hasattr(result,'available')
assert not hasattr(result,'execution_record')
assert isinstance(result.dns_context,probe.DnsContextObservations)
assert not hasattr(result.dns_context,'qualified') and not hasattr(result.dns_context,'available')
try:
    result.content_digest = 'changed'
except FrozenInstanceError:
    pass
else:
    raise AssertionError('Mutable observation')
assert owner.closed and owner.started and not owner.handles
assert not owner.file_pins.handles and not owner.pins.handles
with identity._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
    assert identity._profile_path(sid) is None
assert not owner.reservation.path.exists()
assert not list((root/'cache'/'.readers').iterdir()) and not launch._pending_cleanup
assert not list(work.iterdir())
print('PRIVATE_NETWORK_CONTROLS_OK')
""",
    )
    assert "PRIVATE_NETWORK_CONTROLS_OK" in output


@pytest.mark.parametrize(
    "ports",
    [
        None,
        [],
        (1,),
        (1, 2, 3, True),
        (0, 2, 3, 4),
        (65536, 2, 3, 4),
        (1, 2, 3, "4"),
        (1, 2, 3, 4, 5),
        {"host": "127.0.0.1"},
    ],
)
def test_probe_ports_are_fixed_data_only(ports):
    with pytest.raises(WindowsRuntimeError, match = "network ports"):
        probe.validate_network_ports(ports)


@pytest.mark.parametrize(
    "change",
    [
        "nonce",
        "version",
        "float_version",
        "checks",
        "extra",
        "duplicate",
        "excess",
        "dns_missing",
        "dns_boolean",
        "dns_extra",
    ],
)
def test_probe_observations_reject_unbound_or_malformed_output(change):
    nonce, version = b"a" * 32, (3, 12, 10)
    value = dict(
        probe_nonce = nonce.hex(),
        version = list(version),
        checks = list(probe.CORE_CHECKS + probe.HOST_CHECKS),
        dns_context = [1, 1],
    )
    if change == "nonce":
        value["probe_nonce"] = "b" * 64
    elif change == "version":
        value["version"] = [3, 13, 10]
    elif change == "float_version":
        value["version"] = [3.0, 12, 10]
    elif change == "checks":
        value["checks"].pop()
    elif change == "extra":
        value["qualified"] = True
    elif change == "dns_missing":
        value.pop("dns_context")
    elif change == "dns_boolean":
        value["dns_context"] = [True, True]
    elif change == "dns_extra":
        value["dns_context"] = [1, 1, "qualified"]
    data = json.dumps(value).encode()
    if change == "duplicate":
        data = b'{"probe_nonce":"other",' + data[1:]
    elif change == "excess":
        data += b" " * (probe.MAX_PROBE_OUTPUT + 1)
    with pytest.raises(WindowsRuntimeError):
        probe._parse_probe_output(data, nonce, version, network = True)


def test_valid_probe_output_is_not_a_qualification():
    nonce, version = b"a" * 32, (3, 12, 10)
    expected = probe.CORE_CHECKS + probe.HOST_CHECKS
    data = json.dumps(
        dict(
            probe_nonce = nonce.hex(),
            version = list(version),
            checks = list(expected),
            dns_context = [1, 1],
        )
    ).encode()
    checks, context = probe._parse_probe_output(data, nonce, version, network = True)
    assert checks == expected
    assert context == probe.DnsContextObservations(1, 1)
    assert context.require_usable() is None
    assert not hasattr(context, "qualified") and not hasattr(context, "available")


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        (),
        [1],
        [1, 1, 1],
        [True, 1],
        [1, False],
        [1.0, 1],
        ["1", 1],
        [-1, 1],
        [1, 0x100000000],
    ],
)
def test_dns_context_rejects_malformed_observations(value):
    with pytest.raises(WindowsRuntimeError, match = "Invalid DNS context"):
        probe._parse_dns_context(value)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", -1, 0x100000000])
def test_dns_context_direct_construction_cannot_bypass_validation(value):
    with pytest.raises(WindowsRuntimeError, match = "Invalid DNS context"):
        probe.DnsContextObservations(value, value)


@pytest.mark.parametrize(
    "value", [[0, 1], [1, 0], [0, 0], [1, 2], [None, 1], [1, None], [None, None]]
)
def test_dns_context_failure_is_not_denial_or_qualification(value):
    from dataclasses import FrozenInstanceError

    original = list(value)
    context = probe._parse_dns_context(value)
    assert value == original
    with pytest.raises(WindowsRuntimeError) as error:
        context.require_usable()
    assert error.value.code == "WINDOWS_SANDBOX_DNS_CONTEXT_UNAVAILABLE"
    assert "unverified" in str(error.value)
    with pytest.raises(FrozenInstanceError):
        context.default_compartment = 1
    assert not hasattr(context, "qualified") and not hasattr(context, "available")


@pytest.mark.parametrize("present", [(True, True), (False, True), (True, False), (False, False)])
def test_dns_context_missing_api_stays_unavailable(present):
    from types import SimpleNamespace

    names = ("GetDefaultCompartmentId", "GetCurrentThreadCompartmentId")
    library, calls = SimpleNamespace(), []
    for name, exists in zip(names, present):
        if exists:

            def query(name = name):
                calls.append(name)
                return 1

            setattr(library, name, query)

    def load(name, **kwargs):
        assert name == "iphlpapi" and kwargs == dict(use_last_error = True, winmode = 0x800)
        return library

    namespace = dict(ctypes = SimpleNamespace(WinDLL = load), W = probe.W)
    # This is a binding-shape unit test of fixed source, not enforcement evidence.
    exec(probe._DNS_CONTEXT_SOURCE, namespace)
    context = probe._parse_dns_context(namespace["dns_context"])
    assert calls == [name for name, exists in zip(names, present) if exists]
    if all(present):
        assert context.require_usable() is None
    else:
        with pytest.raises(WindowsRuntimeError, match = "WINDOWS_SANDBOX_DNS_CONTEXT_UNAVAILABLE"):
            context.require_usable()


@pytest.mark.parametrize("mode", ["excess", "malformed", "nonzero", "timeout", "cancel"])
def test_fixed_probe_output_failure_reaps_and_removes_private_identity(
    installed_runtime, tmp_path, mode
):
    payload = {
        "excess": "import os; os.write(1,b'x'*16384)",
        "malformed": "print('{not-json')",
        "nonzero": "raise SystemExit(17)",
        "timeout": "import time; time.sleep(60)",
        "cancel": "import time; time.sleep(60)",
    }[mode]
    worker = f"""
import sys
sys.path.insert(0,{str(BACKEND)!r})
from core.inference.windows_sandbox import probe
from core.inference.windows_sandbox.preparation_worker import main
probe.probe_source = lambda *args: {payload!r}
raise SystemExit(main())
"""
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.windows_sandbox import probe,launch,preparation,identity
owners = []
original_init,original_run,original_collect = launch._PythonLaunch.__init__,preparation._run_worker,probe._collect_probe_output
wrapper = root/'fixed-probe-output-worker.py'
wrapper.write_text({worker!r},encoding='utf-8')
cancel = threading.Event()
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def intercept(argv,*args,**kwargs):
    if argv[4].endswith('preparation_worker.py'):
        argv = [*argv[:4],str(wrapper),*argv[5:]]
    return original_run(argv,*args,**kwargs)
def collect(owner,process):
    if {mode!r} == 'timeout':
        owner.deadline = time.monotonic()+.1
    elif {mode!r} == 'cancel':
        cancel.set()
    return original_collect(owner,process)
launch._PythonLaunch.__init__,preparation._run_worker,probe._collect_probe_output = capture,intercept,collect
try:
    try:
        probe.run_python_probe(sys.executable,root/'cache',cancel=cancel)
    except launch.WindowsRuntimeError as error:
        expected = {{'excess':'WINDOWS_SANDBOX_PROBE_INVALID',
            'malformed':'WINDOWS_SANDBOX_PREPARATION_FAILED','nonzero':'WINDOWS_SANDBOX_PROBE_FAILED',
            'timeout':'WINDOWS_SANDBOX_STARTUP_TIMEOUT','cancel':'WINDOWS_SANDBOX_CANCELLED'}}[{mode!r}]
        assert error.code == expected,str(error)
    else:
        raise AssertionError('Failed fixed probe returned observations')
finally:
    launch._PythonLaunch.__init__,preparation._run_worker,probe._collect_probe_output = original_init,original_run,original_collect
    for owner in owners:
        owner.cleanup()
assert len(owners) == 1 and owners[0].closed and owners[0].started
owner = owners[0]
assert not owner.handles and not owner.pins.handles and not owner.file_pins.handles
with identity._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
    assert identity._profile_path(sid) is None
assert not owner.reservation.path.exists() and not launch._pending_cleanup
assert not list((root/'cache'/'.readers').iterdir())
assert not list(work.iterdir())
print('PROBE_OUTPUT_FAILED_CLOSED')
""",
    )
    assert "PROBE_OUTPUT_FAILED_CLOSED" in output


@pytest.mark.parametrize("mode", ["host_packet", "host_grant", "cleanup_failure", "late_cancel"])
def test_probe_rejects_failed_controls_or_cleanup(installed_runtime, tmp_path, mode):
    worker = f"""
import sys
from pathlib import Path
sys.path.insert(0,{str(BACKEND)!r})
from core.inference.windows_sandbox import probe,launch
from core.inference.windows_sandbox.preparation_worker import main
original = launch._PythonLaunch.prepare_files
def grant(owner):
    original(owner)
    path = str(Path(owner.identity.profile_folder)/probe.HOST_CONTROL_FILENAME)
    launch.lpac._grant_read_execute(path,owner.identity.sid)
launch._PythonLaunch.prepare_files = grant
raise SystemExit(main())
"""
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
import socket
from core.inference.os_sandbox import SandboxUnavailableError
from core.inference.windows_sandbox import probe,launch,preparation,identity
owners,restore = [],[]
original_init,original_run,original_collect = launch._PythonLaunch.__init__,preparation._run_worker,probe._collect_probe_output
wrapper = root/'fixed-probe-grant-worker.py'
wrapper.write_text({worker!r},encoding='utf-8')
cancel = threading.Event()
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def intercept(argv,*args,**kwargs):
    if {mode!r} == 'host_grant' and argv[4].endswith('preparation_worker.py'):
        argv = [*argv[:4],str(wrapper),*argv[5:]]
    return original_run(argv,*args,**kwargs)
def collect(owner,process):
    data = original_collect(owner,process)
    if {mode!r} == 'host_packet':
        # Real packet from a trusted negative-control fixture, not a fake verdict.
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as client:
            client.sendto(b'negative-control',('127.0.0.1',owner.probe_ports[1]))
    elif {mode!r} == 'cleanup_failure':
        raw = owner.stdout.buffer.raw
        restore.append((raw,'_close_handle',raw._close_handle))
        def fail(handle):
            raise OSError('injected probe handle cleanup failure')
        raw._close_handle = fail
    elif {mode!r} == 'late_cancel':
        original_cleanup = owner.cleanup
        restore.append((owner,'cleanup',original_cleanup))
        def cleanup():
            original_cleanup()
            cancel.set()
        owner.cleanup = cleanup
    return data
launch._PythonLaunch.__init__,preparation._run_worker,probe._collect_probe_output = capture,intercept,collect
try:
    try:
        probe.run_python_probe(sys.executable,root/'cache',cancel=cancel)
    except (launch.WindowsRuntimeError,SandboxUnavailableError) as error:
        if {mode!r} == 'host_packet':
            assert 'host network endpoint' in str(error),str(error)
        elif {mode!r} == 'host_grant':
            assert error.code == 'WINDOWS_SANDBOX_PROBE_FAILED',str(error)
            assert 'host_read_denied' in str(error),str(error)
        elif {mode!r} == 'cleanup_failure':
            assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED',str(error)
            assert error.retained_launch is owners[0]
            assert owners[0] in launch._pending_cleanup and not owners[0].closed
        else:
            assert error.code == 'WINDOWS_SANDBOX_CANCELLED',str(error)
    else:
        raise AssertionError('Failed host controls/cleanup returned observations')
finally:
    launch._PythonLaunch.__init__,preparation._run_worker,probe._collect_probe_output = original_init,original_run,original_collect
    for owner,name,original in restore:
        setattr(owner,name,original)
    for owner in owners:
        owner.cleanup()
assert len(owners) == 1 and owners[0].closed
owner = owners[0]
assert not owner.handles and not owner.pins.handles and not owner.file_pins.handles
with identity._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
    assert identity._profile_path(sid) is None
assert not owner.reservation.path.exists() and not launch._pending_cleanup
assert not list((root/'cache'/'.readers').iterdir()) and not list(work.iterdir())
print('PROBE_CONTROL_FAILURE_REJECTED')
""",
    )
    assert "PROBE_CONTROL_FAILURE_REJECTED" in output
