# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Installed bootstrap handle exclusion with real objects and leak controls."""

import sys

import pytest

from test_launch import run_harness, installed_runtime, runtime_wheel, LAUNCH

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows handle inheritance")


HOST_RESOURCES = r"""
from contextlib import contextmanager
import ctypes, socket, _winapi
from ctypes import wintypes as W
from core.inference.windows_sandbox import probe
kernel = probe._host_pipe_api()
kernel.CreateEventW.argtypes = [ctypes.c_void_p,W.BOOL,W.BOOL,W.LPCWSTR]
kernel.CreateEventW.restype = W.HANDLE
kernel.GetHandleInformation.argtypes = [W.HANDLE,ctypes.POINTER(W.DWORD)]
kernel.GetHandleInformation.restype = W.BOOL
kernel.SetHandleInformation.argtypes = [W.HANDLE,W.DWORD,W.DWORD]
kernel.SetHandleInformation.restype = W.BOOL
kernel.GetProcessId.argtypes = [W.HANDLE]
kernel.GetProcessId.restype = W.DWORD
kernel.GetFileType.argtypes = [W.HANDLE]
kernel.GetFileType.restype = W.DWORD
kernel.SetEvent.argtypes = [W.HANDLE]
kernel.SetEvent.restype = W.BOOL

@contextmanager
def resources():
    raw, padding = set(),[]
    connection = None
    handles = {}
    def own(handle):
        assert handle not in (None,0,ctypes.c_void_p(-1).value)
        raw.add(handle)
        return handle
    def close(handle):
        assert kernel.CloseHandle(handle)
        raw.remove(handle)
    try:
        # Bound the fixture; targets stay well above ordinary startup handles.
        for _ in range(8192):
            padding.append(own(kernel.CreateEventW(None,True,False,None)))
            if padding[-1] >= 0x4000:
                break
        else:
            raise AssertionError('High handle fixture exceeded its bound')
        external=root/'host-owned-file'
        external.write_bytes(b'owned fixture only')
        handles['file']=own(kernel.CreateFileW(str(external),0x80000000,7,None,3,0,None))
        handles['directory']=own(kernel.CreateFileW(str(root),1,7,None,3,0x02000000,None))
        reader,writer=_winapi.CreatePipe(None,4096)
        handles['pipe_read'],handles['pipe_write']=own(reader),own(writer)
        handles['process']=own(kernel.OpenProcess(0x1000,False,os.getpid()))
        handles['event']=own(kernel.CreateEventW(None,True,False,None))
        connection=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        connection.bind(('127.0.0.1',0))
        connection.settimeout(1)
        connection.set_inheritable(True)
        handles['socket']=connection.fileno()
        for handle in handles.values():
            assert handle >= 0x4000, ('Target slot not high',handle)
            assert kernel.SetHandleInformation(handle,1,1)
        for handle in padding:
            close(handle)
        padding.clear()
        def positive():
            for handle in handles.values():
                flags=W.DWORD()
                assert kernel.GetHandleInformation(handle,ctypes.byref(flags))
                assert flags.value & 1
            assert kernel.GetFileType(handles['file']) == 1
            assert kernel.GetFileType(handles['directory']) == 1
            assert kernel.GetProcessId(handles['process']) == os.getpid()
            assert kernel.SetEvent(handles['event'])
            assert kernel.WaitForSingleObject(handles['event'],0)==0
            marker=b'owned pipe control'
            assert _winapi.WriteFile(writer,marker)[0]==len(marker)
            assert _winapi.ReadFile(reader,len(marker))[0]==marker
            connection.sendto(marker,connection.getsockname())
            assert connection.recv(128)==marker
            assert external.read_bytes()==b'owned fixture only'
        positive()
        yield handles,connection.getsockname(),positive
    finally:
        if connection is not None:
            connection.close()
        for handle in tuple(raw):
            close(handle)
        for handle in handles.values():
            flags=W.DWORD()
            assert not kernel.GetHandleInformation(handle,ctypes.byref(flags))
            assert ctypes.get_last_error()==6
"""


CHILD_CONTROL = r"""
import ctypes, socket
from ctypes import wintypes as W
kernel=ctypes.WinDLL('kernel32',use_last_error=True,winmode=0x800)
kernel.GetHandleInformation.argtypes=[W.HANDLE,ctypes.POINTER(W.DWORD)]
kernel.GetHandleInformation.restype=W.BOOL
for name,handle in HANDLES.items():
    flags=W.DWORD()
    assert kernel.GetHandleInformation(handle,ctypes.byref(flags)),name
with socket.socket(fileno=HANDLES['socket']) as connection:
    assert connection.getsockname()==ENDPOINT
print('HOST_INHERITANCE_POSITIVE',flush=True)
"""


PAYLOAD = r"""
import ctypes, socket
from ctypes import wintypes as W
kernel=ctypes.WinDLL('kernel32',use_last_error=True,winmode=0x800)
kernel.GetHandleInformation.argtypes=[W.HANDLE,ctypes.POINTER(W.DWORD)]
kernel.GetHandleInformation.restype=W.BOOL
kernel.GetProcessMitigationPolicy.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
kernel.GetProcessMitigationPolicy.restype=W.BOOL
policy=W.DWORD()
assert kernel.GetProcessMitigationPolicy(W.HANDLE(-1),3,ctypes.byref(policy),ctypes.sizeof(policy))
for name,handle in HANDLES.items():
    flags=W.DWORD()
    try:
        valid=kernel.GetHandleInformation(handle,ctypes.byref(flags))
    except OSError as error:
        assert policy.value & 1 and error.winerror & 0xffffffff == 0xc0000008,(name,repr(error),policy.value)
    else:
        assert not valid,('HANDLE_LEAK',name)
        assert ctypes.get_last_error()==6,(name,ctypes.get_last_error())
print('NATIVE_HANDLES_EXCLUDED',flush=True)
if EXERCISE_SOCKET:
    print('SOCKET_REFERENCE_STRICT_'+str(policy.value & 1),flush=True)
    try:
        connection=socket.socket(fileno=HANDLES['socket'])
    except OSError as error:
        assert error.winerror==10038,repr(error)
    else:
        connection.close()
        raise AssertionError('Host socket leaked')
print('POSTDROP_HANDLES_EXCLUDED',flush=True)
"""


@pytest.mark.parametrize("mode", ["normal", "leak_control", "invalid_socket"])
def test_installed_bootstrap_excludes_live_host_handles(installed_runtime, tmp_path, mode):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
{HOST_RESOURCES}
from core.inference.windows_sandbox import launch
with resources() as (handles,endpoint,positive):
    source='HANDLES='+repr(handles)+chr(10)+'ENDPOINT='+repr(endpoint)+chr(10)+{CHILD_CONTROL!r}
    startup=subprocess.STARTUPINFO()
    startup.lpAttributeList={{'handle_list':list(handles.values())}}
    control=subprocess.run([sys.executable,'-I','-c',source],startupinfo=startup,close_fds=True,
        capture_output=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    assert control.returncode==0,control.stderr
    assert control.stdout.strip()==b'HOST_INHERITANCE_POSITIVE'
    positive()
    script.write_text('HANDLES='+repr(handles)+chr(10)+'EXERCISE_SOCKET='+repr({mode!r}=='invalid_socket')+chr(10)+{PAYLOAD!r},encoding='utf-8')
    {LAUNCH.replace(chr(10), chr(10) + "    ")}
    original=launch.create_suspended_host
    def leak(*args,**kwargs):
        kwargs['control_handles']=(*kwargs['control_handles'],handles['file'])
        return original(*args,**kwargs)
    if {mode!r}=='leak_control':
        launch.create_suspended_host=leak
    try:
        process=spawn_prepared_launch(prepared,**kwargs)
        assert prepared.execution_record is None
        data=process.stdout.read()
        code=process.wait(timeout=10)
        if {mode!r}=='leak_control':
            assert code!=0 and "('HANDLE_LEAK', 'file')" in data,data
            assert 'POSTDROP_HANDLES_EXCLUDED' not in data,data
        elif {mode!r}=='invalid_socket' and code & 0xffffffff == 0xc0000008:
            # Winsock's native C call has no ctypes SEH boundary. Strict handle
            # policy terminates that intentionally invalid operation instead.
            assert data.splitlines()==['NATIVE_HANDLES_EXCLUDED','SOCKET_REFERENCE_STRICT_1'],(code,data)
        else:
            expected=['NATIVE_HANDLES_EXCLUDED']
            if {mode!r}=='invalid_socket':
                expected.append('SOCKET_REFERENCE_STRICT_1')
            expected.append('POSTDROP_HANDLES_EXCLUDED')
            assert code==0 and data.splitlines()==expected,(code,data)
    finally:
        launch.create_suspended_host=original
        prepared.cleanup()
    assert owner.closed and not prepared.cleanup_diagnostics
    assert not manifest.exists() and not launch._pending_cleanup
    assert not owner.handles and not owner.file_pins.handles and not owner.pins.handles
    assert not list((root/'cache'/'.readers').iterdir())
    positive()
print('INHERITANCE_CHECK_COMPLETE')
""",
    )
    assert "INHERITANCE_CHECK_COMPLETE" in output


def test_host_handle_fixture_cleans_up_after_failure(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
{HOST_RESOURCES}
try:
    with resources() as (_,_,positive):
        positive()
        raise RuntimeError('owned fixture failure')
except RuntimeError as error:
    assert str(error)=='owned fixture failure'
else:
    raise AssertionError('Failure fixture did not run')
print('HANDLE_FIXTURE_CLEANED')
""",
    )
    assert "HANDLE_FIXTURE_CLEANED" in output
