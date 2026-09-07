import json
import platform
import subprocess
import sys

import pytest

from core.inference import srt_seccomp as guard


def evaluate(machine, arch, number, family=0):
    code = guard.program(machine)
    data = {0: number, 4: arch, 16: family}
    accumulator = 0
    index = 0
    while index < len(code):
        op, yes, no, value = code[index]
        if op == 0x20:
            accumulator = data[value]
        elif op == 0x15:
            index += yes if accumulator == value else no
        elif op == 0x45:
            index += yes if accumulator & value else no
        elif op == 0x06:
            return value
        else:
            raise AssertionError(op)
        index += 1
    raise AssertionError("filter fell through")


@pytest.mark.parametrize("machine", guard.ABIS)
def test_exact_syscall_predicates(machine):
    arch, sock, pair = guard.ABIS[machine]
    for number in range(0, 550):
        for family in (0, 1, 2, 10, 40):
            expected = guard.DENY if number in (425,426,427) or (number in (sock,pair) and family == 40) else guard.ALLOW
            assert evaluate(machine, arch, number, family) == expected
    assert evaluate(machine, arch ^ 1, sock) == guard.KILL
    if machine == "x86_64":
        for number in (0, sock, pair, 425, 426, 427):
            assert evaluate(machine, arch, number | 0x40000000) == guard.KILL


def test_unsupported_architecture():
    with pytest.raises(RuntimeError):
        guard.program("i686")


def test_unavailable_install_fails_closed(monkeypatch):
    monkeypatch.setattr(guard, "_prctl", None)
    with pytest.raises(RuntimeError):
        guard.install()


def test_install_order_and_descriptor(monkeypatch):
    calls = []
    monkeypatch.setattr(guard, "_prctl", lambda *args: calls.append(args) or 0)
    guard.install()
    assert calls == [(38,1,0,0,0), (22,2,guard._pointer,0,0)]


@pytest.mark.parametrize("results", [[-1], [0,-1]])
def test_install_failure_is_closed(monkeypatch, results):
    calls = []
    def fake(*args):
        calls.append(args)
        return results[len(calls)-1]
    monkeypatch.setattr(guard, "_prctl", fake)
    with pytest.raises(RuntimeError):
        guard.install()
    assert len(calls) == len(results)
    assert calls[0] == (38,1,0,0,0)


@pytest.mark.skipif(sys.platform != "linux" or platform.machine().lower() not in guard.ABIS,
                    reason="native Linux child required")
def test_native_child_vsock_uring_denied_unix_allowed():
    code = r'''
import ctypes, errno, json, platform, socket
from core.inference import srt_seccomp as guard
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
arch, sock, pair = guard.ABIS[platform.machine().lower()]
results = []
for nr, args in [(sock,(40,1,0)), (pair,(40,1,0,0)), (425,(0,0)), (426,(-1,0,0,0,0,0)), (427,(-1,0,0,0))]:
    ctypes.set_errno(0)
    value = libc.syscall(nr, *args)
    assert value == -1 and ctypes.get_errno() == errno.EPERM, (nr,value,ctypes.get_errno())
    results.append(nr)
a,b = socket.socketpair()
a.sendall(b'private')
assert b.recv(7) == b'private'
a.close(); b.close()
print(json.dumps(results))
'''
    child = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=15,
                           preexec_fn=guard.install)
    assert child.returncode == 0, child.stderr
    assert len(json.loads(child.stdout)) == 5
