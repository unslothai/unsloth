"""Security suite fixtures: an autouse network blocker refuses non-loopback socket.connect() so a regression reaching the internet fails loudly."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest


# Make `scripts/` importable so tests can grab scanner constants directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_LOOPBACK_PREFIXES = ("127.", "::1", "localhost")


def _is_loopback(host: str | bytes) -> bool:
    if isinstance(host, bytes):
        try:
            host = host.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if not host:
        return False
    host = host.strip()
    if host in {"::1", "localhost", "0.0.0.0"}:
        return True
    return host.startswith("127.")


class _BlockedSocket(socket.socket):
    """Socket subclass that refuses any non-loopback connect()."""

    def connect(self, address):  # type: ignore[override]
        host = None
        if isinstance(address, tuple) and address:
            host = address[0]
        if not _is_loopback(host or ""):
            raise RuntimeError(
                f"network access blocked by tests/security/conftest.py "
                f"(attempted connect to {address!r}); the scanner suite "
                "must run fully offline"
            )
        return super().connect(address)

    def connect_ex(self, address):  # type: ignore[override]
        host = None
        if isinstance(address, tuple) and address:
            host = address[0]
        if not _is_loopback(host or ""):
            raise RuntimeError(
                f"network access blocked by tests/security/conftest.py "
                f"(attempted connect_ex to {address!r})"
            )
        return super().connect_ex(address)


@pytest.fixture(autouse = True)
def network_blocker():
    """Swap socket.socket for the blocker, restored after each test.

    Per test, not per session. A session-scoped fixture in a directory conftest
    applies only to this directory, but it tears down when the SESSION ends, so
    the patch outlived the suite that wanted it and every later test that
    reaches the network died on a socket this file replaced. `security` sorts
    before `version_compat` and `vllm_compat`, whose pinned-symbol checks fetch
    upstream sources, so a full run lost about 1300 of them:

        RuntimeError: network access blocked by tests/security/conftest.py

    They passed alone and failed together, in that order only.
    """
    original = socket.socket
    socket.socket = _BlockedSocket  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original  # type: ignore[assignment]


@pytest.fixture(scope = "session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope = "session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


# The wheel and sdist fixtures embed the May-12 IOC literal on purpose, so the scanner tests can
# prove scan_packages.py trips on it. That makes them true positives for other vendors too: on
# VirusTotal malicious_sdist.tar.gz scores 2/60 and malicious_wheel.whl 2/65 (Tencent, Rising),
# and their presence is what makes Panda report Exploit/CVE-2014-6271 against GitHub's
# unsloth-main.zip. unslothai/unsloth#10060 took the test tree out of the PyPI artifacts, but the
# repository archive contains tests by construction, so committing the built archives kept the
# detections alive there and users kept re-reporting them (discussion #9577).
#
# _build.py is deterministic (SOURCE_DATE_EPOCH = 0, and zip dates pinned to the 1980 DOS epoch),
# so building at session start gives the same bytes the committed copies had. Nothing about what
# the tests assert changes; the archives simply stop existing in git.
_GENERATED_ARCHIVES = ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz")


@pytest.fixture(scope = "session", autouse = True)
def _build_archive_fixtures() -> None:
    """Build the wheel/sdist fixtures into `fixtures/` before any test reads them.

    Session-scoped and autouse because roughly a dozen call sites reach for
    `FIXTURES / "malicious_wheel.whl"` directly, and generating in place keeps every one of them
    working unchanged. Unlike `network_blocker` above, this patches nothing and has no teardown,
    so session scope carries none of the leak-past-the-suite risk that docstring describes.

    Idempotent: a rebuild writes identical bytes, so a re-run over a warm checkout is a no-op in
    content even though it rewrites the files.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures"
    if str(fixtures) not in sys.path:
        sys.path.insert(0, str(fixtures))
    import _build  # noqa: PLC0415 -- fixtures/ is only importable once the path is set above

    try:
        _build.build_all()
    except OSError as exc:
        raise RuntimeError(
            f"could not build the archive fixtures in {fixtures}: {exc}. They are generated "
            "rather than committed (see the comment above this fixture), so the security suite "
            "needs that directory to be writable."
        ) from exc

    missing = [name for name in _GENERATED_ARCHIVES if not (fixtures / name).is_file()]
    if missing:
        raise RuntimeError(
            f"_build.build_all() did not produce {missing}; the fixture builder and the names "
            "the tests read have drifted apart."
        )
