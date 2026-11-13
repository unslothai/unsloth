"""Security suite fixtures: an autouse network blocker refuses non-loopback socket.connect() so a regression reaching the internet fails loudly."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest


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
    """Per test, not per session: a session-scoped patch outlived tests/security and broke later suites."""
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


# The fixtures embed the May-12 IOC on purpose, so committed archives trip AV on GitHub's zip.
# _build.py is deterministic, so building at session start reproduces the same bytes.
_GENERATED_ARCHIVES = ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz")


@pytest.fixture(scope = "session", autouse = True)
def _build_archive_fixtures() -> None:
    """Autouse and session-scoped because many tests read fixtures/ paths directly; it patches nothing."""
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
