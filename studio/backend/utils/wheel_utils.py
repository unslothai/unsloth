# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable

from utils.native_path_leases import child_env_without_native_path_secret
from utils.child_stdio import utf8_child_env
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

_logger = logging.getLogger(__name__)

FLASH_ATTN_RELEASE_BASE_URL = "https://github.com/Dao-AILab/flash-attention/releases/download"


# No arch gate here, deliberately: has_blackwell_gpu() skipped flash-attn while upstream shipped
# no sm_100+ wheels (#5420), then became the bug once it did (#6961). A gate encodes a snapshot of
# what upstream ships and goes stale silently both ways; the post-install import check catches a
# wheel that will not load whatever the cause.


# No arch gate, deliberately: has_blackwell_gpu() skipped flash-attn before sm_100+ wheels existed (#5420) and became
# the bug once they did (#6961). The import check catches a bad wheel.
# #6961 was B200 hosts denied a working wheel, and an arch gate encodes a snapshot of what upstream ships and goes stale
# silently both ways.
def wheel_platform_tag() -> str | None:
    """pip platform tag for this host, or None where nothing we resolve is published.

    Windows is included because download.pytorch.org publishes CUDA-matched
    ``win_amd64`` xFormers wheels (see ``xformers_wheel_url``). It is NOT included for
    flash-attn / causal-conv1d / mamba-ssm, whose upstreams publish Linux assets only --
    ``probe_torch_wheel_env`` keeps that gate, not this function.
    """
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            return "linux_x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
    elif sys.platform == "win32":
        if machine in {"x86_64", "amd64"}:
            return "win_amd64"
        # Windows on ARM: no CUDA, and no win_arm64 wheel on any index.
    # No prebuilt wheels published for macOS
    return None


def probe_torch_wheel_env(
    *, timeout: int | None = None, include_windows: bool = False
) -> dict[str, str] | None:
    """Describe the resident torch build for wheel-URL resolution, or None.

    Windows is opt-in via ``include_windows``: every existing caller resolves a
    flash-attn / causal-conv1d / mamba-ssm asset, and those projects publish no
    win_amd64 wheels at all, so returning an env there would only build 404s.
    """
    platform_tag = wheel_platform_tag()
    if platform_tag is None:
        return None
    if platform_tag == "win_amd64" and not include_windows:
        return None

    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys, re, torch; "
                    "parts = torch.__version__.split('+', 1)[0].split('.')[:2]; "
                    "minor = re.sub(r'[^0-9].*', '', parts[1]) if len(parts) > 1 else '0'; "
                    "torch_mm = parts[0] + '.' + minor; "
                    "print(json.dumps({"
                    "'python_tag': f'cp{sys.version_info.major}{sys.version_info.minor}', "
                    "'torch_mm': torch_mm, "
                    # xFormers publishes one wheel per exact torch PATCH and per CUDA MINOR, so 'torch_mm' /
                    # 'cuda_major' cannot pick between them.
                    # Full release + full CUDA version: cu126 and cu128 are different builds of the same version string.
                    "'torch_version': str(torch.__version__), "
                    "'cuda_version': str(torch.version.cuda) if torch.version.cuda else '', "
                    "'cuda_major': str(int(str(torch.version.cuda).split('.', 1)[0])) if torch.version.cuda else '', "
                    "'hip_version': str(torch.version.hip) if getattr(torch.version, 'hip', None) else '', "
                    "'cxx11abi': str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()"
                    "}))"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            env = utf8_child_env(child_env_without_native_path_secret()),
            **windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return None

    if probe.returncode != 0:
        return None

    try:
        env = json.loads(probe.stdout.strip())
    except json.JSONDecodeError:
        return None
    env["platform_tag"] = platform_tag
    return env


# torch 2.11/2.12 ship no native prebuilt flash-attn / causal-conv1d / mamba-ssm wheels, but the
# torch2.10 CUDA wheels load and pass each project's own suite on both, so they are reused. The
# window is bounded, not open ended: torch broke extension ABI between 2.9 and 2.10, so every new
# key here must be measured against the real wheels before it is added.
# Measured on B200, py3.12, torch 2.12.1+cu130: causal-conv1d 9412 passed / 3888 skipped / 0 failed, mamba tests/ops 20
# passed, flash-attn splitkv+qkvpacked 848 passed, identical to a torch 2.10 control. The torch2.9 flash-attn .so raises
# "undefined symbol" on torch 2.10 and 2.12 alike.
_PREBUILT_WHEEL_TORCH_MM = {"2.11": "2.10", "2.12": "2.10"}


def prebuilt_wheel_torch_mm(torch_mm: str) -> str:
    """Map a torch major.minor to the one whose prebuilt accelerator wheels to use."""
    return _PREBUILT_WHEEL_TORCH_MM.get(torch_mm, torch_mm)


def direct_wheel_url(
    *,
    filename_prefix: str,
    package_version: str,
    release_tag: str,
    release_base_url: str,
    env: dict[str, str] | None,
) -> str | None:
    if env is None or not env.get("cuda_major"):
        return None

    filename = (
        f"{filename_prefix}-{package_version}"
        f"+cu{env['cuda_major']}torch{prebuilt_wheel_torch_mm(env['torch_mm'])}"
        f"cxx11abi{env['cxx11abi']}-{env['python_tag']}-{env['python_tag']}"
        f"-{env['platform_tag']}.whl"
    )
    return f"{release_base_url}/{release_tag}/{filename}"


# xformers/_C is linked against ONE exact (torch, CUDA) pair, and a mismatch is only a log
# warning, so the import "succeeds" with memory-efficient attention silently gone. PyPI publishes
# one win_amd64 flavour whose CUDA family churns across releases, which is why this resolves an
# exact download.pytorch.org URL instead of pinning a version. Keyed on the `torch` field of
# cpp_lib.json, not `cuda`, which is the NVCC toolkit version and does not separate flavours.
# Rows are exact, never interpolated: the extension ABI does not survive a torch minor bump, and
# an unlisted pair means "install nothing", the safe answer. cu118/cu121/cu124 are absent because
# they stop before the cp39-abi3 switch at 0.0.31, so one filename template cannot name them.
# The PyPI win wheel has been cu124 (0.0.29.post2), cu126 (0.0.30), cu128 (0.0.32), cu130 (0.0.33) and cu128 again
# (0.0.33.post1 onward); download.pytorch.org's cu126 0.0.34 also reports 1208, so only the `torch` field
# ("2.10.0+cu128") separates flavours. Every row was HEAD-verified live, e.g.
# cu130/xformers-0.0.34-cp39-abi3-win_amd64.whl reports {"torch": "2.10.0+cu130"}. Keying on the CUDA MINOR is stricter
# than the ABI needs (cu126 and cu128 both link libcudart.so.12; only a major bump changes it), but it names a real
# directory, so torch 2.10.0+cu129 on Linux resolves to nothing. torch 2.11+ maps to 0.0.35, compiled against 2.10.0 and
# compatible with any later version since xFormers moved to the stable API/ABI in 0.0.34. Keep in step with
# $script:XformersWheelVersions in install.ps1 and the matrix in tests/python/test_windows_xformers_wheel_match.py.
# ── xFormers ──────────────────────────────────────────────────────────────────
PYTORCH_WHEEL_INDEX_BASE_URL = "https://download.pytorch.org/whl"


def pytorch_wheel_index_base_url() -> str:
    """Where torch-family wheels are fetched from: ``UNSLOTH_PYTORCH_MIRROR`` when set.

    Read per call rather than frozen at import: this module is imported early, and the
    mirror is the one setting an air-gapped deployment has. The whole installer stack
    already honours it (``install_python_stack._PYTORCH_WHL_BASE``, install.sh, setup.ps1),
    so a direct-URL install that hard-coded download.pytorch.org was the one path that
    could not reach a mirror-only host -- it failed the explicit xFormers request and
    dropped the user back to native attention.
    """
    return (os.environ.get("UNSLOTH_PYTORCH_MIRROR") or PYTORCH_WHEEL_INDEX_BASE_URL).rstrip("/")


_XFORMERS_WHEEL_VERSIONS: dict[str, dict[str, str]] = {
    # torch 2.7.0 is deliberately absent: it predates the stable-ABI switch, so it ships one wheel per interpreter and
    # stops at cp312, while Unsloth's default interpreter is 3.13.
    # Supporting it would mean a per-interpreter gate here and a second one in install.ps1, for a torch that resolves to
    # nothing on the default install anyway (xFormers 0.0.30).
    "2.7.1": {"cu126": "0.0.31.post1", "cu128": "0.0.31.post1"},
    "2.8.0": {"cu126": "0.0.32.post2", "cu128": "0.0.32.post2", "cu129": "0.0.32.post2"},
    "2.9.0": {"cu126": "0.0.33.post1", "cu128": "0.0.33.post1", "cu130": "0.0.33.post1"},
    "2.9.1": {"cu126": "0.0.33.post2", "cu128": "0.0.33.post2", "cu130": "0.0.33.post2"},
    "2.10.0": {"cu126": "0.0.34", "cu128": "0.0.34", "cu130": "0.0.34"},
    # Stable-ABI era: one wheel serves every torch from 2.11 on. The rows stay listed so a
    # future exact-pinned release can displace a single one of them, but they are no longer
    # the only way in: _XFORMERS_STABLE_ABI below covers the patch releases between them.
    "2.11.0": {"cu126": "0.0.35", "cu128": "0.0.35", "cu130": "0.0.35"},
    "2.12.0": {"cu126": "0.0.35", "cu128": "0.0.35", "cu130": "0.0.35"},
    "2.13.0": {"cu126": "0.0.35", "cu128": "0.0.35", "cu130": "0.0.35"},
}

# The stable-ABI floor and what serves it: every torch STRICTLY ABOVE this maps to this release,
# per CUDA family, with exact rows above still winning. An exact-key table alone refused the patch
# releases (2.10.1, 2.11.1, 2.12.1), which cannot be enumerated because they ship after this code;
# 0.0.35 targets 2.10.0 and upstream states later versions stay compatible. Below the floor there
# is no stable ABI, so an unlisted pair must keep resolving to nothing.
_XFORMERS_STABLE_ABI_FLOOR: tuple[int, ...] = (2, 10, 0)
_XFORMERS_STABLE_ABI_VERSIONS = {"cu126": "0.0.35", "cu128": "0.0.35", "cu130": "0.0.35"}

# The interpreter tag in the wheel FILENAME, which xFormers has changed twice: 0.0.30 and earlier ship one wheel per
# cpXY (and stop at cp312), 0.0.31..0.0.34 ship a single cp39-abi3 wheel, and 0.0.35 switched to py39-none. That last
# switch is a PACKAGING change, not an architectural one: 0.0.35's setup.py drops py_limited_api=True and force-tags the
# wheel through a custom bdist_wheel, since the extension is loaded by torch.ops.load_library and its _C.so defines no
# PyInit. The wheel still carries a per-CUDA _C.pyd; it just dropped the bundled flash_attn_3 kernels, the whole 103 MB
# -> 2.6 MB difference. Ranges, not an open-ended floor: an unknown release resolves to nothing until somebody checks
# the real filename.
_XFORMERS_FILENAME_PYTHON_TAGS: tuple[tuple[tuple[int, ...], tuple[int, ...], str], ...] = (
    ((0, 0, 31), (0, 0, 34), "cp39-abi3"),
    ((0, 0, 35), (0, 0, 35), "py39-none"),
)

# platform_tag from wheel_platform_tag() -> the leaf in the wheel filename.
# aarch64 and macOS are absent because download.pytorch.org publishes no xFormers wheel for them.
_XFORMERS_PLATFORM_LEAVES = {
    "linux_x86_64": "manylinux_2_28_x86_64",
    "win_amd64": "win_amd64",
}


def _xformers_version_tuple(version: str) -> tuple[int, ...]:
    """'0.0.33.post1' -> (0, 0, 33). Stops at the first non-numeric component."""
    parts: list[int] = []
    for chunk in str(version).split("."):
        digits = re.sub(r"[^0-9].*", "", chunk)
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def xformers_filename_python_tag(version: str) -> str | None:
    """The interpreter tag in an xFormers wheel filename, or None for an unknown release."""
    parsed = _xformers_version_tuple(version)
    if not parsed:
        return None
    for low, high, tag in _XFORMERS_FILENAME_PYTHON_TAGS:
        if low <= parsed <= high:
            return tag
    return None


def xformers_cuda_family(cuda_version: str | None) -> str | None:
    """torch.version.cuda -> the download.pytorch.org index leaf ('12.8' -> 'cu128').

    None for a ROCm / CPU / XPU torch, which has no xFormers wheel anywhere.
    """
    if not cuda_version:
        return None
    parts = str(cuda_version).strip().split(".")
    try:
        major = int(re.sub(r"[^0-9].*", "", parts[0]))
        minor = int(re.sub(r"[^0-9].*", "", parts[1])) if len(parts) > 1 else 0
    except (IndexError, ValueError):
        return None
    return f"cu{major}{minor}"


def xformers_wheel_version(torch_version: str | None, cuda_family: str | None) -> str | None:
    """The xFormers release for this (torch, CUDA family), else None.

    An exact row wins. Failing that, any release above the stable-ABI floor resolves to the
    wheel that serves that whole era: the exact table cannot list patch releases that are
    published after this code ships, and refusing them left supported builds (2.11.1,
    2.12.1) with no xFormers at all.
    """
    if not torch_version or not cuda_family:
        return None
    # '2.10.0+cu130' -> '2.10.0'. A dev/rc torch has no wheel and must miss the table.
    release = str(torch_version).split("+", 1)[0].strip()
    exact = _XFORMERS_WHEEL_VERSIONS.get(release, {}).get(cuda_family)
    if exact is not None:
        return exact
    # A dev/nightly/rc suffix ('2.11.0.dev20260101') is not a released torch, so it stays
    # out: _xformers_version_tuple stops at the first non-numeric chunk, which would read
    # it as the release itself.
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", release):
        return None
    if _xformers_version_tuple(release) > _XFORMERS_STABLE_ABI_FLOOR:
        return _XFORMERS_STABLE_ABI_VERSIONS.get(cuda_family)
    return None


def xformers_wheel_url(env: dict[str, str] | None) -> str | None:
    """Direct URL of the xFormers wheel matching ``env``'s torch build, else None.

    None means "no matched wheel exists" and callers must install nothing rather than
    fall back to an unpinned resolve -- an unpinned install is what produces the
    mismatched extension in the first place.
    """
    if env is None:
        return None
    platform_leaf = _XFORMERS_PLATFORM_LEAVES.get(str(env.get("platform_tag") or ""))
    if platform_leaf is None:
        return None
    family = xformers_cuda_family(env.get("cuda_version"))
    version = xformers_wheel_version(env.get("torch_version"), family)
    if version is None:
        return None
    python_tag = xformers_filename_python_tag(version)
    if python_tag is None:
        return None
    return join_wheel_url(
        pytorch_wheel_index_base_url(),
        f"{family}/xformers-{version}-{python_tag}-{platform_leaf}.whl",
    )


def join_wheel_url(base: str, path: str) -> str:
    """``base`` + ``path``, with any ?query / #fragment kept at the end.

    UNSLOTH_PYTORCH_MIRROR is allowed to authenticate by query string
    (``https://mirror/whl?token=abc``), and appending after the query put the wheel path
    INSIDE the token value -- leaving the request path at /whl and the token unusable. The
    tokenized private mirror this setting exists for was the one shape that could not
    resolve a wheel.
    """
    cut = min([i for i in (base.find("?"), base.find("#")) if i >= 0], default = -1)
    if cut < 0:
        return f"{base.rstrip('/')}/{path}"
    return f"{base[:cut].rstrip('/')}/{path}{base[cut:]}"


def redact_url_credentials(url: str) -> str:
    """A URL safe to log: no userinfo, no query, no fragment.

    UNSLOTH_PYTORCH_MIRROR is allowed to be a private index, and people put credentials in
    it -- ``https://user:token@mirror/whl`` or ``...?token=``. The wheel URL built from it
    is handed to pip AND printed, so without this the secret lands in the backend log the
    first time Unsloth installs (or fails to install) xFormers. Same rule as the installer's
    Remove-IndexUrlCredentials, so both sides redact identically.
    """
    separator = url.find("://")
    if separator < 0:
        return url
    scheme, rest = url[:separator], url[separator + 3 :]
    cut = min([i for i in (rest.find("?"), rest.find("#")) if i >= 0], default = -1)
    if cut >= 0:
        rest = rest[:cut]
    slash = rest.find("/")
    authority, path = (rest[:slash], rest[slash:]) if slash >= 0 else (rest, "")
    at = authority.rfind("@")
    if at >= 0:
        authority = authority[at + 1 :]
    return f"{scheme}://{authority}{path}"


def flash_attn_package_version(torch_mm: str) -> str | None:
    if torch_mm == "2.10":
        # Newest flash-attn release still carrying the full torch2.10 asset matrix. Do not bump to "the
        # latest release": v2.8.3 publishes only cu13/cp312 for torch2.10 and v2.8.3.post1 dropped every
        # torch2.10 asset, 404ing most users into a source build.
        # The full matrix is cu12 + cu13, cp312 + cp313, x86_64 + aarch64, and post1's newest tag is torch2.9, which
        # will not load here at all.
        return "2.8.1"
    try:
        major, minor = (int(part) for part in torch_mm.split(".", 1))
    except ValueError:
        return None
    if major == 2 and 4 <= minor <= 9:
        return "2.8.3"
    return None


def flash_attn_wheel_url(env: dict[str, str] | None) -> str | None:
    if env is None:
        return None
    package_version = flash_attn_package_version(prebuilt_wheel_torch_mm(env["torch_mm"]))
    if package_version is None:
        return None
    return direct_wheel_url(
        filename_prefix = "flash_attn",
        package_version = package_version,
        release_tag = f"v{package_version}",
        release_base_url = FLASH_ATTN_RELEASE_BASE_URL,
        env = env,
    )


def install_wheel(
    wheel_url: str,
    *,
    python_executable: str,
    use_uv: bool,
    uv_needs_system: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[tuple[str, subprocess.CompletedProcess[str]]]:
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []

    # Try uv first if available, then fall back to pip
    if use_uv and shutil.which("uv"):
        uv_cmd = ["uv", "pip", "install"]
        if uv_needs_system:
            uv_cmd.append("--system")
        uv_cmd.extend(["--python", python_executable, "--no-deps", wheel_url])
        result = run(
            uv_cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = child_env_without_native_path_secret(),
        )
        attempts.append(("uv", result))
        if result.returncode == 0:
            return attempts

    pip_cmd = [python_executable, "-m", "pip", "install", "--no-deps", wheel_url]
    result = run(
        pip_cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        # Make the Python child emit the UTF-8 we decode above.
        env = utf8_child_env(child_env_without_native_path_secret()),
    )
    attempts.append(("pip", result))
    return attempts


def url_exists(url: str) -> bool:
    try:
        request = urllib.request.Request(url, method = "HEAD")
        with urllib.request.urlopen(request, timeout = 10):
            return True
    except urllib.error.HTTPError as exc:
        _logger.debug("url_exists(%s): HTTP %s", url, exc.code)
    except (urllib.error.URLError, TimeoutError) as exc:
        _logger.debug("url_exists(%s): %s", url, exc)
    return False
