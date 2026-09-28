# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load ``oci://`` (CNCF ModelPack) models through a running ``llmman serve``.

The daemon pulls (``POST /api/pull``, NDJSON); ``llmman resolve --no-pull``
then prints the extracted local directory. Stdlib only.
"""

import ipaddress
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request

SCHEME = "oci://"
HOST_ENV = "LLMMAN_HOST"
BIN_ENV = "UNSLOTH_LLMMAN_BIN"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17434
PROBE_TIMEOUT_SECONDS = 5


def is_oci_ref(model_name) -> bool:
    # Explicit scheme only: a bare registry/name:tag looks like an HF repo id.
    return bool(model_name) and str(model_name).lower().startswith(SCHEME)


def strip_scheme(model_name) -> str:
    text = str(model_name)
    return text[len(SCHEME) :] if is_oci_ref(text) else text


def endpoint() -> str:
    """Daemon origin from LLMMAN_HOST (``[scheme://]host[:port][/path]``)."""
    raw = os.getenv(HOST_ENV, "").strip().strip("\"'")
    raw = raw.split("://", 1)[-1].split("/", 1)[0]
    host, port = raw, DEFAULT_PORT
    if raw.startswith("[") and "]" in raw:
        host, rest = raw[1:].split("]", 1)
        if rest[1:].isdigit():
            port = int(rest[1:])
    elif raw.count(":") == 1 and raw.rsplit(":", 1)[1].isdigit():
        host, port = raw.rsplit(":", 1)
        port = int(port)
    host = host or DEFAULT_HOST
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return f"http://{host}:{port}"
    if ip.is_unspecified:  # a wildcard bind is not connectable
        ip = ipaddress.ip_address("127.0.0.1" if ip.version == 4 else "::1")
    return f"http://{f'[{ip}]' if ip.version == 6 else ip}:{port}"


def check_daemon(base: str) -> None:
    """Fail unless an llmman daemon answers ``GET /api/version``."""
    try:
        with urllib.request.urlopen(base + "/api/version", timeout = PROBE_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read())
    except OSError as exc:  # URLError, timeouts, resets
        raise RuntimeError(
            f"no llmman daemon reachable at {base} ({getattr(exc, 'reason', exc)}). Start one with "
            f"`llmman serve`, or point {HOST_ENV} at an existing daemon."
        ) from exc
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or not payload.get("version"):
        raise RuntimeError(f"the server at {base} is not an llmman daemon")


def pull(
    base: str,
    reference: str,
    progress = None,
) -> None:
    """Stream ``POST /api/pull``; errors can arrive in-band at HTTP 200."""
    fail = f"llmman pull of {reference!r} failed"
    req = urllib.request.Request(
        base + "/api/pull",
        data = json.dumps({"model": reference}).encode(),
        headers = {"Content-Type": "application/json"},
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                try:
                    obj = json.loads(line)
                except ValueError:  # blank or non-JSON diagnostic line
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("error"):
                    raise RuntimeError(f"{fail}: {obj['error']}")
                status = obj.get("status")
                if status == "success":
                    return
                if progress is not None and status:
                    progress(status, obj.get("completed", 0), obj.get("total", 0))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors = "replace").strip()
        raise RuntimeError(f"{fail}: HTTP {exc.code} {body}".rstrip()) from exc
    except OSError as exc:
        raise RuntimeError(f"{fail}: {getattr(exc, 'reason', exc)}") from exc
    raise RuntimeError(f"llmman pull of {reference!r} ended without reporting success")


def parse_resolve_output(stdout: str, reference: str) -> str:
    """Return the existing ``path`` from ``llmman resolve``'s last JSON line."""
    fail = f"llmman resolve {reference!r}"
    lines = stdout.strip().splitlines()
    try:
        payload = json.loads(lines[-1])
    except (IndexError, ValueError):
        raise RuntimeError(f"{fail}: unparseable output {stdout!r}") from None
    path = payload.get("path") if isinstance(payload, dict) else None
    if not isinstance(path, str) or not os.path.isdir(path):
        raise RuntimeError(f"{fail}: no model directory at {path!r}")
    return path


def resolve(reference: str) -> str:
    """Local path of an already-pulled reference (``--no-pull``: no network)."""
    binary = os.getenv(BIN_ENV, "").strip() or "llmman"
    if shutil.which(binary) is None:
        raise RuntimeError(
            f"{binary!r} not found. Install llmman "
            "(https://github.com/llmmanorg/llmman) and put it on PATH, or set "
            f"{BIN_ENV} to its location."
        )
    cmd = [binary, "resolve", "--no-pull", reference]
    proc = subprocess.run(cmd, capture_output = True, stdin = subprocess.DEVNULL, text = True)
    if proc.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` exited {proc.returncode}: {proc.stderr.strip()}")
    return parse_resolve_output(proc.stdout, reference)


def resolve_model(model_name) -> str:
    """Pull an ``oci://`` reference through llmman and return its local dir."""
    reference = strip_scheme(model_name).strip()
    if not reference:
        raise ValueError(f"empty OCI model reference: {model_name!r}")
    base = endpoint()
    check_daemon(base)
    print(f"Unsloth: pulling {reference} via llmman at {base}")
    last = None

    def _progress(status, completed, total):
        nonlocal last
        if status != last:  # one line per phase, not per byte update
            last = status
            print(f"Unsloth: llmman {status}")

    pull(base, reference, _progress)
    return resolve(reference)


def maybe_resolve(model_name, use_exact_model_name = False):
    """Map ``oci://`` to a local dir and skip the HF-only name mappers."""
    if not is_oci_ref(model_name):
        return model_name, use_exact_model_name
    return resolve_model(model_name), True
