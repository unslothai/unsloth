"""Client for a running ``llmman serve`` daemon.

Used to acquire models published as CNCF ModelPack
(https://github.com/modelpack/model-spec) OCI artifacts. The daemon owns the
registry work -- ModelPack media types, registry auth, resumable blob download
and a content-addressed store -- so it is not reimplemented here.

Contract (from llmman's src/cmd/serve.rs and src/daemon.rs):
  - LLMMAN_HOST is ``[scheme://]host[:port][/path]``, default 127.0.0.1:17434.
    A wildcard bind host (0.0.0.0, ::) is rewritten to loopback, since a client
    cannot connect to "every interface".
  - ``GET /api/version`` -> ``{"version":..., "exe":..., "pid":...}``.
  - ``POST /api/pull`` ``{"model": ref}`` -> NDJSON stream of ``{"status":...}``
    objects, terminated by ``{"status":"success"}`` or ``{"error":"..."}``.
    An error can arrive in-band at HTTP 200.
  - ``llmman resolve --no-pull <ref>`` -> one line of JSON carrying ``path``.
"""

import ipaddress
import json
import logging
import os
import shutil
import subprocess
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

HOST_ENV = "LLMMAN_HOST"
BIN_ENV = "UNSLOTH_LLMMAN_BIN"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17434

PROBE_TIMEOUT_SECONDS = 5


def _connectable_host(host: str) -> str:
    """Rewrite a wildcard bind host to its loopback equivalent."""
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return host
    if not ip.is_unspecified:
        return host
    return "127.0.0.1" if ip.version == 4 else "::1"


def endpoint() -> str:
    """The http origin of the llmman daemon, honouring LLMMAN_HOST."""
    raw = os.getenv(HOST_ENV, "").strip().strip("\"'")
    if not raw:
        return f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

    if "://" in raw:
        raw = raw.split("://", 1)[1]
    raw = raw.split("/", 1)[0]

    host, port = raw, DEFAULT_PORT
    if raw.startswith("["):  # bracketed IPv6, optionally with :port
        close = raw.find("]")
        if close != -1:
            host = raw[: close + 1]
            rest = raw[close + 1 :]
            if rest.startswith(":") and rest[1:].isdigit():
                port = int(rest[1:])
    elif raw.count(":") == 1:
        maybe_host, maybe_port = raw.rsplit(":", 1)
        if maybe_port.isdigit():
            host, port = maybe_host, int(maybe_port)

    host = host or DEFAULT_HOST
    resolved = _connectable_host(host)
    if ":" in resolved and not resolved.startswith("["):
        resolved = f"[{resolved}]"
    return f"http://{resolved}:{port}"


def llmman_bin() -> str:
    """The llmman executable name, overridable per project."""
    return os.getenv(BIN_ENV, "").strip() or "llmman"


def check_daemon(base: str) -> None:
    """Confirm an llmman daemon is listening and is actually llmman."""
    url = base + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout = PROBE_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"llmman daemon at {base} answered /api/version with HTTP {resp.status}"
                )
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"no llmman daemon reachable at {base} ({exc.reason}). Start one with "
            f"`llmman serve`, or point {HOST_ENV} at an existing daemon."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"the server at {base} is not an llmman daemon (unparseable /api/version)"
        ) from exc

    if not isinstance(payload, dict) or not payload.get("version"):
        raise RuntimeError(
            f"the server at {base} is not an llmman daemon (no version in /api/version)"
        )


def pull(
    base: str,
    reference: str,
    progress = None,
) -> None:
    """Stream POST /api/pull until the daemon reports success.

    ``progress`` receives ``(status, completed, total)``. An error can arrive
    in-band at HTTP 200, and a stream that ends without ``success`` is also a
    failure -- neither is treated as a completed pull.
    """
    body = json.dumps({"model": reference}).encode("utf-8")
    req = urllib.request.Request(
        base + "/api/pull",
        data = body,
        headers = {"Content-Type": "application/json"},
        method = "POST",
    )

    succeeded = False
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 200:
                raise RuntimeError(f"llmman pull of {reference!r} failed: HTTP {resp.status}")
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a non-JSON diagnostic rather than aborting a
                    # pull that may still be progressing.
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("error"):
                    raise RuntimeError(f"llmman pull of {reference!r} failed: {obj['error']}")
                status = obj.get("status")
                if status == "success":
                    succeeded = True
                    continue
                if progress is not None and status:
                    progress(status, obj.get("completed", 0), obj.get("total", 0))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"llmman pull of {reference!r} failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llmman pull of {reference!r} failed: {exc.reason}") from exc

    if not succeeded:
        raise RuntimeError(f"llmman pull of {reference!r} ended without reporting success")


def parse_resolve_output(stdout: str, reference: str) -> str:
    """Parse ``llmman resolve`` stdout into the resolved local path."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"llmman resolve {reference!r}: no output on stdout")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"llmman resolve {reference!r}: could not parse output as JSON: {lines[-1]}"
        ) from exc

    if not isinstance(payload, dict):
        # A protocol violation rather than a caller type error, so RuntimeError
        # keeps every llmman failure one exception type for callers.
        raise RuntimeError(  # noqa: TRY004
            f"llmman resolve {reference!r}: expected a JSON object, got {lines[-1]}"
        )

    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise RuntimeError(f"llmman resolve {reference!r}: returned an empty path")
    if not os.path.exists(path):
        raise RuntimeError(f"llmman resolve {reference!r}: reported path {path!r} does not exist")
    return path


def resolve(reference: str) -> str:
    """Ask the CLI where the daemon's pull left the model on disk.

    ``--no-pull`` guarantees this only reports on bytes ``/api/pull`` already
    fetched, so the daemon stays the only thing that touches the network.
    """
    binary = llmman_bin()
    if shutil.which(binary) is None and not os.path.isfile(binary):
        raise RuntimeError(
            f"{binary!r} not found. Install llmman "
            "(https://github.com/llmmanorg/llmman) and put it on PATH, or set "
            f"{BIN_ENV} to its location."
        )

    completed = subprocess.run(
        [binary, "resolve", "--no-pull", reference],
        capture_output = True,
        stdin = subprocess.DEVNULL,
        text = True,
        check = False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"`{binary} resolve --no-pull {reference}` failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return parse_resolve_output(completed.stdout, reference)


def pull_and_resolve(reference: str, progress = None) -> str:
    """Full acquisition: probe the daemon, pull through it, report the path."""
    base = endpoint()
    check_daemon(base)
    logger.info("Pulling %s via llmman daemon at %s", reference, base)
    pull(base, reference, progress)
    return resolve(reference)


SCHEME = "oci://"


def is_oci_ref(model_name) -> bool:
    """Whether ``model_name`` carries the ``oci://`` scheme.

    An explicit scheme is required rather than sniffing a bare
    ``registry/name:tag``: that shape is indistinguishable from a HuggingFace
    repo id, so guessing would silently hijack existing model names.
    """
    if not model_name:
        return False
    return str(model_name).lower().startswith(SCHEME)


def strip_scheme(model_name) -> str:
    """Drop the ``oci://`` prefix, leaving the bare registry reference."""
    text = str(model_name)
    if is_oci_ref(text):
        return text[len(SCHEME) :]
    return text


def resolve_model(model_name) -> str:
    """Pull an ``oci://`` reference through llmman and return the local path."""
    reference = strip_scheme(model_name).strip()
    if not reference:
        raise ValueError(f"empty OCI model reference: {model_name!r}")

    def _progress(status, completed, total):
        if total:
            logger.info("llmman: %s (%s/%s bytes)", status, completed, total)
        else:
            logger.info("llmman: %s", status)

    return pull_and_resolve(reference, progress = _progress)
