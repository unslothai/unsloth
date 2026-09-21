# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-warmed ``torch.compile`` cache for the diffusion denoiser (Mega-cache).

The regional compile of the repeated denoiser block pays a one-time 25-58s compile on
the FIRST image after a load. This lets that cost be paid ONCE (by the distributor ahead
of time or the user on first run) and reused on every later load via torch's portable
Mega-cache (``save_cache_artifacts`` / ``load_cache_artifacts``, torch >= 2.7).

PORTABILITY IS NOT UNIVERSAL: an artifact is only valid for the SAME torch/Triton/CUDA
build, GPU arch, and model graph (family, dtype, quant, attention backend, compile
kwargs, shape bucket). torch validates these on load; a mismatch yields no hit, not an
error. Hence an EXACT-MATCH fingerprint with SILENT FALLBACK to local compile (a miss is
normal), and per-arch bundles keyed by the full fingerprint. See
``outputs/compile_cache/DISTRIBUTION.md``.

GGUF loads participate too (fingerprinted ``quant="gguf"``, a different compiled graph
than the dense family): ``torch.compile`` mode="default" runs clean over diffusers'
GGUF dequant path (measured on torch 2.10 / diffusers 0.39), but the cold warmup is
HEAVY at batched shapes -- ~159 s first-pass wall on a 12B-class 4-step model at
batch 32, up to ~655 s on a 20B CFG-batched model at 1024px -- which is exactly the
cost this bundle amortises to once-ever. Batched generation registers every distinct
(width, height, batch) chunk shape it runs via ``register_shape`` so a static-compile
bundle grows to cover the batch sizes actually used, OOM-backoff halves included.

Lifecycle (around ``_compile_repeated_blocks``): ``begin`` builds the fingerprint, points
``TORCHINDUCTOR_CACHE_DIR`` at a per-key dir, and loads a matching bundle (before the
first compiled forward); ``save`` writes the bundle + manifest after the warmup forward
(on by default, a hit skips the rewrite); ``restore`` resets the inductor dir on unload.
All env-gated and best-effort; torch imported lazily.

The generate path calls ``save_async``, not ``save``: the write is pure bookkeeping for
the NEXT process, so making a user wait on it buys them nothing. Measured on a B200
(Z-Image-Turbo, speed=max, torch 2.11): a 42.7 MB bundle costs 0.088 s to serialise plus
0.033 s to write, a 71.1 MB one 0.130 s plus 0.057 s, so roughly 2.6 ms per MB, and the
docstring's batched-GGUF bundles are far larger than either. ``save`` itself is unchanged
and still synchronous, for tests and for anyone who needs the write to have happened by
the time the call returns.

ON-DISK LAYOUT, per key dir: ``manifest.json`` plus one or more ``bundle-<sha16>.bin``.
The bundle is CONTENT-ADDRESSED and the manifest names the one it is paired with, so the
manifest is the single commit point: a save writes a file no reader is using, then
publishes the manifest naming it, then collects the bundles no manifest names. Killed
anywhere in between, what is on disk is still a matching pair, either the old one (the
new bundle is an orphan) or the new one. That matters because a save runs on a daemon
thread that interpreter exit can kill mid-write: writing one fixed ``cache.bin`` in place
would leave the old manifest paired with a new bundle, and the sha256 check on load would
then throw away a warm start that was perfectly good. Manifests without a ``bundle`` key
are pre-content-addressing and name ``cache.bin``, so old bundles keep hitting; the format
version is deliberately NOT bumped, since bumping it is what would invalidate them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

# UNSLOTH_DIFFUSION_COMPILE_CACHE: auto (default) | 0 | 1. auto loads a matching bundle and saves one after the first
# compiled generation; 1 also re-saves on a hit; 0 disables it. Measured on Qwen-Image (B200): compile hitch 29.1 ->
# 22.2 s, bit-identical, 7.9 MB bundle. The rest is dynamo tracing + guards, which Mega-cache does not capture.
# UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR: root dir for bundles (default under the workspace).
# UNSLOTH_DIFFUSION_COMPILE_CACHE_SAVE: 0 disables the auto save (load-only); 1 keeps it.
# UNSLOTH_DIFFUSION_COMPILE_CACHE_SYNC: 1 makes save_async write inline on the calling thread, as it did before the
# background worker existed. For tests and for debugging a save that looks like it never ran.
_ENV_MODE = "UNSLOTH_DIFFUSION_COMPILE_CACHE"
_ENV_DIR = "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR"
_ENV_SAVE = "UNSLOTH_DIFFUSION_COMPILE_CACHE_SAVE"
_ENV_SYNC = "UNSLOTH_DIFFUSION_COMPILE_CACHE_SYNC"

_LEGACY_ROOT = Path.home() / ".cache" / "unsloth" / "diffusion_compile_cache"

_MANIFEST_NAME = "manifest.json"
# The pre-content-addressing bundle name. Still read (a manifest without a "bundle" key names it, which is every
# bundle written before this scheme) and still collected once a newer pair supersedes it, never written.
_BUNDLE_NAME = "cache.bin"
_BUNDLE_PREFIX = "bundle-"
_BUNDLE_SUFFIX = ".bin"
# What _atomic_write names its in-progress file, as a suffix: ".<final name>.<random>.tmp".
_TEMP_SUFFIX = ".tmp"
_FORMAT_VERSION = 1

# A bundle file this new is NOT collectable even when the manifest does not name it: another process may have just
# published it and not yet committed its manifest, and the whole point of the split is that the loser of that race
# keeps a loadable pair. Only ever delays a delete.
_GC_GRACE_SECONDS = 60.0


def _bundle_name(digest: str) -> str:
    """Content-addressed file name, so writing a bundle can never damage the live one."""
    return f"{_BUNDLE_PREFIX}{digest[:16]}{_BUNDLE_SUFFIX}"


def _manifest_bundle(cdir: Path, manifest: dict[str, Any]) -> Path:
    """The bundle file a manifest names. Absent "bundle" means a pre-content-addressing manifest."""
    name = str(manifest.get("bundle") or _BUNDLE_NAME)
    # Defence against a manifest naming something outside its own directory.
    return cdir / Path(name).name


def _read_manifest(path: Path) -> Optional[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 - unreadable/absent/half-written reads as "no manifest"
        return None
    return loaded if isinstance(loaded, dict) else None


def cache_mode() -> str:
    """``off`` | ``auto`` | ``on`` from the environment. ``auto`` is the default."""
    raw = (os.environ.get(_ENV_MODE) or "auto").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes"):
        return "on"
    return "auto"


def _save_enabled(mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "on":
        return True
    # auto: save by default (without a saved bundle no user gets a warm restart); the SAVE env overrides: "0" load-only,
    # "1" on.
    return (os.environ.get(_ENV_SAVE) or "").strip().lower() not in ("0", "off", "false", "no")


def _portable_mode() -> bool:
    try:
        from utils.paths.storage_roots import portable_mode
    except ImportError:
        return False
    try:
        return bool(portable_mode())
    except Exception:  # noqa: BLE001 - never fail a cache lookup over this
        return False


def _toolchain_path_unparseable(value: str) -> bool:
    """storage_roots' test, imported per call like _default_root's.

    The fallback repeats the rule rather than narrowing to whitespace, or the same path would be
    refused or accepted depending only on whether studio/backend happened to be on sys.path when
    a CLI entry point reached this module. test_the_import_fallback_matches_the_resolver sweeps
    both against every printable character, so the copy cannot drift.
    """
    try:
        from utils.paths.storage_roots import toolchain_path_unparseable
    except ImportError:
        return (
            any(ch.isspace() for ch in value)
            or "'" in value
            or '"' in value
            or (os.name != "nt" and "\\" in value)
        )
    return toolchain_path_unparseable(value)


def _parseable_cache_fallback(key: str, intended: str) -> str | None:
    """storage_roots' ready-to-publish fallback, or None when this module is reached without
    studio/backend on sys.path, where there is no safe directory to offer."""
    try:
        from utils.paths.storage_roots import parseable_cache_fallback
    except ImportError:
        return None
    return parseable_cache_fallback(key, intended)


def _default_root() -> Path:
    """Resolved per call, not a module constant: this module is imported before startup sets
    UNSLOTH_STUDIO_HOME, which storage_roots reads."""
    try:
        from utils.paths.storage_roots import cache_root as studio_cache_root
    except ImportError:
        return _LEGACY_ROOT
    return studio_cache_root() / "diffusion_compile_cache"


def sync_saves() -> bool:
    """Whether ``save_async`` must write inline instead of handing off to the worker."""
    return (os.environ.get(_ENV_SYNC) or "").strip().lower() in ("1", "on", "true", "yes")


def cache_root() -> Path:
    """The root every bundle, manifest and inductor artifact is WRITTEN under."""
    root = os.environ.get(_ENV_DIR)
    if root:
        return Path(root)
    return _default_root()


def legacy_cache_root() -> Optional[Path]:
    """The pre-relocation root, when it is still worth READING old bundles from.

    Read-only on purpose: returning it as the write root would pin an upgraded install to the home
    directory forever. Skipped under an explicit dir override (it names one exact directory) and
    in portable mode (the host's home is not part of the install).
    """
    if os.environ.get(_ENV_DIR) or _portable_mode():
        return None
    # Equal when storage_roots is unavailable and the default IS the legacy root.
    if _LEGACY_ROOT == _default_root():
        return None
    # This root is the HOST's home, not the install, so it is the one that may be on a mount the
    # new cache does not depend on. Path.exists raises for EACCES and EIO before 3.14, and an
    # optional migration source we cannot inspect is a miss, never a failed generation.
    try:
        if not _LEGACY_ROOT.exists():
            return None
    except OSError:
        return None
    return _LEGACY_ROOT


def _triton_version() -> Optional[str]:
    try:
        import triton  # noqa: PLC0415
        return str(getattr(triton, "__version__", None))
    except Exception:  # noqa: BLE001 - triton optional
        return None


def _diffusers_version() -> Optional[str]:
    try:
        import diffusers  # noqa: PLC0415
        return str(getattr(diffusers, "__version__", None))
    except Exception:  # noqa: BLE001
        return None


def environment_fingerprint() -> dict[str, Any]:
    """HARD-portability dimensions: any difference invalidates a bundle.

    Mirrors what torch's inductor cache keys on (torch + triton + CUDA + GPU type) plus
    diffusers (the graph source), surfaced explicitly so the manifest is self-describing.
    """
    fp: dict[str, Any] = {
        "format": _FORMAT_VERSION,
        "torch": None,
        "torch_cuda": None,
        "triton": _triton_version(),
        "diffusers": _diffusers_version(),
        "gpu_name": None,
        "gpu_capability": None,
    }
    try:
        import torch  # noqa: PLC0415

        fp["torch"] = str(torch.__version__)
        fp["torch_cuda"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            # The CURRENT device, not 0: a load pinned to another card compiles for that architecture, and keying the
            # bundle by GPU 0 lets two cards share or overwrite each other's supposedly architecture-specific
            # artifacts.
            index = torch.cuda.current_device()
            fp["gpu_name"] = torch.cuda.get_device_name(index)
            cap = torch.cuda.get_device_capability(index)
            fp["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
    except Exception:  # noqa: BLE001 - best-effort
        pass
    return fp


def model_fingerprint(
    *,
    family: Any,
    transformer: Any,
    dtype: Any,
    quant: Any,
    attention_backend: Any,
    compile_kwargs: dict[str, Any],
    shape_bucket: Any = None,
) -> dict[str, Any]:
    """MODEL-graph dimensions that change the compiled artifact.

    Reads ``transformer``'s class name + ``_repeated_blocks`` so the key tracks exactly
    what gets compiled.
    """
    blocks = list(getattr(transformer, "_repeated_blocks", []) or [])
    return {
        "family": str(family),
        "transformer_cls": type(transformer).__name__ if transformer is not None else None,
        "repeated_blocks": sorted(str(b) for b in blocks),
        "dtype": str(dtype),
        "quant": str(quant) if quant is not None else "none",
        "attention_backend": str(attention_backend) if attention_backend is not None else "default",
        "compile_kwargs": {k: compile_kwargs[k] for k in sorted(compile_kwargs)},
        "shape_bucket": shape_bucket,
    }


def cache_key(env_fp: dict[str, Any], model_fp: dict[str, Any]) -> str:
    payload = json.dumps({"env": env_fp, "model": model_fp}, sort_keys = True, default = str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclasses.dataclass
class CacheContext:
    """Per-load cache state carried between ``begin`` and ``save``/``restore``.

    ``shapes`` tracks the (width, height, batch) tuples whose STATIC-compile artifacts the
    bundle covers (persisted in the manifest). A dynamic compile reuses one artifact; a
    static compile produces NEW artifacts per shape, so the caller registers each shape
    and clears ``saved`` on a new one so the next ``save`` rewrites the enriched set."""

    key: str
    dir: Path
    bundle: Path
    manifest_path: Path
    env_fp: dict[str, Any]
    model_fp: dict[str, Any]
    mode: str
    hit: bool = False
    saved: bool = False
    prev_inductor_dir: Optional[str] = None
    prev_inductor_dir_set: bool = False
    shapes: set = dataclasses.field(default_factory = set)
    # Bumped by every ``register_shape`` that dirties the context. A background save clears ``saved`` -> True only
    # when this still reads what it read before it started, so a shape registered WHILE that save ran cannot be
    # marked persisted by it.
    dirty_seq: int = 0
    # A bundle this context LOADED and rejected on its checksum. The save below must overwrite that
    # file rather than take its exists() shortcut: the recompiled artifacts usually hash to the same
    # digest, so the shortcut would leave the corrupt bytes in place under a manifest that names
    # them, and every future start would reject the cache again with no way back.
    rejected_bundle: Optional[str] = None


def begin(
    *,
    family: Any,
    transformer: Any,
    dtype: Any,
    quant: Any,
    attention_backend: Any,
    compile_kwargs: dict[str, Any],
    shape_bucket: Any = None,
    logger: Any = None,
) -> Optional[CacheContext]:
    """Point inductor at a per-key dir and load a matching bundle, BEFORE compile.

    Returns a ``CacheContext`` for ``save``/``restore``, or ``None`` when disabled or torch
    lacks the Mega-cache API. Never raises.
    """
    mode = cache_mode()
    if mode == "off":
        return None
    try:
        import torch  # noqa: PLC0415
        if not (
            hasattr(torch.compiler, "save_cache_artifacts")
            and hasattr(torch.compiler, "load_cache_artifacts")
        ):
            _warn(logger, "Mega-cache API unavailable (need torch >= 2.7); skipping")
            return None
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"torch import failed: {exc}")
        return None

    env_fp = environment_fingerprint()
    model_fp = model_fingerprint(
        family = family,
        transformer = transformer,
        dtype = dtype,
        quant = quant,
        attention_backend = attention_backend,
        compile_kwargs = compile_kwargs,
        shape_bucket = shape_bucket,
    )
    key = cache_key(env_fp, model_fp)
    cdir = cache_root() / key
    ctx = CacheContext(
        key = key,
        dir = cdir,
        bundle = cdir / _BUNDLE_NAME,
        manifest_path = cdir / _MANIFEST_NAME,
        env_fp = env_fp,
        model_fp = model_fp,
        mode = mode,
    )

    # Same reason as restore(): the previous load's save reads the inductor dir that is about to be repointed.
    if not wait_for_saves():
        _warn(
            logger,
            f"compile-cache: background save still running after {_SAVE_JOIN_TIMEOUT:.0f}s; continuing",
        )

    try:
        cdir.mkdir(parents = True, exist_ok = True)
        inductor_dir = str(cdir / "inductor")
        # Startup applies this same test before pinning TORCHINDUCTOR_CACHE_DIR, and this
        # assignment used to overwrite whatever it decided. A Studio root the C++ builders cannot
        # parse therefore came back on the first compiled diffusion run, having looked fine in
        # the environment right after launch. The bundle still lives under cdir either way; only
        # the Inductor pin is withheld, which returns it to the temporary directory it picks on
        # its own.
        if _toolchain_path_unparseable(inductor_dir):
            # Leaving the pin alone is not neutral any more. Startup publishes ONE parseable
            # fallback for the whole process, so keeping it here means save_cache_artifacts
            # serialises that shared cache into every fingerprinted bundle and each model
            # accumulates the others'. Ask for a fallback keyed on THIS cdir instead, which is
            # per-key, so the isolation the per-key directory was for survives. If none can be
            # had safely the pin stays as it was, which is the old behaviour.
            isolated = _parseable_cache_fallback("TORCHINDUCTOR_CACHE_DIR", inductor_dir)
            if isolated is None:
                _warn(
                    logger,
                    f"compile-cache: leaving TORCHINDUCTOR_CACHE_DIR as it is: {inductor_dir} "
                    "holds a character the C++ builders cannot paste into a command line "
                    "unquoted, and no safe replacement is available",
                )
            else:
                ctx.prev_inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
                ctx.prev_inductor_dir_set = True
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = isolated
        else:
            ctx.prev_inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            ctx.prev_inductor_dir_set = True
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"could not set TORCHINDUCTOR_CACHE_DIR: {exc}")

    # The MANIFEST decides which bundle is live: it is published last and only ever names a bundle that was
    # completely written, so an interrupted save leaves the previous pair addressed and loadable.
    published = _read_manifest(ctx.manifest_path)
    if published is not None:
        ctx.bundle = _manifest_bundle(cdir, published)

    # Collect here as well as after a save, because the grace window can otherwise leak a bundle for
    # good: two saves for one key inside 60 s leave the first one too young to collect, and the
    # second save is the last thing that ever looks. By the time this key is opened again that
    # orphan is minutes or days old, the live bundle is spared by name, and anything a concurrent
    # process is mid-save on is still inside its grace, so the same two rules decide it.
    _collect_superseded(cdir, logger)

    # Try an exact-match load. A miss/mismatch is normal and non-fatal.
    # Guarded like _load_from_legacy's probe: Path.exists() raises rather than returning False when a parent denies
    # traversal, and this branch moves the write root to a directory the process may not own. Unguarded, that
    # exception left begin() before the legacy fallback below could run.
    try:
        pair_present = published is not None and ctx.bundle.exists()
    except OSError as exc:
        _warn(logger, f"compile-cache: cannot check the bundle at {ctx.dir}: {exc}")
        pair_present = False
    if pair_present:
        ctx.hit = _try_load(ctx, logger)
    if not ctx.hit:
        # An install that predates the relocation may still hold this key under the old root.
        ctx.hit = _load_from_legacy(ctx, logger)
    if not ctx.hit:
        _info(logger, f"compile-cache: no bundle for key {key} (will compile locally)")
    elif mode != "on":
        # Loaded artifacts == on-disk artifacts, so nothing to save. A new static-compile shape
        # re-dirties via register_shape; mode "on" keeps saving.
        ctx.saved = True
    return ctx


def _load_from_legacy(ctx: CacheContext, logger: Any) -> bool:
    """Load the same key's bundle from the pre-relocation root, then migrate it.

    The key already covers every portability dimension, so a legacy bundle under it is the same
    artifact this run would have written. The copy stops the read fallback becoming permanent.
    Best-effort, and skipped when saving is off, since that mode promises a read-only cache."""
    root = legacy_cache_root()
    if root is None:
        return False
    ldir = root / ctx.key
    manifest_path = ldir / _MANIFEST_NAME
    # The manifest is the commit point here too, so it decides which legacy bundle is live; a
    # pre-content-addressing one names cache.bin.
    published = _read_manifest(manifest_path)
    if published is None:
        return False
    bundle = _manifest_bundle(ldir, published)
    try:
        if not bundle.exists():
            return False
    except OSError:
        # Same reason as legacy_cache_root: a pair we cannot even stat is a miss.
        return False
    if not _try_load(ctx, logger, bundle = bundle, manifest_path = manifest_path):
        return False
    if _save_enabled(ctx.mode):
        try:
            ctx.dir.mkdir(parents = True, exist_ok = True)
            # Under the name the copied manifest names, which is not ctx.bundle unless the legacy
            # pair predates content addressing. Published like begin() does, so the context goes
            # on naming the live bundle in the write root.
            migrated = _manifest_bundle(ctx.dir, published)
            _atomic_copy(bundle, migrated)
            # Bundle first: a manifest without one reads as a miss, not an unservable hit.
            _atomic_copy(manifest_path, ctx.manifest_path)
            ctx.bundle = migrated
            _info(logger, f"compile-cache: migrated legacy bundle for key {ctx.key}")
        except OSError as exc:
            _warn(logger, f"compile-cache: could not migrate legacy bundle: {exc}")
    return True


def register_shape(ctx: Optional[CacheContext], shape: Any, *, static: bool) -> None:
    """Record a generation's (width, height, batch) against the bundle coverage.

    Only meaningful for a STATIC compile: each new shape triggers its own compile, so the
    existing bundle lacks those artifacts -- clear ``saved`` so the next ``save`` rewrites
    the enriched set. Dynamic compiles reuse one artifact and never dirty the context.
    Never raises."""
    if ctx is None or not static:
        return
    try:
        key = tuple(shape)
        # Under the dirty lock so a background save cannot read dirty_seq, lose the race to this block, and then
        # publish saved = True over the clear below.
        with _dirty_lock:
            if key not in ctx.shapes:
                ctx.shapes.add(key)
                ctx.saved = False
                ctx.dirty_seq += 1
    except Exception:  # noqa: BLE001 - bookkeeping only
        pass


def _try_load(
    ctx: CacheContext,
    logger: Any,
    *,
    bundle: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> bool:
    """Validate and load a bundle pair, defaulting to the context's own (write-root) one."""
    bundle = bundle if bundle is not None else ctx.bundle
    manifest_path = manifest_path if manifest_path is not None else ctx.manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: unreadable manifest: {exc}")
        return False

    # A manifest that decoded but is not an object. json.loads happily returns [] or null, and
    # the .get() below would then raise AttributeError out of a function whose whole contract is
    # that a bad cache entry is a miss. The legacy root makes this reachable: the bundle being
    # validated was written by an older build, on a disk this run has never checked.
    if not isinstance(manifest, dict):
        _warn(logger, "compile-cache: manifest is not an object; ignoring")
        return False

    # Exact-match guard (defence in depth: torch also validates internally on load).
    if manifest.get("env") != ctx.env_fp or manifest.get("model") != ctx.model_fp:
        _warn(logger, "compile-cache: fingerprint mismatch; falling back to local compile")
        return False

    try:
        data = bundle.read_bytes()
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: cannot read bundle: {exc}")
        return False

    # Integrity check (corruption / truncation; not a security signature).
    digest = hashlib.sha256(data).hexdigest()
    if manifest.get("sha256") and manifest["sha256"] != digest:
        _warn(logger, "compile-cache: bundle checksum mismatch; ignoring")
        ctx.rejected_bundle = ctx.bundle.name
        return False

    try:
        import torch  # noqa: PLC0415

        info = torch.compiler.load_cache_artifacts(data)
        if info is None:
            _warn(logger, "compile-cache: load_cache_artifacts returned None (no hit)")
            return False
        try:
            ctx.shapes = {tuple(s) for s in manifest.get("shapes", [])}
        except Exception:  # noqa: BLE001 - coverage bookkeeping only
            ctx.shapes = set()
        _info(logger, f"compile-cache: loaded bundle for key {ctx.key}")
        return True
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: load failed: {exc}")
        return False


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file in the SAME directory plus ``os.replace``.

    A save runs on a daemon thread, so interpreter exit can kill it anywhere. Renaming a
    fully written temp file into place means a half-written file is never visible under a
    name anything reads. Paired with the content-addressed bundle names above, that is
    what makes an interrupted save leave a matching manifest/bundle pair rather than a
    good bundle the sha256 check has to reject.
    """
    tmp: Optional[str] = None
    try:
        fd, tmp = tempfile.mkstemp(
            dir = str(path.parent), prefix = f".{path.name}.", suffix = _TEMP_SUFFIX
        )
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _atomic_copy(src: Path, dst: Path) -> None:
    """``shutil.copyfile`` into a temp file in *dst*'s directory, then ``os.replace``.

    Same rule and same reason as _atomic_write: a plain copyfile onto the live name is visible
    while it is still partial, and the migration below runs on the same interruptible path as a
    save. A torn manifest costs only a miss, but a miss here means the cold compile the migration
    exists to avoid, and two backends migrating one key would otherwise interleave their writes
    into a single destination file.
    """
    tmp: Optional[str] = None
    try:
        fd, tmp = tempfile.mkstemp(dir = str(dst.parent), prefix = f".{dst.name}.", suffix = _TEMP_SUFFIX)
        os.close(fd)
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _collect_superseded(cdir: Path, logger: Any) -> list[str]:
    """Delete bundles no manifest names any more. Returns the names removed. Never raises.

    Two rules keep this from deleting a bundle somebody is about to need:

    1. The live name is re-read from the manifest ON DISK, not from the context that just wrote it, so a manifest
       another process committed in between decides what survives rather than our stale idea of it.
    2. A file younger than the grace window is spared regardless. Between a racing process publishing its bundle
       and committing its manifest, that bundle is named by nothing; deleting it there would hand the loser of the
       race exactly the broken pair this whole scheme exists to prevent.
    """
    removed: list[str] = []
    try:
        manifest = _read_manifest(cdir / _MANIFEST_NAME)
        live = _manifest_bundle(cdir, manifest).name if manifest is not None else None
        cutoff = time.time() - _GC_GRACE_SECONDS
        for path in cdir.iterdir():
            name = path.name
            if name == live or not path.is_file():
                continue
            if not (
                name == _BUNDLE_NAME
                or (name.startswith(_BUNDLE_PREFIX) and name.endswith(_BUNDLE_SUFFIX))
                # An _atomic_write that the interpreter killed mid-write leaves its temp file
                # behind: the finally that removes it does not run when the daemon save thread is
                # torn down inside fh.write, and the file can be gigabytes. The same grace window
                # covers it, so a write actually in flight is never the one taken.
                or (name.startswith(".") and name.endswith(_TEMP_SUFFIX))
            ):
                continue
            try:
                if path.stat().st_mtime > cutoff:
                    continue
                path.unlink()
                removed.append(name)
            except OSError:
                # Another process got there first, or it is busy on Windows. Next save tries again.
                continue
    except Exception as exc:  # noqa: BLE001 - housekeeping must never fail a save
        _warn(logger, f"compile-cache: could not collect superseded bundles: {exc}")
    return removed


def _write_bundle(ctx: CacheContext, logger: Any) -> bool:
    """The save itself. Runs on the caller's thread (``save``) or the worker (``save_async``)."""
    if not _save_enabled(ctx.mode) or ctx.saved:
        return False
    # Read BEFORE the artifacts are collected: anything registered from here on is not in `data`, and iterating the
    # live set while the request thread adds to it is a "set changed size during iteration" away from failing.
    with _dirty_lock:
        seq = ctx.dirty_seq
        shapes = sorted(list(s) for s in ctx.shapes)
    try:
        import torch  # noqa: PLC0415
        result = torch.compiler.save_cache_artifacts()
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: save_cache_artifacts failed: {exc}")
        return False
    if not result or result[0] is None:
        _warn(logger, "compile-cache: nothing to save (empty artifacts)")
        return False

    data = result[0]
    try:
        ctx.dir.mkdir(parents = True, exist_ok = True)
        digest = hashlib.sha256(data).hexdigest()
        bundle = ctx.dir / _bundle_name(digest)
        manifest = {
            "format": _FORMAT_VERSION,
            "key": ctx.key,
            "created": time.time(),
            "bytes": len(data),
            "sha256": digest,
            # The bundle THIS manifest is paired with. Content-addressed, so a save writes a file no reader is
            # using and the previous pair stays whole and addressed until the manifest below commits.
            "bundle": bundle.name,
            "env": ctx.env_fp,
            "model": ctx.model_fp,
            # Static-compile shape coverage (register_shape); unused by dynamic compiles.
            "shapes": shapes,
        }
        # Bundle first, under its own name, THEN the manifest: the manifest is the single commit point, and it only
        # ever names a bundle already fully on disk. An exit between the two leaves the OLD manifest still naming
        # the OLD bundle, which is untouched, so the previous warm start survives and the new file is just an
        # orphan the next successful save collects.
        if not bundle.exists() or bundle.name == ctx.rejected_bundle:
            _atomic_write(bundle, data)
        _atomic_write(
            ctx.manifest_path,
            json.dumps(manifest, indent = 2, sort_keys = True, default = str).encode("utf-8"),
        )
        ctx.bundle = bundle
        _collect_superseded(ctx.dir, logger)
        with _dirty_lock:
            # A shape registered while this ran is NOT in the bundle just written, so leave the context dirty for
            # the save its own register_shape queued.
            if ctx.dirty_seq == seq:
                ctx.saved = True
        _info(logger, f"compile-cache: saved bundle ({len(data)} bytes) for key {ctx.key}")
        return True
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: could not write bundle: {exc}")
        return False


def save(ctx: Optional[CacheContext], *, logger: Any = None) -> bool:
    """Persist compiled artifacts to the bundle + manifest, AFTER a warmup forward.

    No-op unless save is enabled and the context is dirty: a bundle HIT starts clean
    (rewriting the just-loaded artifacts costs ~0.5 s for no change); a new static-compile
    shape re-dirties via ``register_shape`` so the bundle grows to cover every shape used.
    Returns True if a bundle was written. Synchronous: the write has happened when this
    returns. ``save_async`` is what the generate path uses.
    """
    if ctx is None:
        return False
    return _write_bundle(ctx, logger)


# One daemon worker for the whole process, created on first use. Saves are serialised through it, so no two threads
# ever touch the same context and two contexts cannot interleave their save_cache_artifacts() calls.
_dirty_lock = threading.RLock()
_worker_cv = threading.Condition(threading.Lock())
_worker_thread: Optional[threading.Thread] = None
_worker_queue: list[tuple[CacheContext, Any]] = []
_worker_active: Optional[CacheContext] = None

# How long restore()/begin() will wait for an in-flight save before going ahead anyway. The worst save measured on a
# B200 was 0.19 s for a 71 MB bundle (~2.6 ms/MB), so this is ~150x the observed cost and still covers a multi-GB
# bundle; past it, an unload that keeps blocking reads to a user as a wedged app, which is worse than a bundle that
# lands a moment late (the rename keeps whatever it writes atomic either way).
_SAVE_JOIN_TIMEOUT = 30.0


def _worker_loop() -> None:
    global _worker_active
    while True:
        with _worker_cv:
            while not _worker_queue:
                _worker_cv.wait()
            ctx, logger = _worker_queue.pop(0)
            _worker_active = ctx
        try:
            _write_bundle(ctx, logger)
        except Exception as exc:  # noqa: BLE001 - a cache write must never reach a render
            _warn(logger, f"compile-cache: background save failed: {exc}")
        finally:
            with _worker_cv:
                _worker_active = None
                _worker_cv.notify_all()


def _start_worker_locked() -> None:
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(
        target = _worker_loop, name = "unsloth-compile-cache-save", daemon = True
    )
    _worker_thread.start()


def save_async(ctx: Optional[CacheContext], *, logger: Any = None) -> bool:
    """Queue ``save`` on the shared worker and return at once. Never raises.

    Returns True when a save was queued (or, under the SYNC env, written). Already queued
    is False: the pending save reads the context when it runs, so it covers every shape
    registered up to that point. A context whose save is IN FLIGHT does get queued again,
    since the running save cannot contain what was registered after it started.
    """
    if ctx is None or not _save_enabled(ctx.mode) or ctx.saved:
        return False
    if sync_saves():
        return _write_bundle(ctx, logger)
    try:
        with _worker_cv:
            if any(queued is ctx for queued, _ in _worker_queue):
                return False
            _worker_queue.append((ctx, logger))
            _start_worker_locked()
            _worker_cv.notify_all()
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort, exactly like the write itself
        _warn(logger, f"compile-cache: could not queue background save: {exc}")
        return False


def wait_for_saves(timeout: float = _SAVE_JOIN_TIMEOUT) -> bool:
    """Block until the worker is idle. True if it drained, False on timeout."""
    deadline = time.monotonic() + timeout
    with _worker_cv:
        while _worker_queue or _worker_active is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _worker_cv.wait(remaining)
    return True


def restore(ctx: Optional[CacheContext], *, logger: Any = None) -> None:
    """Restore ``TORCHINDUCTOR_CACHE_DIR`` to its pre-load value. Call on unload."""
    if ctx is None or not ctx.prev_inductor_dir_set:
        return
    # Before the env var moves: save_cache_artifacts() reads the inductor cache this load pointed at, so a save
    # still running when the dir is handed back would be collecting against a directory that is about to mean
    # something else, and the unload below goes on to tear that directory's owner down.
    if not wait_for_saves():
        _warn(
            logger,
            f"compile-cache: background save still running after {_SAVE_JOIN_TIMEOUT:.0f}s; continuing",
        )
    try:
        if ctx.prev_inductor_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = ctx.prev_inductor_dir
    except Exception:  # noqa: BLE001
        pass


def _warn(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.warning("diffusion.compile_cache: %s", msg)


def _info(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.compile_cache: %s", msg)
