# Vendored third-party source

## truststore 0.10.4 (MIT)

- Upstream: https://github.com/sethmlarson/truststore
- Release: https://pypi.org/project/truststore/0.10.4/
- Taken from `truststore-0.10.4-py3-none-any.whl`,
  sha256 `adaeaecf1cbb5f4de3b1959b42d41f6fab57b2b1666adb59e89cb0b53361d981`
- Licence: `LICENSE` beside this file, copied unmodified from the wheel.

`utils/native_tls.py` uses it to verify TLS against the OS trust store, so Unsloth works behind a
corporate TLS-inspecting proxy. See that module's docstring for the why.

### Why the source is checked in rather than installed

`utils/third_party_source.py` is the usual way this repo consumes pinned third-party source, but it
downloads over `urllib` at first use. That cannot work here: behind the proxy this exists to fix, the
download of truststore would itself fail with `CERTIFICATE_VERIFY_FAILED`. The copy has to be present
before the network is. pip vendors truststore for the same reason.

The files are byte-identical to upstream, except the one local patch below, so they carry no
Unsloth licence header. The linters and
formatters are configured to skip this directory (`[tool.ruff] extend-exclude` in `pyproject.toml`
and the `ruff-format-with-kwargs` hook's `exclude` in `.pre-commit-config.yaml`); without both,
`scripts/enforce_kwargs_spacing.py` rewrites them and they stop matching upstream.

### How it is imported

Only ever by appending this directory to `sys.path` and importing the top-level name:

```python
sys.path.append(".../studio/backend/vendor")
import truststore
```

Never `from studio.backend.vendor import truststore`. This directory has no `__init__.py` precisely
so that dotted route does not exist: it would load the same files under a second module name, and
each copy of `inject_into_ssl()` would wrap an already-wrapped `ssl.SSLContext`. Appending rather
than prepending also means a real installed truststore still wins.

### Local patch

`truststore/_api.py` carries one Unsloth patch (upstream issue
https://github.com/sethmlarson/truststore/issues/209, open as of 0.10.4). On macOS and Windows,
`wrap_socket()` and `wrap_bio()` switch OpenSSL verification off on the shared context while a
handshake is in flight and verify against the OS store afterwards. Two overlapping handshakes on one
context could save and restore each other's temporary `CERT_NONE`, leaving the context unverified,
and the OS verification read the same shared flags and skipped itself. The patch keeps one window
per context with a reference count, holds the caller's `check_hostname` / `verify_mode` aside while
it is open on macOS and Windows (the public getters and setters use those, with the same rules as
`ssl.SSLContext`), and verifies against them. On Linux the flags are never flipped, so settings
still go straight to the context. Handshakes still run in parallel. `truststore_manifest.json` records the upstream hash under `patches`;
`tests/test_truststore_concurrent_policy.py` pins the behaviour. Drop the patch once a release
fixes the issue.

### Updating

This is a static copy. There is no refresh step and nothing updates it automatically, which is the
point: the bytes that verify certificates only change when someone decides they should.

To move to another release, replace `truststore/` with that version's wheel contents, copy its
`LICENSE`, and update `version`, `wheel`, `wheel_sha256` and the per-file hashes in
`truststore_manifest.json`. `tests/test_vendored_truststore.py` fails until the manifest matches, so
a half-finished swap cannot land. Read upstream's changelog first: a 0.x minor is where truststore
has changed verification behaviour, which here applies process-wide.

## laya 0.3.5 (Apache-2.0)

- Upstream: https://huggingface.co/convaiinnovations/laya
- Release: https://pypi.org/project/laya/0.3.5/
- Taken from `laya-0.3.5-py3-none-any.whl`,
  sha256 `4c57f64cbaf893bb5c7b4affddc2bf21a819f55df51941689f11868583be2903`
- Licence: `LICENSE.laya` beside this file, copied unmodified from the wheel. The wheel ships no
  `NOTICE` file, and the source files are unmodified.

`core/systemone/laya_runtime.py` uses it to serve the Decision API (`POST /v1/systemone`). It is pure
Python over torch, transformers, safetensors, huggingface_hub and numpy, all of which Unsloth Studio already
installs, so shipping the source replaces the `laya` pin in `requirements/extras-no-deps.txt` and the
runtime `pip install` the Decision API used to fall back to.

### How it is imported

Only through `laya_runtime._laya()`, which loads `vendor/laya/__init__.py` by file path and registers
it as the top-level `laya` module (its files import each other relatively). It never goes through
`sys.path`, so a `laya` left in the venv by an older Unsloth Studio, or installed by the user, does not replace
this copy: `laya_runtime` drives laya internals (`Agent._to_internal`, `laya.common.collate_items`),
so the version it runs must be the one it was written against.

### Updating

Replace `laya/` with the new wheel's `laya/` directory, copy its licence to `LICENSE.laya`, and update
`version`, `wheel`, `wheel_sha256` and the per-file hashes in `laya_manifest.json`. Then check every
laya internal `laya_runtime.py` reaches into, and run `SYSTEMONE_TEST_LAYA=<snapshot> pytest
tests/test_systemone.py`, which compares the fast path against `laya.Agent.predict`.

## diffusers MiniMax-H3-Image models (Apache-2.0)

- Upstream: https://github.com/huggingface/diffusers/pull/15230, merged into diffusers `main`
- Commit: `875373ac7f5856071e345a3e394c74efccc6b9e6` (the merge; recorded in `diffusers_minimax_h3_image_manifest.json`). Upstream's
  two files are byte-identical at the PR head `529403c21a36d03551f09bfd4d432cb64ab6d450` they were first vendored from.
- Files: `diffusers_minimax_h3_image/transformer_minimax_h3_image.py` (`MiniMaxH3ImageTransformer2DModel`) and
  `diffusers_minimax_h3_image/autoencoder_kl_minimax_h3_image.py` (`AutoencoderKLMiniMaxH3Image`), from
  `src/diffusers/models/` of that commit, with their Apache-2.0 headers.
- Licence: `LICENSE.diffusers` beside this file, copied unmodified from the diffusers repository at that commit.

`core/inference/diffusion_minimax_h3_image.py` uses them only when the installed diffusers does not ship the two
classes (feature-detected by name), so MiniMax-H3-Image runs on the pinned diffusers today and moves to diffusers'
own classes as soon as a release carries them. The pipeline itself (canvas rule, packed layout, sampler, decode) is
ported into that module, not vendored: the PR ships it as Modular Diffusers blocks.

### Local patch

Relative imports (`from ...configuration_utils`, `from ..attention`, `from .vae`) are rewritten onto the installed
`diffusers` package, and `PackedColwiseParallel` (only read by diffusers' tensor-parallel hooks, which diffusers
before 0.43 does not have) falls back to a stub when it cannot be imported. Both files also gain `from __future__ import annotations` after the
licence header: upstream annotates with PEP 604 unions (`int | None`), which Python 3.9 (the backend's floor) cannot
evaluate at import. diffusers' `register_to_config` and `ConfigMixin` read parameter names and defaults through
`inspect.signature`, never the annotations, so the string annotations change nothing at runtime. Nothing else
differs; the upstream hashes are under `patches` in the manifest.

### How it is imported

Only through `diffusion_minimax_h3_image._load_vendored()`, which loads each file by path under a private module
name (`unsloth_studio_vendor_minimax_h3_image.*`), never through `sys.path`. `utils/native_tls.py` appends this
directory to `sys.path` for truststore, so `diffusers_minimax_h3_image/__init__.py` (Unsloth's, not upstream's, listed
under `guard_files` in the manifest) raises `ImportError`: the directory is never importable as a top-level package,
which would load the same classes a second time under another name. `_load_vendored()` never runs it.

### Updating

When diffusers releases the classes, the vendored copy stops being used without a change here; drop it once the
pinned diffusers carries them. To move to a newer PR commit, copy the two files from it, re-apply the import and
`__future__` patch, and update the hashes and `upstream_commit` in `diffusers_minimax_h3_image_manifest.json`;
`tests/test_vendored_truststore.py` and `tests/test_diffusion_minimax_h3_image.py` check the manifest.
