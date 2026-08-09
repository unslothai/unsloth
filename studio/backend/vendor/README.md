# Vendored third-party source

## truststore 0.10.4 (MIT)

- Upstream: https://github.com/sethmlarson/truststore
- Release: https://pypi.org/project/truststore/0.10.4/
- Taken from `truststore-0.10.4-py3-none-any.whl`,
  sha256 `adaeaecf1cbb5f4de3b1959b42d41f6fab57b2b1666adb59e89cb0b53361d981`
- Licence: `LICENSE` beside this file, copied unmodified from the wheel.

`utils/native_tls.py` uses it to verify TLS against the OS trust store, so Studio works behind a
corporate TLS-inspecting proxy. See that module's docstring for the why.

### Why the source is checked in rather than installed

`utils/third_party_source.py` is the usual way this repo consumes pinned third-party source, but it
downloads over `urllib` at first use. That cannot work here: behind the proxy this exists to fix, the
download of truststore would itself fail with `CERTIFICATE_VERIFY_FAILED`. The copy has to be present
before the network is. pip vendors truststore for the same reason.

The files are byte-identical to upstream, so they carry no Unsloth licence header. The linters and
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

### Updating

```bash
python scripts/sync_vendored_truststore.py --version <new version>
```

It re-downloads the wheel, verifies the hash against PyPI, rewrites this tree and
`truststore_manifest.json`, and `tests/test_vendored_truststore.py` then checks the result.
Read upstream's changelog first: a 0.x minor is where truststore has changed verification behaviour,
which here applies process-wide.
