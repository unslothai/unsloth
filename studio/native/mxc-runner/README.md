# Unsloth MXC runner (Windows Preview)

This is the narrow native supervisor for Studio's Windows-only MXC vertical slice. It accepts a
bounded trusted request on stdin, connects to Studio's authenticated private control channel, and
uses the pinned `mxc-sdk` streaming API. Workload stdout/stderr never carry lifecycle authority.

The current slice supports Python and native Windows Terminal launch plans through the same policy
and supervisor. It preserves the selected runtime and constructed argv, an explicit environment,
the session workdir, streaming output, timeout, authenticated cancellation, and a trusted
completion/cleanup receipt. Host preparation, CUDA qualification, and frontend controls remain
follow-up work.

The dependency is pinned to MXC revision
`ca7ea12ac6bd9f5420d6adecb37e32a8158da476` and schema `0.8.0-alpha`. A packageable build must use the preparation tool, which verifies the upstream commit, approved patch SHA-256, complete post-patch Git tree, and patched Cargo lock before invoking Cargo:

```powershell
python tools\prepare_build.py build --output target\reproducible-runtime
python tools\prepare_build.py package `
  --artifact target\reproducible-runtime `
  --destination package\windows-x86_64
```

The output contains the runner and its canonical runtime manifest. Studio validates the runner digest,
source identity, protocol, architecture, profile, MXC revision, patch/tree identities, and required
admission API. Packaged Studio installs immutable directories under `bin/generations/`, selects one
with an atomic pointer, and separately allowlists its manifest and runner digests. Source checkouts may
use only `target/reproducible-runtime`; an arbitrary `target/release` executable or PATH entry is never
selected and is not production-ready.

Windows setup calls `studio/install_mxc_runtime.py` against that fixed packaged artifact. The
lifecycle supports explicit install, verify, repair, update, rollback, garbage collection, and
uninstall. Mutations are serialized by a dedicated file lock; active launches hold generation
leases and runner handles, so pointer changes never replace or delete their executable in place.
The selected generation is published with an atomic `current.json` replacement only after the
staged runner, manifest, identity, and digest allowlist all validate.

The package and installed runtime live under Studio's installation tree, not a chat workdir,
temporary directory, model cache, or PATH location. The trust model assumes the Studio install
tree is writable only by the installing user/administrator and is not granted writable access to
MXC workloads. SHA-256 allowlisting detects accidental or post-install byte changes; it is not a
substitute for protecting that application-owned directory from an already-compromised process
running as the installation owner. Setup and lifecycle operations do not perform host-policy
changes or request elevation.

## Required upstream extension

The pinned revision's stable v0.8 parser and ProcessContainer dispatcher already enforce
`fallback.allowDaclMutation=false`, but its public streaming builder cannot set that value and its
live handle does not expose the selected tier. The minimal generic MXC extension used by this
runner is recorded in `upstream/mxc-ca7ea12-no-dacl-tier.patch`. It adds a typed request setter and
structured execution-tier evidence without changing the dispatcher or parsing serialized bytes.

Builds made from an unextended `ca7ea12...` checkout compile without the feature but refuse every
launch before spawn, and must not be packaged as the Studio Preview backend. Once Microsoft
publishes an equivalent API, update the pinned revision, remove the local upstream patch, and keep
the runner tests that prove the request carries disabled DACL fallback and STARTED is bound to the
live BaseContainer tier.

The trusted lifecycle transport is a random, single-instance Windows named pipe with a protected
owner/System DACL. Protocol authentication and workload stdout/stderr remain separate from it.

`mxc-test-failure-injection` is a test-only Cargo feature. It reads failure stages only from the
supervisor process environment and is excluded by the packaged feature allowlist; launch requests
cannot enable it.

MXC is an early preview and is not represented by this integration as a security boundary.
