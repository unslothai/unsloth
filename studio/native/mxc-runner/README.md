# Unsloth MXC runner (Windows Preview)

This is the narrow native supervisor for Studio's Windows-only MXC vertical slice. It accepts a
bounded trusted request on stdin, connects to Studio's authenticated private control channel, and
uses the pinned `mxc-sdk` streaming API. Workload stdout/stderr never carry lifecycle authority.

The current slice supports Python and native Windows Terminal launch plans through the same policy
and supervisor. It preserves the selected runtime and constructed argv, an explicit environment,
the session workdir, streaming output, timeout, authenticated cancellation, and a trusted
completion/cleanup receipt. Runtime installation/repair, host preparation, CUDA qualification, and
frontend controls remain follow-up work.

The dependency is pinned to MXC revision
`ca7ea12ac6bd9f5420d6adecb37e32a8158da476` and schema `0.8.0-alpha`. Build on Windows with:

```powershell
cargo build --release --locked --features mxc-no-dacl-api
```

Source checkouts may use `target/release/unsloth-mxc-runner.exe`. Packaged Studio builds must place
the approved, digest-verified artifact at `bin/unsloth-mxc-runner.exe`; Studio never searches PATH.

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

MXC is an early preview and is not represented by this integration as a security boundary.
