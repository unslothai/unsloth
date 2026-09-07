# Python and Terminal isolation

Studio uses the pinned `@anthropic-ai/sandbox-runtime` **0.0.75** helper for its Linux Required mode. Availability depends on the actual selected runtime and a successful live confinement probe. Available Linux support is **Preview**, with `qualified=false`; this is not comprehensive security or compatibility qualification.

| Platform | Required mode |
|---|---|
| Linux x86-64 / arm64 | Preview only after the live probe succeeds; otherwise execution is refused |
| Windows | Unavailable: strict DNS/read isolation has not been established |
| macOS | Unavailable: strict DNS/read isolation has not been established |

Required never retries a refused or failed command on the host. Limited needs an authenticated, short-lived consent grant for the current page session and capability generation. Limited uses software safeguards without an OS isolation claim. Full is the separately confirmed unrestricted mode. Tool protection records originate from the backend; an SRT launch is labelled only after successful completion, because launching the wrapper alone does not prove confinement succeeded.

## Installation and offline use

Normal Studio setup runs `studio/install_srt_runtime.py` on Linux, independently of the frontend build. It uses the setup-selected Node and npm, requires **Node >=20.11**, runs `npm ci --ignore-scripts --no-audit --no-fund` against the committed lock, applies the reviewed `apply_patch.mjs` changes to the exact pinned SRT source, then verifies the patched dependency contents against the reviewed integrity manifest. These changes provide an empty filesystem root and cap each network forwarder at 32 children. The patch rejects unknown source bytes; upgrades require renewed source review and integrity pins. Setup does not replace the selected Python environment or install packages into it. Missing Node/npm, failed downloads, a bad lock, missing patch or integrity failure leaves Required unavailable.

Linux also needs `bubblewrap`, `socat`, `ripgrep` and usable OS confinement facilities. Install missing prerequisites with your distribution's package manager if the capability message requests them; Studio does not change host security policy to enable them. Existing Studio setup manages an isolated Node where supported; a desktop frontend bundle alone does not supply the backend helper.

For an explicit reinstall from an already populated npm cache:

```sh
python studio/install_srt_runtime.py --offline
```

This command fails if cached packages are missing. Normal tool calls perform no dependency installation and require no npm registry access. Preserve the installed `srt_runtime/node_modules` when preparing an offline image. After an upgrade, rerun setup so the lock and integrity manifest match. The helper lives inside the Studio installation and is removed with that installation; no global npm package, account, CA, WFP rule or filesystem grant is installed.

The live confinement probe requires working host DNS as a positive control. A host without DNS cannot establish that denial result and Required remains unavailable, even when the helper is installed from an offline cache.

Windows setup does not invoke SRT's privileged account/WFP installer. macOS and Windows users see the measured unavailable reason and may explicitly choose Limited or Full under their existing consent flow.

## Current limits

Required denies networking by default. HTTPS allowlists become available only when the separate private-transport probe passes. The configured hosts come from `UNSLOTH_STUDIO_TOOL_NETWORK_ALLOWLIST`; each launch receives its own private proxy sockets. The proxy permits HTTPS CONNECT on port 443 with host, public-address and TLS SNI checks. It refuses cleartext HTTP and SOCKS, while direct host networking and system DNS remain blocked. No interception certificate or global proxy is installed. Native controls have exercised real HTTPS success and these network denials; this evidence does not establish full platform qualification.

Private Unix sockets and multiprocessing resource sharing are supported after live positive and negative controls pass. Host filesystem sockets are excluded by the filesystem boundary and host abstract sockets by the network namespace. A small inherited seccomp filter denies VSOCK and io_uring before starting the helper. Native tests exercise spawned Python workers and descriptor transfer. CUDA, broad native package compatibility and every platform's full confidentiality matrix remain unqualified. Do not interpret skipped or unavailable probes as successful denials.

The sandbox supplies an ephemeral private `/tmp`; its short path keeps multiprocessing sockets within Unix pathname limits even for long session directories. Native Studio tool runs exercised seven unmodified Pillow formats, NumPy 2.5.3 and CPU PyTorch 2.14.0 tensor sharing with spawned workers. These Linux measurements do not qualify Windows Python or CUDA.

The helper uses a bounded protocol and separate control/user output channels. Missing or corrupted helpers, malformed control output and failed probes refuse execution. The selected Python and shell remain the requested runtimes; Studio does not substitute WSL or another interpreter for compatibility.

The pinned Linux patch starts from an empty filesystem root and restores explicit runtime reads and the private workdir. It avoids SRT's original enumeration of existing host root entries, which could expose an entry created after wrapping. A controlled unprivileged outer namespace reproduced that exposure with the published source and confirmed denial with the patch, while Python and private writes continued working. This check did not modify the actual host root and does not replace the wider qualification matrix.
