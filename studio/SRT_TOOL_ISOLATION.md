# Python and Terminal isolation

Studio uses the pinned `@anthropic-ai/sandbox-runtime` **0.0.75** helper for native Required mode. Availability depends on the selected runtime, platform setup and a successful live probe. Support is **Preview**; Windows follows upstream's **alpha** model. These labels do not imply comprehensive security or compatibility qualification.

| Platform | Required mode |
|---|---|
| Linux x86-64 / arm64 | Preview only after the live probe succeeds; otherwise execution is refused |
| Windows x64 / arm64 | Upstream sandbox account, restricted token, job object, ACL grants and WFP; one-time elevated setup required |
| macOS | Native Seatbelt profile and upstream proxy model; ripgrep required |

Required never retries a refused or failed command on the host. Limited needs an authenticated, short-lived consent grant for the current page session and capability generation. Limited uses software safeguards without an OS isolation claim. Full is the separately confirmed unrestricted mode. Tool protection records originate from the backend; an SRT launch is labelled only after successful completion, because launching the wrapper alone does not prove confinement succeeded.

## Installation and offline use

Normal Studio setup runs `studio/install_srt_runtime.py` on Linux, macOS and Windows, independently of the frontend build. It uses the setup-selected Node and npm, requires **Node >=20.11**, runs `npm ci --ignore-scripts --no-audit --no-fund` against the committed lock, applies the reviewed `apply_patch.mjs` changes to the exact pinned SRT source, then verifies the patched dependency contents against the reviewed integrity manifest. The Linux changes provide an empty filesystem root and cap each network forwarder at 32 children. The patch rejects unknown source bytes; upgrades require renewed source review and integrity pins. Setup does not replace the selected Python environment or install packages into it. Missing Node/npm, failed downloads, a bad lock, missing patch or integrity failure leaves Required unavailable.

Linux also needs `bubblewrap`, `socat`, `ripgrep` and usable OS confinement facilities. Install missing prerequisites with your distribution's package manager if the capability message requests them; Studio does not change host security policy to enable them. Existing Studio setup manages an isolated Node where supported; a desktop frontend bundle alone does not supply the backend helper.

Installing `bubblewrap` is not always enough. Ubuntu 23.10 and newer ship `kernel.apparmor_restrict_unprivileged_userns=1`, which denies the user namespace `bwrap` needs unless an AppArmor profile permits it, so the live probe fails on an otherwise complete install. Sandbox details names this case when it applies. Grant the permission with a profile, for example `/etc/apparmor.d/bwrap-userns-restrict` from the `apparmor-profiles` package, then retry. Studio reports the condition and never relaxes it for you.

macOS uses the operating system's `sandbox-exec`/Seatbelt implementation. Required retains the upstream native read policy: commands can read host files accessible to Studio, including unrelated documents and credentials. It restricts writes but does not confine reads to the runtime and workdir. Capability responses, execution records and Sandbox details disclose `host_files_readable`. The native probe checks workdir writes and a denied host write; it does not qualify read confinement. Install `ripgrep` with `brew install ripgrep` if it is missing; no separate account or WFP setup applies.

For an explicit reinstall from an already populated npm cache:

```sh
python studio/install_srt_runtime.py --offline
```

This command fails if cached packages are missing. Normal tool calls perform no dependency installation and require no npm registry access. Preserve the installed `srt_runtime/node_modules` when preparing an offline image. After an upgrade, rerun setup so the lock and integrity manifest match. The helper lives inside the Studio installation; helper installation alone does not provision a sandbox account or machine policy.

The Linux live confinement probe uses owned local TCP, UDP and Unix socket positive controls and checks that the sandbox cannot reach them. It requires no public DNS or Internet connection. Failed controls, launch verification or a probe timeout still leave Required unavailable.

The bridge accepts up to 1,024 explicit read roots, since Linux may enumerate hundreds of individual shared libraries instead of granting their parent directories. Write and deny lists remain limited to 128 entries, and the entire request remains capped at 256 KiB. Exceeding a limit fails with a count or size error; paths are never dropped or replaced with broader grants to fit.

## Diagnostics and recovery

An unavailable capability includes a stable `reason_code` and a bounded `diagnostic`: the stage, a named dependency where known, and policy entry counts where applicable. Probe output, environment values and private paths are not copied into the capability response. Unknown failures stay unavailable with `probe_failed`; they do not imply that bubblewrap is missing or that a weaker configuration would work.

| Reason | Recovery |
|---|---|
| `runtime_missing`, `runtime_invalid` | Install or repair the pinned helper using Studio's selected Python |
| `dependency_missing` | Install or repair the named dependency in the execution environment |
| `policy_invalid`, `policy_oversized` | Repair policy generation or runtime layout; never discard paths or broaden grants to fit |
| `operation_unsupported` | Use an environment supporting the required isolation operations |
| `probe_timeout`, `enforcement_failed`, `probe_failed` | Review Diagnostic details and recheck after resolving the failure |

**Check again** runs a fresh capability check. It does not replay the original command. Limited remains a separate session consent followed by a manual retry; cancelling consent starts no tool command. Runtime/container identity changes invalidate cached probes and consent generations.

Colab wording is selected from server-side runtime facts, never from the browser URL or request labels. This identifies the environment, not a qualified outer security boundary. Limited capability, consent and execution details disclose that tools run with Studio's permissions and can access its files, credentials and network without an additional Studio OS sandbox.

Container-compatible SRT is a separate, explicitly consented Linux variant. It is offered only when a trusted normal probe fails and fixed differential controls show that reusing the container's `/proc` addresses the restriction. Missing bubblewrap, policy errors, unrelated failures and timeouts do not qualify for the offer. Consent forces the selected-runtime nested probe; execution rechecks eligibility, the session-bound grant and the selected runtime before launch. Standard Required remains unchanged, and there is no automatic retry or unsandboxed fallback.

The pinned patch drops all capabilities in the nested path. Its live probe checks capability removal, unrelated-file and `/proc/<pid>/root` read denials, workdir writes, private IPC, and denial of controlled host network endpoints. The variant exposes outer process information through the shared `/proc`, relies partly on the outer container, and supports only the deny network policy. GPU device access is not provided. Studio keeps its selected Python and installed packages; it does not replace the Python/CUDA environment. These bounded checks do not establish broad native-package or full platform qualification. Backend execution records identify the effective variant and carry its authority disclosure.

## Windows machine setup

After helper installation, explicitly run this command once per machine:

```powershell
python studio/install_srt_runtime.py --windows-install
```

The pinned upstream installer requests elevation through UAC. It creates the `srt-sandbox` local user and `sandbox-runtime-users` group, stores the encrypted credential and setup state in `HKLM\SOFTWARE\sandbox-runtime`, installs WFP rules keyed to the sandbox user SID, and stamps upstream's ambient write-deny ACLs. The default permitted proxy ports are loopback **60080–60089**. Re-running rotates the sandbox password and reconciles the filters; conflicting configuration is replaced only with the explicit `--windows-force` option. Declined elevation or incomplete setup leaves Required unavailable. No logout is required.

If Windows has reserved those ports, choose another bindable range of 2–100 ports. For example, after verifying all ports in this range are free:

```powershell
python studio/install_srt_runtime.py --windows-install --windows-proxy-port-range 55080 55089 --windows-force
```

This changes SRT's permitted proxy range, not Windows port reservations. After successful provisioning, `srt_runtime/installed-runtime-settings.json` records the range so the runtime proxy matches the WFP rules. Normal helper reinstalls preserve and validate this machine-local file; wheels exclude it. Other Studio installations using the same machine-wide SRT account must use the same provisioned range.

Commands keep Studio's selected Python and shell. The sandbox account needs explicit read grants for per-user runtime installations; Studio does not replace them with a machine-wide Python or WSL. Runtime filesystem grants follow upstream's session ACL lifecycle. Windows system DNS resolution is not fenced, and the shared sandbox account is not a strict separation boundary between concurrent sessions. macOS system services can likewise resolve DNS outside the process network restrictions. These are upstream model limits, not failed Linux-denial tests.

Windows TLS interception and sandbox-user CA trust are upstream features separate from account/WFP installation. Studio's current native Windows adapter does not enable TLS interception or CA setup. Windows and macOS currently expose only the deny network policy; the Linux HTTPS allowlist option is not available on these platforms.

To remove upstream's machine setup using the already installed pinned helper:

```powershell
node studio/backend/core/inference/srt_runtime/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js windows-uninstall
```

This elevated operation removes the upstream account/profile, group, WFP filters and registry state. Upstream leaves `%ProgramData%\sandbox-runtime` CA material and per-user `%LOCALAPPDATA%\sandbox-runtime` state behind. Removing Studio's files alone does not uninstall this machine setup. On a shared installation, coordinate uninstall with other users of SRT.

## Current limits

Linux Required denies networking by default. Linux HTTPS allowlists become available only when the separate private-transport probe passes. The configured hosts come from `UNSLOTH_STUDIO_TOOL_NETWORK_ALLOWLIST`; each launch receives its own private proxy sockets. The proxy permits HTTPS CONNECT on port 443 with host, public-address and TLS SNI checks. It refuses cleartext HTTP and SOCKS, while direct host networking and system DNS remain blocked. Linux installs no interception certificate or global proxy. Windows and macOS use upstream native platform enforcement and report measured capabilities of that model. Linux native controls have exercised real HTTPS success and these network denials; this evidence does not establish full platform qualification.

HTTPS launches preserve the selected CA file and snapshot hashed certificate/revocation entries from a single selected certificate directory into private read-only storage. This preserves symlink-backed trust stores without exposing their target directories or changing global trust. Explicit empty or missing stores remain empty or missing. Multi-directory `SSL_CERT_DIR` configurations remain unqualified.

Private Unix sockets and multiprocessing resource sharing on Linux are supported after live positive and negative controls pass. Host filesystem sockets are excluded by the filesystem boundary and host abstract sockets by the network namespace. A small inherited seccomp filter denies VSOCK and io_uring before starting the helper. Native tests exercise spawned Python workers and descriptor transfer. CUDA, broad native package compatibility and every platform's full confidentiality matrix remain unqualified. Do not interpret skipped or unavailable probes as successful denials.

The sandbox supplies an ephemeral private `/tmp`; its short path keeps multiprocessing sockets within Unix pathname limits even for long session directories. Native Studio tool runs exercised seven unmodified Pillow formats, NumPy 2.5.3 and CPU PyTorch 2.14.0 tensor sharing with spawned workers. These Linux measurements do not qualify Windows Python or CUDA.

The helper uses a bounded protocol and separate control/user output channels. Missing or corrupted helpers, malformed control output and failed probes refuse execution. The selected Python and shell remain the requested runtimes; Studio does not substitute WSL or another interpreter for compatibility.

The pinned Linux patch starts from an empty filesystem root and restores explicit runtime reads and the private workdir. It avoids SRT's original enumeration of existing host root entries, which could expose an entry created after wrapping. A controlled unprivileged outer namespace reproduced that exposure with the published source and confirmed denial with the patch, while Python and private writes continued working. This check did not modify the actual host root and does not replace the wider qualification matrix.
