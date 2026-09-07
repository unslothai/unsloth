# Studio SRT bridge

This private Node helper runs one Linux tool invocation using the published
`@anthropic-ai/sandbox-runtime@0.0.75` package. Node 20.11+, `/usr/bin/bwrap`,
`/bin/bash`, and the bundled x64/arm64 seccomp helper are required. Installed
Studio must provision these dependencies before launch; the helper never invokes
npm, downloads a runtime, or falls back to host execution.

Install through `studio/install_srt_runtime.py`, or run `npm ci --ignore-scripts`
followed by `node apply_patch.mjs` in this directory. The lockfile pins the npm dependency graph. Before
importing SRT, the bridge verifies the committed integrity ledger and the content
of its installed package files. Updating SRT requires inspecting the release and
native artifacts, updating the ledger and its digest in `bridge.mjs`, and rerunning
native security and compatibility checks. A missing or changed file fails before
the workload is started.

The bridge intentionally uses the pinned package's exported
`dist/sandbox/linux-sandbox-utils.js` function. SRT's public manager always grants
additional paths including `/tmp/claude` and caller log directories. Its Linux
implementation restores these write grants inside broad read-denied regions, so
the manager cannot express Studio's private-workdir read contract. This internal
module dependency is isolated here and must be checked when upgrading SRT.

Protocol version 1 accepts one bounded JSON request on stdin. Python supplies the
absolute executable, argument array, explicit environment, runtime read roots,
private workdir/write roots, operation, timeout, and control descriptor. The helper
emits bounded JSON records on that separate descriptor; the workload receives
only stdin/stdout/stderr. User output cannot create trusted records. `spawned`
acknowledges the sandbox launcher, not successful payload execution. Consumers
must not treat that record as workload success. `probe` runs the caller-supplied
controlled command through the same path as `run`.

The exact-source patch replaces SRT's initial host-root bind with an empty tmpfs
when the read policy denies `/`. This prevents later-created host root entries
from appearing in the sandbox. Original and patched native runs reproduce the
before/after behavior inside an unprivileged, controlled outer root; the actual
host root is never changed by that test. The patch accepts only the audited
original or patched source hashes. It also caps each internal socat forwarder at
32 children, so idle connections cannot cause unbounded process creation.

The Linux policy starts from that empty root, masks denied nested paths, and restores only specified
runtime reads and private work writes, creates fresh PID/network namespaces,
drops capabilities, and inherits Studio's small syscall filter denying VSOCK
and io_uring. Studio requests private Unix IPC only after installing that filter;
the bridge additionally requires inherited seccomp mode. Standalone requests
default to SRT's stricter Unix-socket-denying native helper. Native checks prove
private socket pairs and multiprocessing spawn/resource sharing work while
positive-control host filesystem and abstract sockets cannot be reached.
Optional HTTPS transport restores only two private, owned Unix sockets: the
existing Studio HTTPS CONNECT proxy and a SOCKS refusal listener. Its authority
is the private socket, so no proxy credential enters a command line. The
payload commands set numeric-loopback proxy URLs after SRT's generated
environment, avoiding hostname resolution inside the DNS-denied namespace.
Native transport testing covers allowed HTTPS, disallowed hosts, cleartext HTTP, SOCKS,
direct TCP, and system DNS. Linux remains Preview: the tested isolated CPU Python
environment does not qualify the user's Windows Python/CUDA environment, GPU
execution, or Windows/macOS strict isolation.
The host and installed runtime remain trusted; admitted runtime and workdir
contents are intentionally visible, including files created inside admitted roots.
Existing hardlinks inside allowed paths remain accessible as those files;
callers must prepare workdirs without importing unrelated host hardlinks.

The helper directly owns bubblewrap, which uses `--die-with-parent`; normal exit,
timeouts, and cancellation wait for the launcher to exit before cleanup. Native
tests exercise a detached descendant on timeout and uncatchable bridge death.
Run `npm test` for protocol/integrity checks. On a prepared native Linux host,
run `UNSLOTH_SRT_NATIVE_TESTS=1 npm test` for execution, output pressure, private
file boundaries, and descendant cleanup. Skipped native tests are not platform
qualification.

Pinned release source:
https://github.com/anthropics/sandbox-runtime/tree/40804af269e1616092e9971de12a1f358f58eba9
