# Windows SRT bridge

Studio uses `@anthropic-ai/sandbox-runtime@0.0.75` on Windows only. Node 20.11+
is installed or reused by Windows setup. Linux uses native
bubblewrap and macOS uses native Seatbelt, without SRT or backend Node/npm.

Windows Studio setup runs `python studio/install_srt_runtime.py --windows-install`
with Studio's selected Python to install and verify the pinned runtime. This option
reuses an existing installation after a live probe and read-grant cleanup succeed.
Otherwise it provisions the upstream sandbox account and WFP filters and may request elevation.
Declining approval fails setup. Rerun Studio setup to retry; repair never retries a tool.

The lockfile and integrity ledger pin the upstream package. The retained exact
source patch is part of that ledger, but Studio never selects its Linux backend.
The Windows helper uses a bounded authenticated control socket, checks the selected
Python and Terminal, and requires a trusted completion receipt. Workload output
cannot supply control records. Read-grant leases are reused while runtime inputs
remain unchanged and released during shutdown. Startup checks never install software.

API requests and Studio chat default to Auto on every platform and preserve saved
Required selections. Auto permits software safeguards when a live capability check
reports that OS isolation is unavailable. Required refuses. Preparation, launch and cancellation
errors never trigger host replay. Full access uses the existing explicit permission
confirmation. The Windows preview retains upstream system DNS and shared-account
grant limitations; it is not full sandbox qualification.
