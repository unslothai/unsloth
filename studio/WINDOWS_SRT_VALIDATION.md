# Windows SRT stacked validation

This draft depends on #10526 (`studio-minimal-sandbox`). It preserves that
branch's native bubblewrap and Seatbelt implementations and selectively ports
the Windows SRT adapter from #10441. It is a tested platform split, not full
sandbox qualification.

## Revisions

- Base: #10526, `4b98ad5e507b30b1dc0aa40605d611f5704a4843`.
- Windows source: #10441, `b81fd06afaa77871464e7467c45a80d70bdc6436`.
- Tested implementation: `bec101d292663b54ca69f3a3af5a12dc524bc8f8`.
- UI component capture: `ab4289284`; its component sources match the tested implementation.
- Runtime: published `@anthropic-ai/sandbox-runtime@0.0.75`, committed lockfile
  and integrity ledger. The experimental performance branch is not used.
- Both source PR heads were rechecked and still matched their planning pins.
  #10441 and existing checkouts were not modified.

## Behavior

Auto is the default on every platform. An Auto request with a negative capability
result permits one execution with software
safeguards; Required refuses. Preparation, startup, cancellation and
uncertain planning errors do not replay the command on the host. Trusted
isolation refusals terminate the model's tool retry loop. Full access remains
behind the existing permission confirmation and cannot be selected through the
public isolation-mode field.

Only Windows imports/probes/installs SRT. Windows Studio setup installs SRT,
uses Studio's selected Python, may request administrator approval, and never
retries the original command. Native Linux/macOS keep unrestricted networking.
Windows execution records retain effective SRT policy and limitations; the UI
shows a compact execution label. Labels come from the backend launch
channel, not capability badges or model arguments.

The request boundary reads the current isolation selection after dispatch waits
and on authentication retry. Chat/session or permission changes refuse the
pending send. New profiles use Auto; saved Required selections remain
Required; obsolete selections retain Required pending reselection and clear
legacy stored grants. The API rejects obsolete isolation values.

## Verification on Windows, 2026-09-09

| Check | Result |
| --- | --- |
| Backend mode/routing, streaming, tool loops, permissions, setup and leases | 2,124 passed, 26.15 seconds |
| #10526 native Linux/macOS and launch-wiring suites under WSL | 169 passed, 10 skipped, 10.04 seconds |
| Real HTTP serialization through a 401 refresh; saved-mode migration | 2 passed |
| Node control/read-lease/platform tests | 5 passed |
| Python Ruff checks on changed backend files | Passed |
| Frontend application and test type checks | Passed |
| Production frontend build | Passed |
| New frontend modules / chat API lint | No errors; 3 Fast Refresh warnings |
| Broader touched frontend files | Same 15 lint errors at base and head: 14 in chat-adapter, 1 in Terminal UI |
| Isolated wheel build and archive inspection | Passed; required helper assets present, node_modules and local installed settings absent |
| UI evidence driver guards | 6 passed |

Backend command (from `studio/backend`):

```text
python -m pytest tests/test_windows_platform_split.py tests/test_tool_isolation_modes.py tests/test_srt_setup_interpreter.py tests/test_srt_control.py tests/test_srt_diagnostics.py tests/test_srt_windows_read_lease.py tests/test_tool_stream_events.py tests/test_studio_tool_loop.py tests/test_safetensors_tool_loop.py tests/test_permission_mode.py -q --disable-warnings
```

WSL used Linux-native source and temporary storage, running:

```text
python -m pytest tests/test_sandbox_linux.py tests/test_sandbox_macos.py tests/test_sandbox_probe.py tests/test_tool_sandbox_wiring.py -q --disable-warnings
```

Frontend and packaging commands:

```text
node --experimental-strip-types --test tests/isolation-http-boundary.test.ts
npm run typecheck
npm run build
node --test windows-read-lease.test.mjs windows-platform.test.mjs
python -m build --wheel --outdir <evidence>/dist
```

## Native observations

Required Python preserved the selected Python 3.12 interpreter, PyTorch
`2.10.0+cu128`, CUDA availability and one CUDA device. Required Terminal ran
`git --version`. A one-second tool timeout completed in about two seconds.

Direct launch-adapter controls denied an out-of-workdir write and a connection
to an owned loopback listener. The outside sentinel remained unchanged. A
two-second timeout observed one Python child and left that child no longer
alive. These controls used the real adapter and process cleanup; the ordinary
Python tool independently rejected the probe snippets during code analysis.
No administrator setup action was needed on this already provisioned host.

| Measurement, seconds | New branch | Pinned #10441 |
| --- | ---: | ---: |
| Cold capability | 21.951 | 23.221 |
| Warm capability | 0.014 | 0.011 |
| git call 1 | 2.219 | 2.332 |
| git call 2 | 46.265 | 44.799 |
| git call 3 | 1.020 | 1.032 |
| Python / packages / CUDA query | 2.781 | 2.696 |
| One-second timeout | 1.984 | 1.992 |

These are single sequential runs, not a statistical performance claim. Both
branches show the second-call spike. Runtime-input changes invalidate cached
checks; unchanged warm/concurrent checks are covered separately by regression
tests.

## UI evidence and limits

Local artifacts are under `E:/unsloth_git/windows-srt-evidence`:
`ui-before.png`, `ui-after.png`, `ui-composite.png`, `ui-facts.json`, native logs,
test/build logs and `lint-comparison.json`. The labelled composite was manually
inspected. At 680×900 per side, the real permission components show four base
permission rows versus six after adding Auto and Require sandbox. Required
persists after selection. The harness stubs the permission store and supplies a
recorded native capability response: it is component evidence, not a full
authenticated Studio session. Screenshots/raw logs are local artifacts rather
than publicly hosted evidence.

Unavailable: native macOS execution; native Linux OS-boundary qualification in
this WSL environment; container qualification; full authenticated UI/admission
and history-reload proof. Execution badges are currently session-local. DNS and
cross-session shared-account grant qualification remain explicitly incomplete.
No upstream CI result is claimed by these local checks.

After #10526 merges, reconcile this branch against its actual merged revision
and retarget the draft to `main`. Do not retarget or merge it ahead of that event.


## Review remediation, 2026-09-09

Reviewed head `2eb2f1809a12db0a9b049b8d538f721fdf7ce73a` was followed by
remediation code at `25662f2e12dc02e62543086a982ed29d44c2f012`:

- Both safetensors wrappers forward the requested isolation mode.
- The Codex policy accepts and forwards isolation to the shared tool loop.
- Output draining with no captured process group no longer references launch-local metadata.
- Execution disclosures use the pane and run-unique tool part ID, retaining first-turn
  records after autosave/settlement without sharing records with another run or pane.

Focused backend regressions: 26 passed. Neighboring safetensors/shared tool-loop
suites: 386 passed. Frontend controls: 4 passed. Frontend application/test typecheck
and production build passed. Changed backend Ruff checks passed; the execution-record
module has no ESLint errors and retains three Fast Refresh warnings.

Commands, from the respective backend/frontend directory:

```text
python -m pytest tests/test_tool_isolation_forwarding.py tests/test_tool_output_streaming.py::test_drain_process_output_without_posix_process_group_apis tests/test_tool_isolation_modes.py tests/test_tool_stream_events.py -q --disable-warnings
python -m pytest tests/test_conversation_search_safetensors_loop.py tests/test_studio_tool_loop.py tests/test_safetensors_tool_loop.py -q --disable-warnings
node --experimental-strip-types --test tests/tool-execution-record.test.ts tests/isolation-http-boundary.test.ts
npm run typecheck
npm run build
```

These checks address the four retained review mechanisms. The broader review's
native-platform, authenticated-browser and structural closure gaps remain open;
this is not a completed whole-PR review or sandbox qualification.
