# Windows Smart App Control probe

Collects evidence about Windows code integrity blocking the Unsloth Studio
llama.cpp runtime, on a real machine, in a form that can be attached to an issue
or a pull request.

## What we are trying to settle

Users report Studio installing and launching, then failing to load a model, with
the failure clearing when Smart App Control is turned off and returning after a
reboot. The dialog is:

```
llama-server.exe - Bad Image
C:\Users\...\.unsloth\llama.cpp\build\bin\Release\llama-common.dll is either not
designed to run on Windows or it contains an error. Error status 0xc0e90002.
```

`0xc0e90002` is a code integrity refusal, not a corrupt file. Every PE we ship in
the Windows bundles is unsigned today (47 of 47 in the CUDA bundle, 51 of 52 in
the CPU bundle, the one exception being Microsoft's own OpenMP runtime), and
VirusTotal has never seen any of them, so they carry no cloud reputation either.
Smart App Control decides per file, on signature first and reputation second.

So far this rests on one screenshot. This probe is how it becomes a measurement.

## What this does not do

It never turns Smart App Control on or off. Switching it through Settings is a
one-way operation that cannot be undone without reinstalling Windows, and the
registry route needs BitLocker suspended and a boot into recovery. The script
only adds and removes an **audit** App Control policy, which is reversible, and
reports whichever mode it finds the machine in.

## Requirements

- Windows 11, elevated PowerShell (right click Windows Terminal, Run as administrator)
- Unsloth Studio installed and running
- Any Python 3 on `PATH`. The scenario uses only the standard library: no pip install, no browser
- Optional but strongly preferred: `SmartAppControlAuditNoISG.bin` from <https://aka.ms/sacauditpolicies>

The NoISG policy is the useful one. It checks signatures and skips the cloud
reputation lookup, which means it works **even with Smart App Control off**, so
any machine can produce evidence. It logs event 3076 for every file it would
have refused. That is exactly the half of the verdict that signing changes.

## Running it

Four stages, in order. Use a distinct `-Label` per cell of the matrix below so
runs do not overwrite each other.

```powershell
cd scripts\windows_sac_probe

.\sac-probe.ps1 -Stage prepare -Label custom-sac-on -AuditPolicy C:\path\to\SmartAppControlAuditNoISG.bin
.\sac-probe.ps1 -Stage run     -Label custom-sac-on
.\sac-probe.ps1 -Stage collect -Label custom-sac-on
.\sac-probe.ps1 -Stage revert  -Label custom-sac-on
```

If Studio has already been opened on that machine, its password has been rotated
and the bootstrap file is gone, so pass yours through:

```powershell
$env:UNSLOTH_STUDIO_PASSWORD = 'your studio password'
.\sac-probe.ps1 -Stage run -Label custom-sac-on
```

`prepare` is slow the first time because it runs `winget upgrade --all` and
`Update-MpSignature`. Add `-SkipUpdates` to skip both.

### What each stage does

| stage | what it changes | how it is undone |
| --- | --- | --- |
| `prepare` | records a baseline; updates packages and Defender signatures; turns Defender real-time, MAPS advanced, cloud block level high and PUA on; raises the CodeIntegrity log to 64 MB; applies the audit policy if given | `revert` |
| `run` | nothing; inventories signatures and drives Studio | n/a |
| `collect` | nothing; exports events and zips the evidence | n/a |
| `revert` | removes the audit policy, restores the Defender settings from the baseline | n/a |

`revert` reads `baseline.json` from the same `-Label` directory, so use the label
you prepared with.

## The matrix

One run is not a result. Run all four cells, ideally across the machines
available:

| | Smart App Control enforcing | Smart App Control off |
| --- | --- | --- |
| our build (`unslothai/llama.cpp`) | C-on | C-off |
| upstream build (`ggml-org/llama.cpp`) | U-on | U-off |

The upstream row separates "our additive-merge build is the problem" from "the
whole unsigned llama.cpp ecosystem is the problem". Upstream is also 0 of N
signed, and so is `lemonade-sdk/llamacpp-rocm` (72 PE files, 0 signed), so U-on
is what tells us whether download prevalence alone carries a build through.

To pin a specific runtime before a run:

```powershell
$env:UNSLOTH_LLAMA_RELEASE_TAG = 'b10715-mix-86bd2d3'   # an older, more established build
```

## Reading the output

`collect` writes a zip to `%USERPROFILE%\unsloth-sac-probe\`. Inside:

- `signature-inventory.csv` and `.json`: every PE under `~\.unsloth\llama.cpp` with its Authenticode `Status`, `StatusMessage`, signer subject, thumbprint and SHA-256. Record `Status`, not merely whether a certificate is present: an unsigned file and one whose chain did not build both report `UnknownError`, and `StatusMessage` is what separates them
- `code-integrity-events.json` and `.txt`: events 3033, 3076, 3077, 3089 and 3090 to 3099 in the run window
- `CodeIntegrity-Operational.evtx`: the raw log
- `scenario-results.json`: every HTTP call with its duration, plus the status-poll summary
- `studio-logs\`: Studio's own backend and llama-server logs
- `baseline.json`, `sac-state-after.json`: what the machine looked like before and after

**Grade the events correctly.** They are not interchangeable:

- **3077** is an enforced block. This is proof.
- **3076** is audit only: "would have been blocked". Strong, but not enforcement.
- **3089** carries the signature detail for a block. Correlate it to a 3077 by `ActivityID`, never by timestamp alone.
- **3090 to 3092** are allow and origin context, and prove nothing on their own.

Also keep "unsigned" and "low reputation" apart. Smart App Control may allow a
validly signed file whose reputation is inconclusive, which is the behaviour the
signing fix depends on. Search the events for **every** PE path in the bundle:
the file that gets blocked can be a dependent `ggml`, HIP, OpenMP or VC runtime
DLL rather than `llama-server.exe` itself.

Do not enable Windows test-signing mode. It changes enforcement conditions and
invalidates the run.

## The status timing half

`scenario-results.json` also answers a separate question. Field diagnostics show
`/api/inference/status` taking over 80 seconds, which starves `/api/health` and
gets the backend killed by the desktop watchdog at its 3 strike, roughly 75
second budget, surfacing as "Server stopped unexpectedly". Those stalls were
observed with **no model loaded**, so the cause is not yet established. The
scenario polls status at the same 5 second cadence the frontend uses and reports
`max_ms`, `over_10s` and `over_75s`, which is the measurement that has been
missing.

## What to send back

Attach the zip. It contains your Windows user name inside file paths. Redacting
the user name is fine; keep the file names and the rest of each path, because
which file was blocked is the entire point.
