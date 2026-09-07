# Windows Smart App Control probe

Collects evidence about Windows code integrity blocking the Unsloth Studio
llama.cpp runtime, on a real machine, in a form that can be attached to an issue
or a pull request.

## What we are trying to settle

**Start with an AMD machine if you have one.** Every Smart App Control report so
far has been AMD, not NVIDIA: unslothai/unsloth#6588 (unsigned ROCm libraries),
unslothai/unsloth#6648 (`rocfft.dll`, with the verdict changing on a byte
identical file), and the report that prompted this. The ROCm bundle also has by
far the largest unsigned surface, 75 PE files against 47 for CUDA, because it
carries AMD's own TheRock runtime DLLs (`amdhip64_*.dll`, `rocblas.dll`,
`hipblas.dll`, `amd_comgr.dll`, `origami.dll` and more) which ship unsigned.

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
- Any Python 3 on `PATH`. The scenario uses only the standard library: no pip install, no browser
- Optional but strongly preferred: `SmartAppControlAuditNoISG.bin` from <https://aka.ms/sacauditpolicies>

The NoISG policy is the useful one. It checks signatures and skips the cloud
reputation lookup, which means it works **even with Smart App Control off**, so
any machine can produce evidence. It logs event 3076 for every file it would
have refused. That is exactly the half of the verdict that signing changes.

## Getting it onto the machine

```powershell
git clone --branch windows-sac-probe --depth 1 https://github.com/unslothai/unsloth C:\unsloth-probe
cd C:\unsloth-probe\scripts\windows_sac_probe
```

These scripts are not Authenticode signed, so if you download them individually
rather than cloning, Windows marks them with Mark-of-the-Web and PowerShell
refuses to run them under the default `RemoteSigned` policy. Either unblock them
or relax the policy for this session only:

```powershell
Get-ChildItem . -Recurse | Unblock-File
# or, for this process only, reverted when the window closes:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

Do not change the machine-wide execution policy. It is part of what we are
measuring.

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

`run` needs the Studio password, always:

```powershell
$env:UNSLOTH_STUDIO_PASSWORD = 'your studio password'
.\sac-probe.ps1 -Stage run -Label custom-sac-on
```

On a Studio that has already been opened, that is the password you sign in
with. On one that has never been opened, the bootstrap credential is still on
disk and the scenario rotates it to the value you give here; the rotation is
permanent and `revert` does not undo it, which is why there is no default. The
password is never written to the evidence.

If Studio runs from a custom home (`UNSLOTH_STUDIO_HOME`) or a custom runtime
(`UNSLOTH_LLAMA_CPP_PATH`), set the same variables in the shell that runs the
probe, so it inventories the runtime Studio actually loads and reads the logs
Studio actually writes.

`prepare` runs `Update-MpSignature`; add `-SkipUpdates` to skip it. It does not
upgrade packages unless you pass `-UpgradePackages`: `winget upgrade --all` changes
every managed package on the machine and `revert` cannot put them back, so it is
not part of the reversible run.

Running `prepare` again with the same label (after a failure, or to add
`-AuditPolicy`) keeps the baseline the first pass captured, so `revert` still
restores the machine as it was before the first pass. `collect` refuses a label
that was never prepared rather than exporting unrelated events.

### On a machine with no Unsloth on it

Nothing extra to do. `prepare` detects that Studio is absent and installs it with
the documented command, then starts it and waits for it to answer:

```
=== Unsloth Studio ===
Studio is not installed on this machine.

=== Install Unsloth Studio ===
irm https://unsloth.ai/install.ps1 | iex
...
managed interpreter: C:\Users\you\.unsloth\studio\unsloth_studio\Scripts\python.exe
starting Studio on port 8888
Studio answering on port 8888 after 65s
```

The install happens **after** the event log window opens, on purpose: installing
is what downloads the llama.cpp bundle, and first launch is what loads it, so
both are inside the window `collect` exports. On an affected machine the block
may well appear during `prepare` rather than during `run`.

Studio is started through the managed interpreter, not the generated
`unsloth.exe`. Windows materialises that console script as an unsigned PE and
Application Control denies it (unslothai/unsloth#8490) while the signed
interpreter beside it keeps running, so on exactly the machines this probe
targets, launching the normal way would fail for a reason unrelated to what is
being measured. The supported entry point is documented in
`unsloth_cli/__main__.py`:

```powershell
python -X utf8 -I -m unsloth_cli studio -p 8888
```

Pass `-SkipInstall` to refuse to install, or `-Port` if 8888 is taken. `run`
re-checks and restarts Studio too, since a machine may have been rebooted
between stages, which is itself part of the reported behaviour.

### What each stage does

| stage | what it changes | how it is undone |
| --- | --- | --- |
| `prepare` | records a baseline; updates packages and Defender signatures; turns Defender real-time, MAPS advanced, cloud block level high and PUA on; raises the CodeIntegrity log to 64 MB; applies the audit policy if given (a pre-existing policy with the same GUID is saved first); installs Unsloth Studio if it is missing, then starts it | `revert` (Studio is left installed) |
| `run` | nothing on the machine beyond restarting Studio if it stopped; inventories signatures and drives Studio (rotates a never-used bootstrap password to yours, see above) | n/a |
| `collect` | nothing; exports events and zips the evidence | n/a |
| `revert` | removes the audit policy (or restores the pre-existing one), restores the CodeIntegrity log settings and the Defender settings from the baseline; the EFI partition is never left mounted | n/a |

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

To pin a specific runtime for a cell, the runtime has to be **installed** with
that pin and Studio restarted before `run`; `UNSLOTH_LLAMA_RELEASE_TAG` is read
by the installer only, and `run` neither installs nor restarts anything:

```powershell
$env:UNSLOTH_LLAMA_RELEASE_TAG = 'b10715-mix-86bd2d3'   # an older, more established build
# re-run the Studio installer (studio\setup.ps1) so it fetches that release,
# then restart Studio, then:
.\sac-probe.ps1 -Stage run -Label custom-b10715-sac-on
```

For the upstream row, point Studio at an upstream build through its custom
llama.cpp folder setting (or `UNSLOTH_LLAMA_CPP_PATH`) and restart it. Either
way, check `signature-inventory.json` afterwards: the SHA-256 of each file is
what says which build a cell actually exercised, not the label.

## Reading the output

`collect` writes a zip to `%USERPROFILE%\unsloth-sac-probe\`. Inside:

- `signature-inventory.csv` and `.json`: every PE under the runtime Studio loads (`UNSLOTH_LLAMA_CPP_PATH`, else `<Studio home>\llama.cpp`, else `~\.unsloth\llama.cpp`) with its Authenticode `Status`, `StatusMessage`, signer subject, thumbprint and SHA-256. Record `Status`, not merely whether a certificate is present: an unsigned file and one whose chain did not build both report `UnknownError`, and `StatusMessage` is what separates them
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
