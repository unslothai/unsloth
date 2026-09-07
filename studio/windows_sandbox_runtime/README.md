# Windows Python bootstrap runtime

The LPAC rewrite is under construction. These components are **not selected by
the production backend**, and the ABI registry is not a support/qualification
matrix. Required execution continues to fail closed when the existing backend's
live probe fails.

The [implementation plan](IMPLEMENTATION_PLAN.md) separates component delivery
from activating the expanded Python profile. This directory now includes the
experimental protected-runtime and native-bootstrap infrastructure, adapted to
the current PR's separate production Windows launchers. The production backend
selector does not import this bootstrap.

### Pillow activation-context component

`activation_manifest.py` inventories exact PE resource bytes and admits only a
single empty assembly manifest at resource 2. It retains resource language,
codepage and content hashes; unknown manifest semantics receive
`WINDOWS_SANDBOX_ACTIVATION_UNSUPPORTED`. Package filenames and CPython ABI
suffixes are discovered, not embedded in the adapter.

`src/activation_context.c` compares admitted bytes against a resource-only mapping,
pins the image against writes/deletion, prepares an activation context, and uses
Microsoft Detours to intercept `KernelBase!CreateActCtxW`. Its matcher uses the
actual mapped `hModule`, not a caller-supplied path. It retains context ownership
and the hook through process termination; each substituted result receives its
own reference. The caller must retain the protected snapshot's directory lease
and prevent thread creation during installation. These are integration
requirements, not protections supplied by the hook itself.

Build the standalone component test driver with an explicit checkout of Detours
at `adb07604aa56508448b95bf037c2a6d0d3b6831a`:

```powershell
python build_activation_context.py --detours-source C:\dev\Detours `
  --output C:\build\activation-control `
  --vs-root "C:\Program Files\Microsoft Visual Studio\2022\Community" `
  --sdk-root "C:\Program Files (x86)\Windows Kits\10"
$env:UNSLOTH_ACTIVATION_DRIVER = 'C:\build\activation-control\activation_driver.exe'
python -m pytest -q tests/test_activation_manifest.py tests/test_activation_context.py
```

The builder copies committed Detours source into the selected output directory,
compiles it there, and records source/binary provenance. It does not download or
install Detours. Use the pinned compiler/SDK documented in `build.py`.

The component is **not yet linked into the packaged Python host**. Native driver
tests exercise resource-only preparation, wrong-module refusal, changed-manifest
refusal and real repeated Pillow DLL unload/reload in an ordinary host process.
A separate local LPAC diagnostic also passed the six-format Pillow suite with
zero payload capabilities using this adapter, but that diagnostic still composes
private-catalog startup outside the packaged launcher. Neither result qualifies
DNS, remaining retained authority, CUDA, Terminal or full runtime activation.

The intended Python profile runs one workload process per tool invocation.
Imports and threads remain available; worker creation raises
`WINDOWS_SANDBOX_CHILD_PROCESS_DISABLED`. Native Job limits must enforce this
policy. The Python guards only improve diagnostics and must never be installed
in the Studio broker, Terminal, Limited, or Full execution.

Current components:

- `core/inference/windows_sandbox/profiles.py`: versioned bootstrap policy and
  the single CPython 3.11/3.12/3.13 x64 release ABI build registry.
- `runtime.py` and `dependencies.py`: static selected-interpreter inventory,
  content hashes and bounded PE parsing. Discovery executes no interpreter,
  activation scripts, package imports, `.pth`, or `sitecustomize`. An inventory
  remains payload-only; it does not approve privileged initialization.
- `policy.py`: explicit post-drop Python process-creation diagnostics.
- `content*`, `admission.py`, and `native_plan.py`: bounded snapshot admission,
  protected generations and leases.
- `native_bindings.py` and `native_compat.py`: standard-library-only Win32 and
  private invocation ownership. They do not share production container cleanup.
- `preparation*`, `launch*`, and `native_process.py`: isolated preparation workers,
  creation-time Job assignment, explicit cancellation and retained cleanup owners.
- `src/python_host.c`, `gate*`, and `host_config*`: matching-ABI native hosts and
  the startup/drop protocol. Nonzero host returns use hard termination to avoid
  DLL teardown callbacks on startup failure.
- `build.py`, `package_runtime.py`, and `artifacts.py`: explicit development
  compilation, offline companion-wheel assembly and source/artifact checks.
- Python and Terminal owners set `ToolLaunchPlan.execution_kind` explicitly.

## Development tests

Install pytest in a dedicated development environment, then install the pinned
parser from this directory's `requirements-test.txt`. From `tests/`, run:

```powershell
python -m pip install -r ../requirements-test.txt
python -m pytest -q test_contract.py test_discovery.py test_job_policy.py
```

These tests cover static discovery and Python diagnostic behavior. Their host
subprocesses run fixed test code, **not LPAC**, so passing them does not establish
token, file, network, handle, or Job enforcement. The Windows-only discovery
test reads the actual interpreter's PE metadata without launching it.

`test_job_policy.py` separately uses fixed, ordinary Windows Python controls to
prove that the shared Job owner permits a child at limit 2, denies it at limit 1,
and denies breakaway. It verifies the installed native limits and an external
child-entry sentinel. This is component enforcement evidence, not LPAC or
production bootstrap qualification.

For native bootstrap tests, build matching hosts with `build.py` and set
`UNSLOTH_TEST_PYTHON_HOST`, `UNSLOTH_TEST_PYTHON_EXECUTABLE`, and
`UNSLOTH_TEST_GATE_BINARY`. The installed-runtime fixtures additionally require
`UNSLOTH_TEST_RUNTIME_WHEEL` and the offline dependency wheels documented by
`tests/test_artifacts.py`. These fixtures deliberately fail on missing build
prerequisites; platform skips and standalone adapter skips never qualify a host.

Still required before selection: the expanded private-catalog/activation-plan
connection, per-runtime authority and network qualification, per-kind production
selection, and the complete installed-package Windows matrix. Unsupported
ABIs/layouts must never select a nearby adapter or gain additional capabilities.
