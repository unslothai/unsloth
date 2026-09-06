# Windows Python bootstrap runtime

The LPAC rewrite is under construction. These components are **not selected by
the production backend**, and the ABI registry is not a support/qualification
matrix. Required execution continues to fail closed when the existing backend's
live probe fails.

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

Still required before selection: protected content generations and dependency
plans; native ABI artifacts; supported-API startup token construction; bounded
control protocol and irreversible drop gate; native one-process enforcement;
per-runtime production qualification; real streaming and lifecycle integration;
and installed-package delivery and the full Windows test matrix. Unsupported
ABIs/layouts must never select a nearby adapter or gain additional capabilities.
