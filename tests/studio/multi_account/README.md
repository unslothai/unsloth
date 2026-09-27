# Account isolation integration gates

The backend suite is `studio/backend/tests/multi_account/`. It uses real authentication
and router handlers without running the GPU/download lifespan. The upgrade probe imports
the real `main.app` in a fresh process after seeding old SQLite schemas and binary files.

`legacy_studio_schema.sql` is the schema of `mu/base~1` (`cc0cdab40e`), generated once from
that local Git revision. Test-time seeding uses SQLite and files only. The old password
hash and encrypted credential vector are deterministic test data. The auth migration
must preserve the old columns' values, while other seeded files remain byte-identical.

The route inventory imports every module below `routes/`, including hidden routes and
nested routers. It records router-relative paths, not the mount aliases in `main.py`.
Regenerate the checked-in report from `studio/backend`:

```bash
PYTHONPATH=. python -m tests.multi_account.inventory --output ../../artifacts/route_inventory.md
```

Extend `FACTORIES` in `factories.py` with the inventory's `module:METHOD:/path` key.
Each registration gets five real-credential cases automatically, plus an owner success
case. Missing factories intentionally fail under a strict xfail and are explicitly
uncovered, not evidence that the endpoint is isolated. `WORKERS` in `inventory.py` is
a provisional mapping because the other worker prompts were not supplied. Confirm that
mapping when integrating. The first-use database schema regression awaits worker 02;
remove its strict xfail when per-account schema initialization lands.
The deactivated API-key regression awaits worker 01; unlike JWT authentication on the
contract branch, API-key authentication still admits an inactive account.

The five matrix actors target Alice's existing resource. The owner and Bob must get 404,
Alice must succeed, missing credentials must get 401/403, and a previously issued JWT
must get 401 after Alice is deactivated. Rejected requests also preserve Alice's logical
database contents. Factory workspaces are prepopulated to separate authorization from
the independently tested first-use migration problem.

Run the standalone timing gate from the repo root, with the Studio Python environment
active and no concurrent test job:

```bash
python tests/studio/multi_account/perf/compare.py --output artifacts/perf.json
```

Each of three rounds makes 2,000 status calls with an owner JWT header and 200 authenticated
chat-history listings of 100 saved threads on each revision. Status remains a public
endpoint, so attaching a JWT does not add a dependency that production does not have.
`test_hot_path_cost.py` separately measures a GET through the real authentication dependency.
The benchmark keeps a persistent TestClient portal, warms all routes 100 times, alternates
revision order, and compares the median round p50 and p95 independently against 5%.
All round measurements are retained. It exits nonzero if any percentile exceeds the limit;
it does not retry until a pass, widen the tolerance, or silently xfail latency regressions.

The base snapshot comes only from local Git objects, is extracted below this clone's
`.tmp/`, and is deleted after timing. The head is the working tree, and the artifact
records its current commit. Python bytecode writes are disabled. Keep environment caches
and pytest temporary directories within this clone, as in the task's test command.

Windows tests are lexical `ntpath`/`PureWindowsPath` tests, not native Windows execution.
Account UUIDs add 42 UTF-16 units to owner-relative leaves. MAX_PATH includes the trailing
NUL; arbitrarily long installation paths still require Windows long-path support or a
shorter root. The tests do not impose a new path restriction on existing owners.
