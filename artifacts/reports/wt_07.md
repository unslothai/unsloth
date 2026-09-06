Worker 07 implemented account-scoped chat tool and MCP state in commit `709f484ee8` (`Isolate chat tool sandboxes and MCP state by account`). Final verification passed 4,190 tests, with six skips. HTTP response admission still needs the route integration described below.

Changes and rationale:

- Sandboxes use the acting account's workspace. The owner retains `studio_root()/sandbox`; managed sandbox overrides append `accounts/<account_id>`. Workdir memoization, in-flight sessions, deferred removals, orphan records, output records, and recovery are account-scoped. Managed accounts cannot inherit or migrate owner legacy sandboxes.
- Managed project tool sessions use their account sandbox while multi-user policy applies. Edit-file resolution rejects absolute paths, traversal, and symlinks reaching another account. Cleanup, retrieval, and tool worker threads retain the acting account.
- `state.tool_policy.require_tool_access(...)` centralizes refusal of full access, bypass permissions, and disabled sandboxes. GGUF, safetensors, and the shared hosted/Codex loop use its permission normalizer. Direct tool dispatch also checks it. Rejection carries HTTP status 400 and the message: `Full access is unavailable while more than one account exists.`
- `_build_safe_env` already excludes credentials through an allowlist, so its implementation remains unchanged. Added coverage checks the requested credential names and wildcard examples for the owner, Alice, and Bob.
- MCP configuration databases initialize their schemas independently. Session keys, close generations, tool caches, failure cooldowns, and OAuth token stores are account-scoped. An account's session eviction or configuration invalidation does not evict another account's sessions or cached schemas.
- Managed accounts in multi-user mode cannot register or connect to stdio or private-network HTTP MCP servers. HTTP connections validate all DNS answers and pin public destinations while preserving Host/TLS identity. Redirects and OAuth HTTP requests use the same public-network transport; owner transport behavior remains unchanged.
- Added `tests/test_account_tool_isolation.py`. Updated one existing sandbox test to use its pytest temporary directory instead of writing to a hardcoded external temporary path.

Single-user regression evidence:

The frozen account contract and existing sandbox, permissions, MCP, project, retrieval, and tool-loop tests pass. Owner sandbox paths, override handling, OAuth token location, local MCP registration, and permission normalization are covered. The safe environment builder is unchanged, and the complete environment compares identically across account contexts when given the same workdir. Ordinary sandboxed admission performs no installation-policy lookup; owner MCP address validation performs no DNS lookup. Request-model validators remain unchanged and policy-free. No latency benchmark was run; account identity lookups and tuple keys introduce Python bookkeeping.

Tests ran from `studio/backend` with this setup, using the supplied environment read-only and keeping generated files inside this clone:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export UNSLOTH_STUDIO_HOME="$PWD/../../.tmp/home" HF_HOME="$PWD/../../.tmp/hf" TMPDIR="$PWD/../../.tmp/tmp" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$UNSLOTH_STUDIO_HOME" "$HF_HOME" "$TMPDIR"
```

Focused development command:

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_tool_isolation.py -q -n 8 --timeout=330
```

Successive runs as tests were added and corrected: 98 passed/7 failed; 118 passed/3 failed; 122 passed/2 failed; 127 passed/0 failed. Failures were incorrect new-test fixtures, arguments, or helper imports. The final broad run includes the subsequently added recovery test, for 128 passing contract/isolation cases.

Negative control, with only implementation source files temporarily replaced by their pre-change HEAD contents and restored in a `finally` block:

```bash
PYTHONPATH=. /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/python -m pytest tests/test_account_tool_isolation.py -q -n 8 --timeout=330
```

The first negative-control run produced 69 failed/38 passed. After adding recovery coverage, it produced 70 failed/38 passed. These expected failures demonstrate that the tests detect the missing account boundaries; the compatibility cases already pass on the base.

Broad regression command:

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_tool_isolation.py tests/test_mcp*.py tests/test_sandbox*.py tests/test_tool_sandbox_per_thread.py tests/test_edit_file_tool.py tests/test_full_access_tool_prompt.py tests/test_tool_policy*.py tests/test_llama_cpp_tool_loop.py tests/test_safetensors_tool_loop.py tests/test_studio_tool_loop.py tests/test_tool_output_streaming.py tests/test_tool_stream_generator_drain.py tests/test_gguf_tool_non_streaming.py tests/test_secure_tools_execute.py tests/test_tool_confirm_loop.py tests/test_tool_loop_exception_contracts.py -q -n 8 --timeout=330
```

Initial run: 2,023 passed, two failed, six skipped. Both failures were imports in the new account tests. Final run: 2,029 passed, zero failed, six skipped in 120.52 seconds. Two warnings came from the existing coroutine-contract test and a multiprocessing fork test.

Additional existing regression command:

```bash
PYTHONPATH=. python -m pytest tests/test_bypass_permissions.py tests/test_permission_mode.py tests/test_project_workspace_location.py tests/test_tool_result_fits_window.py tests/test_session_guard_exception_paths.py tests/test_rag_retrieval.py tests/test_openai_codex_subscription.py -q -n 8 --timeout=330
```

Result: 2,161 passed, zero failed in 6.63 seconds.

Validation from the clone root, repeated before commits:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export PYTHONDONTWRITEBYTECODE=1
changed_py=(
  studio/backend/core/inference/tools.py
  studio/backend/core/inference/mcp_client.py
  studio/backend/core/inference/studio_tool_loop.py
  studio/backend/core/inference/safetensors_agentic.py
  studio/backend/core/inference/llama_cpp.py
  studio/backend/state/tool_policy.py
  studio/backend/storage/mcp_servers_db.py
  studio/backend/tests/test_account_tool_isolation.py
  studio/backend/tests/test_tool_sandbox_per_thread.py
)
ruff check "${changed_py[@]}"
python3 scripts/enforce_kwargs_spacing.py "${changed_py[@]}"
python3 scripts/verify_import_hoist.py
python3 scripts/verify_import_hoist.py --audit "${changed_py[@]}"
git diff --check
```

All checks passed. The explicit import audit covered nine files with zero syntax skips, analyzer errors, or false positives. No frontend or Rust files changed; their tests were not run.

Integration notes and assumptions:

- Auth must bind the immutable account context before invoking these helpers or starting a loop. Outer generation threads must use `account_thread` or `run_as`; the changes here preserve that context across the inner tool, retrieval, and cleanup workers.
- The inference-route domain must call `require_tool_access(payload.permission_mode, bypass_permissions = bool(payload.bypass_permissions), disable_sandbox = bool(getattr(payload, "disable_sandbox", False)))` after authentication and before constructing a streaming response or dispatching generation. The loops refuse unsafe execution already, but an exception after streaming starts cannot change the response status to HTTP 400. `routes/inference.py` was outside the explicit file list and remains an integration dependency.
- MCP route validation should call `validate_mcp_address(url)` in its shared URL validator. Storage and transport currently refuse unsafe endpoints. Calling the helper during route validation additionally preserves per-entry import errors and preflight HTTP 400 responses. Without that wiring, an unsafe HTTP import entry can terminate the batch with HTTP 400 after earlier valid entries were saved, and a connection preflight can return an unsuccessful probe result instead of HTTP 400.
- Account deletion should cancel and drain the account's work, then use `run_as(account, close_mcp_sessions)` and `run_as(account, invalidate_tool_cache)` before renaming its directories. `close_mcp_sessions(all_accounts = True)` is reserved for process shutdown. Include the managed sandbox override directory in rename-aside handling when an override is configured.
- Project and file-serving callers should resolve managed tool files through `get_sandbox_workdir`/`resolve_sandbox_workdir` under the account context. Managed tool sessions deliberately do not use arbitrary host project workspace paths.
- I interpreted the named `storage/mcp_servers_db.py` as allowed and left route callers untouched. The implementation assumes account IDs and roles come from the frozen authenticated account contract. Neither referenced comparison PR was accessed.

Known gaps:

The route admission and MCP validation handoffs above are outstanding. The existing Python/terminal execution sandbox still relies on code/command checks, environment filtering, and resource limits; this patch does not add operating-system filesystem confinement. Account directory separation and edit-file containment alone are not a complete boundary against hostile Python/terminal code reading host files. OAuth browser sign-in and native Windows/macOS execution were not exercised end to end. These limitations must be resolved or accepted before claiming complete isolation for untrusted accounts.

The implementation and this report are committed on `mu/wt-07`; temporary test artifacts were removed and the working tree is clean.
