Implemented ownership-aware GPU switching on `mu/wt-06`, committed as `f9a76e63ae` (`Protect shared GPU generations during account model switches`). Training teardown integration and complete media-generation registration remain required from the other domains.

The changes:

- Forced model switching cancels only the caller's generations. Foreign generations produce HTTP 409 `gpu_busy`, including during a post-cancel drain whose timeout has expired. Preflight checks leave cancellation events untouched.
- The shared arbiter guard protects replacements within one backend as well as transfers between chat, images and video. Registration callbacks identify actual loads; ordinary same-model ownership assertions remain reusable across accounts. Request boundaries convert arbiter refusals into retryable HTTP errors.
- Chat and media auto-switch preserve their endpoint error envelopes, including late refusals during load preparation. Diffusion checks ownership before engine activation, which can unload a resident pipeline before the final acquisition.
- Retry hints inspect existing admission queues only on refusal. They estimate 15 seconds per occupied/queued wave, bounded to 120 seconds, with a 15-second fallback. Responses expose no foreign model, account or conversation identifiers.
- Durable generation registrations can borrow an existing entry only within the same account.
- Keep-warm behavior and admission scheduling remain unchanged. Every account using a backend refreshes its existing global activity clock; the last active account keeps that shared resident model warm.

`test_gpu_arbitration_sim.py` contains 133 CPU-only cases. Its small backend simulator uses the real arbiter and registry, enumerates three accounts and all backend pairs, and covers idle swaps, busy refusals followed by successful retries, a third account, scoped cancellation, deletion and guarded training teardown. Async stream tests exercise the real admission queue: two accounts occupy distinct slots concurrently, excess work queues, and cancellation of either a running or queued request leaves other accounts running.

Validation commands ran from `studio/backend`, using this environment:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export UNSLOTH_STUDIO_HOME="$PWD/../../.tmp/home" HF_HOME="$PWD/../../.tmp/hf" TMPDIR="$PWD/../../.tmp/tmp" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$PWD/../../.tmp/cache"
```

All writable test directories were inside this clone. Test output was also captured under `.tmp/`.

1. Initial contract and simulator run: **143 passed, 0 failed**. Ten more cases were added afterward.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_gpu_arbitration_sim.py -q -n 8 --timeout=330
```

2. Combined regression command, run twice during development:

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_gpu_arbitration_sim.py tests/test_gpu_arbiter.py tests/test_active_generations.py tests/test_openai_auto_switch.py tests/test_media_auto_switch.py tests/test_media_keepwarm.py tests/test_keepwarm_tick_off_event_loop.py tests/test_llama_admission*.py tests/test_anthropic_admission.py tests/test_chat_load_during_training.py tests/test_diffusion_routes.py tests/test_video_routes.py tests/test_tunnel_safe_long_post.py -q -n 8 --timeout=330
```

The first run had **1,535 passed, 7 failed, 7 skipped, 2 subtests passed**. Six failures came from passing a new keyword to existing test adapters; load calls now retain their original arguments. The remaining failure was the preview test-order issue described below.

The second run had **1,550 passed, 2 failed, 7 skipped, 2 subtests passed**. One failure was the same preview issue; the other was an xdist worker segmentation fault during garbage collection in `rich`. Subsequent runs completed without a worker crash.

3. New engine-activation regression test, before adding the earlier diffusion guard: **3 passed, 3 failed as expected**. The failures showed engine activation preceding refusal. All six cases pass in the final suite.

```bash
PYTHONPATH=. python -m pytest tests/test_gpu_arbitration_sim.py -k actual_media_load_routes -q -n 8 --timeout=330
```

4. Final regression command, run twice successfully, including after the final source edit: **1,534 passed, 0 failed, 7 skipped, 2 subtests passed** each time.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_gpu_arbitration_sim.py tests/test_gpu_arbiter.py tests/test_active_generations.py tests/test_openai_auto_switch.py tests/test_media_auto_switch.py tests/test_media_keepwarm.py tests/test_keepwarm_tick_off_event_loop.py tests/test_llama_admission*.py tests/test_anthropic_admission.py tests/test_chat_load_during_training.py tests/test_diffusion_routes.py tests/test_video_routes.py -q -n 8 --timeout=330
```

5. Tunnel suite in its own process group: **18 passed, 0 failed**. Together with the final regression run: **1,552 passed, 7 skipped, 2 subtests passed**. The contract suite retains its existing unawaited-coroutine warning.

```bash
PYTHONPATH=. python -m pytest tests/test_tunnel_safe_long_post.py -q -n 8 --timeout=330
```

6. Negative control against the frozen contract: **36 passed, 27 failed as expected**. A temporary `baseline_seams` pytest plugin compiled the old `acquire_for`, the three permitted inference guard/switch functions, and `ActiveGeneration.__enter__` from local commit `d010387b19` into their module namespaces. These failures demonstrate missing same-backend protection, scoped cancellation, foreign drain refusal, account-aware registration borrowing and chat auto-switch protection.

```bash
PYTHONPATH=.:../../.tmp python -m pytest tests/test_gpu_arbitration_sim.py -p baseline_seams -k 'sim_different_models_busy_then_idle_swap or route_preflight_and_force_respect_accounts or wait_refuses_foreign or borrowing_a_durable or chat_auto_switch_keeps_resident' -q -n 8 --timeout=330
```

7. Existing preview test-order issue reproduced using the same frozen-function plugin: **1 passed, 1 failed**. The first existing test leaves the admitted-inference counter incremented; the following preview test reports a busy model. Neither test file is in this worker's allowlist.

```bash
PYTHONPATH=.:../../.tmp python -m pytest tests/test_openai_auto_switch.py::test_preview_scope_disables_auto_switch tests/test_tunnel_safe_long_post.py::test_preview_chat_waits_for_a_slow_checkpoint_load -p baseline_seams -q -n 0 --timeout=330
```

Before each commit, the required checks passed: `ruff check <changed .py files>`, `python3 scripts/enforce_kwargs_spacing.py <changed .py files>`, and `python3 scripts/verify_import_hoist.py`. An additional import comparison against the staged files also passed: `python3 scripts/verify_import_hoist.py --before d010387b19 --after '' <changed .py files>`. Its initial alias-target blockers were resolved by naming `acquire_for_request` explicitly. `git diff --check` passed.

Single-account regression evidence: all 20 frozen contract tests pass. The simulator checks 24 owner-only arbiter combinations and all eight active/force/cancel combinations against legacy decisions. Owner drains retain their timeout behavior; idle clocks retain their timing; resident chat and media fast paths perform no new account-policy lookup or foreign-generation scan. Admission adds no per-account bookkeeping. No new `gpu_busy` refusal occurs with only owner generations. The existing non-forced `active_generations` 409 confirmation remains intact.

Integration notes and assumptions:

- Worker 05 should retain the small hunks in `_raise_or_cancel_active_generations`, `_wait_for_model_switch_idle`, `_maybe_auto_switch_model`, the chat/image load paths' arbiter usage, and the video acquisition import/call.
- Destructive callers can use `raise_if_other_accounts_active(account_id)` or its HTTP boundary `require_no_foreign_generations(account_id, path = ...)`. Invoke the guard under the lifecycle gate before teardown. `acquire_for_request(owner, register)` provides the final guarded acquisition and HTTP conversion; `acquire_for(..., replacing = True)` covers real replacements without a registration callback.
- Training must call the shared guard before its cleanup exception handlers, while holding admission/lifecycle protection through teardown and reservation. Current training routes unload resources before calling `release()`, so changing `release()` cannot prevent that teardown. The simulator verifies the required guarded sequence and that existing DIFFUSION/VIDEO release calls leave CHAT and its cancellation events alone. Actual training-route protection is not complete in this branch.
- Generation-owning domains must register chat, media and background work with the immutable account ID, preserve context across execution boundaries, and retain registrations until cleanup completes. Media routes currently have work that is not registered here; complete tracking is required for account-aware cross-backend protection. Lifecycle gates must prevent new execution during destructive handoffs.
- Account deletion should call `active_generations.cancel_all(account_id)` and let each registered stream unwind. Cancellation is cooperative and does not immediately remove registry entries. Each run needs its own cancellation event.

Known limits: no GPU, real llama-server, or training subprocess was exercised. Retry timing is a queue-based hint, not a predicted completion time. CPU-only media switches retain their existing busy probes. The existing preview test-order failure and the one intermittent interpreter crash are documented above. All source changes stay within the allowlist; the report is committed separately.
