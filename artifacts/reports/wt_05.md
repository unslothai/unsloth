Worker 05 implemented inference and media account boundaries on `mu/wt-05`. Final verification: 5,182 passed, 77 skipped; four independently reproduced baseline offline failures excluded. Changes stay within the assigned source and test allowlist. No code was fetched or copied from PR #10102 or PR #10250.

Commits:

- `6bd0f58904` Scope media galleries, signed links and job registries to accounts.
- `2c28d5a88b` Restrict model access, downloads and inference control to accounts.

Changes and rationale:

- Image, audio and video galleries use the account workspace. Object reads, flags, deletion, clearing, video exports and OpenAI video jobs resolve within that workspace. Search-image registries, fences and thumbnail caches are account scoped. Signed media links bind the account and object together; existing owner link formats remain valid.
- Resident chat, image and video identity is hidden from other managed accounts. GPU leases supply ownership; a route-level record covers CPU residents. Generation, preview, load and unload paths check account access. Foreign active generations produce HTTP 409 with `{"error":"gpu_busy","retry_after":1}` and `Retry-After: 1`. Forced cancellation addresses the caller's registrations. Cancel IDs and load-request IDs are namespaced by immutable account ID.
- Shared model scans are filtered after reading shared caches. Managed catalog and advertised-path caches are separate. Models require an account grant or anonymous Hub confirmation. Public results are cached for five minutes; negative or unreachable results for thirty seconds. Unknown and gated metadata does not establish public access. Snapshot paths, companion overrides, adapter aliases and symlinks use the same access checks.
- Successful model and dataset downloads record durable grants in the initiating account's `studio.db`, under `app_settings.model_grants`. Entries use `model:org/repo` or `dataset:org/repo`. Transactions preserve simultaneous completions. Download authorization precedes shared-cache reuse, and gated repositories also require a Hub authorization check. Download progress and cancellation are scoped, including standalone speech-model downloads.
- Managed requests cannot inherit ambient HF credentials. Provider credentials, provider configuration locks, MCP route lookups and remote-code approvals use account identity. A caller-side schema initializer supports fresh account databases even when legacy storage modules have already initialized the owner's database. Shared-cache deletion, host folder suggestions and desktop reveal operations are restricted to the owner in multi-account mode.

Testing used the provided Python environment, with all writable test paths inside this clone. From `studio/backend`:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export UNSLOTH_STUDIO_HOME="$PWD/../../.tmp/home"
export HF_HOME="$PWD/../../.tmp/hf"
export TMPDIR="$PWD/../../.tmp/tmp"
export XDG_CACHE_HOME="$PWD/../../.tmp/xdg"
export HF_HUB_OFFLINE=1
mkdir -p "$UNSLOTH_STUDIO_HOME" "$HF_HOME" "$TMPDIR"
```

These argument groups express the exact test selections used, without repeating long lists for reruns:

```bash
A="tests/test_account_contract.py tests/test_account_model_access.py tests/test_account_inference_isolation.py tests/test_account_provider_isolation.py tests/test_account_media_isolation.py"
M="tests/test_account_contract.py tests/test_account_media_isolation.py tests/test_image_gallery.py tests/test_audio_gallery.py tests/test_video_gallery.py tests/test_gallery_flags.py tests/test_search_images.py tests/test_openai_videos_route.py tests/test_openai_images_generations_route.py"
I="tests/test_account_contract.py tests/test_active_generations.py tests/test_gpu_arbiter.py tests/test_diffusion_routes.py tests/test_video_routes.py tests/test_inference_status_route.py tests/test_inference_status_loaded_gguf.py tests/test_load_progress_throttle.py tests/test_load_progress_ready_fraction.py tests/test_hub_download_ambient_token.py tests/test_hub_download_transport_auto.py tests/test_cached_gguf_routes.py tests/test_gguf_variants_local_resolution.py tests/test_preview_routes.py tests/test_preview_followups.py tests/test_chat_load_during_training.py tests/test_gguf_load_cache_reuse.py tests/test_load_subdirs_stay_offline.py"
R=$(rg --files tests | rg 'test_(inference|gguf|stt|model|local|cached|hub|scoped|download|browse|credential|mcp_servers|provider|trc_approval|openai_models|diffusion_routes|video_routes|active_generations|gpu_arbiter|preview|load_|chat_load|audio_gallery|image_gallery|video_gallery|gallery_flags|search_images)')
E="tests/test_openai_images_generations_route.py tests/test_openai_videos_route.py tests/test_diffusion_lora.py tests/test_diffusion_controlnet.py"
B="tests/test_hub_token_caller_identity.py::test_seed_inspection_derives_its_policy_from_the_caller tests/test_hub_token_caller_identity.py::test_an_anonymous_config_read_does_not_strip_the_process_credential tests/test_hub_token_caller_identity.py::test_the_config_probes_do_not_go_local_only_for_an_anonymous_caller tests/test_inference_default_models_non_blocking.py"

PYTHONPATH=. python -m pytest $M -q -n 8 --timeout=330
PYTHONPATH=. python -m pytest $I tests/test_scoped_load_cancel.py -q -n 8 --timeout=330
PYTHONPATH=. python -m pytest $I -q -n 8 --timeout=330
PYTHONPATH=. python -m pytest $A -q -n 8 --timeout=330
PYTHONPATH=. python -m pytest $A $R -q -n 8 --timeout=330
PYTHONPATH=. python -m pytest $A tests/test_openai_images_generations_route.py tests/test_openai_videos_route.py tests/test_openai_models_media.py tests/test_cached_gguf_routes.py tests/test_diffusion_lora.py tests/test_diffusion_controlnet.py -q -n 8 --timeout=330
PYTHONPATH=. python -m pytest $A $R $E \
  --deselect=tests/test_hub_token_caller_identity.py::test_seed_inspection_derives_its_policy_from_the_caller \
  --deselect=tests/test_hub_token_caller_identity.py::test_an_anonymous_config_read_does_not_strip_the_process_credential \
  '--deselect=tests/test_hub_token_caller_identity.py::test_the_config_probes_do_not_go_local_only_for_an_anonymous_caller[False-False]' \
  --deselect=tests/test_inference_default_models_non_blocking.py::test_default_models_returns_static_defaults_before_top_fetch \
  -q -n 8 --timeout=330
# The preceding backend-relative deselections did not match repository-root node IDs.
# Reran with these corrected selectors:
PYTHONPATH=. python -m pytest $A $R $E \
  --deselect=studio/backend/tests/test_hub_token_caller_identity.py::test_seed_inspection_derives_its_policy_from_the_caller \
  --deselect=studio/backend/tests/test_hub_token_caller_identity.py::test_an_anonymous_config_read_does_not_strip_the_process_credential \
  '--deselect=studio/backend/tests/test_hub_token_caller_identity.py::test_the_config_probes_do_not_go_local_only_for_an_anonymous_caller[False-False]' \
  --deselect=studio/backend/tests/test_inference_default_models_non_blocking.py::test_default_models_returns_static_defaults_before_top_fetch \
  -q -n 8 --timeout=330
PYTHONPATH=. python -m pytest tests/test_inference_default_models_non_blocking.py --collect-only -q
```

Run results, including intermediate failures:

| Selection | Passed | Failed | Skipped | Notes |
| --- | ---: | ---: | ---: | --- |
| M, initial | 400 | 3 | 0 | New test fixtures corrected |
| M, rerun | 403 | 0 | 0 | First commit validation |
| I plus nonexistent scoped-load test | 0 | 0 | 0 | Exit 5; no tests collected |
| I | 896 | 5 | 3 | Fixed compatibility delegation, owner thread factory and local inventory regression |
| A, initial | 80 | 8 | 0 | Corrected dependency fixtures and managed provider deletion behavior |
| A, rerun | 88 | 0 | 0 | Account tests passed |
| A, expanded | 97 | 0 | 0 | Added speech, CPU and credential coverage |
| A + R | 4,981 | 4 | 77 | Four baseline offline failures |
| A + R, expanded | 4,987 | 5 | 77 | Four baseline failures and CPU image unload ownership gap |
| Follow-up selection | 553 | 1 | 1 | Reproduced CPU image unload gap; subsequently fixed |
| Frozen-base B | 2 | 4 | 0 | Same four offline failures on `mu/base` |
| A + R + E, ineffective backend-relative exclusions | 5,182 | 4 | 77 | Only the four baseline failures remained |
| Final A + R + E, corrected exclusions | 5,182 | 0 | 77 | Four baseline cases excluded; two subtests passed; 109.56 seconds |

The collection-only command collected one test and confirmed the repository-root node ID.

The frozen-base comparison used a local archive, without changing branches or touching another worker's clone:

```bash
# From the clone root:
mkdir -p .tmp/baseline
git archive mu/base studio/backend | tar -x -C .tmp/baseline
cd .tmp/baseline/studio/backend
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export UNSLOTH_STUDIO_HOME="$PWD/../../../../.tmp/home"
export HF_HOME="$PWD/../../../../.tmp/hf"
export TMPDIR="$PWD/../../../../.tmp/tmp"
export XDG_CACHE_HOME="$PWD/../../../../.tmp/xdg"
PYTHONPATH=. python -m pytest $B -q -n 8 --timeout=330
```

The four baseline failures exercise mocked online config/seed probes and asynchronous remote model ranking while the required environment forces offline behavior. They were retained unchanged. The existing account-contract coroutine warning and model-memory syntax warning also remain. No frontend or Rust files changed, so frontend and Rust checks were not run.

Before each commit, the required checks passed from the clone root:

```bash
ruff check <changed .py files>
python3 scripts/enforce_kwargs_spacing.py <changed .py files>
python3 scripts/verify_import_hoist.py
```

`<changed .py files>` included added tests. For the report commit, this was the cumulative list from `git diff --name-only mu/base -- '*.py'`. `git diff --check` also passed. All checks reported zero blockers.

Single-account regression evidence: the existing gallery, signed-link, model status, preview, load, cached-model, provider, MCP and credential tests were exercised. Owner galleries and remote-code approval paths retain their historical locations. Owner model lists bypass grant database reads and public Hub probes; a dedicated test fails if either occurs. Existing owner token fallback, thread construction, scan caches and API response shapes are retained. No new account-policy network calls or executor jobs occur on owner paths. CPU ownership bookkeeping is active only in multi-account installations. Authentication/login, desktop, sandbox, GPU-arbiter, active-generation and keepwarm implementation files were not changed. These tests establish behavioral compatibility; no live GPU performance benchmark was run.

Integration requirements and assumptions:

- Authentication must bind the frozen `AccountContext` for the entire request. Background work must preserve it through `account_thread`, `run_as` or `arun_as`. Unbound work is still the owner by contract.
- Worker 06 must retain account identity in core media/STT workers and keep active generations registered throughout loading, generation and persistence. The route checks complement GPU-arbiter locking; they cannot replace atomic checks inside engine transitions. Core CPU loads that bypass these routes should call `account_access.note_resident_account(modality, *references)`. Other route surfaces can use `gpu_busy_route` for the same retry envelope.
- Dataset and other shared-cache consumers should use `repo_visible(..., repo_type = "dataset")`, `require_model_access(reference, "dataset")` or `filter_model_rows(..., repo_type = "dataset")`. Shared download completion already writes dataset grants. Treat `app_settings.model_grants` as internal state, not a user-editable preference.
- Worker 07 must scope MCP clients, tool-approval state and sandbox execution. These routes perform account-bound DB lookups before MCP calls; session and tool execution implementations were not edited. API-monitor persistence still needs immutable account IDs in its owning domain.
- Account deletion must revoke access and settle background downloads before renaming directories. Shared download cancellation can run in the target account context through `download_lifecycle.cancel_worker`; standalone speech downloads use `_cancel_account_stt_download`. Otherwise a late completion could recreate the retired account directory.
- The caller-side `ensure_account_schema` bridge can be removed when storage schema-readiness caches are keyed by database path throughout the storage domain.

Known limits: managed accounts cannot load arbitrary host paths or owner-local adapters outside their workspace and recognized HF caches. Core LoRA/ControlNet discovery roots are outside this allowlist; their owner files are withheld, while full managed local-adapter discovery needs the core roots made account aware. Public proofs have the documented cache expiry; offline operation withholds ungranted repos. Signed media URLs remain bearer capabilities: possession permits redemption, but changing the signed account or media ID does not. Hub, provider and GPU behavior was tested with mocks and registered generation events, not live services or hardware. Full end-to-end isolation requires the other workers' domains to be integrated.

The source changes and this report are committed. Temporary test files and the untracked task-prompt copy were removed. The working tree is clean.
