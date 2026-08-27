# Unsloth Studio agent workspace QA matrix

Snapshot date: 2026-08-27

Local branch: `feat/codex-agent-workspace`

Feature source commit: `3af2f62e6d72564b2fa0840ca491a48817f12c3d`

Latest source hardening commit: `59995084d`

Full CI matrix hardening commit: `b1e6809d14d7a87d2213093f7a4676022744acfa`

CI matrix contract commit: `abc3b2cf4`

Integrated `upstream/main`: `1fe27b1b5`

The feature branch contains that upstream tip through a normal merge and is reviewable ordinary source. Publication state and remote CI must be verified on PR #9673 after each push. Nothing in this document claims that a packaged build, physical platform, live provider, or release candidate has passed.

Status meanings:

- `PASS`: focused local automation exercised the stated behavior.
- `PARTIAL`: the source path exists and focused checks pass, but part of the acceptance contract needs an unrun platform or live integration check.
- `GAP`: the requested functionality is intentionally unavailable or not implemented.
- `MANUAL`: packaged or physical platform evidence is required.
- `NOT CERTIFIED`: implementation exists, but the full release gate has not passed.

## Release gates

| Gate | Result | Evidence or remaining gate |
| --- | --- | --- |
| G0: reviewable feature diff | PASS on feature branch | Python, TypeScript, React, Rust, tests, workflow, and documentation are committed as ordinary source. Verify the live PR head and changed-file count after publication. |
| G1: current with upstream main | PASS at snapshot | The branch contains fetched `upstream/main` at `1fe27b1b5` through a normal merge. Refresh immediately before publication. |
| G2: ordinary source changes | PASS | Recovery payload files and the unsafe restore workflow are removed. The replacement is directly reviewable source. |
| G3: backend, frontend, and Tauri wiring | PASS locally | Native folder selection, signed grants, persistence, project context, agent workflow routes, and the Agent Workspace panel are connected. |
| G4: feature-specific automation | PASS for the merged feature suites | Exact local counts are recorded below. Full repository, remote CI, packaged app, and live runtime results are not implied. |
| G5: native platform certification | MANUAL | Packaged macOS, Windows, and Linux runs have not been recorded. |
| G6: full Codex parity | NOT CERTIFIED | Windows handle-verified repository and instruction traversal exists, but arbitrary Windows project edits and command execution remain fail-closed. Real model, provider, Codex, GitHub product handoff, packaged app, and physical platform checks remain unrun. |

## 1. Repository workspace

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| WS-01 to WS-05, WS-07, and WS-08 | PASS | Managed projects remain available. Existing folders use a native picker and purpose-bound grant, persist by canonical identity, share project cwd across chats and code tools, support bounded reads, remain separate from RAG Sources, and reuse the existing project record. |
| WS-06 | PARTIAL | Direct project edits use descriptor-relative confinement on macOS and Linux. Windows project edits fail closed pending an equivalent confinement implementation. |
| WS-09 to WS-12 | PASS | Equality and ancestor or descendant overlap are rejected for managed and folder-backed roots. Project deletion never deletes a user-owned repository, including `delete_files=true`. |
| WS-13 to WS-16 | PASS | Missing, read-only, symlink-selected, replayed, and identity-replaced roots fail without corrupting the project list. If optional Source attachment fails after durable creation, the project still opens and reports a retryable Sources-tab recovery instead of pretending creation failed. |
| WS-17 to WS-23 | MANUAL | Removable volume recovery, network and mapped-drive policy, Unicode paths, Windows case folding and long paths, packaged macOS access persistence, and packaged Linux folder access need platform evidence. Windows repository and `AGENTS.md` reads use handle-verified traversal; their Windows CI and packaged evidence remains pending until the remote matrix runs. |
| WS-24 | PARTIAL | Project and worktree writes use process-local cooperative writer slots with deterministic in-process behavior. Separate Studio processes and external Git or editor writers are not serialized by that lock. |

## 2. Durable project goal and `/goal`

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| GOAL-01 to GOAL-12 | PASS | Show, set, shorthand, done, reopen, clear, and help are local commands. They work without loading or calling a model, persist through restart, and remain project-scoped. |
| GOAL-13 to GOAL-17 | PASS | Active goal state enters bounded provider-neutral context in deterministic order, uses escaped structural boundaries, snapshots in-flight runs, and applies revision-aware update semantics to later runs. |
| GOAL-18 | PASS | The backend enforces the configured fresh-verification requirement before completing a goal. The UI cannot bypass this gate. |
| GOAL-19 | PASS | Local command results use normal persisted chat history and survive reload without a model turn. |
| GOAL-20 | PASS in focused automation | Single and compare request construction use the same project context contract. A packaged compare-mode smoke remains part of G5. |

## 3. Repository instructions (`AGENTS.md`)

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| INS-01 to INS-10 | PASS | Root and nested files resolve by target scope from ancestor to descendant. No-follow descriptor reads, size bounds, UTF-8 fallback, refresh on each run, and stable composition with project goal are covered. |
| INS-11 | PASS | User system text, project instructions, goal, and repository instructions have a deterministic tested order. |
| INS-12 | PARTIAL | Chat Completions, Responses, Anthropic, local, provider, and Codex runtime adapters share resolved project context in focused tests. Real runtime and provider calls remain unrun. |

## 4. Repository discovery and context selection

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| DISC-01 to DISC-03 | PASS | Root and nested ignore rules, including negation, are applied relative to their directory. |
| DISC-04 to DISC-08 | PASS | Heavy directories, binaries, oversized files, symlink loops, root escapes, and credential-shaped paths are excluded. |
| DISC-09 to DISC-12 | PASS | File, byte, and token limits are enforced and disclosed. Cancellation, refresh, rename, and deletion remove stale map entries. |
| DISC-13 and DISC-14 | PASS | Non-Git folders produce a bounded map. Nested repositories and submodule-like boundaries are explicitly excluded from the selected repository map. |
| DISC-15 | PASS | Metadata discovery and relevance selection avoid inserting the entire repository into each prompt. |
| DISC-16 | PASS for local fixture | The 100,000-path fixture completed in 0.693 seconds and stopped at 20,000 scanned paths and 20,000 included bytes with `path-limit` disclosure. Packaged platform responsiveness remains part of G5. |

## 5. Verification policy

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| VER-01 to VER-05 | PASS | Ordered test, lint, build, and custom checks persist per project. Saves use optimistic config revisions, foreground and queued runs bind the exact saved revision, invalid selected names fail instead of running a subset, and `/verify` remains local without inference. |
| VER-06 to VER-12 | PASS | Structured evidence records command, result, timing, bounded output, timeout, cancellation, process-tree cleanup, source fingerprint, freshness, and goal-completion eligibility. |
| VER-13 | PASS on supported POSIX paths | Commands use an explicit trusted shell contract inside project confinement. Repository Git configuration, hooks, filters, diff drivers, pagers, editors, credentials, and signing are neutralized for managed Git operations. |
| VER-14 | GAP | Windows PowerShell and cmd project execution is not implemented. The backend rejects it rather than running without an equivalent confinement boundary. |
| VER-15 | PARTIAL | macOS `sandbox-exec` and Linux `bubblewrap` confinement are implemented and focused POSIX checks pass. A Linux-only AF_UNIX check was skipped on the local macOS run, and packaged macOS and Linux smokes remain unrun. |

## 6. Git integration

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| GIT-01 to GIT-04 | PASS | Branch, detached state, staged, unstaged, untracked, and conflict state are bounded and binary-safe. Repository-controlled execution surfaces are neutralized. |
| GIT-05 to GIT-07 | PASS | Studio-owned checkpoint refs and fingerprint-guarded rollback preserve unrelated user changes and refuse stale state. No reset or clean fallback is used. |
| GIT-08 | PASS | Worktree branches use a validated collision-safe Studio namespace. |
| GIT-09 | PASS | Two-phase prepared commits accept reviewed tracked paths, use an expiring one-use confirmation, leave branch, index, and worktree unchanged, and write only a Studio-owned prepared-commit ref. |
| GIT-10 to GIT-12 | PASS | Detached HEAD, non-Git folders, and linked worktrees capability-gate safely with correct common-directory semantics. |

## 7. Durable plans

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| PLAN-01 to PLAN-05 | PASS | Plans have durable IDs, goal snapshots, ordered tasks, explicit states, clearable blocker text, and per-task verification requirements and evidence. Leaving the blocked state clears stale blocker summaries. |
| PLAN-06 to PLAN-09 | PASS | State survives restart, keeps explicit goal relationship semantics, rejects stale revisions, and supports deterministic local `/plan` lifecycle commands. |
| PLAN-10 | PASS | Completion and review summaries include evidence, task state, and remaining blockers. |

## 8. Background execution

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| BG-01 to BG-10 | PASS | Durable queue, atomic claim, bounded scheduling, real inference executor routing, success, failure, cancellation, process-tree cleanup, retry lineage, bounded logs, isolation, and restart-to-interrupted recovery are implemented and focused-tested. |
| BG-11 | PASS in focused automation | Missing or removed repositories fail without recreation or false success. Packaged removable-volume behavior remains manual. |
| BG-12 | PASS | The contract is process-local. App shutdown cancels active execution and restart reconciliation marks unfinished work interrupted, never successful. |

The background executor has production paths for local GGUF, local MLX-compatible runtime routing, external providers, and Codex subscription transport. Focused tests exercise the production seams with controlled runtimes. No real model, provider, or Codex call is claimed.

## 9. Parallel agents and worktrees

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| PAR-01 to PAR-04 | PASS | Marker-backed Studio worktrees use unique branch and path ownership, isolate files, bind goal, plan, task, and runtime context, and run verification in the worktree cwd. |
| PAR-05 and PAR-06 | PASS | Clean merges record their result. Unexpected conflicts remain visible with conflict paths for explicit resolution. There is no destructive abort, reset, clean, or force fallback. |
| PAR-07 | PASS | Queued and running agent tasks can be cancelled, with task state and cleanup eligibility preserved. |
| PAR-08 to PAR-10 | PASS | Cleanup requires canonical Studio ownership and a clean safe state, refuses foreign worktrees, and reconstructs durable task and worktree linkage after restart. |

Process-local writer slots coordinate Studio operations. They do not serialize a separate Studio process or an external editor or Git client.

## 10. Review and pull-request workflow

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| REV-01 to REV-04 | PASS | The Agent Workspace panel and routes expose bounded changed-file, staged, unstaged, untracked, diff, fresh verification, goal, plan, and blocker evidence. |
| REV-05 | PASS | Prepared-commit review is selected-path, two-phase, confirmation-bound, and non-mutating to the user branch and index. |
| REV-06 | PASS | A bounded, redacted PR title and body draft works without GitHub credentials. |
| REV-07 | PARTIAL | The connected GitHub handoff is wired to the exact connector schema with request digest, connector snapshot, and one-use process-local confirmation. Definite pre-dispatch rejections are distinguished from unknown transport or connector outcomes. Tests use a fake connector; no live GitHub request was made, and pending confirmations do not survive restart. |
| REV-08 to REV-10 | PASS | Local paths, credentials, and secrets are redacted; large evidence is bounded; and local review works without a remote. |

## 11. Harness and model portability

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| PORT-01 to PORT-05 | PARTIAL | One project harness routes local GGUF, MLX-compatible local runtimes, OpenAI-compatible providers, Codex subscription transport, Anthropic, and Gemini-style providers. Controlled adapter tests pass, but real runtime and provider calls remain unrun. |
| PORT-06 | PASS in focused automation | Tool-call healing retains the bound project session, cwd, goal, and instruction context. No real small-model quality run is claimed. |
| PORT-07 | PASS | Model changes do not mutate workspace, goal, plan, or task identity. |
| PORT-08 | PASS | Goal, plan, verify, status, and review administration works with no model loaded. |
| PORT-09 | PASS in focused automation | Both compare panes receive the same resolved project contract. Packaged compare UI smoke remains unrun. |
| PORT-10 | PASS | Queued prompts and background tasks carry exact project and worktree session identity without cross-project fallback. |

## 12. Security and destructive operations

| IDs | Result | Evidence or remaining gap |
| --- | --- | --- |
| SEC-01 to SEC-05 | PASS | Raw renderer paths, expired grants, OpenProject token replay, purpose mismatch, and identity swaps are rejected. Reusable model tokens remain purpose-bound for their separate validate and load operations. |
| SEC-06 and SEC-07 | PASS | Filesystem roots, home directories, sensitive credential roots, unsafe network roots, and managed-root overlap are rejected. |
| SEC-08 | PARTIAL | Descriptor-relative edit confinement and POSIX child-process confinement prevent project-root escape on macOS and Linux. Windows repository and instruction reads use verified Win32 handles, while arbitrary Windows project edits and commands fail closed pending an equivalent filesystem sandbox. External processes are outside the process-local writer scheduler. |
| SEC-09 to SEC-11 | PASS | Rollback, project deletion, worktree merge, and cleanup require ownership and fingerprint evidence and never use destructive fallback against user work. |
| SEC-12 | PASS | Goal, instruction, review, task output, and PR-draft boundaries are escaped, bounded, and redacted. |

## Focused validation evidence

The following local results were recorded after merging the fetched upstream tip. Overlapping suites are identified and must not be added into one synthetic total:

- Consolidated POSIX-compatible backend and workflow suite: 683 passed, 1 Linux-only AF_UNIX check skipped on the local macOS host.
- Provider, research, and tool-loop compatibility suite: 1,042 passed.
- Latest upstream GPU integration lane: 224 passed, 4 platform skips, and 6 subtests. The merge also fixes an order-dependent test fixture that previously unloaded imported NumPy and Torch modules through a broad `sys.modules` restoration.
- Security-focused workspace lane: 180 passed, 5 platform skips. The broader repository security run recorded 402 passed and 26 skipped; 5 Linux AppImage inspection checks were environment-blocked on macOS because no `readelf` implementation was installed.
- Full frontend Node test suite: 5,708 passed, 0 failed.
- Frontend production build: passed. Vite reported existing chunk-size and mixed dynamic-import warnings.
- Frontend TypeScript typecheck and targeted ESLint: passed.
- Targeted Biome: exited 0 with 132 warning-level diagnostics. This is not described as warning-free.
- Full Tauri Rust test suite: 413 passed, 0 failed, with one unrelated dead-code warning.
- Changed Rust files passed `rustfmt --check` with child modules skipped deliberately.
- Changed Python files passed Ruff.
- Workflow guard suite: 803 passed, 1 skipped. A dedicated matrix contract prevents regression to targeted frontend, Rust, or incomplete research lanes. The workflow now runs the full frontend and Tauri suites, includes durable Deep Research storage and progress coverage, and bounds Linux apt operations with retries and step timeouts.
- The direct-source workflow parsed with two jobs, and trigger lint scanned 44 workflows without finding an unsafe trigger or cache-key collision.
- `git diff --check` passed.
- Repository discovery fixture: 100,000 paths in 0.693 seconds with the configured bounds enforced and disclosed.

These are source and local integration results. They do not imply remote CI, packaged desktop, physical platform, or live runtime certification.

## Release decision

The source implementation is committed and can enter review while fork CI awaits maintainer approval, but merge readiness still requires the pushed head and approved remote CI to pass. The release remains not certified. Arbitrary Windows project edits and execution, real runtimes, providers, Codex transport, the product's live GitHub connector handoff, packaged builds, and physical platform smokes remain unrun. Do not merge or describe the product as release-ready until the remaining gates have recorded evidence.
