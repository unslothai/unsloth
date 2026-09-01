# Unsloth Studio Codex parity contract

Snapshot date: 2026-08-27

This document describes the product contract implemented by the current local working tree. It complements [CODEX_AGENT_WORKSPACE_MATRIX.md](./CODEX_AGENT_WORKSPACE_MATRIX.md), which records the exact QA rows and focused evidence.

Feature source commit `3af2f62e6d72564b2fa0840ca491a48817f12c3d` is integrated with fetched `upstream/main` at `1fe27b1b5` through a normal merge. Final Deep Research project-context hardening is committed at `59995084d`; the full CI matrix is committed at `b1e6809d1` and locked by the matrix contract at `abc3b2cf4`. The implementation is ordinary source. The live PR head and remote checks must be verified after publication.

The product target is the Codex agent coding experience inside Unsloth, not a visual clone: a durable project root, stable project context, local administrative commands, constrained tools, reviewable Git operations, recoverable background work, isolated worktrees, and one agent contract across local and hosted runtimes.

## Status vocabulary

- `Implemented locally`: ordinary source and focused behavioral tests exist in the working tree.
- `Partial`: the implementation exists, but a platform, live service, or broader product contract remains open.
- `Existing`: upstream Studio behavior is reused without a new agent-workspace implementation.
- `Gap`: the behavior is unavailable or outside this implementation.
- `Unrun`: the path exists, but the named real environment has not been exercised.

## Current 12-section parity

| Area | Local status | Implemented contract | Remaining evidence or gap |
| --- | --- | --- | --- |
| 1. Repository workspace | Implemented locally | Managed and existing-folder projects, one-use signed native selection grants, canonical identity, persistence, shared project cwd, RAG separation, overlap rejection, safe deletion, missing-root isolation, partial source-upload recovery, and in-process writer coordination. Windows repository and instruction reads use handle-verified traversal. | Packaged removable, network, Unicode, case, long-path, and permission smokes are unrun. External processes and separate Studio processes are not serialized. Arbitrary Windows project commands fail closed. |
| 2. Durable goal | Implemented locally | Local `/goal` show, set, shorthand, done, reopen, clear, and help; project scope; a project-level multiline goal surface; durable history; bounded context; stable ordering; revision behavior; and a backend fresh-verification completion gate. | Real model context smoke remains unrun. |
| 3. Repository instructions | Implemented locally | Root and nested `AGENTS.md`, subtree scope, ancestor-to-descendant order, no-follow reads, size and encoding bounds, refresh, and stable composition with project instructions and goal. | Real local and hosted runtime smokes remain unrun. |
| 4. Discovery and context | Implemented locally | Ignore-aware bounded maps, negation, heavy and binary exclusions, symlink safety, cancellation, refresh, non-Git fallback, explicit nested-repository exclusion, and relevance selection. | Packaged responsiveness is unrun. The local 100,000-path fixture completed in 0.693 seconds and stopped at its configured path and byte bounds. |
| 5. Verification | Partial | Durable ordered checks, optimistic config revisions, revision-bound foreground and queued runs, exact selected-name validation, local `/verify`, structured evidence, timeout, cancellation, bounded logs, staleness, process cleanup, and macOS or Linux project confinement. | Windows PowerShell and cmd execution is unavailable and fails closed. Packaged macOS and Linux shell smokes are unrun. One Linux-only AF_UNIX check was skipped on the local macOS host. |
| 6. Git | Implemented locally | Bounded status and diff, binary safety, hostile repository-config neutralization, Studio checkpoints, fingerprint-safe rollback, collision-safe branches, and two-phase prepared commits that preserve branch, index, and worktree. | The source is committed. The pushed PR head and remote checks still require live verification. |
| 7. Durable plans | Implemented locally | Goal-linked plans, ordered tasks, revisions, task states, blockers, verification requirements and evidence, restart recovery, local `/plan`, and completion summaries. | Packaged restart smoke remains unrun. |
| 8. Background execution | Implemented locally | Durable queue, atomic scheduling, production inference executor routing, task isolation, cancellation, retry lineage, bounded output, restart interruption, and missing-root failure. | Real GGUF, MLX, provider, and Codex runs are unrun. |
| 9. Parallel agents and worktrees | Implemented locally | Marker-backed owned worktrees, isolated task sessions, worktree verification, clean merge, retained conflicts, cancellation, safe cleanup, and restart recovery. | Cross-process and external writers remain outside the process-local scheduler. Conflict resolution is explicit, not automatic. |
| 10. Review and pull request | Partial | Changed-file and diff evidence, fresh checks, goal and plan state, prepared-commit review, redacted PR drafting, and confirmation-bound connected GitHub handoff. Definite pre-dispatch rejections remain retryable; only transport and server failures are treated as unknown outcomes. | Tests use a controlled connector. No real product GitHub connector call was made, and pending handoff confirmations are process-local rather than restart-durable. Publication of this source and remote PR checks are separate live gates. |
| 11. Harness portability | Partial | One project-context contract across local, provider, Anthropic-style, Gemini-style, OpenAI-compatible, and Codex routes; model switching; local administrative commands; compare context; and queued task isolation. | Real llama.cpp, MLX, small-model healing, hosted provider, and Codex subscription runs are unrun. |
| 12. Security and destructive safety | Partial | Raw path rejection, expiring one-use purpose grants, identity checks, sensitive-root rejection, descriptor-relative POSIX edits, POSIX process confinement, handle-verified Windows reads, project-safe Git, identity-checked deletion of Studio-owned storage, preservation of user-owned repositories, and bounded escaped prompt or review content. | Arbitrary Windows project edits and commands lack an equivalent filesystem sandbox and fail closed. Physical destructive-safety smokes remain unrun. |

## Product behavior

### Project and context

Every project has one authoritative workspace identity. Managed roots retain existing behavior. Existing-folder roots remain user-owned and are never deleted by project removal. Project chats, compare requests, foreground tools, queued prompts, background tasks, verification, and worktrees carry explicit project session identity. Sources remain optional RAG inputs and are never implied by opening a repository.

The request context order is deterministic: user system configuration, project instructions, active goal, then resolved repository instructions. Each structured segment is escaped and bounded. Repository discovery supplies bounded metadata and selected context instead of blindly embedding repository contents.

### Local command plane

`/goal`, `/plan`, `/verify`, `/status`, and `/review` are intercepted before model selection or inference. Their state changes use the same typed backend operations as the visible Agent Workspace panel and their responses persist through normal chat history.

The current contract does not claim a complete Codex command palette. `/init`, `/compact`, `/side`, slash-driven project or model switching, and personality profiles remain outside the 12-section acceptance implementation.

### Tools and verification

On POSIX, project file edits resolve descriptor-relative paths and reject root escape. Project commands run with an explicit cwd and environment contract. On macOS they use `sandbox-exec`; on Linux they use `bubblewrap`; child network access is denied. Windows repository and instruction reads verify Win32 handles, root identity, containment, long paths, case-insensitive identity, and reparse-point rejection. Arbitrary Windows project edits and commands fail closed because no equivalent filesystem sandbox is implemented.

Verification persists ordered test, lint, build, and custom checks with timestamps, exit status, bounded output, source fingerprints, freshness, cancellation, and timeout. Goal completion can require current passing evidence. Background agent tasks use the same project or worktree identity and evidence contract.

### Git, review, and parallel work

Managed Git calls neutralize repository-controlled hooks, filters, diff drivers, text conversion, fsmonitor, pagers, editors, credential helpers, signing, and aliases. Checkpoints and rollback operate only on Studio-owned refs and matching fingerprints. Prepared commits use reviewed tracked paths, an expiring one-use confirmation, and a Studio-owned ref without mutating the user's branch, index, or worktree.

Worktree tasks use unique branches, durable ownership markers, and explicit task binding. Successful merges record the resulting head. Unexpected conflicts remain in place with recorded paths for explicit resolution. Cleanup refuses foreign, dirty, active, or identity-mismatched worktrees and never falls back to reset, clean, force, or broad deletion.

The connected GitHub flow prepares a bounded redacted request, freezes connector identity and request digest, and requires one-use confirmation. A definite client rejection is reported as rejected before submission. Transport and connector failures remain unknown outcomes so the user can check GitHub before retrying. Focused tests use a controlled connector. No live request is claimed.

### Runtime portability

The background executor has production routes for local GGUF, local MLX-compatible runtimes, external providers, and Codex subscription transport. Runtime snapshots exclude credentials. Every tool call revalidates the exact project or worktree session. Unattended background execution accepts only project-bound off or full tool modes; interactive approval modes are rejected because no user is present to approve them.

This is source and adapter parity, not runtime certification. Real local models, hosted providers, Codex transport, and small-model healing remain unrun.

## Broader Codex features outside this matrix

These items appeared in the earlier roadmap but are not part of the user-provided 12-section release matrix. They must not be described as implemented by this working tree.

| Feature | Status | Boundary |
| --- | --- | --- |
| `/init` repository bootstrap | Gap | AGENTS resolution exists, but guided instruction generation and approval do not. |
| `/compact` durable handoff | Gap | Normal context truncation exists; no explicit project handoff artifact is claimed. |
| `/side` lightweight side thread | Gap | Multiple project chats exist; no dedicated side-thread contract is claimed. |
| Unified slash palette for project, model, reasoning, and personality | Gap | The local agent commands are implemented; the broader palette is not. |
| Per-turn mutation ledger and hunk-level stage or revert UI | Gap | Bounded diffs and prepared paths exist; a general edit ledger and hunk actions are not claimed. |
| User-authored persistent command rules | Gap | Project confinement and permission modes exist; a Codex-style rule file is not claimed. |
| Installable project skills and lifecycle hooks | Gap | MCP exists; a first-class skill and hook system is not claimed. |
| Scheduled project automations | Gap | Durable background tasks exist; scheduling is not claimed. |
| First-class child-agent delegation API with role and budget | Gap | Durable agent tasks and worktrees exist; a product-level spawn contract is not claimed. |

## Architectural rules

1. The renderer cannot authorize a raw filesystem path. The desktop shell issues a purpose-bound grant and the backend verifies it once against current folder identity.
2. A user-selected workspace is user-owned. Project deletion cannot delete, rename, empty, or replace it.
3. RAG Sources and workspace files are separate concepts.
4. One provider-neutral project contract serves local and hosted runtimes.
5. Goal, plan, verification, task, worktree, and Git evidence is durable and project-scoped. Review summaries are regenerated, and pending GitHub handoff confirmations are process-local.
6. Studio mutations are bounded, project-scoped, cancellable where applicable, and protected from stale state. Authentication does not imply a durable actor audit ledger.
7. Parallel work uses owned worktrees. Process-local locks coordinate Studio writers but do not claim control over external processes.
8. Git operations never trust repository-controlled executable configuration.
9. Conflict and cleanup paths preserve user work and do not use destructive fallback.
10. Unsupported platform security boundaries fail closed.

## Certification boundary

The implementation can enter source review while fork CI awaits maintainer approval. Merge readiness still requires the pushed PR head and approved remote CI to pass. It is not release-ready. Certification still requires packaged macOS, Linux, and Windows runs, a secure Windows project-command implementation, real llama.cpp and MLX runs, at least one real hosted provider, Codex transport, a live product GitHub handoff, physical removable and destructive-safety checks, and recorded evidence for every manual row.
