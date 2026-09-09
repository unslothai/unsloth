# Task test and build commands

This independent source layer connects opt-in project tasks to the existing
Linux command supervisor. It requires #10655 task ownership, #10633 lifecycle,
#10577 secure edits, #10594 Git/worktrees, #10658 task execution/UI, and #10636
command supervision. The workflow composes exact prerequisite revisions before
qualifying the implementation. The task and supervisor PRs contain small optional
backend hooks; the existing task panel owns the capability checkbox and evidence
view. The Git layer exposes its existing owned-path proof for trusted callers
already holding its project execution fence.

## User behavior

Enable **Allow implementers to run tests and builds** when starting a task.
It is off by default, resets on project changes, and is unavailable unless the
server certifies supervised Linux bubblewrap support. macOS and Windows refuse
execution. A coordinator or reviewer cannot run commands. Implementer children
inherit the captured opt-in, while explicit retries recheck host support.

Each attempt may reserve six commands, with at most 120 seconds of execution per
command and a 64 KiB output capture. A task deadline or cancellation also stops
command preparation/execution. Reservations are durable and never refunded after
failure, interruption or cancellation. Each retry is a new attempt with a fresh
checkout and command allowance. No network access or dependency installation is
provided. Tests and builds must use tools/dependencies already visible in the
supervisor's restricted system/Python runtime. Commands can change their assigned
checkout, so enabling them grants that mutation capability to implementers.

Expand **Test and build results** on an implementer task to see argv, exit status,
time limit, observed output bytes and bounded output. The list returns 4,096-character
previews; **Show captured output** retrieves that command's full bounded capture.
Pass, fail, timeout, cancellation, unavailable execution and unconfirmed cleanup
are distinct outcomes. A completed task does not imply its checks passed, and a
passing command covers only its invocation; files may change afterward.

## Ownership and failure behavior

- Model arguments contain only an argv array and integer timeout. No raw checkout,
  environment, sandbox, identity, worker token or command receipt is accepted.
- A live private task context, durable implementer role and captured opt-in are
  required. The root is resolved from the task's worktree binding, marker proof,
  Git registration and stored device/inode. Provider/runtime and task ownership
  are rechecked before dispatch and at physical spawn/release boundaries.
- The supervisor owns the process lease and mutation slot through quarantine.
  The existing project execution flock is inherited by the bubblewrap monitor;
  Git operations and project retirement already require the same lock. Final
  binding checks reuse the supervisor's held lock rather than nesting another
  flock or resolving a caller-supplied root.
- The writable mount is the owned checkout. Its root `.git` file is masked by a
  read-only empty mount; original Git metadata remains untouched and primary
  project files, host credentials and network access remain outside the boundary.
- Every execution receives a durable evidence row before dispatch. Bounded,
  redacted evidence is written even after task cancellation. Worker loss without
  a confirmed result displays an unconfirmed outcome, never a pass. A containment
  failure cancels further task tools and retains the supervisor's existing
  quarantine ownership. Cleanup-unconfirmed evidence stays conservative; normal
  Git operations resume only when the process fence can actually be acquired.
- No automatic commit, merge, worktree cleanup, live-model benchmark or inference
  quality qualification is introduced.

## Validation

The source split checks missing prerequisites. The composed suite covers role
and opt-in policy, request injection, atomic command counts, evidence persistence,
authenticated task/command scope, output bounds, final binding checks and a
simulated model through the real Studio tool loop. Native Linux cases execute
real confined commands and check pass/fail/timeout/cancel behavior, output floods,
Git marker protection, primary-file isolation and network refusal. Adjacent task,
Git, lifecycle, secure-edit and process-supervisor regressions remain in the same
matrix. Frontend API tests, type checks, build and bundle budget run on the composed
frontend. Browser fixtures exercise rendering only and do not execute a model or
native command.


Exact prerequisites used by this layer's CI:

| Layer | Revision |
| --- | --- |
| #10655 task engine | `191ac8ba86943eb0bd54a9f1532c5d2ed384c6c5` |
| #10633 lifecycle | `41e6153ddfce1ed41a6b4e32e9e1ded8c8042e2b` |
| #10577 secure edits | `c6118ec5a653e94158bd637bc59bc18656ca6257` |
| #10594 Git/worktrees | `016da23650c227ccfb99d8e3fca16473143f5bb9` |
| #10658 task execution/UI | `5e79f9deb8af6a014ad29d43c598d8cd2ef2ec36` |
| #10636 command supervisor | `354135952c8f48dfd224b874916805ea3a4833c4` |
