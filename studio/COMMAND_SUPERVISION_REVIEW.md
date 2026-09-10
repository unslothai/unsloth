# Managed-project command supervision

This source split contains the command supervisor, its execution and process-fence
support, Python/terminal routing, and focused native tests. Confined file edits are
owned by #10577, which this PR requires. Supervisor-specific shutdown retries live
with their implementation here. The standalone survivor-gated breadcrumb fix is
#10641, and approval UI remains separate.

## User-visible behavior

Sandboxed Python and terminal tools in a managed project now require Linux
bubblewrap plus an identity-verified PID namespace and pidfd lifecycle support.
The namespace restricts filesystem access and networking. Workspace admission and
mutation locks remain held until descendant cleanup is proven. Cleanup failures
retain their locks in quarantine.

When a command times out, its captured output remains in the completed tool result,
followed by the timeout status. Truncation notices and artifact cards remain visible.
The same bounded-output helper serves ordinary and supervised commands. Native
cleanup tests also confirm that the child ran before checking that no detached
descendant survives workspace lease and mutation-slot release.

On native macOS, a sandboxed project Python or terminal call returns:

```
Execution error: Secure supervised project commands require Linux bubblewrap process isolation. macOS sandbox-exec cannot prove detached descendants are gone.
```

This is a behavior change for macOS users whose project commands previously ran.
No user process is launched. Windows also refuses project commands before launch,
with its platform-unavailable explanation. Ordinary conversation commands and
explicit Full access continue through their existing execution paths. The existing
macOS filesystem sandbox alone does not establish the descendant-lifetime guarantee
used by this supervisor; passing macOS refusal tests is not macOS execution support.

A missing #10577 edit boundary makes command execution unavailable before probing or
spawning. CI composes its pinned source and separately exercises Linux execution,
macOS refusal and Windows refusal. Linux hosts that disallow the required namespace
also refuse safely.

## Validation

The native matrix runs the project-command, edit-prerequisite and supervisor tests.
Local macOS tests exercise the actual Python and terminal entrypoints and install a
Popen sentinel: the unsupported path must return before it can spawn. The full-access
and ordinary-conversation tests cover their separate existing paths.

Native receipts are reported for the published head in the PR description. Model
integration, packaged desktop behavior and release qualification are separate gates.

## Optional task routing

A private task-context hook lets the separate task-command layer resolve a durable
owned-worktree binding. Public command APIs still accept no root, environment or
boundary injection. Task execution masks the root `.git` marker with a read-only
empty mount and retains the same supervised process lease, mutation slot and
monitor-held project flock through cleanup or quarantine. Ordinary project
commands keep their existing behavior.

Shutdown retries quarantined processes before and after the generic process sweep.
The recovery breadcrumb is retained until both sweeps prove cleanup. These
supervisor-specific retries live here; #10641 contains only the independently
useful survivor check on the existing generic sweep. Native CI includes both
shutdown recovery and process-lifetime tests.
