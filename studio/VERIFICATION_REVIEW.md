# Project verification and reviewed tool hooks

Third source split from #9673. This branch starts at upstream main
`29b87ecac` and composes with the secure command runner in #10577. The pinned
prerequisite is `d79c53fdba726fec0b27dc4385667ba3798be277`; the CI workflow
merges that exact commit before qualifying native execution.

## User flow

Open a managed project's **Checks & hooks** tab, add named test, lint,
typecheck, build, or custom shell commands, and save the profile. Select
**Run checks**, or send `/verify` in a project conversation. Comparison sends
execute the command once and display the same response in both panes.

A run records the profile and workspace revisions, command order, required
versus optional checks, bounded output, exit status, timing, and cancellation.
Runs continue independently of the chat and can be cancelled from the panel.
History survives a backend restart; expired owners become interrupted rather
than successful. Saving a profile does not run it.

The run does not snapshot project source. The UI and API explicitly label
source freshness **unverified**, including after a passing result. Git source
snapshots and checkpoints belong to the subsequent split.

## Reviewed hooks

Place a configuration such as this in the project's `.codex/hooks.json`:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write|Bash",
      "hooks": [{
        "type": "command",
        "command": "python .codex/check.py",
        "timeout": 30,
        "additionalContextLimit": 1000
      }]
    }]
  }
}
```

Review the displayed commands and full hash before trusting the current file.
Trust, revocation, and per-handler preferences require a UI session and a
matching revision. Changing the file or workspace invalidates effective trust.
Trust covers the configuration bytes; commands can invoke mutable project
scripts. All execution retains the native project boundary.

This split runs synchronous `PreToolUse` and `PostToolUse` hooks for `edit_file`,
`python`, and `terminal`. Matchers accept their Studio names and the
`Edit`/`Write`, `Python`, and `Bash`/`Terminal` aliases. Session, permission,
compaction, delegation, stop, and asynchronous declarations remain visible but
inactive. The trust dialog states this scope.

Hooks receive bounded JSON on stdin with the event, project, original tool name
and arguments, and (for post-hooks) a bounded tool response. A pre-hook failure,
timeout, truncated result, `decision: block`, `continue: false`, or
`permissionDecision: deny` prevents the tool. Allow decisions and argument
rewrites do not grant authority or alter the tool. Post-hook failure is reported
alongside the completed tool result. Each event has a combined 60-second budget;
each handler has a 16 KiB output capture limit. Additional context uses a
conservative UTF-8 byte allowance within the configured token limit, capped at
2 KiB per handler, and shares the tool's final result budget.

## Execution and retirement

Review, configuration, and history work without #10577. Commands fail closed
when its reviewed-command and process-tree fencing support is absent. With that
prerequisite, execution is available only on Linux with a working bubblewrap
and user-namespace facility. macOS can review hook files and profiles; Windows
supports verification configuration/history but secure hook file discovery and
project command execution are unavailable.

The command runner revalidates the persisted workspace, profile/trust revision,
exact configuration hash, and cancellation/retirement authority before native
spawn and again before releasing user code. Revocation and project retirement
also cancel running hooks. Archive/delete establish a durable admission fence,
cancel verification, and retain the inherited process-tree fence through the
storage mutation. A failed cleanup does not release that fence.

Verification uses strict SQLite schemas, revision checks, owner leases,
monotonic evidence/history revisions, bounded persistence, and a write
attestation layer. These checks cover its owned tables; arbitrary access to
SQLite internals or deletion/recreation of the parent project row is outside
that authority boundary.

## Review map

- `backend/core/agent_workspace/verification_state.py`: durable profiles, runs,
  owner recovery, cancellation, retirement, evidence validation, and pruning.
- `verification.py` and `verification_process.py`: background orchestration and
  the restrictive adapter to the prerequisite runner.
- `hooks.py`, `hook_runtime.py`, and `backend/storage/project_hook_trust_db.py`:
  bounded discovery, exact-file trust, matching, and synchronous tool hooks.
- `backend/routes/project_verification.py` and `project_hooks.py`: authenticated
  review APIs under `/api/agent/projects/{project_id}`.
- `frontend/src/features/chat/components/project-checks-panel.tsx`: lazy-loaded
  entry point for profile/run controls and hook review.

## Validation

`.github/workflows/studio-verification-ci.yml` tests both standalone and composed
source on Linux, macOS, and Windows. The composed Linux lane requires actual
native execution, including durable output, optional-check aggregation,
detached descendant cancellation, trust revocation before command release, and
a pre-hook blocking the public edit tool. Unsupported-platform behavior is
explicitly tested; a skipped native test is not execution certification.

Local checks include the focused backend suite, adjacent project/history/edit
regressions, the full frontend suite, typecheck, production build, bundle budget,
workflow guards, Python 3.10 floor, Ruff, and a browser inspection using fixture
data. The source split contains no generated frontend assets. Packaged desktop
UI, live model/provider loops, folder-backed workspaces, other lifecycle events,
and release qualification remain separate work.
