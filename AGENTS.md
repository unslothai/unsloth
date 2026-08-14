# Repository Editing Guidelines

## Goal

Implement features in a way that minimizes future merge conflicts with upstream
and with work from other contributors.

## Required workflow

- Before starting work, read this file and report that you have done so.
- Inspect the existing architecture and extension points before editing code.
- Prefer adding isolated modules over rewriting existing core modules.
- Keep changes narrowly scoped to the requested feature.
- Do not perform unrelated refactors, renames, formatting, or import reordering.
- Avoid modifying generated files, vendored code, lock files, and build artifacts
  unless the task explicitly requires it.
- Reuse existing hooks, registries, adapters, configuration mechanisms, and public
  interfaces whenever possible.
- Preserve backward compatibility unless the task explicitly authorizes a breaking
  change.
- If a frequently modified core file must be changed, keep the integration patch
  as small as possible and place the main implementation in a separate module.
- Do not overwrite or revert changes that are unrelated to the current task.
- Keep logical changes separated so they can be reviewed or reverted independently.
- Add or update focused tests for changed behavior without rewriting unrelated tests.
- Before finishing, inspect the final diff, remove unrelated changes, and report
  that no unrelated diff remains.

## When uncertain

If implementing the feature requires a large core rewrite, broad formatting changes,
or changes to public APIs, stop and explain the conflict risk before proceeding.
Propose the smallest isolated alternative first.
