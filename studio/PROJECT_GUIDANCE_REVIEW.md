# Project guidance: first context layer from #9673

This change is independently refreshed against upstream `main` at
`95feb6979`. It uses Studio's existing managed
project sandbox. The choice between existing-folder PRs #9859 and #9670 remains
with the maintainer; neither implementation is part of this patch.

## Behavior

- `/init` creates a starter `AGENTS.md` without invoking a model or overwriting
  an existing `AGENTS.md` or `AGENTS.override.md`. Compare mode executes it once
  and shows the same result in both panes.
- The project landing page has an **Instructions & skills** tab. It shows the
  resolved root instructions, discovered skill metadata, and excluded paths.
- Existing stored project instructions retain the frontend's exact
  `<project_instructions>` serialization for sends and token counting. With no
  AGENTS file or skills, the backend preserves model messages and system bytes.
  The backend resolves root agent instructions and project skills for each request. Caller-supplied server envelopes are
  replaced with current server-resolved guidance. Ordinary chat IDs that happen
  to match `project-<id>` are not treated as project authority.
- Automatic prompt injection reads root instructions. The authenticated
  instructions endpoint also accepts a relative `target` to inspect the
  applicable root-to-target instruction chain, preferring `AGENTS.override.md`
  within each scope. Automatic nested instruction discovery during tool edits
  is not claimed by this layer.
- Skills are discovered under `.agents/skills/<package>/SKILL.md`. The catalog
  includes metadata; a `$name` request includes that skill's body. Ambiguous
  names, malformed files, and traversal outside the workspace fail closed.
- OpenAI chat, Responses, Anthropic messages, provider forwarding, and token
  counting carry the project session. Local preflight parsing and MLX token
  counting see the same guidance as completion dispatch.
- Existing durable chat generation holds the project deletion fence for the
  stream lifetime. Cancellation while acquiring that fence releases an entry
  that completes after cancellation.

## Boundaries

File traversal uses bounded, descriptor-relative POSIX reads, verifies directory
identity, and rejects symlinks and nonregular files. This patch introduces no
folder selection, database migration, native path grants, Git mutation APIs,
verification engine, child-agent executor, or task scheduler.

On Windows, stored project instructions continue to work; secure repository
instruction/skill traversal and `/init` are unavailable pending the separate
Windows filesystem layer. Deep Research retains its existing client-generated
project instructions; new repository-guidance parity for that subsystem belongs
to the continuity follow-up.

## Local validation

The review follow-up adds tests for prompt byte preservation, external project
session metadata, acquiring the deletion fence before validation, and deletion
during workspace revalidation. The focused guidance backend suite passes 57 tests.
The counts below are historical validation of the prior head, not new-head CI.

- Project guidance, cancellation, durable chat, OpenAI passthrough, and Responses:
  **772 passed**, one warning.
- Full frontend suite: **7,119 passed**, zero failures.
- Frontend typecheck, production build, startup bundle budget, Python lint,
  pinned Python formatter, and whitespace checks pass.
- Broader model-routing/media selection: **784 passed, 2 skipped, 2 failed**.
  Both failures reproduce on untouched `d530284a0`: the case-variant cache test
  assumes a case-sensitive filesystem, and the API-key detection test attempts
  to open the real auth database under the host sandbox. Neither is reported as
  a passing gate.
- ESLint on the modified existing frontend surfaces has upstream diagnostics.
  Both baseline and candidate report 64 errors and 12 warnings, with no added
  diagnostics after normalizing shifted line numbers. The new guidance modules
  pass. This is not a clean repository lint run.

These are local source checks. Hosted CI, packaged platform tests, physical UI
checks, and real model/provider runs are not certified by this receipt.

## Remaining split order

1. Project context (`/init`, repository instructions, project skills): this patch.
2. Secure cross-platform edits and commands, including Windows qualification.
3. Verification workflows and hooks.
4. Git worktrees, status/diff, hunk review, checkpoints, and GitHub handoff.
5. Child-agent delegation.
6. Background tasks and scheduling.
7. Chat continuity (`/compact`, `/side`, Deep Research).

Each subsequent branch needs a fresh upstream/source audit and its own evidence.
Historical aggregate validation on #9673 or #9987 does not certify a split head.
