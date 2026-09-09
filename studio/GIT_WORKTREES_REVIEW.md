# Git review and owned worktrees

This source split adds the **Git & worktrees** tab to project landing pages on
current Studio. It operates on a Git repository already present in a persisted
managed project workspace. Folder selection and native path grants remain in
the existing-folder workspace PR; this split does not accept a renderer-supplied
filesystem root.

Read-only review works independently. Git mutations require project lifecycle
support from #10633; they refuse before workspace access without that support.
OAuth GitHub handoff additionally requires #10632's configuration-check capability.
These shared changes are separate PRs and are not included in this source diff.

## Review path

1. `backend/core/agent_workspace/git_review.py` produces bounded status and
   structured file/hunk manifests for all, staged, and unstaged changes. Two
   captures must agree. Conflicts, output limits, and incomplete untracked
   content remain visible and cannot certify a complete snapshot.
2. `git_service.py` uses the same trusted Git executable, private environment,
   empty hooks directory, and executable-driver overrides for mutations.
   Included configuration and linked-worktree configuration are inspected.
   Repository-local settings apply; system/global Git configuration is not
   inherited, including global line-ending conversion and ignore files.
   Output is bounded while the process runs. Git helpers, filters, external
   diffs, signing, automatic maintenance, and network protocols are disabled.
3. `git_guard.py` holds project storage, the shared secure-tools mutation slot
   when installed, a process-shared project fence, and a repository fence keyed
   by Git's common directory. `process_fence.py` is identical to the secure-tools
   prerequisite. Windows mutations refuse before launching Git.
4. `worktrees.py` combines an ownership marker, its secret digest in SQLite, and
   Git registration. Creation pins the starting commit and reserves durable
   state first. Cleanup requires matching ownership and a clean checkout,
   rejects ignored files, and retains the branch. Merge preflight reports
   conflicts; an uncertain real merge leaves the checkout for manual recovery.
   It never performs reset, clean, force-removal, or automatic merge abort.
5. `checkpoints.py` records selected-file snapshots under private Git refs with
   a temporary index. Rollback requires the current fingerprint and unchanged
   checkpoint identity. `prepared_commits.py` presents the actual selected diff,
   then consumes a durable, expiring token to create a prepared ref. Neither
   operation advances the active branch or changes the staging area. An
   uncertain ref publication retains its candidate SHA and recovery record.
6. `github_handoff.py` binds an expiring, one-use preview to the connector
   configuration, tool contract, local branch, commit, and complete content
   fingerprint. Confirmation verifies the published head with the connector's
   `get_commit` before invoking `create_pull_request`. The OAuth/one-shot MCP
   path requires the separately installed configuration checks before handoff.
   Ambiguous submission errors require checking GitHub before
   retrying; the UI never retries automatically.
7. `git_retirement.py` prevents archive/delete from racing guarded operations.
   Project deletion refuses while owned worktrees, checkpoint refs, or prepared
   refs still need recovery. The UI exposes explicit removal of recovery refs.
   Startup reconciles worktree and checkpoint ownership conservatively.
   When the verification split is installed, this retirement entry point also
   cancels its runs and delegates the shared process fence to verification.
   Integration must retain this one wrapper around archive/delete, rather than
   nesting both splits' retirement wrappers.

All mutation routes require an authenticated UI session and workspace revision.
API keys retain read access but cannot create, restore, merge, remove, or submit.
The validated worktree location is available only through a UI-authenticated
read; ownership secrets are never returned.

## Platform and scope

| Surface | Linux/macOS | Windows |
| --- | --- | --- |
| Status and structured tracked diffs | Available | Available |
| Untracked content | Descriptor-relative reads | Requires the secure-tools split's native reader |
| Checkpoints, prepared refs, worktrees | Native Git with guarded operations | Refused before mutation |
| GitHub handoff | UI-reviewed connector path | Unavailable until mutation/review boundary is qualified |

The native workflow composes pinned lifecycle support in both variants and the
pinned secure-tools source in the second variant. Shared UI insertions from the other review splits may require
integration when those PRs land. No generated frontend files are included.

Prepared refs are recovery objects; they are not branch commits or a push.
Handoff requires an already published branch and compatible `get_commit` and
`create_pull_request` tools. Remote-head verification is a point-in-time check,
not a lock against another GitHub writer. These advisory locks coordinate Studio
writers; they cannot make multi-command Git operations atomic against arbitrary
external filesystem or Git writers. Source changes observed during review or
confirmation cause rejection.

Packaged desktop behavior, real model/provider tool loops, existing-folder
integration, and a live authenticated GitHub handoff remain separate qualification
gates. Headless browser fixtures exercise rendering and confirmation invalidation
without publishing a pull request through the product.

## Reproduce validation

From `studio/backend`:

```sh
python -m pytest tests/test_project_git_review.py tests/test_project_git_review_routes.py \
  tests/test_agent_workspace_worktrees_focused.py tests/test_project_git_safety.py \
  tests/test_project_git_review_regressions.py tests/test_project_git_prerequisites.py -q --timeout=120
```

Compose the pinned lifecycle commit in the workflow before testing mutations.
The prerequisite tests explicitly cover refusal when that support is absent.

From `studio/frontend`: `npm test`, `npm run typecheck`, `npm run build`, and
`npm run bundle:check`. The dedicated native workflow is
`.github/workflows/studio-git-review-ci.yml`.
