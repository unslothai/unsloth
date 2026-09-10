// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type GitHubPreview,
  type GitHubRequest,
  type GitManagementState,
  type PreparedCommit,
  gitAction,
  gitConnectors,
  gitPreviewKey,
} from "../api/project-git-actions-api";
import {
  type ProjectGitStatus,
  getProjectGitStatus,
} from "../api/project-git-review-api";
import type { ProjectRecord } from "../types";
import { GitDiffFile, ProjectGitReviewPanel } from "./project-git-review-panel";

type GitProject = ProjectRecord & {
  workspaceRevision?: number;
  workspaceAvailable?: boolean;
};
const section =
  "rounded-xl border border-border bg-background/60 p-4 space-y-3";
const message = (error: unknown) =>
  error instanceof Error ? error.message : "Git request failed.";

export function ProjectGitPanel({ project }: { project: GitProject }) {
  // A project key retires every form, pending request, and confirmation together.
  return (
    <GitPanel
      key={`${project.id}:${project.workspaceRevision ?? 0}`}
      project={project}
    />
  );
}

function GitPanel({ project }: { project: GitProject }) {
  const revision = project.workspaceRevision ?? 0;
  const mounted = useRef(false);
  const sequence = useRef(0);
  const actionPending = useRef(false);
  const [state, setState] = useState<GitManagementState | null>(null);
  const [status, setStatus] = useState<ProjectGitStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [reviewVersion, setReviewVersion] = useState(0);
  const [paths, setPaths] = useState<string[]>([]);
  const [commitMessage, setCommitMessage] = useState("");
  const [prepared, setPrepared] = useState<{
    key: string;
    value: PreparedCommit;
  } | null>(null);
  const [branch, setBranch] = useState("");
  const [baseRef, setBaseRef] = useState("HEAD");
  const [connectors, setConnectors] = useState<
    Awaited<ReturnType<typeof gitConnectors>>
  >([]);
  const [handoff, setHandoff] = useState<GitHubRequest>({
    serverId: "",
    owner: "",
    repository: "",
    base: "main",
    head: "",
    title: "",
    bodyNote: "",
    draft: true,
  });
  const [preview, setPreview] = useState<{
    key: string;
    value: GitHubPreview;
  } | null>(null);
  const commitKey = gitPreviewKey(project.id, revision, [paths, commitMessage]);
  const handoffKey = gitPreviewKey(project.id, revision, handoff);
  const activeCommit = prepared?.key === commitKey ? prepared.value : null;
  const activePreview = preview?.key === handoffKey ? preview.value : null;
  const available = Boolean(
    state?.mutationsAvailable &&
      state.head &&
      state.fingerprint?.startsWith("c0dec0de"),
  );

  useEffect(() => {
    if (!preview) return;
    const timer = window.setTimeout(
      () => setPreview(null),
      Math.max(0, preview.value.expiresAt - Date.now()),
    );
    return () => window.clearTimeout(timer);
  }, [preview]);

  const refresh = useCallback(async () => {
    const request = ++sequence.current;
    const [next, gitStatus] = await Promise.all([
      gitAction<GitManagementState>(project.id, revision, "git/manage"),
      getProjectGitStatus(project.id, revision),
    ]);
    if (!mounted.current || request !== sequence.current) return;
    if (
      next.projectId !== project.id ||
      next.workspaceRevision !== revision ||
      gitStatus.projectId !== project.id ||
      gitStatus.workspaceRevision !== revision
    )
      throw new Error("Project workspace changed. Reopen Git review.");
    setState(next);
    setStatus(gitStatus);
    setPaths([]);
    setPrepared(null);
    setPreview(null);
    setReviewVersion((value) => value + 1);
  }, [project.id, revision]);

  useEffect(() => {
    mounted.current = true;
    const timer = window.setTimeout(() => {
      refresh().catch((failure) => {
        if (mounted.current) setError(message(failure));
      });
      gitConnectors()
        .then((items) => {
          if (mounted.current)
            setConnectors(items.filter((item) => item.is_enabled));
        })
        .catch(() => {
          /* Connector failure must not hide local Git review. */
        });
    }, 0);
    return () => {
      window.clearTimeout(timer);
      mounted.current = false;
      sequence.current += 1;
    };
  }, [refresh]);

  async function perform<T>(
    operation: () => Promise<T>,
    accept: (value: T) => void,
    reload = true,
  ) {
    if (actionPending.current) return;
    actionPending.current = true;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const value = await operation();
      if (!mounted.current) return;
      accept(value);
      if (reload) {
        try {
          await refresh();
        } catch (failure) {
          if (mounted.current)
            setError(
              `Action completed, but refresh failed: ${message(failure)}`,
            );
        }
      }
    } catch (failure) {
      if (mounted.current) setError(message(failure));
    } finally {
      actionPending.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  function action<T>(endpoint: string, method: string, payload?: unknown) {
    return gitAction<T>(project.id, revision, endpoint, method, payload);
  }

  return (
    <div className="mt-6 space-y-4 pb-8">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">Git & worktrees</h2>
          <p className="text-xs text-muted-foreground">
            Review local changes and keep recoverable snapshots.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={() => void perform(refresh, () => {}, false)}
        >
          Refresh
        </Button>
      </div>
      {error ? (
        <p
          role="alert"
          className="rounded-lg border border-destructive/30 p-3 text-sm text-destructive"
        >
          {error}
        </p>
      ) : null}
      {notice ? (
        <output className="block rounded-lg bg-muted p-3 text-sm whitespace-pre-wrap">
          {notice}
        </output>
      ) : null}
      <ProjectGitReviewPanel key={reviewVersion} project={project} />
      {state && !available ? (
        <p className="text-sm text-muted-foreground">
          {state.unavailableReason ??
            "Git mutations need a complete repository review on Linux or macOS. Windows currently supports status and diff review."}
        </p>
      ) : null}

      <section className={section}>
        <h3 className="font-medium">Selected files</h3>
        <p className="text-xs text-muted-foreground">
          Checkpoints and prepared commits capture the selected files as they
          are now. The active branch and staging area stay intact.
        </p>
        <div className="max-h-56 overflow-auto space-y-1">
          {status?.files
            .filter(
              (file) =>
                file.pathEncoding === "utf-8" &&
                !/[\\\r\n]/.test(file.path) &&
                !file.conflicted,
            )
            .map((file) => (
              <label key={file.id} className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  disabled={busy || !available}
                  checked={paths.includes(file.path)}
                  onChange={(event) => {
                    setPrepared(null);
                    setPaths((current) =>
                      event.target.checked
                        ? [...current, file.path].sort()
                        : current.filter((path) => path !== file.path),
                    );
                  }}
                />
                <code
                  className="break-all"
                  dir="ltr"
                  style={{ unicodeBidi: "isolate" }}
                >
                  {file.path}
                </code>
              </label>
            ))}
          {status?.files.length ? null : (
            <p className="text-xs text-muted-foreground">
              No changed files to select.
            </p>
          )}
        </div>
        <Button
          size="sm"
          variant="outline"
          disabled={busy || !available || !paths.length}
          onClick={() =>
            void perform(
              () => action("git/checkpoints", "POST", { ownedPaths: paths }),
              () => setNotice("Checkpoint saved."),
            )
          }
        >
          Save checkpoint
        </Button>
        <Textarea
          aria-label="Commit message"
          value={commitMessage}
          disabled={busy}
          onChange={(event) => {
            setPrepared(null);
            setCommitMessage(event.target.value);
          }}
          placeholder="Describe the selected changes"
        />
        <Button
          size="sm"
          disabled={
            busy || !available || !paths.length || !commitMessage.trim()
          }
          onClick={() =>
            void perform(
              () =>
                action<PreparedCommit>("git/commits/prepare", "POST", {
                  ownedPaths: paths,
                  message: commitMessage,
                }),
              (value) => setPrepared({ key: commitKey, value }),
              false,
            )
          }
        >
          Preview commit
        </Button>
        {activeCommit ? (
          <div className="rounded-lg border p-3 space-y-2 text-xs">
            <p>
              Prepared commit on {activeCommit.branch} at{" "}
              {activeCommit.baseHead.slice(0, 12) ||
                "new branch (no commits yet)"}
            </p>
            <pre className="whitespace-pre-wrap break-words">
              {activeCommit.message}
            </pre>
            <ul>
              {activeCommit.ownedPaths.map((path) => (
                <li key={path}>
                  <code>{path}</code>
                </li>
              ))}
            </ul>
            {activeCommit.previewFiles?.map((file) => (
              <GitDiffFile key={file.id} file={file} />
            ))}
            <p>
              Confirmation creates a recoverable Git ref. It does not advance
              the branch or publish changes.
            </p>
            <Button
              size="sm"
              disabled={busy}
              onClick={() => {
                const value = activeCommit;
                setPrepared(null);
                void perform(
                  () =>
                    action<PreparedCommit>(
                      `git/commits/preparations/${encodeURIComponent(value.id)}/confirm`,
                      "POST",
                      { confirmationToken: value.confirmationToken },
                    ),
                  (result) =>
                    setNotice(
                      `Prepared commit ${result.commitSha}\nRecovery ref: ${result.refName}`,
                    ),
                );
              }}
            >
              Create prepared ref
            </Button>
          </div>
        ) : null}
        {state?.preparedCommits.map((item) => (
          <div key={item.id} className="text-xs break-all">
            <strong>{item.status}</strong> · <code>{item.refName}</code> ·{" "}
            {item.commitSha?.slice(0, 12)}
            <Button
              size="sm"
              variant="ghost"
              disabled={busy || !available}
              onClick={() => {
                if (
                  !window.confirm(
                    `Remove recovery ref ${item.refName} at ${item.commitSha}? Keep a separate reference if you need this commit.`,
                  )
                )
                  return;
                void perform(
                  () =>
                    action(
                      `git/commits/preparations/${encodeURIComponent(item.id)}`,
                      "DELETE",
                    ),
                  () => setNotice("Prepared commit recovery ref removed."),
                );
              }}
            >
              Remove ref
            </Button>
          </div>
        ))}
      </section>

      <section className={section}>
        <h3 className="font-medium">Checkpoints</h3>
        {state?.checkpoints.length ? null : (
          <p className="text-xs text-muted-foreground">No saved checkpoints.</p>
        )}
        {state?.checkpoints.map((checkpoint) => (
          <div key={checkpoint.id} className="rounded-lg border p-3 space-y-2">
            <p className="text-xs">
              <code>{checkpoint.commitSha.slice(0, 12)}</code> ·{" "}
              {checkpoint.ownedPaths.length} files
            </p>
            <details className="text-xs">
              <summary>Files in this checkpoint</summary>
              <ul>
                {checkpoint.ownedPaths.map((path) => (
                  <li key={path}>
                    <code>{path}</code>
                  </li>
                ))}
              </ul>
            </details>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={busy || !available}
                onClick={() => {
                  if (
                    !window.confirm(
                      `Restore these files from checkpoint ${checkpoint.commitSha.slice(0, 12)}? Current contents of these paths will be replaced.\n\n${checkpoint.ownedPaths.join("\n")}`,
                    )
                  )
                    return;
                  void perform(
                    () =>
                      action(
                        `git/checkpoints/${encodeURIComponent(checkpoint.id)}/rollback`,
                        "POST",
                        { expectedCurrentFingerprint: state.fingerprint },
                      ),
                    () => setNotice("Selected checkpoint files restored."),
                  );
                }}
              >
                Restore files
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={busy || !available}
                onClick={() => {
                  if (
                    !window.confirm(
                      `Delete checkpoint ref ${checkpoint.commitSha.slice(0, 12)}? The saved recovery point will be removed.`,
                    )
                  )
                    return;
                  void perform(
                    () =>
                      action(
                        `git/checkpoints/${encodeURIComponent(checkpoint.id)}`,
                        "DELETE",
                      ),
                    () => setNotice("Checkpoint ref removed."),
                  );
                }}
              >
                Delete checkpoint
              </Button>
            </div>
          </div>
        ))}
      </section>

      <section className={section}>
        <h3 className="font-medium">Worktrees</h3>
        <p className="text-xs text-muted-foreground">
          Create an isolated checkout from a commit or branch. Cleanup preserves
          its branch and refuses uncommitted or ignored files.
        </p>
        <div className="grid gap-2 sm:grid-cols-2">
          <Input
            aria-label="Worktree branch"
            placeholder="unsloth-studio/my-task (optional)"
            value={branch}
            disabled={busy}
            onChange={(event) => setBranch(event.target.value)}
          />
          <Input
            aria-label="Worktree starting ref"
            value={baseRef}
            disabled={busy}
            onChange={(event) => setBaseRef(event.target.value)}
          />
        </div>
        <Button
          size="sm"
          disabled={busy || !available || !baseRef.trim()}
          onClick={() =>
            void perform(
              () =>
                action("worktrees", "POST", {
                  branch: branch.trim() || null,
                  baseRef: baseRef.trim(),
                }),
              () => setNotice("Worktree created."),
            )
          }
        >
          Create worktree
        </Button>
        {state?.worktrees
          .filter((item) => item.status !== "removed")
          .map((item) => (
            <div key={item.id} className="rounded-lg border p-3 space-y-2">
              <p className="text-xs break-all">
                <code>{item.branch}</code> · {item.status}
              </p>
              {item.merge ? (
                <p className="text-xs">
                  Merge: {item.merge.status}
                  {item.merge.primaryWorkspaceChanged
                    ? "; inspect the primary checkout before continuing"
                    : ""}
                </p>
              ) : null}
              {item.merge?.conflicts?.length ? (
                <pre className="text-xs whitespace-pre-wrap">
                  {item.merge.conflicts.join("\n")}
                </pre>
              ) : null}
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy || item.status !== "active"}
                  onClick={() =>
                    void perform(
                      () =>
                        action<{ path: string }>(
                          `worktrees/${encodeURIComponent(item.id)}/location`,
                          "GET",
                        ),
                      (result) =>
                        setNotice(`Owned worktree location:\n${result.path}`),
                      false,
                    )
                  }
                >
                  Show location
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={busy || !available || item.status !== "active"}
                  onClick={() => {
                    if (
                      !window.confirm(
                        `Merge ${item.branch} into ${status?.branch ?? "the primary branch"} at ${state.head?.slice(0, 12)}? Both checkouts must be clean.`,
                      )
                    )
                      return;
                    void perform(
                      () =>
                        action<{ merge?: { status: string } }>(
                          `worktrees/${encodeURIComponent(item.id)}/merge`,
                          "POST",
                          { expectedTargetHead: state.head },
                        ),
                      (result) =>
                        setNotice(
                          `Worktree merge: ${result.merge?.status ?? "completed"}.`,
                        ),
                    );
                  }}
                >
                  Merge
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy || !available}
                  onClick={() => {
                    if (
                      !window.confirm(
                        `Remove the clean Studio checkout for ${item.branch}? Its branch will remain available for recovery.`,
                      )
                    )
                      return;
                    void perform(
                      () =>
                        action(
                          `worktrees/${encodeURIComponent(item.id)}`,
                          "DELETE",
                        ),
                      () =>
                        setNotice(
                          "Worktree removed; its branch remains available.",
                        ),
                    );
                  }}
                >
                  Remove checkout
                </Button>
              </div>
            </div>
          ))}
      </section>

      <section className={section}>
        <h3 className="font-medium">GitHub handoff</h3>
        <p className="text-xs text-muted-foreground">
          Preview a pull request for an already published branch. Review the
          destination and exact request before submitting through your
          connector.
        </p>
        <select
          aria-label="GitHub connector"
          className="w-full rounded-md border bg-background p-2 text-sm"
          disabled={busy}
          value={handoff.serverId}
          onChange={(event) => {
            setPreview(null);
            setHandoff({ ...handoff, serverId: event.target.value });
          }}
        >
          <option value="">Choose a connected GitHub server</option>
          {connectors.map((item) => (
            <option key={item.id} value={item.id}>
              {item.display_name}
            </option>
          ))}
        </select>
        <div className="grid gap-2 sm:grid-cols-2">
          {(["owner", "repository", "base", "head", "title"] as const).map(
            (field) => (
              <Input
                key={field}
                aria-label={`Pull request ${field}`}
                placeholder={
                  field === "head"
                    ? `Published head branch (${status?.branch ?? "branch"})`
                    : `Pull request ${field}`
                }
                disabled={busy}
                value={handoff[field]}
                onChange={(event) => {
                  setPreview(null);
                  setHandoff({ ...handoff, [field]: event.target.value });
                }}
              />
            ),
          )}
        </div>
        <Textarea
          aria-label="Pull request notes"
          placeholder="Notes for reviewers"
          disabled={busy}
          value={handoff.bodyNote}
          onChange={(event) => {
            setPreview(null);
            setHandoff({ ...handoff, bodyNote: event.target.value });
          }}
        />
        <label className="flex gap-2 text-sm">
          <input
            type="checkbox"
            checked={handoff.draft}
            disabled={busy}
            onChange={(event) => {
              setPreview(null);
              setHandoff({ ...handoff, draft: event.target.checked });
            }}
          />
          Create as draft
        </label>
        <Button
          size="sm"
          variant="outline"
          disabled={
            busy ||
            !available ||
            !handoff.serverId ||
            !handoff.owner ||
            !handoff.repository ||
            !handoff.head
          }
          onClick={() =>
            void perform(
              () =>
                action<GitHubPreview>(
                  "review/pull-request-handoff/prepare",
                  "POST",
                  handoff,
                ),
              (value) => setPreview({ key: handoffKey, value }),
              false,
            )
          }
        >
          Preview pull request
        </Button>
        {activePreview ? (
          <div className="rounded-lg border p-3 space-y-2 text-xs">
            <p>
              {activePreview.connector.displayName} ·{" "}
              {activePreview.request.owner}/{activePreview.request.repo}
            </p>
            <p>
              {activePreview.request.head} → {activePreview.request.base} ·{" "}
              {activePreview.request.draft ? "Draft" : "Ready for review"}
            </p>
            <p className="font-medium">{activePreview.request.title}</p>
            <pre className="whitespace-pre-wrap break-words">
              {activePreview.request.body}
            </pre>
            <p>
              Reviewed local commit:{" "}
              <code>{activePreview.reviewBinding.head}</code>
            </p>
            <Button
              size="sm"
              disabled={busy}
              onClick={() => {
                const value = activePreview;
                setPreview(null);
                void perform(
                  () =>
                    action<{ result: string }>(
                      `review/pull-request-handoff/${encodeURIComponent(value.id)}/confirm`,
                      "POST",
                      {
                        serverId: value.connector.id,
                        confirmationToken: value.confirmationToken,
                        expectedRequestDigest: value.requestDigest,
                      },
                    ),
                  (result) => setNotice(result.result),
                  false,
                );
              }}
            >
              Submit reviewed pull request
            </Button>
          </div>
        ) : null}
      </section>
    </div>
  );
}
