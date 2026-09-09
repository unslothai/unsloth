// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

import {
  PROJECT_HOOK_EVENTS,
  type ProjectHookEvent,
  type ProjectHookGroup,
  type ProjectHookHandler,
  type ProjectHooks,
  ProjectHooksRequestGuard,
  type ProjectHooksStoredTrust,
  type ProjectHooksTrustIdentity,
  type ProjectHooksTrustState,
  getProjectHooks,
  getProjectHooksTrustState,
  projectHooksTrustIdentityMatches,
  revokeProjectHooks,
  setProjectHookHandlerEnabled,
  trustProjectHooks,
  visibleProjectHookText,
} from "../api/project-hooks-api";
import type { ProjectRecord as StoredProjectRecord } from "../types";

type ProjectRecord = StoredProjectRecord & { workspaceAvailable?: boolean };
import { projectHookHandlerPresentation } from "./project-hook-handler-presentation";

type MutationKey = "trust" | "revoke" | `handler:${string}`;

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Project hooks request failed.";
}

function VisibleProjectHookText({
  value,
  fallback,
  className,
}: {
  value: string | null | undefined;
  fallback?: string;
  className: string;
}) {
  const visible = visibleProjectHookText(value ?? null);
  return (
    <pre
      dir="ltr"
      className={`whitespace-pre-wrap break-words ${className}`}
      style={{ unicodeBidi: "isolate" }}
    >
      {visible || fallback || ""}
    </pre>
  );
}

function storedTrustFor(
  hooks: ProjectHooks | null,
  trustState: ProjectHooksTrustState | null,
): ProjectHooksStoredTrust | null {
  if (hooks && Object.prototype.hasOwnProperty.call(hooks, "storedTrust")) {
    return hooks.storedTrust ?? null;
  }
  return trustState?.storedTrust ?? null;
}

function trustStateFromHooks(hooks: ProjectHooks): ProjectHooksTrustState {
  return {
    storedTrust: hooks.storedTrust ?? null,
    revision: hooks.revision,
  };
}

function validWorkspaceRevision(
  workspaceRevision: number | null | undefined,
): workspaceRevision is number {
  return (
    workspaceRevision !== null &&
    workspaceRevision !== undefined &&
    Number.isSafeInteger(workspaceRevision) &&
    workspaceRevision >= 0
  );
}

function trustIdentityFor(
  projectId: string,
  hooks: ProjectHooks | null,
): ProjectHooksTrustIdentity | null {
  const workspaceRevision = hooks?.workspaceRevision;
  if (
    !(hooks?.exists && hooks.contentHash) ||
    hooks.trusted ||
    !validWorkspaceRevision(workspaceRevision)
  ) {
    return null;
  }
  return {
    projectId,
    workspaceRevision,
    contentHash: hooks.contentHash,
    revision: hooks.revision,
  };
}

function currentWorkspaceAvailable(
  projectAvailable: boolean | undefined,
  hooks: ProjectHooks | null,
): boolean {
  return hooks === null
    ? projectAvailable !== false
    : hooks.workspaceAvailable !== false;
}

export function ProjectHooksPanel({ project }: { project: ProjectRecord }) {
  const [hooks, setHooks] = useState<ProjectHooks | null>(null);
  const [trustState, setTrustState] = useState<ProjectHooksTrustState | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mutation, setMutation] = useState<MutationKey | null>(null);
  const [trustConfirmation, setTrustConfirmation] =
    useState<ProjectHooksTrustIdentity | null>(null);
  const requestGuard = useRef(new ProjectHooksRequestGuard());

  const refresh = useCallback(async () => {
    const requestRevision = requestGuard.current.begin();
    setTrustConfirmation(null);
    setMutation(null);
    setHooks(null);
    setTrustState(null);
    setLoading(true);
    setError(null);
    const [hooksResult, trustStateResult] = await Promise.allSettled([
      getProjectHooks(project.id),
      getProjectHooksTrustState(project.id),
    ]);
    if (!requestGuard.current.accepts(requestRevision)) {
      return;
    }
    if (hooksResult.status === "fulfilled") {
      if (
        trustStateResult.status === "fulfilled" &&
        trustStateResult.value.revision !== hooksResult.value.revision
      ) {
        setHooks(null);
        setTrustState(trustStateResult.value);
        setError("Project hook trust changed during refresh. Refresh again.");
      } else {
        setHooks(hooksResult.value);
        setTrustState(trustStateFromHooks(hooksResult.value));
      }
    } else {
      setHooks(null);
      setTrustState(
        trustStateResult.status === "fulfilled" ? trustStateResult.value : null,
      );
      setError(errorMessage(hooksResult.reason));
    }
    setLoading(false);
  }, [project.id]);

  useEffect(() => {
    const guard = requestGuard.current;
    guard.activate();
    const initialRefresh = window.setTimeout(() => {
      refresh().catch(() => undefined);
    }, 0);
    return () => {
      window.clearTimeout(initialRefresh);
      guard.retire();
    };
  }, [refresh]);

  async function applyMutation(
    key: MutationKey,
    request: () => Promise<ProjectHooks>,
  ): Promise<void> {
    if (mutation !== null || loading) {
      return;
    }
    const requestRevision = requestGuard.current.begin();
    setTrustConfirmation(null);
    setHooks(null);
    setTrustState(null);
    setMutation(key);
    setLoading(true);
    setError(null);
    try {
      const next = await request();
      if (requestGuard.current.accepts(requestRevision)) {
        setHooks(next);
        setTrustState(trustStateFromHooks(next));
      }
    } catch (nextError) {
      let fallbackTrustState: ProjectHooksTrustState | null = null;
      try {
        fallbackTrustState = await getProjectHooksTrustState(project.id);
      } catch {
        // The mutation error remains the primary actionable failure.
      }
      if (requestGuard.current.accepts(requestRevision)) {
        setTrustState(fallbackTrustState);
        setError(errorMessage(nextError));
      }
    } finally {
      if (requestGuard.current.accepts(requestRevision)) {
        setMutation(null);
        setLoading(false);
      }
    }
  }

  function requestTrust(): void {
    if (loading || mutation !== null) {
      return;
    }
    const identity = trustIdentityFor(project.id, hooks);
    if (identity === null) {
      setError(
        "Project workspace identity is unavailable. Refresh the project.",
      );
      return;
    }
    setTrustConfirmation(identity);
  }

  async function trustCurrentFile(): Promise<void> {
    const identity = trustConfirmation;
    setTrustConfirmation(null);
    if (identity === null) {
      return;
    }
    if (
      loading ||
      mutation !== null ||
      !projectHooksTrustIdentityMatches(identity, {
        projectId: project.id,
        hooks,
      })
    ) {
      setHooks(null);
      setError("Project hooks changed. Refresh and review the current file.");
      return;
    }
    await applyMutation("trust", () =>
      trustProjectHooks(identity.projectId, {
        workspaceRevision: identity.workspaceRevision,
        contentHash: identity.contentHash,
        revision: identity.revision,
      }),
    );
  }

  async function revokeTrust(): Promise<void> {
    const storedTrust = storedTrustFor(hooks, trustState);
    const revision = hooks?.trusted ? hooks.revision : storedTrust?.revision;
    if (revision === undefined) {
      return;
    }
    await applyMutation("revoke", () =>
      revokeProjectHooks(project.id, { revision }),
    );
  }

  async function setHandlerEnabled(
    handler: ProjectHookHandler,
    enabled: boolean,
  ): Promise<void> {
    if (!(hooks?.trusted && hooks.contentHash)) {
      return;
    }
    const workspaceRevision = hooks.workspaceRevision;
    if (!validWorkspaceRevision(workspaceRevision)) {
      setHooks(null);
      setError(
        "Project workspace identity is unavailable. Refresh the project.",
      );
      return;
    }
    const contentHash = hooks.contentHash;
    const revision = hooks.revision;
    const projectId = project.id;
    await applyMutation(`handler:${handler.id}`, () =>
      setProjectHookHandlerEnabled(projectId, handler.id, {
        workspaceRevision,
        contentHash,
        revision,
        enabled,
      }),
    );
  }

  const busy = mutation !== null;
  const workspaceAvailable = currentWorkspaceAvailable(
    project.workspaceAvailable,
    hooks,
  );
  const storedTrust = storedTrustFor(hooks, trustState);

  return (
    <>
      <div className="mt-3 border-t border-border/70 pt-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold text-foreground">
              Project hooks
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Review configured hook commands and manage exact-file trust.
              Synchronous PreToolUse and PostToolUse hooks run around project
              edits and commands. Other lifecycle and background hooks are inactive.
            </p>
          </div>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={loading || busy}
            onClick={refresh}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
        </div>

        {hooks?.execution && !hooks.execution.available ? (
          <p className="mt-2 text-xs text-muted-foreground" role="status">
            {hooks.execution.reason ?? "Secure project hook execution is unavailable."}
          </p>
        ) : null}
        {error ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {error}
          </p>
        ) : null}

        {loading && hooks === null ? (
          <p className="mt-2 text-xs text-muted-foreground">
            Loading project hooks...
          </p>
        ) : null}

        {!loading && hooks === null ? (
          <div className="mt-2">
            <p className="text-xs text-muted-foreground">
              Project hooks are unavailable.
            </p>
            {storedTrust ? (
              <StoredTrustNotice
                storedTrust={storedTrust}
                message="The current hook file could not be inspected. Stored trust is inactive until the exact file and workspace can be verified."
                loading={loading}
                busy={busy}
                mutation={mutation}
                onRevokeTrust={revokeTrust}
              />
            ) : null}
          </div>
        ) : null}

        {!loading && hooks && !workspaceAvailable ? (
          <UnavailableWorkspaceHooks
            storedTrust={storedTrust}
            loading={loading}
            busy={busy}
            mutation={mutation}
            onRevokeTrust={revokeTrust}
          />
        ) : null}

        {!loading && workspaceAvailable && hooks && !hooks.exists ? (
          <div className="mt-3 rounded-xl bg-background/70 px-3 py-2">
            <p className="text-xs font-medium text-foreground">
              No project hook file found
            </p>
            <VisibleProjectHookText
              value={hooks.sourcePath}
              className="mt-1 text-[11px] text-muted-foreground"
            />
            {storedTrust ? (
              <StoredTrustNotice
                storedTrust={storedTrust}
                message="Stored trust is inactive while the project hook file is absent."
                loading={loading}
                busy={busy}
                mutation={mutation}
                onRevokeTrust={revokeTrust}
              />
            ) : null}
          </div>
        ) : null}

        {workspaceAvailable && hooks?.exists ? (
          <HooksBrowser
            hooks={hooks}
            storedTrust={storedTrust}
            loading={loading}
            busy={busy}
            mutation={mutation}
            onRequestTrust={requestTrust}
            onRevokeTrust={revokeTrust}
            onSetHandlerEnabled={setHandlerEnabled}
          />
        ) : null}
      </div>

      <AlertDialog
        open={trustConfirmation !== null}
        onOpenChange={(open) => {
          if (!open) {
            setTrustConfirmation(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Trust project hooks?</AlertDialogTitle>
            <AlertDialogDescription>
              Trust approves enabled synchronous PreToolUse and PostToolUse commands
              for this exact file and workspace. They run around project file edits,
              Python, and terminal tools, and pre-tool failures stop the tool.
              Session, delegation, and background hooks stay inactive in this version.
              Hooks cannot grant permissions or rewrite tool arguments. Trusting
              resets handler preferences and enables all configured handlers.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {trustConfirmation ? (
            <div className="space-y-1 rounded bg-muted px-2 py-1">
              <p className="text-[10px] text-muted-foreground">
                Workspace revision {trustConfirmation.workspaceRevision}
              </p>
              <VisibleProjectHookText
                value={trustConfirmation.contentHash}
                className="font-mono text-[11px] text-foreground"
              />
            </div>
          ) : null}
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={loading || busy || trustConfirmation === null}
              onClick={() => {
                trustCurrentFile().catch(() => undefined);
              }}
            >
              Trust current file
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

function StoredTrustNotice({
  storedTrust,
  message,
  loading,
  busy,
  mutation,
  onRevokeTrust,
}: {
  storedTrust: ProjectHooksStoredTrust;
  message: string;
  loading: boolean;
  busy: boolean;
  mutation: MutationKey | null;
  onRevokeTrust: () => Promise<void>;
}) {
  return (
    <div className="mt-2 rounded-lg border border-amber-500/30 px-3 py-2">
      <p className="text-xs text-amber-600 dark:text-amber-400">{message}</p>
      <p className="mt-1 text-[10px] text-muted-foreground">Stored SHA-256</p>
      <VisibleProjectHookText
        value={storedTrust.contentHash}
        className="font-mono text-[11px] text-foreground"
      />
      <Button
        className="mt-2"
        type="button"
        size="sm"
        variant="outline"
        disabled={loading || busy}
        onClick={() => {
          onRevokeTrust().catch(() => undefined);
        }}
      >
        {mutation === "revoke" ? "Revoking..." : "Revoke trust"}
      </Button>
    </div>
  );
}

function UnavailableWorkspaceHooks({
  storedTrust,
  loading,
  busy,
  mutation,
  onRevokeTrust,
}: {
  storedTrust: ProjectHooksStoredTrust | null;
  loading: boolean;
  busy: boolean;
  mutation: MutationKey | null;
  onRevokeTrust: () => Promise<void>;
}) {
  return (
    <div className="mt-3 rounded-xl bg-background/70 px-3 py-3">
      <p className="text-xs text-muted-foreground">
        Reconnect the project folder to inspect the current hook file.
      </p>
      {storedTrust ? (
        <StoredTrustNotice
          storedTrust={storedTrust}
          message="Stored trust is inactive while the workspace is unavailable."
          loading={loading}
          busy={busy}
          mutation={mutation}
          onRevokeTrust={onRevokeTrust}
        />
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">
          No stored hook trust exists for this project.
        </p>
      )}
    </div>
  );
}

function HooksBrowser({
  hooks,
  storedTrust,
  loading,
  busy,
  mutation,
  onRequestTrust,
  onRevokeTrust,
  onSetHandlerEnabled,
}: {
  hooks: ProjectHooks;
  storedTrust: ProjectHooksStoredTrust | null;
  loading: boolean;
  busy: boolean;
  mutation: MutationKey | null;
  onRequestTrust: () => void;
  onRevokeTrust: () => Promise<void>;
  onSetHandlerEnabled: (
    handler: ProjectHookHandler,
    enabled: boolean,
  ) => Promise<void>;
}) {
  return (
    <div className="mt-3 rounded-xl bg-background/70 px-3 py-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <VisibleProjectHookText
            value={hooks.sourcePath}
            className="font-mono text-[11px] text-foreground"
          />
          <p className="mt-1 text-[10px] text-muted-foreground">SHA-256</p>
          <VisibleProjectHookText
            value={hooks.contentHash}
            fallback="none"
            className="font-mono text-[11px] text-foreground"
          />
          <p
            className={
              hooks.trusted
                ? "mt-1 text-xs text-emerald-600 dark:text-emerald-400"
                : "mt-1 text-xs text-amber-600 dark:text-amber-400"
            }
          >
            {hooks.trusted
              ? "Trusted for this exact content hash."
              : "This file is untrusted. No configured hook can run until this exact file and workspace are trusted."}
          </p>
        </div>
        {hooks.trusted ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={loading || busy}
            onClick={() => {
              onRevokeTrust().catch(() => undefined);
            }}
          >
            {mutation === "revoke" ? "Revoking..." : "Revoke trust"}
          </Button>
        ) : (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={loading || busy || !hooks.contentHash}
            onClick={onRequestTrust}
          >
            {mutation === "trust" ? "Trusting..." : "Trust current file"}
          </Button>
        )}
      </div>

      {!hooks.trusted && storedTrust ? (
        <StoredTrustNotice
          storedTrust={storedTrust}
          message="Stored trust does not match the current hook file and workspace, so it is inactive."
          loading={loading}
          busy={busy}
          mutation={mutation}
          onRevokeTrust={onRevokeTrust}
        />
      ) : null}

      {hooks.description ? (
        <VisibleProjectHookText
          value={hooks.description}
          className="mt-2 font-sans text-xs text-muted-foreground"
        />
      ) : null}

      <p className="mt-2 text-[10px] text-muted-foreground">
        {hooks.groupCount} matcher group
        {hooks.groupCount === 1 ? "" : "s"}, {hooks.handlerCount} handler
        {hooks.handlerCount === 1 ? "" : "s"}
      </p>

      <div className="mt-3 space-y-2">
        {PROJECT_HOOK_EVENTS.map((event) => (
          <EventHooks
            key={event}
            event={event}
            groups={hooks.hooks[event] ?? []}
            trusted={hooks.trusted}
            loading={loading}
            busy={busy}
            mutation={mutation}
            onSetHandlerEnabled={onSetHandlerEnabled}
          />
        ))}
      </div>
    </div>
  );
}

function EventHooks({
  event,
  groups,
  trusted,
  loading,
  busy,
  mutation,
  onSetHandlerEnabled,
}: {
  event: ProjectHookEvent;
  groups: ProjectHookGroup[];
  trusted: boolean;
  loading: boolean;
  busy: boolean;
  mutation: MutationKey | null;
  onSetHandlerEnabled: (
    handler: ProjectHookHandler,
    enabled: boolean,
  ) => Promise<void>;
}) {
  const handlerCount = groups.reduce(
    (count, group) => count + group.hooks.length,
    0,
  );
  return (
    <details className="rounded-lg border border-border/70 px-3 py-2">
      <summary className="cursor-pointer text-xs font-medium text-foreground">
        {event} ({handlerCount})
      </summary>
      {groups.length > 0 ? (
        <div className="mt-2 space-y-3">
          {groups.map((group) => (
            <div
              key={group.hooks.map((handler) => handler.id).join(":")}
              className="space-y-2"
            >
              <p className="text-[10px] text-muted-foreground">Matcher:</p>
              <VisibleProjectHookText
                value={group.matcher}
                fallback="all"
                className="font-mono text-[10px] text-muted-foreground"
              />
              {group.hooks.map((handler) => (
                <HandlerRow
                  key={handler.id}
                  handler={handler}
                  trusted={trusted}
                  disabled={loading || busy}
                  updating={mutation === `handler:${handler.id}`}
                  onSetEnabled={onSetHandlerEnabled}
                />
              ))}
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-[10px] text-muted-foreground">
          No handlers configured.
        </p>
      )}
    </details>
  );
}

function HandlerRow({
  handler,
  trusted,
  disabled,
  updating,
  onSetEnabled,
}: {
  handler: ProjectHookHandler;
  trusted: boolean;
  disabled: boolean;
  updating: boolean;
  onSetEnabled: (
    handler: ProjectHookHandler,
    enabled: boolean,
  ) => Promise<void>;
}) {
  const presentation = projectHookHandlerPresentation({
    trusted,
    enabled: handler.enabled,
    active: handler.active,
    background: handler.async,
  });
  return (
    <div className="rounded-lg bg-muted/40 px-3 py-2">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <VisibleProjectHookText
            value={handler.command}
            className="font-mono text-[11px] text-foreground"
          />
          {handler.commandWindows ? (
            <div className="mt-1">
              <p className="text-[10px] text-muted-foreground">Windows:</p>
              <VisibleProjectHookText
                value={handler.commandWindows}
                className="font-mono text-[10px] text-muted-foreground"
              />
            </div>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-2 text-[10px] text-muted-foreground">
          <Checkbox
            aria-label={`Enable preference for ${visibleProjectHookText(handler.id)}`}
            checked={handler.enabled}
            disabled={!trusted || disabled}
            onCheckedChange={(checked) => {
              onSetEnabled(handler, checked === true).catch(() => undefined);
            }}
          />
          {updating ? "Saving..." : presentation.state}
        </div>
      </div>
      <p className="mt-1 text-[10px] text-muted-foreground">
        Timeout {handler.timeout}s, {presentation.execution}, context limit{" "}
        {handler.additionalContextLimit}
      </p>
      {handler.statusMessage ? (
        <VisibleProjectHookText
          value={handler.statusMessage}
          className="mt-1 font-sans text-[10px] text-muted-foreground"
        />
      ) : null}
      <VisibleProjectHookText
        value={handler.id}
        className="mt-1 font-mono text-[9px] text-muted-foreground/80"
      />
    </div>
  );
}
