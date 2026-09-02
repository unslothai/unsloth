// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import {
  Check,
  Copy,
  MoreHorizontal,
  RefreshCw,
  ShieldCheck,
  Trash2,
  Users,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  type CreatedManagedAccount,
  type CurrentAccount,
  type ManagedAccount,
  createAccount,
  deleteAccount,
  fetchAccounts,
  fetchCurrentAccount,
  regenerateSetupCode,
} from "../api/accounts";

type AccountAction = {
  type: "regenerate" | "delete";
  username: string;
};

function SetupCodeReveal({
  account,
  onDone,
  onError,
}: {
  account: CreatedManagedAccount;
  onDone: () => void;
  onError: (message: string) => void;
}) {
  const [copied, setCopied] = useState(false);

  async function copyCode() {
    if (!(await copyToClipboard(account.setup_code))) {
      onError("Could not copy the setup code. Select and copy it manually.");
      return;
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border bg-card p-3" data-testid="setup-code-card">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-500">
          <Check className="size-3.5" strokeWidth={2.25} />
        </span>
        <div className="min-w-0">
          <h3 className="text-sm font-medium text-foreground">Account created</h3>
          <p className="text-xs text-muted-foreground">
            Send this setup code to <span className="font-medium text-foreground">{account.username}</span>.
          </p>
        </div>
      </div>

      <button
        type="button"
        onClick={() => void copyCode()}
        className={cn(
          "flex w-full items-center justify-between gap-3 rounded-md border bg-muted/35 px-3 py-2.5 transition-colors hover:bg-muted/55",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          copied && "border-emerald-500/30 bg-emerald-500/5",
        )}
        aria-label={copied ? "Setup code copied" : "Copy setup code"}
      >
        <code
          className="min-w-0 flex-1 select-all text-left font-mono text-sm font-semibold tracking-wide text-foreground"
          data-testid="setup-code"
          data-reload-snapshot-sensitive
        >
          {account.setup_code}
        </code>
        <span className="flex shrink-0 items-center gap-1.5 text-xs font-medium text-muted-foreground">
          {copied ? "Copied" : "Copy"}
          {copied ? <Check className="size-3.5 text-emerald-600" /> : <Copy className="size-3.5" />}
        </span>
      </button>

      <div className="flex items-center justify-between gap-3">
        <p className="text-ui-11 text-muted-foreground">
          Shown once · Expires in 60 minutes · Used as the first password
        </p>
        <Button type="button" size="sm" variant="outline" onClick={onDone}>
          Done
        </Button>
      </div>
    </div>
  );
}

function accountStatus(account: ManagedAccount) {
  if (account.is_admin) {
    return { label: "Private workspace ready", dot: "bg-emerald-500" };
  }
  if (account.setup_code_expired) {
    return { label: "Setup code expired", dot: "bg-amber-500" };
  }
  if (account.must_change_password) {
    return { label: "Awaiting first sign-in", dot: "bg-amber-500" };
  }
  return { label: "Private workspace ready", dot: "bg-emerald-500" };
}

function AccountRow({
  account,
  busy,
  onAction,
}: {
  account: ManagedAccount;
  busy: boolean;
  onAction: (action: AccountAction) => void;
}) {
  const status = accountStatus(account);

  return (
    <div
      className="group flex min-h-14 items-center gap-3 border-b border-border/60 px-3 py-2.5 last:border-b-0 transition-colors hover:bg-accent/30"
      data-testid={`managed-account-${account.username}`}
    >
      <span className={cn("size-1.5 shrink-0 rounded-full", status.dot)} aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium text-foreground">{account.username}</span>
          {account.is_admin && (
            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-ui-10 font-medium text-muted-foreground">
              <ShieldCheck className="size-3" /> Owner
            </span>
          )}
        </div>
        <p className="text-ui-11 text-muted-foreground">{status.label}</p>
      </div>

      {!account.is_admin && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="size-7 p-0 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 max-sm:!opacity-100 max-sm:size-9"
              aria-label={`Account actions for ${account.username}`}
              disabled={busy}
            >
              <MoreHorizontal className="size-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {account.must_change_password && (
              <DropdownMenuItem
                onSelect={() => onAction({ type: "regenerate", username: account.username })}
              >
                <RefreshCw className="mr-2 size-3.5" /> Generate new setup code
              </DropdownMenuItem>
            )}
            <DropdownMenuItem
              className="text-destructive focus:text-destructive"
              onSelect={() => onAction({ type: "delete", username: account.username })}
            >
              <Trash2 className="mr-2 size-3.5" /> Delete account
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </div>
  );
}

export function AccountsTab() {
  const [current, setCurrent] = useState<CurrentAccount | null>(null);
  const [accounts, setAccounts] = useState<ManagedAccount[]>([]);
  const [username, setUsername] = useState("");
  const [createdAccount, setCreatedAccount] = useState<CreatedManagedAccount | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [pendingAction, setPendingAction] = useState<AccountAction | null>(null);
  const [actionBusy, setActionBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const me = await fetchCurrentAccount();
      setCurrent(me);
      setAccounts(me.is_admin ? await fetchAccounts() : []);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to load accounts");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadInitial() {
      try {
        const me = await fetchCurrentAccount();
        const initialAccounts = me.is_admin ? await fetchAccounts() : [];
        if (cancelled) return;
        setCurrent(me);
        setAccounts(initialAccounts);
        setError(null);
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : "Failed to load accounts");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void loadInitial();
    return () => {
      cancelled = true;
    };
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const created = await createAccount(username);
      setCreatedAccount(created);
      setUsername("");
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to create account");
    } finally {
      setSaving(false);
    }
  }

  async function confirmAction() {
    if (!pendingAction) return;
    setActionBusy(true);
    setError(null);
    try {
      if (pendingAction.type === "regenerate") {
        const created = await regenerateSetupCode(pendingAction.username);
        setCreatedAccount(created);
      } else {
        await deleteAccount(pendingAction.username);
        if (createdAccount?.username === pendingAction.username) setCreatedAccount(null);
      }
      setPendingAction(null);
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to update account");
    } finally {
      setActionBusy(false);
    }
  }

  const actionIsDelete = pendingAction?.type === "delete";

  return (
    <div className="flex min-w-0 max-w-2xl flex-col gap-6">
      <header className="flex flex-col gap-1" data-settings-label="Accounts">
        <div className="flex items-center gap-2">
          <Users className="size-5" />
          <h1 className="text-xl font-semibold font-heading">Accounts</h1>
        </div>
        <p className="text-xs text-muted-foreground">
          Give each person a private workspace. Model downloads are shared to save disk space.
        </p>
      </header>

      {error && (
        <div role="alert" className="rounded-md border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive">
          {error}
        </div>
      )}

      {loading ? (
        <div className="h-24 animate-pulse rounded-lg bg-muted/40" />
      ) : current && !current.is_admin ? (
        <div className="rounded-lg border bg-card p-4 text-sm">
          Signed in as <span className="font-semibold">{current.username}</span>. Only the
          installation owner can manage accounts.
        </div>
      ) : (
        <>
          <section className="space-y-3">
            <div>
              <h2 className="text-sm font-semibold text-foreground">Add account</h2>
              <p className="text-xs text-muted-foreground">
                They’ll sign in with a setup code, then choose their own password.
              </p>
            </div>

            {createdAccount ? (
              <SetupCodeReveal
                account={createdAccount}
                onDone={() => setCreatedAccount(null)}
                onError={setError}
              />
            ) : (
              <form onSubmit={submit} className="flex flex-wrap items-center gap-2">
                <Input
                  aria-label="Username"
                  value={username}
                  onChange={(event) => setUsername(event.target.value.toLowerCase())}
                  minLength={3}
                  maxLength={64}
                  pattern="[a-z0-9][a-z0-9._-]*"
                  placeholder="Username, e.g. alice"
                  autoComplete="off"
                  className="h-9 min-w-[220px] flex-1 text-sm"
                  required
                />
                <Button type="submit" size="sm" disabled={saving || !username.trim()}>
                  {saving ? "Creating…" : "Create account"}
                </Button>
              </form>
            )}
          </section>

          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-foreground">People with access</h2>
              <span className="text-ui-11 tabular-nums text-muted-foreground">{accounts.length}</span>
            </div>
            <div className="overflow-hidden rounded-lg border bg-card">
              {accounts.map((account) => (
                <AccountRow
                  key={account.username}
                  account={account}
                  busy={actionBusy}
                  onAction={setPendingAction}
                />
              ))}
            </div>
          </section>
        </>
      )}

      <AlertDialog
        open={pendingAction !== null}
        onOpenChange={(open) => !open && !actionBusy && setPendingAction(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {actionIsDelete ? `Delete ${pendingAction?.username}?` : "Generate a new setup code?"}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {actionIsDelete
                ? "Their login and active sessions will be removed. Workspace files are kept on disk under a dated folder, but adding the same username again starts an empty workspace."
                : `The previous code and any first-login session for ${pendingAction?.username ?? "this account"} will stop working.`}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={actionBusy}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={actionBusy}
              onClick={(event) => {
                event.preventDefault();
                void confirmAction();
              }}
              className={cn(actionIsDelete && "bg-destructive text-destructive-foreground hover:bg-destructive/90")}
            >
              {actionBusy
                ? actionIsDelete
                  ? "Deleting…"
                  : "Generating…"
                : actionIsDelete
                  ? "Delete account"
                  : "Generate code"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
