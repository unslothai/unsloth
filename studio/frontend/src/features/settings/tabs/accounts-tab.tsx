// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  Check,
  Copy,
  RefreshCw,
  ShieldCheck,
  Trash2,
  UserPlus,
  Users,
  X,
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

export function AccountsTab() {
  const [current, setCurrent] = useState<CurrentAccount | null>(null);
  const [accounts, setAccounts] = useState<ManagedAccount[]>([]);
  const [username, setUsername] = useState("");
  const [createdAccount, setCreatedAccount] =
    useState<CreatedManagedAccount | null>(null);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [regenerating, setRegenerating] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const me = await fetchCurrentAccount();
      setCurrent(me);
      setAccounts(me.is_admin ? await fetchAccounts() : []);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Failed to load accounts",
      );
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
          setError(
            cause instanceof Error ? cause.message : "Failed to load accounts",
          );
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
      setCopied(false);
      setUsername("");
      await load();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Failed to create account",
      );
    } finally {
      setSaving(false);
    }
  }

  async function copySetupCode() {
    if (!createdAccount) return;
    try {
      if (!(await copyToClipboard(createdAccount.setup_code))) {
        throw new Error("Clipboard write failed");
      }
      setCopied(true);
    } catch {
      setError("Could not copy the setup code. Select and copy it manually.");
    }
  }

  async function regenerate(usernameToRegenerate: string) {
    if (
      !window.confirm(
        `Generate a new setup code for ${usernameToRegenerate}? The previous code and any first-login session will stop working.`,
      )
    ) {
      return;
    }
    setRegenerating(usernameToRegenerate);
    setError(null);
    try {
      const created = await regenerateSetupCode(usernameToRegenerate);
      setCreatedAccount(created);
      setCopied(false);
      await load();
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Failed to regenerate setup code",
      );
    } finally {
      setRegenerating(null);
    }
  }

  async function remove(usernameToDelete: string) {
    if (
      !window.confirm(
        `Delete ${usernameToDelete}'s login? Workspace files will be retained. ` +
          `Creating ${usernameToDelete} again will restore access to those files.`,
      )
    ) {
      return;
    }
    setDeleting(usernameToDelete);
    setError(null);
    try {
      await deleteAccount(usernameToDelete);
      await load();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Failed to delete account",
      );
    } finally {
      setDeleting(null);
    }
  }

  return (
    <div className="flex min-w-0 max-w-2xl flex-col gap-6">
      <header className="flex flex-col gap-1" data-settings-label="Accounts">
        <div className="flex items-center gap-2">
          <Users className="size-5" />
          <h1 className="text-xl font-semibold font-heading">Accounts</h1>
        </div>
        <p className="text-xs text-muted-foreground">
          Each account gets private chats, settings, credentials, uploads,
          projects, and outputs. Model downloads remain shared to save disk
          space.
        </p>
      </header>

      {error && (
        <div
          role="alert"
          className="rounded-lg border border-destructive/25 bg-destructive/5 p-3 text-sm text-destructive"
        >
          {error}
        </div>
      )}

      {loading ? (
        <div className="h-28 animate-pulse rounded-xl bg-muted/40" />
      ) : current && !current.is_admin ? (
        <div className="rounded-xl border bg-card p-4 text-sm">
          Signed in as <span className="font-semibold">{current.username}</span>
          . Only the installation owner can create or remove accounts.
        </div>
      ) : (
        <>
          <form
            onSubmit={submit}
            className="space-y-4 rounded-xl border bg-card p-4"
          >
            <div className="flex items-center gap-2">
              <UserPlus className="size-4" />
              <h2 className="font-semibold">Add account</h2>
            </div>
            <p className="text-xs text-muted-foreground">
              We generate a one-time setup code. The user enters it as their
              password, then chooses a permanent password.
            </p>
            <div className="max-w-sm space-y-2">
              <Label htmlFor="account-username">Username</Label>
              <Input
                id="account-username"
                value={username}
                onChange={(event) =>
                  setUsername(event.target.value.toLowerCase())
                }
                minLength={3}
                maxLength={64}
                pattern="[a-z0-9][a-z0-9._-]*"
                placeholder="e.g. alice"
                autoComplete="off"
                required={true}
              />
            </div>
            <Button type="submit" disabled={saving}>
              {saving ? "Creating…" : "Create account"}
            </Button>
          </form>

          {createdAccount && (
            <section
              className="space-y-3 rounded-xl border border-primary/30 bg-primary/5 p-4"
              data-testid="setup-code-card"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold">
                    Setup code for {createdAccount.username}
                  </h2>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Share this securely. It is shown only here and expires in 60
                    minutes.
                  </p>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label="Dismiss setup code"
                  onClick={() => setCreatedAccount(null)}
                >
                  <X className="size-4" />
                </Button>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <code
                  className="select-all rounded-md border bg-background px-3 py-2 font-mono text-base font-semibold tracking-wide"
                  data-testid="setup-code"
                >
                  {createdAccount.setup_code}
                </code>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => void copySetupCode()}
                >
                  {copied ? (
                    <Check className="mr-1.5 size-4" />
                  ) : (
                    <Copy className="mr-1.5 size-4" />
                  )}
                  {copied ? "Copied" : "Copy"}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                {createdAccount.username} signs in normally with this code in
                the Password field, then they will be asked to choose a new
                password.
              </p>
            </section>
          )}

          <section className="space-y-2">
            <h2 className="text-sm font-semibold">People with access</h2>
            <div className="overflow-hidden rounded-xl border bg-card">
              {accounts.map((account) => (
                <div
                  key={account.username}
                  className="flex min-h-14 items-center justify-between gap-3 border-b px-4 py-3 last:border-b-0"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium">
                        {account.username}
                      </span>
                      {account.is_admin && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                          <ShieldCheck className="size-3" /> Owner
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {account.setup_code_expired
                        ? "Setup code expired"
                        : account.must_change_password
                          ? "Awaiting first sign-in"
                          : "Private workspace ready"}
                    </p>
                  </div>
                  {!account.is_admin && (
                    <div className="flex items-center gap-1">
                      {account.must_change_password && (
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          aria-label={`Regenerate setup code for ${account.username}`}
                          aria-busy={regenerating === account.username}
                          disabled={deleting !== null || regenerating !== null}
                          onClick={() => void regenerate(account.username)}
                        >
                          <RefreshCw
                            className={`size-4 text-muted-foreground ${
                              regenerating === account.username
                                ? "animate-spin"
                                : ""
                            }`}
                          />
                        </Button>
                      )}
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        aria-label={
                          deleting === account.username
                            ? `Deleting ${account.username}`
                            : `Delete ${account.username}`
                        }
                        aria-busy={deleting === account.username}
                        disabled={deleting !== null || regenerating !== null}
                        onClick={() => void remove(account.username)}
                      >
                        {deleting === account.username ? (
                          <span className="text-xs text-muted-foreground">
                            …
                          </span>
                        ) : (
                          <Trash2 className="size-4 text-muted-foreground" />
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
