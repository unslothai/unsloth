// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ShieldCheck, Trash2, UserPlus, Users } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  createAccount,
  deleteAccount,
  fetchAccounts,
  fetchCurrentAccount,
  type CurrentAccount,
  type ManagedAccount,
} from "../api/accounts";

export function AccountsTab() {
  const [current, setCurrent] = useState<CurrentAccount | null>(null);
  const [accounts, setAccounts] = useState<ManagedAccount[]>([]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
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
      await createAccount(username, password);
      setUsername("");
      setPassword("");
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to create account");
    } finally {
      setSaving(false);
    }
  }

  async function remove(usernameToDelete: string) {
    if (!window.confirm(`Delete ${usernameToDelete}'s login? Workspace files will be retained.`)) {
      return;
    }
    setError(null);
    try {
      await deleteAccount(usernameToDelete);
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to delete account");
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
          Each account gets private chats, settings, credentials, uploads, projects, and outputs.
          Model downloads remain shared to save disk space.
        </p>
      </header>

      {error && (
        <div role="alert" className="rounded-lg border border-destructive/25 bg-destructive/5 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading ? (
        <div className="h-28 animate-pulse rounded-xl bg-muted/40" />
      ) : current && !current.is_admin ? (
        <div className="rounded-xl border bg-card p-4 text-sm">
          Signed in as <span className="font-semibold">{current.username}</span>. Only the
          installation owner can create or remove accounts.
        </div>
      ) : (
        <>
          <form onSubmit={submit} className="space-y-4 rounded-xl border bg-card p-4">
            <div className="flex items-center gap-2">
              <UserPlus className="size-4" />
              <h2 className="font-semibold">Add account</h2>
            </div>
            <p className="text-xs text-muted-foreground">
              The user must replace this temporary password when they first sign in.
            </p>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="account-username">Username</Label>
                <Input
                  id="account-username"
                  value={username}
                  onChange={(event) => setUsername(event.target.value.toLowerCase())}
                  minLength={3}
                  maxLength={64}
                  pattern="[a-z0-9][a-z0-9._-]*"
                  placeholder="e.g. alice"
                  autoComplete="off"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="account-password">Temporary password</Label>
                <Input
                  id="account-password"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  minLength={8}
                  pattern="[^\\s]+"
                  autoComplete="new-password"
                  required
                />
              </div>
            </div>
            <Button type="submit" disabled={saving}>
              {saving ? "Creating…" : "Create account"}
            </Button>
          </form>

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
                      <span className="truncate text-sm font-medium">{account.username}</span>
                      {account.is_admin && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                          <ShieldCheck className="size-3" /> Owner
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {account.must_change_password
                        ? "Password change required"
                        : "Private workspace ready"}
                    </p>
                  </div>
                  {!account.is_admin && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      aria-label={`Delete ${account.username}`}
                      onClick={() => void remove(account.username)}
                    >
                      <Trash2 className="size-4 text-muted-foreground" />
                    </Button>
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
