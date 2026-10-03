// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState, type SyntheticEvent } from "react";
import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useIsAccountOwner } from "@/features/auth";
import {
  getInferenceStatus,
  listCachedModels,
  unloadModel,
  type CachedModelRepo,
} from "@/features/chat/api/chat-api";
import { deleteCachedModel } from "@/features/hub/inventory/api";
import {
  createAccount,
  deleteAccount,
  fetchAccounts,
  regenerateSetupCode,
  setAccountActive,
  type AccountSetupCode,
  type StudioAccount,
} from "@/features/settings/api/accounts";
import { fetchBlockedModels, saveBlockedModels } from "./api";

function errorText(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Something went wrong";
}

function formatSize(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

function Section({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border p-4">
      <h2 className="text-base font-semibold">{title}</h2>
      <p className="mb-3 text-sm text-muted-foreground">{description}</p>
      {children}
    </section>
  );
}

function UsersSection({ onError }: { onError: (message: string | null) => void }) {
  const [accounts, setAccounts] = useState<StudioAccount[] | null>(null);
  const [username, setUsername] = useState("");
  const [setup, setSetup] = useState<AccountSetupCode | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setAccounts(await fetchAccounts());
    } catch (reason) {
      onError(errorText(reason));
    }
  }, [onError]);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function run(action: () => Promise<void>) {
    onError(null);
    setBusy(true);
    try {
      await action();
      await refresh();
    } catch (reason) {
      onError(errorText(reason));
    } finally {
      setBusy(false);
    }
  }

  function create(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!username.trim()) return;
    void run(async () => {
      setSetup(await createAccount(username));
      setUsername("");
    });
  }

  return (
    <Section
      title="Users"
      description="Create accounts, disable or remove them, and reset passwords. A reset signs the user out everywhere and issues a one-time code (valid 60 minutes) they use to sign in and choose a new password. You never see their password."
    >
      <form onSubmit={create} className="mb-3 flex gap-2">
        <Input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="New username"
          aria-label="New username"
        />
        <Button type="submit" disabled={busy || !username.trim()}>
          Create account
        </Button>
      </form>
      {setup && (
        <div className="mb-3 rounded-md border border-border bg-muted p-3 text-sm">
          Setup code for <b>{setup.username}</b>:{" "}
          <code className="select-all font-mono">{setup.setup_code}</code>
          <div className="text-muted-foreground">
            Shown only once. Share it with the account holder; it expires {new Date(setup.expires_at).toLocaleString()}.
          </div>
          <Button variant="ghost" size="sm" onClick={() => setSetup(null)}>
            Done
          </Button>
        </div>
      )}
      {accounts === null ? (
        <Spinner />
      ) : (
        <ul className="divide-y divide-border">
          {accounts.map((account) => {
            const isOwner = account.role === "owner";
            return (
              <li key={account.account_id} className="flex items-center gap-3 py-2">
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium">{account.username}</div>
                  <div className="text-xs text-muted-foreground">
                    {isOwner ? "Installation owner" : "Private account"} · created{" "}
                    {new Date(account.created_at).toLocaleDateString()}
                  </div>
                </div>
                {!isOwner && (
                  <>
                    <Switch
                      checked={account.is_active}
                      disabled={busy}
                      aria-label={`${account.is_active ? "Disable" : "Enable"} ${account.username}`}
                      onCheckedChange={(next) =>
                        void run(() => setAccountActive(account.account_id, next))
                      }
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={busy}
                      onClick={() =>
                        void run(async () => setSetup(await regenerateSetupCode(account.account_id)))
                      }
                    >
                      Reset password
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      disabled={busy}
                      onClick={() => {
                        if (window.confirm(`Delete ${account.username}? Their files are retired, not erased.`)) {
                          void run(() => deleteAccount(account.account_id));
                        }
                      }}
                    >
                      Delete
                    </Button>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Section>
  );
}

function ModelsSection({ onError }: { onError: (message: string | null) => void }) {
  const [loaded, setLoaded] = useState<string[] | null>(null);
  const [cached, setCached] = useState<CachedModelRepo[] | null>(null);
  const [blocked, setBlocked] = useState<string[] | null>(null);
  const [newBlock, setNewBlock] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [status, cachedModels, blockedModels] = await Promise.all([
        getInferenceStatus(),
        listCachedModels(),
        fetchBlockedModels(),
      ]);
      setLoaded(status.loaded ?? (status.active_model ? [status.active_model] : []));
      setCached(cachedModels);
      setBlocked(blockedModels);
    } catch (reason) {
      onError(errorText(reason));
    }
  }, [onError]);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function run(action: () => Promise<unknown>) {
    onError(null);
    setBusy(true);
    try {
      await action();
      await refresh();
    } catch (reason) {
      onError(errorText(reason));
    } finally {
      setBusy(false);
    }
  }

  const isBlocked = (id: string) => !!blocked?.some((b) => b.toLowerCase() === id.toLowerCase());
  const setBlockedFor = (id: string, block: boolean) =>
    run(async () => {
      const rest = (blocked ?? []).filter((b) => b.toLowerCase() !== id.toLowerCase());
      setBlocked(await saveBlockedModels(block ? [...rest, id] : rest));
    });

  function addBlock(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    const id = newBlock.trim();
    if (!id) return;
    setNewBlock("");
    void setBlockedFor(id, true);
  }

  return (
    <Section
      title="Models"
      description="See what is loaded, unload or delete models, and block models from other accounts. Blocked models can still be used by the owner."
    >
      <h3 className="mb-1 text-sm font-medium">Loaded now</h3>
      {loaded === null ? (
        <Spinner />
      ) : loaded.length === 0 ? (
        <p className="mb-3 text-sm text-muted-foreground">No model is loaded.</p>
      ) : (
        <ul className="mb-3 divide-y divide-border">
          {loaded.map((id) => (
            <li key={id} className="flex items-center gap-3 py-2">
              <span className="min-w-0 flex-1 truncate font-mono text-sm">{id}</span>
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={() => void run(() => unloadModel({ model_path: id, force_cancel_active: true }))}
              >
                Unload
              </Button>
            </li>
          ))}
        </ul>
      )}

      <h3 className="mb-1 text-sm font-medium">Blocked for other accounts</h3>
      <form onSubmit={addBlock} className="mb-2 flex gap-2">
        <Input
          value={newBlock}
          onChange={(e) => setNewBlock(e.target.value)}
          placeholder="org/model-name"
          aria-label="Model to block"
        />
        <Button type="submit" disabled={busy || !newBlock.trim()}>
          Block
        </Button>
      </form>
      {blocked && blocked.length > 0 && (
        <ul className="mb-3 divide-y divide-border">
          {blocked.map((id) => (
            <li key={id} className="flex items-center gap-3 py-2">
              <span className="min-w-0 flex-1 truncate font-mono text-sm">{id}</span>
              <Button variant="outline" size="sm" disabled={busy} onClick={() => void setBlockedFor(id, false)}>
                Unblock
              </Button>
            </li>
          ))}
        </ul>
      )}

      <h3 className="mb-1 text-sm font-medium">Downloaded models</h3>
      {cached === null ? (
        <Spinner />
      ) : cached.length === 0 ? (
        <p className="text-sm text-muted-foreground">No downloaded models.</p>
      ) : (
        <ul className="divide-y divide-border">
          {cached.map((model) => (
            <li key={model.repo_id} className="flex items-center gap-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="truncate font-mono text-sm">{model.repo_id}</div>
                <div className="text-xs text-muted-foreground">{formatSize(model.size_bytes)}</div>
              </div>
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                Blocked
                <Switch
                  checked={isBlocked(model.repo_id)}
                  disabled={busy}
                  aria-label={`Block ${model.repo_id}`}
                  onCheckedChange={(next) => void setBlockedFor(model.repo_id, next)}
                />
              </label>
              <Button
                variant="destructive"
                size="sm"
                disabled={busy}
                onClick={() => {
                  if (window.confirm(`Delete ${model.repo_id} from disk?`)) {
                    void run(() => deleteCachedModel(model.repo_id));
                  }
                }}
              >
                Delete
              </Button>
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}

export function AdminPage() {
  const owner = useIsAccountOwner();
  const [error, setError] = useState<string | null>(null);

  if (!owner) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
        <h1 className="text-lg font-semibold">Admin only</h1>
        <p className="text-sm text-muted-foreground">Only the installation owner can open the admin panel.</p>
        <Button asChild variant="outline">
          <Link to="/chat">Back to chat</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 overflow-y-auto p-6">
      <div>
        <h1 className="text-xl font-semibold">Admin panel</h1>
        <p className="text-sm text-muted-foreground">Manage users and models for this installation.</p>
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}
      <UsersSection onError={setError} />
      <ModelsSection onError={setError} />
    </div>
  );
}
