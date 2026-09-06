// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, type SyntheticEvent } from "react";
import { useIsAccountOwner } from "@/features/auth/account-session";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { useT } from "@/i18n";
import {
  createAccount,
  deleteAccount,
  fetchAccounts,
  regenerateSetupCode,
  setAccountActive,
  type AccountSetupCode,
  type StudioAccount,
} from "../api/accounts";

export function AccountsTab() {
  const owner = useIsAccountOwner();
  // No requests or managed-account content even if a persisted tab directly names Accounts.
  return owner ? <OwnerAccountsTab /> : null;
}

function OwnerAccountsTab() {
  const t = useT();
  const [accounts, setAccounts] = useState<StudioAccount[]>([]);
  const [username, setUsername] = useState("");
  const [setup, setSetup] = useState<AccountSetupCode | null>(null);
  const [copied, setCopied] = useState(false);
  const [retiring, setRetiring] = useState<StudioAccount | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;
    void fetchAccounts()
      .then((result) => {
        if (!canceled) setAccounts(result);
      })
      .catch((reason: unknown) => {
        if (!canceled)
          setError(
            reason instanceof Error
              ? reason.message
              : t("settings.accounts.failed"),
          );
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });
    return () => {
      canceled = true;
    };
  }, [t]);

  async function perform(action: () => Promise<void>) {
    setError(null);
    setBusy(true);
    try {
      await action();
      setAccounts(await fetchAccounts());
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : t("settings.accounts.failed"),
      );
    } finally {
      setBusy(false);
    }
  }

  function showSetup(result: AccountSetupCode) {
    setSetup(result);
    setCopied(false);
  }

  function create(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || !username.trim()) return;
    void perform(async () => {
      showSetup(await createAccount(username));
      setUsername("");
    });
  }

  return (
    <div className="space-y-6">
      <div>
        <h2
          className="text-lg font-semibold"
          data-settings-label={t("settings.accounts.title")}
        >
          {t("settings.accounts.title")}
        </h2>
        <p className="text-sm text-muted-foreground">
          {t("settings.accounts.description")}
        </p>
      </div>
      <form className="space-y-2" onSubmit={create}>
        <Label
          htmlFor="new-account-username"
          data-settings-label={t("settings.accounts.create")}
        >
          {t("settings.accounts.create")}
        </Label>
        <div className="flex gap-2">
          <Input
            id="new-account-username"
            aria-label={t("settings.accounts.username")}
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="off"
            autoCapitalize="none"
            spellCheck={false}
            required
          />
          <Button type="submit" disabled={busy || !username.trim()}>
            {t("settings.accounts.create")}
          </Button>
        </div>
      </form>
      {setup && (
        <section
          className="space-y-3 rounded-xl border p-4"
          aria-label={t("settings.accounts.setupCode")}
        >
          <h3 className="font-medium">
            {t("settings.accounts.setupFor", { username: setup.username })}
          </h3>
          <p className="text-sm text-muted-foreground">
            {t("settings.accounts.shownOnce")}
          </p>
          <code
            className="block break-all select-all"
            data-testid="account-setup-code"
          >
            {setup.setup_code}
          </code>
          <p className="text-sm">
            {t("settings.accounts.expires", {
              expiry: new Date(setup.expires_at).toLocaleString(),
            })}
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => {
                void copyToClipboard(setup.setup_code).then((ok) => {
                  setCopied(ok);
                  if (!ok) setError(t("settings.accounts.copyFailed"));
                });
              }}
            >
              {copied
                ? t("settings.accounts.copied")
                : t("settings.accounts.copy")}
            </Button>
            <Button variant="ghost" onClick={() => setSetup(null)}>
              {t("settings.accounts.dismiss")}
            </Button>
          </div>
        </section>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {loading ? (
        <p>{t("common.loading")}</p>
      ) : (
        <ul className="divide-y">
          {accounts.map((account) => (
            <li
              key={account.account_id}
              className="flex flex-wrap items-center gap-3 py-3"
              data-testid={`account-${account.username}`}
            >
              <div className="min-w-0 flex-1">
                <p className="break-all font-medium">{account.username}</p>
                <p className="text-sm text-muted-foreground">
                  {account.role === "owner"
                    ? t("settings.accounts.owner")
                    : account.is_active
                      ? t("settings.accounts.active")
                      : t("settings.accounts.inactive")}
                </p>
              </div>
              {account.role !== "owner" && (
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={busy}
                    onClick={() =>
                      void perform(async () =>
                        showSetup(
                          await regenerateSetupCode(account.account_id),
                        ),
                      )
                    }
                  >
                    {t("settings.accounts.regenerate")}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={busy}
                    onClick={() =>
                      void perform(async () => {
                        await setAccountActive(
                          account.account_id,
                          !account.is_active,
                        );
                        if (setup?.account_id === account.account_id)
                          setSetup(null);
                      })
                    }
                  >
                    {account.is_active
                      ? t("settings.accounts.deactivate")
                      : t("settings.accounts.reactivate")}
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={busy}
                    onClick={() => setRetiring(account)}
                  >
                    {t("settings.accounts.delete")}
                  </Button>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
      {error && (
        <Button
          variant="outline"
          disabled={busy}
          onClick={() => void perform(async () => {})}
        >
          {t("settings.accounts.retry")}
        </Button>
      )}
      <AlertDialog
        open={retiring !== null}
        onOpenChange={(open) => {
          if (!open && !busy) setRetiring(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.accounts.deleteTitle", {
                username: retiring?.username ?? "",
              })}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("settings.accounts.deleteDescription", {
                username: retiring?.username ?? "",
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>
              {t("settings.accounts.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={busy}
              onClick={(event) => {
                event.preventDefault();
                if (!retiring || busy) return;
                void perform(async () => {
                  await deleteAccount(retiring.account_id);
                  if (setup?.account_id === retiring.account_id) setSetup(null);
                  setRetiring(null);
                });
              }}
            >
              {t("settings.accounts.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
