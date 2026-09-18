// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState, type SyntheticEvent } from "react";
import {
  Add01Icon,
  Search01Icon,
  Copy01Icon,
  Delete02Icon,
  Key01Icon,
  MoreHorizontalIcon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Spinner } from "@/components/ui/spinner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useIsAccountOwner } from "@/features/auth";
import { UserAvatar } from "@/features/profile";
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
import { useLocale, useT } from "@/i18n";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
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
  // managed accounts cannot request or render account administration.
  return owner ? <OwnerAccountsTab /> : null;
}

function OwnerAccountsTab() {
  const t = useT();
  const locale = useLocale();
  const actionTrigger = useRef<HTMLButtonElement | null>(null);
  const copyButton = useRef<HTMLButtonElement | null>(null);
  const createTrigger = useRef<HTMLButtonElement | null>(null);
  const [creating, setCreating] = useState(false);
  const [query, setQuery] = useState("");
  const [accounts, setAccounts] = useState<StudioAccount[]>([]);
  const [username, setUsername] = useState("");
  const [setup, setSetup] = useState<AccountSetupCode | null>(null);
  const [copied, setCopied] = useState(false);
  const [resetting, setResetting] = useState<StudioAccount | null>(null);
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

  useEffect(() => {
    if (setup) copyButton.current?.focus();
  }, [setup]);

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
      setQuery("");
    });
  }

  const confirming = resetting ?? retiring;
  const editorOpen = creating || setup !== null;
  const filteredAccounts = accounts.filter((account) =>
    account.username
      .toLocaleLowerCase(locale)
      .includes(query.trim().toLocaleLowerCase(locale)),
  );
  const dateFormatter = new Intl.DateTimeFormat(locale, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
  function createdDate(account: StudioAccount) {
    const date = new Date(account.created_at);
    return Number.isNaN(date.getTime()) ? "—" : dateFormatter.format(date);
  }
  function closeEditor() {
    if (busy) return;
    setCreating(false);
    setSetup(null);
    setUsername("");
    if (!setup) setError(null);
  }

  return (
    <div className="space-y-6">
      <div className="space-y-1 pr-6">
        <h2
          className="font-heading text-base font-semibold"
          data-settings-label={t("settings.accounts.title")}
        >
          {t("settings.accounts.title")}
        </h2>
        <p className="max-w-lg text-xs leading-relaxed text-muted-foreground">
          {t("settings.accounts.description")}
        </p>
      </div>

      <div className="flex items-center gap-3">
        <div className="relative min-w-0 flex-1">
          <HugeiconsIcon
            icon={Search01Icon}
            className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden="true"
          />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("settings.accounts.search")}
            aria-label={t("settings.accounts.search")}
            className="pl-9"
          />
        </div>
        <Button
          ref={createTrigger}
          variant="dark"
          size="sm"
          disabled={busy}
          data-settings-label={t("settings.accounts.create")}
          onClick={() => {
            setError(null);
            actionTrigger.current = null;
            setCreating(true);
          }}
        >
          <HugeiconsIcon
            icon={Add01Icon}
            className="size-3.5"
            aria-hidden="true"
          />
          {t("settings.accounts.create")}
        </Button>
      </div>

      {error && !confirming && !editorOpen && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-destructive/20 bg-destructive/5 p-3">
          <p
            role="alert"
            className="min-w-0 break-words text-xs text-destructive"
          >
            {error}
          </p>
          <Button
            variant="outline"
            size="sm"
            disabled={busy}
            onClick={() => void perform(async () => {})}
          >
            {t("settings.accounts.retry")}
          </Button>
        </div>
      )}

      <section
        aria-label={t("settings.accounts.title")}
        aria-busy={loading || busy}
      >
        {loading ? (
          <div
            className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground"
            role="status"
          >
            <Spinner label={t("common.loading")} />
            {t("common.loading")}
          </div>
        ) : (
          <table className="w-full table-fixed border-separate border-spacing-x-0 border-spacing-y-1 text-left text-xs">
            <thead className="text-muted-foreground">
              <tr className="[&>th]:bg-muted/40">
                <th scope="col" className="rounded-l-xl py-2 pl-5 pr-3 font-normal">
                  {t("settings.accounts.username")}
                </th>
                <th
                  scope="col"
                  className="w-28 py-2 pr-3 font-normal max-sm:hidden"
                >
                  {t("settings.accounts.created")}
                </th>
                <th scope="col" className="w-24 py-2 font-normal">
                  {t("settings.accounts.status")}
                </th>
                <th scope="col" className="w-12 rounded-r-xl py-2">
                  <span className="sr-only">
                    {t("settings.accounts.actions")}
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredAccounts.map((account) => (
                <tr
                  key={account.account_id}
                  className="[&>td]:transition-colors hover:[&>td]:bg-muted/40 focus-within:[&>td]:bg-muted/40"
                  data-testid={`account-${account.username}`}
                >
                  <td className="rounded-l-xl py-3 pl-3 pr-3">
                    <div className="flex items-center gap-2.5">
                      <UserAvatar
                        name={account.username}
                        imageUrl={null}
                        size="sm"
                        className="size-8"
                      />
                      <div className="min-w-0 space-y-0.5">
                        <p className="break-all text-sm font-medium">
                          {account.username}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {account.role === "owner"
                            ? t("settings.accounts.owner")
                            : t("settings.accounts.privateAccount")}
                        </p>
                        <p className="text-xs text-muted-foreground sm:hidden">
                          {createdDate(account)}
                        </p>
                      </div>
                    </div>
                  </td>
                  <td className="py-3 pr-3 text-muted-foreground max-sm:hidden">
                    <time dateTime={account.created_at}>
                      {createdDate(account)}
                    </time>
                  </td>
                  <td className="py-3">
                    <span
                      className={cn(
                        "inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-1 text-[11px] font-medium",
                        account.is_active
                          ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                          : "bg-muted text-muted-foreground",
                      )}
                    >
                      <span
                        className={cn(
                          "size-1.5 rounded-full",
                          account.is_active
                            ? "bg-emerald-500"
                            : "bg-muted-foreground/50",
                        )}
                        aria-hidden="true"
                      />
                      {account.is_active
                        ? t("settings.accounts.active")
                        : t("settings.accounts.inactive")}
                    </span>
                  </td>
                  <td className="rounded-r-xl py-3 pr-3 text-right">
                    {account.role !== "owner" && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            className="max-sm:size-9"
                            onFocus={(event) => {
                              actionTrigger.current = event.currentTarget;
                            }}
                            onPointerDown={(event) => {
                              actionTrigger.current = event.currentTarget;
                            }}
                            disabled={busy}
                            aria-label={t("settings.accounts.actionsFor", {
                              username: account.username,
                            })}
                          >
                            <HugeiconsIcon
                              icon={MoreHorizontalIcon}
                              className="size-4"
                              aria-hidden="true"
                            />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="w-60"
                          onCloseAutoFocus={(event) => {
                            if (confirming) event.preventDefault();
                          }}
                        >
                          <DropdownMenuItem
                            disabled={busy}
                            onSelect={() => {
                              setError(null);
                              setResetting(account);
                            }}
                          >
                            <HugeiconsIcon
                              icon={Key01Icon}
                              className="size-4"
                              aria-hidden="true"
                            />
                            {t("settings.accounts.regenerate")}
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            disabled={busy}
                            onSelect={() =>
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
                            <HugeiconsIcon
                              icon={UserIcon}
                              className="size-4"
                              aria-hidden="true"
                            />
                            {account.is_active
                              ? t("settings.accounts.deactivate")
                              : t("settings.accounts.reactivate")}
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            variant="destructive"
                            disabled={busy}
                            onSelect={() => {
                              setError(null);
                              setRetiring(account);
                            }}
                          >
                            <HugeiconsIcon
                              icon={Delete02Icon}
                              className="size-4"
                              aria-hidden="true"
                            />
                            {t("settings.accounts.delete")}
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!loading &&
          !error &&
          (filteredAccounts.length === 0 ||
            (accounts.length === 1 &&
              accounts[0].role === "owner" &&
              !query.trim())) && (
            <p
              className="py-8 text-center text-xs font-normal leading-relaxed text-muted-foreground"
              role="status"
            >
              {t(
                query.trim()
                  ? "settings.accounts.noResults"
                  : "settings.accounts.empty",
              )}
            </p>
          )}
      </section>

      <Dialog
        open={editorOpen}
        onOpenChange={(open) => {
          if (!open) closeEditor();
        }}
      >
        <DialogContent
          showCloseButton={false}
          className="sm:max-w-md max-sm:max-w-[calc(100%-2rem)] max-sm:top-1/2 max-sm:left-1/2 max-sm:-translate-1/2 max-sm:h-auto max-sm:w-[calc(100%-2rem)] max-sm:max-h-[calc(100dvh-var(--studio-window-chrome-top,0px)-2rem)] max-sm:rounded-2xl"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            const trigger = actionTrigger.current;
            (trigger?.isConnected && !trigger.disabled
              ? trigger
              : createTrigger.current
            )?.focus({ preventScroll: true });
          }}
        >
          <DialogHeader>
            <DialogTitle className="break-words">
              {setup
                ? t("settings.accounts.setupFor", { username: setup.username })
                : t("settings.accounts.create")}
            </DialogTitle>
            <DialogDescription>
              {t(
                setup
                  ? "settings.accounts.shownOnce"
                  : "settings.accounts.createDescription",
              )}
            </DialogDescription>
          </DialogHeader>
          {setup ? (
            <>
              <code
                className="block select-all break-all rounded-lg border border-border/60 bg-muted/30 px-4 py-3 font-mono text-sm leading-relaxed"
                data-testid="account-setup-code"
              >
                {setup.setup_code}
              </code>
              <p className="text-xs text-muted-foreground">
                {t("settings.accounts.loginHint", { username: setup.username })}
              </p>
              <p className="text-xs text-muted-foreground">
                {t("settings.accounts.expires", {
                  expiry: new Date(setup.expires_at).toLocaleString(locale),
                })}
              </p>
              <DialogFooter>
                <Button
                  ref={copyButton}
                  variant="outline"
                  onClick={() => {
                    void copyToClipboard(setup.setup_code).then((ok) => {
                      setCopied(ok);
                      if (!ok) setError(t("settings.accounts.copyFailed"));
                    });
                  }}
                >
                  <HugeiconsIcon
                    icon={copied ? Tick02Icon : Copy01Icon}
                    className="size-3.5"
                    aria-hidden="true"
                  />
                  <span aria-live="polite">
                    {t(
                      copied
                        ? "settings.accounts.copied"
                        : "settings.accounts.copy",
                    )}
                  </span>
                </Button>
                <Button variant="dark" disabled={busy} onClick={closeEditor}>
                  {t("settings.accounts.dismiss")}
                </Button>
              </DialogFooter>
            </>
          ) : (
            <form onSubmit={create} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="new-account-username" className="text-xs">
                  {t("settings.accounts.username")}
                </Label>
                <Input
                  id="new-account-username"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  autoComplete="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  readOnly={busy}
                  required
                />
              </div>
              <DialogFooter>
                <Button
                  type="button"
                  variant="ghost"
                  disabled={busy}
                  onClick={closeEditor}
                >
                  {t("settings.accounts.cancel")}
                </Button>
                <Button
                  type="submit"
                  variant="dark"
                  disabled={busy || !username.trim()}
                >
                  {busy && <Spinner label={t("common.loading")} />}
                  {t("settings.accounts.create")}
                </Button>
              </DialogFooter>
            </form>
          )}
          {error && (
            <p role="alert" className="break-words text-xs text-destructive">
              {error}
            </p>
          )}
        </DialogContent>
      </Dialog>
      <AlertDialog
        open={confirming !== null}
        onOpenChange={(open) => {
          if (open || busy) return;
          setRetiring(null);
          setResetting(null);
        }}
      >
        <AlertDialogContent
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            if (setup) return;
            const trigger = actionTrigger.current;
            const target =
              trigger?.isConnected && !trigger.disabled
                ? trigger
                : createTrigger.current;
            target?.focus({ preventScroll: true });
          }}
        >
          <AlertDialogHeader>
            <AlertDialogTitle className="break-words">
              {t(
                resetting
                  ? "settings.accounts.resetTitle"
                  : "settings.accounts.deleteTitle",
                { username: confirming?.username ?? "" },
              )}
            </AlertDialogTitle>
            <AlertDialogDescription className="break-words">
              {t(
                resetting
                  ? "settings.accounts.resetDescription"
                  : "settings.accounts.deleteDescription",
                { username: confirming?.username ?? "" },
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {error && (
            <p role="alert" className="break-words text-sm text-destructive">
              {error}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>
              {t("settings.accounts.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={busy}
              onClick={(event) => {
                event.preventDefault();
                if (busy) return;
                if (resetting) {
                  void perform(async () => {
                    showSetup(await regenerateSetupCode(resetting.account_id));
                    setResetting(null);
                  });
                  return;
                }
                if (!retiring) return;
                void perform(async () => {
                  await deleteAccount(retiring.account_id);
                  if (setup?.account_id === retiring.account_id) setSetup(null);
                  setRetiring(null);
                });
              }}
            >
              {busy && <Spinner label={t("common.loading")} />}
              {t(
                resetting
                  ? "settings.accounts.regenerate"
                  : "settings.accounts.delete",
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
