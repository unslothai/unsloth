// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { ComputerAddIcon, ComputerIcon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { Copy01Icon } from "@/lib/copy-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  answerPairing,
  fetchMachines,
  fetchPairing,
  removeMachine,
  startPairing,
  type MachinesList,
  type PairedMachine,
  type Pairing,
} from "../api/cluster";

const JOIN_COMMAND = "unsloth worker --join";
const POLL_MS = 1000;

function stateLabel(m: PairedMachine): TranslationKey {
  switch (m.state) {
    case "ready":
      return m.via === "direct internet connection"
        ? "settings.machines.readyInternet"
        : "settings.machines.ready";
    case "busy":
      return "settings.machines.busy";
    case "rpc down":
      return "settings.machines.notServing";
    case "relayed":
      return "settings.machines.relayed";
    case "unreachable":
      return "settings.machines.unreachable";
    default:
      return "settings.machines.checking";
  }
}

function StatusBadge({ machine }: { machine: PairedMachine }) {
  const t = useT();
  const good = machine.state === "ready";
  const warn = machine.state === "busy" || machine.state === "relayed";
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-1 text-ui-11 font-medium",
        good && "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
        warn && "bg-amber-500/10 text-amber-700 dark:text-amber-400",
        !good && !warn && "bg-muted text-muted-foreground",
      )}
      data-testid={`machine-state-${machine.node_id}`}
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          good ? "bg-emerald-500" : warn ? "bg-amber-500" : "bg-muted-foreground/50",
        )}
        aria-hidden="true"
      />
      {t(stateLabel(machine))}
    </span>
  );
}

function secondsLeft(expiresAt: number): number {
  return Math.max(0, Math.round(expiresAt - Date.now() / 1000));
}

function formatLeft(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function AddMachineDialog({
  open,
  onClose,
  onPaired,
}: {
  open: boolean;
  onClose: () => void;
  onPaired: () => void;
}) {
  const t = useT();
  const [code, setCode] = useState<string | null>(null);
  const [pairing, setPairing] = useState<Pairing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [left, setLeft] = useState(0);
  const pairingRef = useRef<Pairing | null>(null);
  pairingRef.current = pairing;
  const generation = useRef(0);

  const begin = useCallback(async () => {
    const mine = ++generation.current;
    setError(null);
    setCopied(false);
    setBusy(true);
    try {
      const { code: fresh, ...state } = await startPairing();
      if (mine !== generation.current) {
        // The dialog closed or restarted while the window was opening (first use downloads the helper).
        void answerPairing(state.id, "cancel").catch(() => undefined);
        return;
      }
      setCode(fresh);
      setPairing(state);
    } catch (e) {
      if (mine === generation.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (mine === generation.current) setBusy(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    void begin();
    return () => {
      generation.current += 1;
      const current = pairingRef.current;
      if (current && (current.state === "waiting" || current.state === "approval")) {
        void answerPairing(current.id, "cancel").catch(() => undefined);
      }
      setCode(null);
      setPairing(null);
    };
  }, [open, begin]);

  const id = pairing?.id;
  const live = pairing?.state === "waiting" || pairing?.state === "approval";
  useEffect(() => {
    if (!id || !live) return;
    const timer = window.setInterval(() => {
      void fetchPairing(id)
        .then((next) => {
          setPairing(next);
          if (next.state === "paired") {
            setCode(null);
            onPaired();
          }
          if (next.state !== "waiting") setCode((c) => (next.state === "approval" ? c : null));
        })
        .catch(() => undefined);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [id, live, onPaired]);

  useEffect(() => {
    if (!pairing || pairing.state !== "waiting") return;
    setLeft(secondsLeft(pairing.expires_at));
    const timer = window.setInterval(() => setLeft(secondsLeft(pairing.expires_at)), 1000);
    return () => window.clearInterval(timer);
  }, [pairing]);

  const answer = async (action: "approve" | "refuse") => {
    if (!pairing) return;
    setBusy(true);
    try {
      setPairing(await answerPairing(pairing.id, action));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const state = pairing?.state;
  const ended = state === "expired" || state === "refused" || state === "failed" || state === "cancelled";
  let title: string = t("settings.machines.add");
  let description: string = t("settings.machines.intro");
  if (state === "approval") {
    title = t("settings.machines.askTitle", { name: pairing?.name ?? "" });
    description = t("settings.machines.askDescription");
  } else if (state === "paired") {
    title = t("settings.machines.pairedTitle", { name: pairing?.name ?? "" });
    description = t("settings.machines.pairedDescription");
  } else if (state === "expired") {
    description = t("settings.machines.expired");
  } else if (state === "refused") {
    description = t("settings.machines.refused", { name: pairing?.name ?? "" });
  } else if (state === "failed" || state === "cancelled") {
    description = t("settings.machines.failed", { error: pairing?.error ?? state });
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
    >
      <DialogContent className="sm:max-w-lg" data-testid="add-machine-dialog">
        <DialogHeader>
          <DialogTitle className="break-words">{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        {error && (
          <p role="alert" className="break-words text-sm text-destructive">
            {error}
          </p>
        )}
        {!pairing && busy && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground" role="status">
            <Spinner label={t("common.loading")} />
            {t("common.loading")}
          </div>
        )}
        {state === "waiting" && code && (
          <div className="space-y-3">
            <code className="block select-all rounded-md border border-border/60 bg-muted/30 px-3 py-2 font-mono text-sm">
              {JOIN_COMMAND}
            </code>
            <code
              className="block max-h-28 select-all overflow-y-auto break-all rounded-md border border-border/60 bg-muted/30 px-3 py-2 font-mono text-xs leading-relaxed"
              data-testid="pairing-code"
            >
              {code}
            </code>
            <p className="text-xs text-muted-foreground">
              {t(pairing?.remote ? "settings.machines.worksAnywhere" : "settings.machines.worksLan", {
                time: formatLeft(left),
              })}
            </p>
            <p className="flex items-center gap-2 text-xs text-muted-foreground" role="status">
              <Spinner label={t("settings.machines.waiting")} />
              {t("settings.machines.waiting")}
            </p>
          </div>
        )}
        {state === "approval" && (
          <p
            className="rounded-md border border-border/60 bg-muted/30 px-3 py-3 text-center font-mono text-lg font-semibold tracking-wide"
            data-testid="pairing-words"
          >
            {(pairing?.words ?? []).join("  ")}
          </p>
        )}
        <DialogFooter>
          {state === "waiting" && code && (
            <Button
              variant="outline"
              onClick={() => {
                void copyToClipboard(code).then((ok) => {
                  setCopied(ok);
                  if (!ok) setError(t("settings.machines.copyFailed"));
                });
              }}
            >
              <HugeiconsIcon
                icon={copied ? Tick02Icon : Copy01Icon}
                className="size-3.5"
                aria-hidden="true"
              />
              <span aria-live="polite">
                {t(copied ? "settings.machines.copied" : "settings.machines.copy")}
              </span>
            </Button>
          )}
          {state === "approval" && (
            <>
              <Button variant="outline" disabled={busy} onClick={() => void answer("refuse")}>
                {t("settings.machines.refuse")}
              </Button>
              <Button variant="dark" disabled={busy} onClick={() => void answer("approve")}>
                {t("settings.machines.allow")}
              </Button>
            </>
          )}
          {ended && (
            <Button variant="outline" disabled={busy} onClick={() => void begin()}>
              {t("settings.machines.startAgain")}
            </Button>
          )}
          {state !== "approval" && (
            <Button variant={state === "paired" ? "dark" : "ghost"} onClick={onClose}>
              {t(state === "paired" ? "settings.machines.done" : "settings.machines.close")}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function MachinesTab() {
  const t = useT();
  const [list, setList] = useState<MachinesList | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [removing, setRemoving] = useState<PairedMachine | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      setList(await fetchMachines(false));
      setChecking(true);
      setList(await fetchMachines(true));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const machines = list?.machines ?? [];

  return (
    <div className="settings-page">
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.machines.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.machines.title")}
        </h1>
        <p
          data-settings-label={t("settings.machines.description")}
          className="text-xs text-muted-foreground"
        >
          {t("settings.machines.description")}
        </p>
      </header>

      {list?.role === "worker" ? (
        <p className="rounded-lg border border-border/60 bg-muted/30 p-3 text-xs text-muted-foreground">
          {t("settings.machines.workerRole")}
        </p>
      ) : (
        <>
          <div className="flex items-center justify-end gap-2">
            <Button
              variant="ghost"
              size="sm"
              disabled={checking}
              onClick={() => void load()}
            >
              {checking ? (
                <Spinner label={t("settings.machines.checking")} />
              ) : (
                <HugeiconsIcon icon={RefreshIcon} className="size-3.5" aria-hidden="true" />
              )}
              {t("settings.machines.refresh")}
            </Button>
            <Button
              variant="dark"
              size="sm"
              data-settings-label={t("settings.machines.add")}
              onClick={() => setAdding(true)}
            >
              <HugeiconsIcon icon={ComputerAddIcon} className="size-3.5" aria-hidden="true" />
              {t("settings.machines.add")}
            </Button>
          </div>

          {error && (
            <p role="alert" className="break-words rounded-lg border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive">
              {error}
            </p>
          )}

          <section aria-label={t("settings.machines.title")} aria-busy={list === null}>
            {list === null ? (
              <div
                className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground"
                role="status"
              >
                <Spinner label={t("common.loading")} />
                {t("common.loading")}
              </div>
            ) : machines.length === 0 ? (
              <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border/70 py-10 text-center">
                <HugeiconsIcon icon={ComputerIcon} className="size-6 text-muted-foreground" aria-hidden="true" />
                <p className="text-sm font-medium">{t("settings.machines.empty")}</p>
                <p className="max-w-sm text-xs text-muted-foreground">{t("settings.machines.emptyHint")}</p>
              </div>
            ) : (
              <table className="w-full table-fixed border-separate border-spacing-x-0 border-spacing-y-1 text-left text-xs">
                <thead className="text-muted-foreground">
                  <tr className="[&>th]:bg-muted/40">
                    <th scope="col" className="rounded-l-full py-2 pl-5 pr-3 font-normal">
                      {t("settings.machines.machine")}
                    </th>
                    <th scope="col" className="w-48 py-2 pr-3 font-normal">
                      {t("settings.machines.status")}
                    </th>
                    <th scope="col" className="w-12 rounded-r-full py-2">
                      <span className="sr-only">{t("settings.machines.remove")}</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {machines.map((m) => (
                    <tr
                      key={m.node_id}
                      className="[&>td]:transition-colors hover:[&>td]:bg-muted/40"
                      data-testid={`machine-${m.name}`}
                    >
                      <td className="rounded-l-md py-3 pl-3 pr-3">
                        <div className="flex items-center gap-2.5">
                          <HugeiconsIcon icon={ComputerIcon} className="size-5 shrink-0 text-muted-foreground" aria-hidden="true" />
                          <div className="min-w-0">
                            <p className="truncate font-medium text-foreground">{m.name}</p>
                            <p className="truncate text-muted-foreground">
                              {m.state === "unreachable" && m.reason
                                ? m.reason
                                : [m.where, ...(m.devices ?? [])].filter(Boolean).join(" · ")}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 pr-3">
                        <StatusBadge machine={m} />
                      </td>
                      <td className="rounded-r-md py-3 pr-3 text-right">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          disabled={busy}
                          aria-label={t("settings.machines.removeNamed", { name: m.name })}
                          onClick={() => setRemoving(m)}
                        >
                          <HugeiconsIcon icon={TrashIcon} className="size-4" aria-hidden="true" />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </>
      )}

      <AddMachineDialog
        open={adding}
        onClose={() => setAdding(false)}
        onPaired={() => void load()}
      />

      <AlertDialog
        open={removing !== null}
        onOpenChange={(open) => {
          if (!open && !busy) setRemoving(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="break-words">
              {t("settings.machines.removeTitle", { name: removing?.name ?? "" })}
            </AlertDialogTitle>
            <AlertDialogDescription className="break-words">
              {t("settings.machines.removeDescription", { name: removing?.name ?? "" })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>{t("settings.machines.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={busy}
              onClick={(event) => {
                event.preventDefault();
                if (!removing) return;
                setBusy(true);
                void removeMachine(removing.node_id)
                  .then(() => {
                    setRemoving(null);
                    return load();
                  })
                  .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
                  .finally(() => setBusy(false));
              }}
            >
              {busy && <Spinner label={t("common.loading")} />}
              {t("settings.machines.remove")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
