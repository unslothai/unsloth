// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { usePlatformStore } from "@/config/env";
import {
  loadLanAccess,
  startLanAccess,
  stopLanAccess,
  updateLanAccessAddresses,
  updateLanAccessAutoStart,
  updateLanAccessPort,
} from "@/features/settings/api/lan-access";
import {
  LAN_ACCESS_POLL_MS,
  type LanAccessStatus,
  defaultLanAccessAddressSelection,
  keylessLanAccessDescription,
  lanAccessAddressChoices,
  lanAccessAddressesReadOnly,
  lanAccessAutoStartReadOnly,
  lanAccessBlockMessage,
  lanAccessErrorMessage,
  lanAccessPortReadOnly,
  lanAccessStopDisconnectsOrigin,
  lanApiUrls,
  sameLanAccessAddresses,
  validLanAccessPort,
} from "@/features/settings/api/lan-access-state";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon, QrCodeIcon, Wifi01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import QRCode from "react-qr-code";
import { SettingsRow } from "./settings-row";

type LanAccessOperation = "start" | "stop" | "auto" | "port" | "addresses";
type PortMode = "automatic" | "custom";
type AddressMode = "automatic" | "chosen";

const STATE_LABEL: Record<LanAccessStatus["state"], string> = {
  off: "Off",
  online: "Online",
  error: "Error",
};

const OWNER_LABEL: Record<
  Exclude<LanAccessStatus["managedBy"], null>,
  string
> = {
  launch: "Launch managed",
  settings: "Settings managed",
};

function stateDotClass(state?: LanAccessStatus["state"]): string {
  if (state === "online") {
    return "bg-emerald-500";
  }
  return state === "error" ? "bg-red-500" : "bg-muted-foreground";
}

function AccessStatus({ status }: { status: LanAccessStatus | null }) {
  const owner = status?.managedBy ? OWNER_LABEL[status.managedBy] : null;
  return (
    <output
      className="flex items-center gap-1.5 text-xs text-muted-foreground"
      aria-live="polite"
    >
      <span
        className={cn("size-2 rounded-full", stateDotClass(status?.state))}
      />
      {status ? STATE_LABEL[status.state] : "Unavailable"}
      {owner ? ` · ${owner}` : ""}
    </output>
  );
}

function CopyLanUrlButton({ url, label }: { url: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const text = copied ? "Copied" : label;
  const copyTimer = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (copyTimer.current !== null) {
        window.clearTimeout(copyTimer.current);
      }
    };
  }, []);
  return (
    <Button
      type="button"
      size="sm"
      variant="outline"
      className="gap-1.5"
      aria-label={`${text} ${url}`}
      onClick={async () => {
        if (!(await copyToClipboard(url))) {
          return;
        }
        setCopied(true);
        if (copyTimer.current !== null) {
          window.clearTimeout(copyTimer.current);
        }
        copyTimer.current = window.setTimeout(() => setCopied(false), 1800);
      }}
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        className="size-3.5"
      />
      {text}
    </Button>
  );
}

function LanUrlQrButton({ url }: { url: string }) {
  return (
    <Dialog>
      <DialogTrigger asChild={true}>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="gap-1.5"
          aria-label={`Show QR code for ${url}`}
        >
          <HugeiconsIcon icon={QrCodeIcon} className="size-3.5" />
          QR
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xs">
        <DialogHeader>
          <DialogTitle>Open on your phone</DialogTitle>
          <DialogDescription>
            Scan from a device on the same network to open this address.
          </DialogDescription>
        </DialogHeader>
        <div className="mx-auto mt-2 rounded-md bg-white p-3">
          <QRCode value={url} size={192} />
        </div>
        <code className="block break-all text-center font-mono text-xs text-muted-foreground">
          {url}
        </code>
      </DialogContent>
    </Dialog>
  );
}

function StatusMessage({
  message,
  destructive,
}: {
  message?: string | null;
  destructive?: boolean;
}) {
  if (!message) {
    return null;
  }
  return (
    <p
      className={cn(
        "border-t border-border/60 px-4 py-2.5 text-xs leading-snug",
        destructive ? "text-destructive" : "text-muted-foreground",
      )}
    >
      {message}
    </p>
  );
}

function LanUrlActions({ url, copyLabel }: { url: string; copyLabel: string }) {
  return (
    <div className="flex shrink-0 items-center gap-2">
      <LanUrlQrButton url={url} />
      <CopyLanUrlButton url={url} label={copyLabel} />
    </div>
  );
}

function LanUrlPanel({ status }: { status: LanAccessStatus | null }) {
  if (!status || status.urls.length === 0) {
    return null;
  }
  const single = status.urls.length === 1;
  return (
    <div className="flex flex-col gap-1.5 border-t border-border/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-foreground">
          {single ? "Network address" : "Network addresses"}
        </span>
        {single ? (
          <LanUrlActions url={status.urls[0]} copyLabel="Copy URL" />
        ) : null}
      </div>
      {status.urls.map((url) => (
        <div key={url} className="flex items-center gap-2">
          <code className="block w-full min-w-0 break-all rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs text-foreground">
            {url}
          </code>
          {single ? null : <LanUrlActions url={url} copyLabel="Copy" />}
        </div>
      ))}
      {status.publicUrls.length > 0 ? (
        <span className="text-xs text-destructive leading-snug">
          {status.publicUrls[0]} is a public internet address, so this reaches
          beyond your local network. Anyone who has the password or an API key
          can sign in and run code on this machine.
        </span>
      ) : (
        <span className="text-xs text-muted-foreground leading-snug">
          {status.servesWebUi
            ? "Anyone on this network who has the password or an API key can sign in and run code on this machine."
            : "This launch serves the API only, so devices on the network can call the API but not open the web UI."}
        </span>
      )}
    </div>
  );
}

export function LanAccessSection() {
  const portErrorId = useId();
  const addressErrorId = useId();
  const [status, setStatus] = useState<LanAccessStatus | null>(null);
  const [busy, setBusy] = useState<LanAccessOperation | null>(null);

  const [portMode, setPortMode] = useState<PortMode>("automatic");
  const [portDraft, setPortDraft] = useState("8888");
  const [portError, setPortError] = useState<string | null>(null);
  const [addressMode, setAddressMode] = useState<AddressMode>("automatic");
  const [addressDraft, setAddressDraft] = useState<string[]>([]);
  const [addressError, setAddressError] = useState<string | null>(null);
  const [pollRevision, setPollRevision] = useState(0);
  const [pollEnabled, setPollEnabled] = useState(true);
  const mutationEpoch = useRef(0);
  const pollSuppressed = useRef(false);
  const selfStopDisconnectExpected = useRef(false);

  const applyStatus = useCallback((next: LanAccessStatus) => {
    setStatus(next);
    usePlatformStore.setState({ lanUrls: lanApiUrls(next) });
  }, []);

  useEffect(() => {
    const configured = status?.configuredPort;
    setPortMode(configured == null ? "automatic" : "custom");
    setPortDraft(String(configured ?? 8888));
    setPortError(null);
  }, [status?.configuredPort]);

  // keyed on the saved value, not the array: every poll returns a fresh one, which would wipe the draft
  const configuredAddressesKey = status?.configuredAddresses?.join(",") ?? null;
  // biome-ignore lint/correctness/useExhaustiveDependencies: configuredAddressesKey stands in for status.configuredAddresses
  useEffect(() => {
    const configured = status?.configuredAddresses ?? null;
    setAddressMode(configured === null ? "automatic" : "chosen");
    setAddressDraft(configured ?? []);
    setAddressError(null);
  }, [configuredAddressesKey]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: pollRevision intentionally restarts polling after a mutation
  useEffect(() => {
    if (!pollEnabled) {
      return;
    }
    let stopped = false;
    let timer: number | null = null;
    const schedule = () => {
      if (!stopped && !pollSuppressed.current) {
        timer = window.setTimeout(poll, LAN_ACCESS_POLL_MS);
      }
    };
    const poll = () => {
      if (pollSuppressed.current) {
        return;
      }
      const epoch = mutationEpoch.current;
      loadLanAccess()
        .then((next) => {
          if (
            !stopped &&
            !pollSuppressed.current &&
            mutationEpoch.current === epoch
          ) {
            selfStopDisconnectExpected.current = false;
            applyStatus(next);
          }
          schedule();
        })
        .catch(() => {
          if (mutationEpoch.current !== epoch) {
            return;
          }
          // a stop from a LAN address kills this page's own origin, so stop polling
          if (selfStopDisconnectExpected.current) {
            setPollEnabled(false);
            return;
          }
          schedule();
        });
    };
    poll();
    return () => {
      stopped = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [applyStatus, pollEnabled, pollRevision]);

  const perform = async (
    operation: LanAccessOperation,
    request: () => Promise<LanAccessStatus>,
    pausePollingAfterSuccess = false,
  ) => {
    mutationEpoch.current += 1;
    pollSuppressed.current = true;
    setBusy(operation);
    try {
      applyStatus(await request());
      if (pausePollingAfterSuccess) {
        selfStopDisconnectExpected.current = true;
      }
    } catch {
      if (operation === "port") {
        setPortError("Could not save the LAN port.");
      }
      if (operation === "addresses") {
        setAddressError("Could not save the LAN addresses.");
      }
      // polling resumes below and reconciles the visible state
    } finally {
      setBusy(null);
      pollSuppressed.current = false;
      setPollEnabled(true);
      setPollRevision((revision) => revision + 1);
    }
  };

  const start = () => perform("start", startLanAccess);
  const stop = () =>
    perform(
      "stop",
      stopLanAccess,
      lanAccessStopDisconnectsOrigin(
        status?.urls ?? [],
        typeof window === "undefined" ? "" : window.location.origin,
      ),
    );
  const setAutoStart = (enabled: boolean) =>
    perform("auto", () => updateLanAccessAutoStart(enabled));

  const portInvalid = portMode === "custom" && !validLanAccessPort(portDraft);
  const portErrorVisible = portInvalid || portError !== null;
  const selectedPort =
    portMode === "custom" && !portInvalid ? Number(portDraft) : null;
  const portDirty = status !== null && selectedPort !== status.configuredPort;
  const savePort = () => {
    if (portInvalid || !portDirty) return;
    setPortError(null);
    void perform("port", () => updateLanAccessPort(selectedPort));
  };

  const addressesReadOnly = busy !== null || lanAccessAddressesReadOnly(status);
  const addressChoices = lanAccessAddressChoices(status, addressDraft);
  const addressesEmpty = addressMode === "chosen" && addressDraft.length === 0;
  const addressErrorVisible = addressesEmpty || addressError !== null;
  const selectedAddresses = addressMode === "chosen" ? addressDraft : null;
  const addressesDirty =
    status !== null &&
    !sameLanAccessAddresses(selectedAddresses, status.configuredAddresses);
  const toggleAddress = (address: string, checked: boolean) => {
    setAddressDraft((draft) =>
      checked
        ? [...draft.filter((entry) => entry !== address), address]
        : draft.filter((entry) => entry !== address),
    );
    setAddressError(null);
  };
  const saveAddresses = () => {
    if (addressesEmpty || !addressesDirty) return;
    setAddressError(null);
    void perform("addresses", () =>
      updateLanAccessAddresses(selectedAddresses),
    );
  };

  const blockMessage = lanAccessBlockMessage(status, isTauri);
  const errorMessage = lanAccessErrorMessage(
    status?.error ?? null,
    status?.configuredPort ?? null,
  );
  const stopAction = status?.state === "online";
  // a start (now or at boot) binds the saved choice, so an unsaved one would expose the addresses just unticked
  const actionDisabled =
    busy !== null ||
    (stopAction ? !status?.canStop : !status?.canStart || addressesDirty);
  const actionLabel =
    busy === "start"
      ? "Starting…"
      : busy === "stop"
        ? "Stopping…"
        : stopAction
          ? "Stop"
          : "Start";

  return (
    <section
      data-settings-label="LAN access"
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <HugeiconsIcon
              icon={Wifi01Icon}
              className="size-4 text-foreground"
            />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="settings-heading text-base font-semibold font-heading">
                LAN access
              </h2>
              <AccessStatus status={status} />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Use Unsloth and its APIs from other devices on your Wi-Fi or wired
              network.
            </p>
          </div>
        </div>
        <Button
          type="button"
          size="sm"
          variant={stopAction ? "outline" : "default"}
          className="min-w-20 shrink-0"
          onClick={stopAction ? stop : start}
          disabled={actionDisabled}
        >
          {actionLabel}
        </Button>
      </div>

      <StatusMessage
        message={blockMessage ?? errorMessage}
        destructive={!blockMessage}
      />
      <LanUrlPanel status={status} />

      <div className="border-t border-border/60 px-4 py-1">
        {status?.portConfigurationSupported ? (
          <SettingsRow
            label="Port"
            description="Automatic tries 8888, then 8889–8908. Custom uses only the selected port. Stop LAN access before changing it."
            // Always mounted, even when empty: a live region has to exist before its
            // text changes for the change to be announced.
            below={
              <span
                id={portErrorId}
                role="status"
                aria-live="polite"
                className="text-xs text-destructive"
              >
                {portErrorVisible
                  ? portInvalid
                    ? "Enter a port from 1 to 65535."
                    : portError
                  : null}
              </span>
            }
          >
            <div className="flex items-center gap-2">
              <Select
                value={portMode}
                disabled={busy !== null || lanAccessPortReadOnly(status)}
                onValueChange={(value) => {
                  setPortMode(value as PortMode);
                  setPortError(null);
                }}
              >
                <SelectTrigger
                  size="sm"
                  className="w-28"
                  aria-label="LAN port mode"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="automatic">Automatic</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
              {portMode === "custom" ? (
                <Input
                  type="number"
                  min={1}
                  max={65535}
                  value={portDraft}
                  disabled={busy !== null || lanAccessPortReadOnly(status)}
                  aria-label="Custom LAN port"
                  aria-invalid={portInvalid}
                  aria-describedby={portErrorVisible ? portErrorId : undefined}
                  className="h-8 w-24"
                  onChange={(event) => {
                    setPortDraft(event.target.value);
                    setPortError(null);
                  }}
                />
              ) : null}
              <Button
                type="button"
                size="sm"
                onClick={savePort}
                disabled={
                  busy !== null ||
                  lanAccessPortReadOnly(status) ||
                  portInvalid ||
                  !portDirty
                }
              >
                Save
              </Button>
            </div>
          </SettingsRow>
        ) : null}
        {status?.addressConfigurationSupported ? (
          <>
            <SettingsRow
              label="Addresses"
              description="Automatic uses every address this machine has, public ones included. Choose uses only the addresses you tick, such as a Tailscale address. Stop LAN access before changing it, and save before starting."
              below={
                <span
                  id={addressErrorId}
                  role="status"
                  aria-live="polite"
                  className="text-xs text-destructive"
                >
                  {addressErrorVisible
                    ? addressesEmpty
                      ? "Tick at least one address, or use Automatic."
                      : addressError
                    : null}
                </span>
              }
            >
              <div className="flex items-center gap-2">
                <Select
                  value={addressMode}
                  disabled={addressesReadOnly}
                  onValueChange={(value) => {
                    const mode = value as AddressMode;
                    setAddressMode(mode);
                    if (mode === "chosen" && addressDraft.length === 0) {
                      setAddressDraft(defaultLanAccessAddressSelection(status));
                    }
                    setAddressError(null);
                  }}
                >
                  <SelectTrigger
                    size="sm"
                    className="w-28"
                    aria-label="LAN address mode"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="automatic">Automatic</SelectItem>
                    <SelectItem value="chosen">Choose</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  type="button"
                  size="sm"
                  onClick={saveAddresses}
                  disabled={
                    addressesReadOnly || addressesEmpty || !addressesDirty
                  }
                >
                  Save
                </Button>
              </div>
            </SettingsRow>
            {addressMode === "chosen" ? (
              <fieldset
                className="flex flex-col gap-2 pb-3"
                aria-describedby={
                  addressErrorVisible ? addressErrorId : undefined
                }
              >
                <legend className="sr-only">LAN addresses to use</legend>
                {addressChoices.length === 0 ? (
                  <span className="text-xs text-muted-foreground">
                    No network address found on this machine.
                  </span>
                ) : null}
                {addressChoices.map((choice) => (
                  <div
                    key={choice.address}
                    className="flex items-center gap-2.5"
                  >
                    <Checkbox
                      id={`${addressErrorId}-${choice.address}`}
                      checked={addressDraft.includes(choice.address)}
                      disabled={addressesReadOnly}
                      onCheckedChange={(checked) =>
                        toggleAddress(choice.address, checked === true)
                      }
                    />
                    <label
                      htmlFor={`${addressErrorId}-${choice.address}`}
                      className="flex cursor-pointer flex-wrap items-center gap-x-2.5 text-sm"
                    >
                      <code className="font-mono text-xs text-foreground">
                        {choice.address}
                      </code>
                      {choice.public ? (
                        <span className="text-xs text-destructive">
                          Public internet address
                        </span>
                      ) : null}
                      {choice.detected ? null : (
                        <span className="text-xs text-muted-foreground">
                          Not on this machine right now
                        </span>
                      )}
                    </label>
                  </div>
                ))}
              </fieldset>
            ) : null}
          </>
        ) : null}
        <SettingsRow
          label="Keyless API status"
          description={keylessLanAccessDescription(status)}
        >
          <span className="text-xs font-medium text-muted-foreground">
            {status?.keylessScope === "inference"
              ? "Inference"
              : status?.keylessScope === "full"
                ? "Local full"
                : "Off"}
          </span>
        </SettingsRow>
        <SettingsRow
          label="Start automatically"
          description="Put Unsloth on the network each time it starts. Stopping LAN access now won’t turn this off."
        >
          <Switch
            checked={status?.autoStart ?? false}
            disabled={
              busy !== null ||
              lanAccessAutoStartReadOnly(status) ||
              (addressesDirty && !status?.autoStart)
            }
            onCheckedChange={setAutoStart}
            aria-label="Start automatically"
          />
        </SettingsRow>
      </div>
    </section>
  );
}
