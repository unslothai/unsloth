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
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { usePlatformStore } from "@/config/env";
import {
  cancelCustomRemoteAccess,
  loadRemoteAccess,
  provisionCustomRemoteAccess,
  startRemoteAccess,
  stopRemoteAccess,
  teardownCustomRemoteAccess,
  updateRemoteAccessAutoStart,
  updateRemoteAccessMethod,
} from "@/features/settings/api/remote-access";
import {
  type RemoteAccessBlockMessageId,
  type RemoteAccessOperation,
  type RemoteAccessProgressStepId,
  type RemoteAccessRequestAxis,
  type RemoteAccessStatus,
  closeUnusedRemoteAccessWindow,
  remoteAccessAutoStartKind,
  remoteAccessAutoStartReadOnly,
  remoteAccessBlockMessageId,
  remoteAccessCustomActionsDisabled,
  remoteAccessCustomOperationInFlight,
  remoteAccessCustomTeardownMessageId,
  remoteAccessDnsConflictHostname,
  remoteAccessHeaderActionDisabled,
  remoteAccessIsReady,
  remoteAccessOperationRevision,
  remoteAccessPollDelay,
  remoteAccessPreferredKind,
  remoteAccessProgressStep,
  remoteAccessRequestMessageId,
  remoteAccessSelfStopPoll,
  remoteAccessSetupDialogShouldOpen,
  remoteAccessShouldClearRequestError,
  remoteAccessShowsCustomPanel,
  remoteAccessStopDisconnectsOrigin,
  remoteAccessTeardownNeedsLocalOrigin,
  remoteAccessUsableUrl,
  remoteAccessUsesStopAction,
} from "@/features/settings/api/remote-access-state";
import { type TranslationKey, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { openLink } from "@/lib/open-link";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  Copy01Icon,
  Globe02Icon,
  QrCodeIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import QRCode from "react-qr-code";
import { ChangePasswordDialog } from "./change-password-dialog";
import { SettingsRow } from "./settings-row";

type Translate = ReturnType<typeof useT>;

const CLOUDFLARE_TUNNELS_URL =
  "https://dash.cloudflare.com/?to=/:account/tunnels";
const CLOUDFLARE_DNS_URL =
  "https://dash.cloudflare.com/?to=/:account/:zone/dns/records";

const BLOCK_MESSAGE_KEYS: Record<RemoteAccessBlockMessageId, TranslationKey> = {
  serverStarting: "settings.general.remoteAccess.blockServerStarting",
  passwordDesktop: "settings.general.remoteAccess.blockPasswordDesktop",
  passwordWeb: "settings.general.remoteAccess.blockPasswordWeb",
  explicitlyDisabled: "settings.general.remoteAccess.blockExplicitlyDisabled",
  launchManaged: "settings.general.remoteAccess.blockLaunchManaged",
  colabManaged: "settings.general.remoteAccess.blockColabManaged",
  colab: "settings.general.remoteAccess.blockColab",
};

function localizedBlockMessage(
  reason: string | null,
  isDesktop: boolean,
  t: Translate,
): string | null {
  const messageId = remoteAccessBlockMessageId(reason, isDesktop);
  return messageId ? t(BLOCK_MESSAGE_KEYS[messageId]) : null;
}

function localizedCustomError(
  status: RemoteAccessStatus,
  t: Translate,
): string | null {
  const teardownMessageId = remoteAccessCustomTeardownMessageId(status);
  if (teardownMessageId === "teardownManual") {
    return t("settings.general.remoteAccess.errorTeardownManual");
  }
  if (teardownMessageId === "teardownFailed") {
    return t("settings.general.remoteAccess.errorTeardownFailed");
  }
  switch (status.customError) {
    case null:
      return null;
    case "dns_conflict": {
      const hostname = remoteAccessDnsConflictHostname(status);
      return hostname
        ? t("settings.general.remoteAccess.dnsConflict", { hostname })
        : t("settings.general.remoteAccess.dnsConflictUnknown");
    }
    case "invalid_hostname":
      return t("settings.general.remoteAccess.errorInvalidHostname");
    case "hostname_not_authorized":
      return t("settings.general.remoteAccess.errorWrongDomain", {
        hostname: status.customErrorDetail ?? status.customHostname ?? "",
      });
    case "certificate_exists": {
      return t("settings.general.remoteAccess.errorCertificateExists", {
        path: status.customErrorDetail ?? "~/.cloudflared/cert.pem",
      });
    }
    case "certificate_state_busy":
    case "connector_in_use":
      return t("settings.general.remoteAccess.errorBusy");
    case "identity_exists":
      return t("settings.general.remoteAccess.errorIdentityExists");
    case "cloudflared_unreachable":
      return t("settings.general.remoteAccess.errorUnavailable");
    case "cancelled":
      return t("settings.general.remoteAccess.errorCancelled");
    case "login_timed_out":
      return t("settings.general.remoteAccess.errorLoginTimeout");
    case "login_failed":
      return t("settings.general.remoteAccess.errorLoginFailed");
    case "setup_record_unreadable":
    case "tunnel_create_failed":
    case "route_failed":
      return t("settings.general.remoteAccess.errorSetupFailed");
    case "connector_stop_failed":
    case "teardown_failed":
      return t("settings.general.remoteAccess.errorTeardownFailed");
    default:
      return t("settings.general.remoteAccess.errorUnknown");
  }
}

function localizedRuntimeError(
  status: RemoteAccessStatus | null,
  t: Translate,
): string | null {
  if (!status?.error) {
    return null;
  }
  if (status.error === "custom_hostname_unreachable") {
    return t("settings.general.remoteAccess.errorHostnameUnreachable", {
      hostname: status.customHostname ?? "",
    });
  }
  return t("settings.general.remoteAccess.runtimeFailed");
}

function localizedRequestError(error: unknown, t: Translate): string {
  const code = error instanceof Error ? error.message : null;
  const messageId = remoteAccessRequestMessageId(code, isTauri);
  if (messageId in BLOCK_MESSAGE_KEYS) {
    return t(BLOCK_MESSAGE_KEYS[messageId as RemoteAccessBlockMessageId]);
  }
  switch (messageId) {
    case "invalidHostname":
      return t("settings.general.remoteAccess.errorInvalidHostname");
    case "busy":
      return t("settings.general.remoteAccess.errorBusy");
    default:
      return t("settings.general.remoteAccess.requestFailed");
  }
}

const PROGRESS_MESSAGE_KEYS: Record<
  RemoteAccessProgressStepId,
  TranslationKey
> = {
  connecting: "settings.general.remoteAccess.progressConnecting",
  openingLink: "settings.general.remoteAccess.progressOpeningLink",
  checkingHostname: "settings.general.remoteAccess.progressCheckingHostname",
  disconnecting: "settings.general.remoteAccess.progressDisconnecting",
};

function ConnectionProgress({ status }: { status: RemoteAccessStatus | null }) {
  const t = useT();
  const ready = remoteAccessIsReady(status);
  const step = remoteAccessProgressStep(status);
  const [showReady, setShowReady] = useState(false);
  useEffect(() => {
    if (!ready) {
      setShowReady(false);
      return;
    }
    setShowReady(true);
    const timer = window.setTimeout(() => setShowReady(false), 3200);
    return () => window.clearTimeout(timer);
  }, [ready]);
  if (showReady) {
    return (
      <output
        className="flex items-center gap-1.5 text-xs text-emerald-600"
        aria-live="polite"
      >
        <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
        {t("settings.general.remoteAccess.ready")}
      </output>
    );
  }
  if (step === null) {
    return null;
  }
  return (
    <output
      className="flex items-center gap-1.5 text-xs text-muted-foreground"
      aria-live="polite"
    >
      <span className="size-2 rounded-full bg-blue-500 animate-pulse" />
      <span>
        {t(
          step === "disconnecting"
            ? "settings.general.remoteAccess.actionStopping"
            : "settings.general.remoteAccess.actionStarting",
        )}
        {" · "}
        {t(PROGRESS_MESSAGE_KEYS[step])}
      </span>
    </output>
  );
}

function CopyRemoteUrlButton({ url }: { url: string }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
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
      {copied
        ? t("settings.general.remoteAccess.copied")
        : t("settings.general.remoteAccess.copyUrl")}
    </Button>
  );
}

function RemoteUrlQrButton({ url }: { url: string }) {
  const t = useT();
  return (
    <Dialog>
      <DialogTrigger asChild={true}>
        <Button type="button" size="sm" variant="outline" className="gap-1.5">
          <HugeiconsIcon icon={QrCodeIcon} className="size-3.5" />
          QR
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xs">
        <DialogHeader>
          <DialogTitle>
            {t("settings.general.remoteAccess.qrTitle")}
          </DialogTitle>
          <DialogDescription>
            {t("settings.general.remoteAccess.qrDescription")}
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
      role={destructive ? "alert" : undefined}
      className={cn(
        "border-t border-border/60 px-4 py-2.5 text-xs leading-snug",
        destructive ? "text-destructive" : "text-muted-foreground",
      )}
    >
      {message}
    </p>
  );
}

function RemoteUrlPanel({ status }: { status: RemoteAccessStatus | null }) {
  const t = useT();
  const url = remoteAccessUsableUrl(status);
  if (!url) {
    return null;
  }
  return (
    <div className="flex flex-col gap-1.5 border-t border-border/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-foreground">
          {t("settings.general.remoteAccess.remoteUrl")}
        </span>
        <div className="flex items-center gap-2">
          <RemoteUrlQrButton url={url} />
          <CopyRemoteUrlButton url={url} />
        </div>
      </div>
      <code className="block w-full break-all rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs text-foreground">
        {url}
      </code>
      <span className="text-xs text-muted-foreground leading-snug">
        {t("settings.general.remoteAccess.urlHint")}
      </span>
      <span className="text-xs text-muted-foreground leading-snug">
        {status?.streamingSupported
          ? t("settings.general.remoteAccess.streamingSupported")
          : t("settings.general.remoteAccess.streamingBuffered")}
      </span>
    </div>
  );
}

// Desktop signs in with a local secret, so the account password exists only for
// remote browsers and is managed here rather than in the General tab.
function RemotePasswordRow({
  status,
  onDone,
}: {
  status: RemoteAccessStatus | null;
  onDone: () => void;
}) {
  const t = useT();
  if (!(isTauri && status)) {
    return null;
  }
  return (
    <SettingsRow
      label={t("settings.general.remoteAccess.passwordLabel")}
      description={t("settings.general.remoteAccess.passwordDescription")}
    >
      <ChangePasswordDialog initial={status.passwordPending} onDone={onDone} />
    </SettingsRow>
  );
}

function RemoteAccessMethodControl({
  method,
  disabled,
  onChange,
}: {
  method: RemoteAccessStatus["method"];
  disabled: boolean;
  onChange: (method: RemoteAccessStatus["method"]) => void;
}) {
  const t = useT();
  const reduced = useReducedMotion();
  const options = [
    {
      value: "temporary" as const,
      label: t("settings.general.remoteAccess.temporaryMethod"),
    },
    {
      value: "custom" as const,
      label: t("settings.general.remoteAccess.customMethod"),
    },
  ];
  return (
    <div className="hub-tab-toggle inline-flex h-8 items-center rounded-full">
      {options.map((option) => {
        const active = option.value === method;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            aria-pressed={active}
            disabled={disabled}
            className={cn(
              "relative flex h-8 items-center rounded-full px-3 text-xs font-medium transition-colors disabled:opacity-50",
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active ? (
              <motion.span
                layoutId="remote-access-method-pill"
                className="hub-tab-toggle-pill absolute inset-0 rounded-full"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 35, mass: 0.5 }
                }
              />
            ) : null}
            <span className="relative z-10">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}

function CustomTunnelSetup({
  status,
  hostname,
  setHostname,
  busy,
  onProvision,
  onCancel,
  requestError,
}: {
  status: RemoteAccessStatus;
  hostname: string;
  setHostname: (hostname: string) => void;
  busy: RemoteAccessOperation | null;
  onProvision: (hostname: string) => Promise<boolean>;
  onCancel: (expectedRevision: number) => Promise<boolean>;
  requestError: string | null;
}) {
  const t = useT();
  const [confirmOpen, setConfirmOpen] = useState(
    remoteAccessSetupDialogShouldOpen(status, null),
  );
  const [requestedHostname, setRequestedHostname] = useState(
    status.customHostname ?? "",
  );
  const cloudflareWindow = useRef<Window | null>(null);
  const openedLoginUrl = useRef<string | null>(null);
  const cancelledSetupRevision = useRef<number | null>(null);
  const disabled = remoteAccessCustomActionsDisabled(status, busy !== null);
  const waiting = busy === "provision" || status.customState === "provisioning";

  useEffect(() => {
    if (status.customState !== "provisioning") {
      if (cancelledSetupRevision.current !== null && busy === null) {
        cancelledSetupRevision.current = null;
        setConfirmOpen(false);
      }
      return;
    }
    if (
      !remoteAccessSetupDialogShouldOpen(status, cancelledSetupRevision.current)
    ) {
      return;
    }
    setConfirmOpen(true);
    if (status.customHostname) {
      setRequestedHostname(status.customHostname);
    }
  }, [busy, status]);

  useEffect(() => {
    const loginUrl = status.loginUrl;
    if (
      !(confirmOpen && loginUrl) ||
      openedLoginUrl.current === loginUrl ||
      cancelledSetupRevision.current === status.customOperationRevision
    ) {
      return;
    }
    if (isTauri) {
      openedLoginUrl.current = loginUrl;
      import("@tauri-apps/plugin-opener").then(({ openUrl }) => {
        openUrl(loginUrl).catch(console.error);
      });
      return;
    }
    if (cloudflareWindow.current && !cloudflareWindow.current.closed) {
      try {
        cloudflareWindow.current.location.replace(loginUrl);
        openedLoginUrl.current = loginUrl;
      } catch {
        // Manual Open remains available.
      }
    }
  }, [confirmOpen, status.loginUrl, status.customOperationRevision]);

  const closePendingWindow = () => {
    if (openedLoginUrl.current === null) {
      closeUnusedRemoteAccessWindow(cloudflareWindow.current);
    }
    cloudflareWindow.current = null;
  };

  const cancelSetup = () => {
    closePendingWindow();
    if (waiting) {
      cancelledSetupRevision.current = status.customOperationRevision;
      onCancel(status.customOperationRevision).then((cancelled) => {
        if (!cancelled) {
          cancelledSetupRevision.current = null;
          setConfirmOpen(true);
        }
      });
      return;
    }
    setConfirmOpen(false);
  };

  return (
    <>
      {waiting ? null : (
        <form
          className="mt-3 flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            const targetHostname = hostname.trim();
            cancelledSetupRevision.current = null;
            setRequestedHostname(targetHostname);
            openedLoginUrl.current = null;
            if (!isTauri) {
              cloudflareWindow.current = window.open("", "_blank");
              if (cloudflareWindow.current) {
                cloudflareWindow.current.opener = null;
              }
            }
            setConfirmOpen(true);
            onProvision(targetHostname).then((started) => {
              if (!started) {
                closePendingWindow();
              }
            });
          }}
        >
          <Input
            value={hostname}
            onChange={(event) => setHostname(event.target.value)}
            placeholder={t("settings.general.remoteAccess.hostnamePlaceholder")}
            aria-label={t("settings.general.remoteAccess.hostnameLabel")}
            disabled={disabled}
          />
          <Button
            type="submit"
            size="sm"
            className="shrink-0"
            disabled={disabled || hostname.trim().length === 0}
          >
            {t("settings.general.remoteAccess.setupAction")}
          </Button>
        </form>
      )}
      <AlertDialog
        open={confirmOpen}
        onOpenChange={(open) => {
          if (open || !waiting) {
            setConfirmOpen(open);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.general.remoteAccess.setupConfirmTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription className="space-y-3">
              <span className="block">
                {t("settings.general.remoteAccess.setupConfirmDescription")}
              </span>
              <code className="block break-all rounded-md bg-muted px-3 py-2 font-mono text-xs text-foreground">
                {requestedHostname}
              </code>
              <span className="block" aria-live="polite">
                {status.loginUrl
                  ? t("settings.general.remoteAccess.setupAuthorize", {
                      hostname: requestedHostname,
                    })
                  : t("settings.general.remoteAccess.setupPreparing")}
              </span>
              {requestError || status.customError ? (
                <span className="block text-destructive" role="alert">
                  {requestError ?? localizedCustomError(status, t)}
                </span>
              ) : null}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={cancelSetup}
              disabled={
                busy === "cancel" ||
                (busy === "provision" && status.customState !== "provisioning")
              }
            >
              {t("common.cancel")}
            </Button>
            {status.loginUrl ? (
              <Button
                type="button"
                onClick={() => openLink(status.loginUrl as string)}
              >
                {t("settings.general.remoteAccess.openCloudflare")}
              </Button>
            ) : (
              <Button type="button" disabled={true}>
                {t("settings.general.remoteAccess.openCloudflare")}
              </Button>
            )}
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

function CustomTunnelRemovalProgress() {
  const t = useT();
  return (
    <div
      className="mt-3 rounded-md border border-border bg-muted/30 p-3"
      aria-live="polite"
      aria-atomic="true"
    >
      <p className="text-xs leading-snug text-muted-foreground">
        {t("settings.general.remoteAccess.removeProgress")}
      </p>
    </div>
  );
}

function CustomTunnelIdentity({
  status,
  busy,
  onTeardown,
}: {
  status: RemoteAccessStatus;
  busy: RemoteAccessOperation | null;
  onTeardown: () => void;
}) {
  const t = useT();
  const teardownDisconnectsOrigin = remoteAccessTeardownNeedsLocalOrigin(
    status,
    typeof window === "undefined" ? "" : window.location.origin,
  );
  return (
    <div
      className="mt-3 flex flex-col gap-3"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="flex items-center gap-2">
        <code className="min-w-0 flex-1 break-all rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs text-foreground">
          {status.customHostname}
        </code>
        <AlertDialog>
          <AlertDialogTrigger asChild={true}>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="shrink-0 text-destructive hover:text-destructive"
              disabled={
                remoteAccessCustomActionsDisabled(status, busy !== null) ||
                teardownDisconnectsOrigin
              }
            >
              {t("settings.general.remoteAccess.removeAction")}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>
                {t("settings.general.remoteAccess.removeTitle")}
              </AlertDialogTitle>
              <AlertDialogDescription>
                {t("settings.general.remoteAccess.removeDescription")}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="space-y-3 text-sm">
              <div className="space-y-2 rounded-md border border-border p-3">
                <p className="text-xs text-muted-foreground">
                  {t("settings.general.remoteAccess.removeTunnelHint")}
                </p>
                <code className="block break-all font-mono text-xs text-foreground">
                  {status.customTunnelName ??
                    t("settings.general.remoteAccess.tunnelNameUnavailable")}
                </code>
                <Button
                  asChild={true}
                  type="button"
                  size="sm"
                  variant="outline"
                >
                  <a
                    href={CLOUDFLARE_TUNNELS_URL}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {t("settings.general.remoteAccess.openCloudflareTunnels")}
                  </a>
                </Button>
              </div>
              <div className="space-y-2 rounded-md border border-border p-3">
                <p className="text-xs text-muted-foreground">
                  {t("settings.general.remoteAccess.removeDnsHint")}
                </p>
                <code className="block break-all font-mono text-xs text-foreground">
                  {status.customHostname}
                </code>
                <Button
                  asChild={true}
                  type="button"
                  size="sm"
                  variant="outline"
                >
                  <a href={CLOUDFLARE_DNS_URL} target="_blank" rel="noreferrer">
                    {t("settings.general.remoteAccess.openCloudflareDns")}
                  </a>
                </Button>
              </div>
            </div>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
              <AlertDialogAction variant="destructive" onClick={onTeardown}>
                {t("settings.general.remoteAccess.removeConfirm")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
      {teardownDisconnectsOrigin ? (
        <p className="text-xs leading-snug text-muted-foreground">
          {t("settings.general.remoteAccess.removeLocally")}
        </p>
      ) : null}
    </div>
  );
}

function CustomTunnelMessages({ status }: { status: RemoteAccessStatus }) {
  const t = useT();
  return (
    <>
      {status.customError ? (
        <div
          className="mt-3 text-xs leading-snug text-destructive"
          role="alert"
        >
          <p>{localizedCustomError(status, t)}</p>
        </div>
      ) : null}
    </>
  );
}

function CustomTunnelPanel({
  status,
  hostname,
  setHostname,
  busy,
  onProvision,
  onCancel,
  onTeardown,
  requestError,
}: {
  status: RemoteAccessStatus;
  hostname: string;
  setHostname: (hostname: string) => void;
  busy: RemoteAccessOperation | null;
  onProvision: (hostname: string) => Promise<boolean>;
  onCancel: (expectedRevision: number) => Promise<boolean>;
  onTeardown: () => void;
  requestError: string | null;
}) {
  const t = useT();
  const provisioning = status.customState === "provisioning";
  const removing = status.customState === "tearing_down";
  const hasIdentity =
    status.customState !== "provisioning" && status.customHostname !== null;

  return (
    <div className="border-t border-border/60 p-4">
      <div className="flex flex-col gap-1">
        <h3 className="text-sm font-medium text-foreground">
          {t("settings.general.remoteAccess.customHostnameTitle")}
        </h3>
        <p className="text-xs leading-snug text-muted-foreground">
          {t("settings.general.remoteAccess.customHostnameDescription")}
        </p>
      </div>

      {hasIdentity || removing ? null : (
        <CustomTunnelSetup
          status={status}
          hostname={hostname}
          setHostname={setHostname}
          busy={busy}
          onProvision={onProvision}
          onCancel={onCancel}
          requestError={requestError}
        />
      )}

      {removing ? <CustomTunnelRemovalProgress /> : null}

      {hasIdentity && !provisioning && !removing ? (
        <CustomTunnelIdentity
          status={status}
          busy={busy}
          onTeardown={onTeardown}
        />
      ) : null}
      {provisioning ? null : <CustomTunnelMessages status={status} />}
    </div>
  );
}

function useRemoteAccessPolling(
  applyStatus: (next: RemoteAccessStatus) => void,
) {
  const [pollRevision, setPollRevision] = useState(0);
  const [pollEnabled, setPollEnabled] = useState(true);
  const mutationEpoch = useRef(0);
  const pollSuppressed = useRef(false);
  const selfStopDisconnectExpected = useRef(false);
  const restartPolling = useCallback(() => {
    setPollEnabled(true);
    setPollRevision((revision) => revision + 1);
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: pollRevision intentionally restarts polling after a mutation
  useEffect(() => {
    if (!pollEnabled) {
      return;
    }
    let stopped = false;
    let timer: number | null = null;
    const schedule = (next: RemoteAccessStatus | null) => {
      if (!(stopped || pollSuppressed.current)) {
        timer = window.setTimeout(poll, remoteAccessPollDelay(next));
      }
    };
    const poll = () => {
      if (pollSuppressed.current) {
        return;
      }
      const epoch = mutationEpoch.current;
      loadRemoteAccess()
        .then((next) => {
          if (
            !(stopped || pollSuppressed.current) &&
            mutationEpoch.current === epoch
          ) {
            const settled = remoteAccessSelfStopPoll(
              next,
              selfStopDisconnectExpected.current,
            );
            selfStopDisconnectExpected.current = settled.expectingDisconnect;
            if (settled.apply) {
              applyStatus(next);
            }
          }
          schedule(next);
        })
        .catch(() => {
          if (mutationEpoch.current !== epoch) {
            return;
          }
          if (selfStopDisconnectExpected.current) {
            setPollEnabled(false);
            return;
          }
          schedule(null);
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

  return {
    mutationEpoch,
    pollSuppressed,
    restartPolling,
    selfStopDisconnectExpected,
  };
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: lifecycle orchestration is intentionally centralized
export function RemoteAccessSection() {
  const t = useT();
  const [status, setStatus] = useState<RemoteAccessStatus | null>(null);
  const [busy, setBusy] = useState<RemoteAccessOperation | null>(null);
  const [hostname, setHostname] = useState("");
  const [requestError, setRequestError] = useState<string | null>(null);
  const requestErrorBaseline = useRef<{
    operation: RemoteAccessRequestAxis;
    revision: string;
  } | null>(null);
  const currentStatus = useRef<RemoteAccessStatus | null>(null);
  const applyStatus = useCallback((next: RemoteAccessStatus) => {
    if (
      remoteAccessShouldClearRequestError(requestErrorBaseline.current, next)
    ) {
      requestErrorBaseline.current = null;
      setRequestError(null);
    }
    currentStatus.current = next;
    setStatus(next);
    usePlatformStore.setState({ cloudflareUrl: remoteAccessUsableUrl(next) });
  }, []);
  const {
    mutationEpoch,
    pollSuppressed,
    restartPolling,
    selfStopDisconnectExpected,
  } = useRemoteAccessPolling(applyStatus);

  // A password change rotates credentials outside this section's requests;
  // discard any in-flight poll and re-read so the block resolves at once.
  const refreshStatus = useCallback(() => {
    mutationEpoch.current += 1;
    restartPolling();
  }, [mutationEpoch, restartPolling]);

  const perform = async (
    requestAxis: RemoteAccessRequestAxis,
    request: () => Promise<RemoteAccessStatus>,
    pausePollingAfterSuccess = false,
  ) => {
    const operation: RemoteAccessOperation =
      requestAxis === "start:temporary" || requestAxis === "start:custom"
        ? "start"
        : requestAxis;
    mutationEpoch.current += 1;
    pollSuppressed.current = true;
    setBusy(operation);
    requestErrorBaseline.current = null;
    setRequestError(null);
    try {
      applyStatus(await request());
      if (pausePollingAfterSuccess) {
        selfStopDisconnectExpected.current = true;
      }
      return true;
    } catch (error) {
      requestErrorBaseline.current = {
        operation: requestAxis,
        revision: remoteAccessOperationRevision(
          currentStatus.current,
          requestAxis,
        ),
      };
      setRequestError(localizedRequestError(error, t));
      return false;
    } finally {
      setBusy(null);
      pollSuppressed.current = false;
      // A stop through the tunnel latches polling off when its own origin dies.
      // Any later action means the user is on a reachable origin, so resume.
      restartPolling();
    }
  };

  const start = () => {
    const kind = remoteAccessPreferredKind(status);
    return perform(`start:${kind}`, startRemoteAccess);
  };
  const stop = () =>
    perform(
      "stop",
      stopRemoteAccess,
      remoteAccessStopDisconnectsOrigin(
        status?.url ?? null,
        typeof window === "undefined" ? "" : window.location.origin,
      ),
    );
  const setAutoStart = (enabled: boolean) =>
    perform("auto", () => updateRemoteAccessAutoStart(enabled));
  const setMethod = (method: RemoteAccessStatus["method"]) =>
    perform("method", () => updateRemoteAccessMethod(method));
  const provision = (targetHostname: string) =>
    perform("provision", () => provisionCustomRemoteAccess(targetHostname));
  const cancel = (expectedRevision: number) =>
    perform("cancel", () => cancelCustomRemoteAccess(expectedRevision));
  const teardown = () => perform("teardown", teardownCustomRemoteAccess);

  const blockMessage = localizedBlockMessage(
    status?.blockReason ?? null,
    isTauri,
    t,
  );
  const autoStartMessage =
    status?.autoStartBlockReason === "custom_tunnel_not_configured"
      ? t("settings.general.remoteAccess.autoStartCustomMissing")
      : localizedBlockMessage(status?.autoStartBlockReason ?? null, isTauri, t);
  const stopAction = remoteAccessUsesStopAction(status);
  const actionDisabled = remoteAccessHeaderActionDisabled(
    status,
    busy !== null,
  );
  const actionLabel = stopAction
    ? t("settings.general.remoteAccess.stopAction")
    : t("settings.general.remoteAccess.startAction");
  const showHeaderAction =
    stopAction ||
    remoteAccessPreferredKind(status) === "temporary" ||
    status?.customRunnable === true;

  return (
    <section
      data-settings-label={t("settings.general.remoteAccess.sectionTitle")}
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <HugeiconsIcon
              icon={Globe02Icon}
              className="size-4 text-foreground"
            />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-base font-semibold font-heading text-foreground">
                {t("settings.general.remoteAccess.sectionTitle")}
              </h2>
              <ConnectionProgress status={status} />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {t("settings.general.remoteAccess.description")}
            </p>
          </div>
        </div>
        {showHeaderAction ? (
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
        ) : null}
      </div>

      <StatusMessage
        message={
          blockMessage ?? requestError ?? localizedRuntimeError(status, t)
        }
        destructive={!blockMessage}
      />
      <output className="sr-only" aria-live="polite">
        {remoteAccessUsableUrl(status)
          ? t("settings.general.remoteAccess.remoteUrlReady")
          : ""}
      </output>
      <RemoteUrlPanel status={status} />

      <div className="border-t border-border/60 px-4 py-1">
        <SettingsRow
          label={t("settings.general.remoteAccess.methodLabel")}
          description={t("settings.general.remoteAccess.methodDescription")}
        >
          <RemoteAccessMethodControl
            method={status?.method ?? "temporary"}
            disabled={
              busy !== null ||
              remoteAccessAutoStartReadOnly(status) ||
              remoteAccessCustomOperationInFlight(status)
            }
            onChange={setMethod}
          />
        </SettingsRow>
      </div>

      {remoteAccessShowsCustomPanel(status) && status ? (
        <CustomTunnelPanel
          status={status}
          hostname={hostname}
          setHostname={setHostname}
          busy={busy}
          onProvision={provision}
          onCancel={cancel}
          onTeardown={teardown}
          requestError={requestError}
        />
      ) : null}

      <div className="border-t border-border/60 px-4 py-1">
        <RemotePasswordRow status={status} onDone={refreshStatus} />

        <SettingsRow
          label={t("settings.general.remoteAccess.autoStartLabel")}
          description={
            remoteAccessAutoStartKind(status) === "custom"
              ? t("settings.general.remoteAccess.autoStartCustom")
              : t("settings.general.remoteAccess.autoStartTemporary")
          }
        >
          <Switch
            checked={status?.autoStart ?? false}
            disabled={
              busy !== null ||
              remoteAccessAutoStartReadOnly(status) ||
              remoteAccessCustomOperationInFlight(status)
            }
            onCheckedChange={setAutoStart}
            aria-label={t("settings.general.remoteAccess.autoStartLabel")}
          />
        </SettingsRow>
        <StatusMessage message={autoStartMessage} />
      </div>
    </section>
  );
}
