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
  type RemoteAccessRequestMessageId,
  type RemoteAccessOperation,
  type RemoteAccessProgressStepId,
  type RemoteAccessRequestAxis,
  type RemoteAccessStatus,
  remoteAccessAuthorizationAction,
  remoteAccessAuthorizationShouldOpen,
  remoteAccessAuthorizationView,
  remoteAccessAutoStartReadOnly,
  remoteAccessBlockMessageId,
  remoteAccessCustomActionsDisabled,
  remoteAccessCustomOperationInFlight,
  remoteAccessCustomTeardownMessageId,
  remoteAccessDnsConflictHostname,
  remoteAccessHeaderActionDisabled,
  remoteAccessHeaderStatus,
  remoteAccessOperationRevision,
  remoteAccessPollDelay,
  remoteAccessPreferredKind,
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
  ArrowUpRight01Icon,
  Copy01Icon,
  Delete02Icon,
  Globe02Icon,
  QrCodeIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { SegmentedControl } from "@/components/segmented-control";
import { Loader2Icon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import QRCode from "react-qr-code";
import { ChangePasswordDialog } from "./change-password-dialog";
import { SettingsRow } from "./settings-row";

type Translate = ReturnType<typeof useT>;

const CLOUDFLARE_TUNNELS_URL =
  "https://dash.cloudflare.com/?to=/:account/tunnels";
const CLOUDFLARE_DNS_URL =
  "https://dash.cloudflare.com/?to=/:account/:zone/dns/records";

const MESSAGE_KEYS: Record<RemoteAccessRequestMessageId, TranslationKey> = {
  invalidHostname: "settings.general.remoteAccess.errorInvalidHostname",
  busy: "settings.general.remoteAccess.errorBusy",
  requestFailed: "settings.general.remoteAccess.requestFailed",
  serverStarting: "settings.general.remoteAccess.blockServerStarting",
  passwordDesktop: "settings.general.remoteAccess.blockPasswordDesktop",
  passwordWeb: "settings.general.remoteAccess.blockPasswordWeb",
  explicitlyDisabled: "settings.general.remoteAccess.blockExplicitlyDisabled",
  launchManaged: "settings.general.remoteAccess.blockLaunchManaged",
  colabManaged: "settings.general.remoteAccess.blockColabManaged",
  colab: "settings.general.remoteAccess.blockColab",
  customNotConfigured: "settings.general.remoteAccess.blockCustomNotConfigured",
};

function localizedBlockMessage(
  reason: string | null,
  isDesktop: boolean,
  t: Translate,
): string | null {
  const messageId = remoteAccessBlockMessageId(reason, isDesktop);
  return messageId ? t(MESSAGE_KEYS[messageId]) : null;
}

// Codes whose message is a plain lookup. The three that interpolate a value,
// and the teardown pre-check, stay in the function below.
const CUSTOM_ERROR_KEYS: Record<string, TranslationKey> = {
  // biome-ignore lint/style/useNamingConvention: backend error codes
  certificate_state_busy: "settings.general.remoteAccess.errorBusy",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  connector_in_use: "settings.general.remoteAccess.errorBusy",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  identity_exists: "settings.general.remoteAccess.errorIdentityExists",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  cloudflared_unreachable: "settings.general.remoteAccess.errorUnavailable",
  cancelled: "settings.general.remoteAccess.errorCancelled",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  login_timed_out: "settings.general.remoteAccess.errorLoginTimeout",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  login_failed: "settings.general.remoteAccess.errorLoginFailed",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  invalid_hostname: "settings.general.remoteAccess.errorInvalidHostname",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  setup_record_unreadable: "settings.general.remoteAccess.errorSetupFailed",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  tunnel_create_failed: "settings.general.remoteAccess.errorSetupFailed",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  route_failed: "settings.general.remoteAccess.errorSetupFailed",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  connector_stop_failed: "settings.general.remoteAccess.errorTeardownFailed",
  // biome-ignore lint/style/useNamingConvention: backend error codes
  teardown_failed: "settings.general.remoteAccess.errorTeardownFailed",
};

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
  const code = status.customError;
  if (code === null) {
    return null;
  }
  if (code === "dns_conflict") {
    const hostname = remoteAccessDnsConflictHostname(status);
    return hostname
      ? t("settings.general.remoteAccess.dnsConflict", { hostname })
      : t("settings.general.remoteAccess.dnsConflictUnknown");
  }
  if (code === "hostname_not_authorized") {
    return t("settings.general.remoteAccess.errorWrongDomain", {
      hostname: status.customErrorDetail ?? status.customHostname ?? "",
    });
  }
  if (code === "certificate_exists") {
    return t("settings.general.remoteAccess.errorCertificateExists", {
      path: status.customErrorDetail ?? "~/.cloudflared/cert.pem",
    });
  }
  return t(
    CUSTOM_ERROR_KEYS[code] ?? "settings.general.remoteAccess.errorUnknown",
  );
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
  return t(MESSAGE_KEYS[remoteAccessRequestMessageId(code, isTauri)]);
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

const STATE_MESSAGE_KEYS: Record<RemoteAccessStatus["state"], TranslationKey> =
  {
    off: "settings.general.remoteAccess.stateOff",
    starting: "settings.general.remoteAccess.stateStarting",
    online: "settings.general.remoteAccess.stateOnline",
    stopping: "settings.general.remoteAccess.stateStopping",
    error: "settings.general.remoteAccess.stateError",
  };

const OWNER_MESSAGE_KEYS: Record<
  Exclude<RemoteAccessStatus["managedBy"], "settings" | null>,
  TranslationKey
> = {
  launch: "settings.general.remoteAccess.ownerLaunch",
  colab: "settings.general.remoteAccess.ownerColab",
};

function stateDotClass(state?: RemoteAccessStatus["state"]): string {
  if (state === "online") {
    return "bg-emerald-500";
  }
  if (state === "starting" || state === "stopping") {
    return "animate-pulse bg-blue-500";
  }
  return state === "error" ? "bg-red-500" : "bg-muted-foreground";
}

function ExternalArrow() {
  return <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />;
}

function openAuthorizedRemoteLink(
  url: string | null,
  authorized: boolean,
  openTauri: (url: string, reopen: boolean) => void,
) {
  if (!(url && authorized)) {
    return;
  }
  if (isTauri) {
    openTauri(url, true);
    return;
  }
  openLink(url);
}

function CustomAuthorizationAction({
  authorized,
  loginUrl,
  authorizationCurrent,
  confirmationDisabled,
  openTauri,
  onConfirm,
}: {
  authorized: boolean;
  loginUrl: string | null;
  authorizationCurrent: boolean;
  confirmationDisabled: boolean;
  openTauri: (url: string, reopen: boolean) => void;
  onConfirm: () => void;
}) {
  const t = useT();
  const action = remoteAccessAuthorizationAction(
    authorized,
    loginUrl,
    authorizationCurrent,
  );
  if (action !== "confirm") {
    // Cloudflare has to hand back the authorization URL before there is anything
    // to open, so the wait is shown rather than left as a button that looks ready
    // and does nothing. Once the URL exists this is a real link, so it can be
    // opened, copied or moved to another browser like any other.
    if (action === "preparing") {
      return (
        <Button type="button" disabled={true}>
          <Loader2Icon
            aria-hidden="true"
            className="size-3.5 shrink-0 animate-spin"
          />
          {t("settings.general.remoteAccess.openCloudflare")}
        </Button>
      );
    }
    return (
      <Button asChild={true} type="button">
        <a
          // "open" is reached only with a URL; the fallback is for the type.
          href={loginUrl ?? undefined}
          target="_blank"
          rel="noreferrer"
          onClick={(event) => {
            if (!isTauri) {
              return;
            }
            event.preventDefault();
            openAuthorizedRemoteLink(loginUrl, authorizationCurrent, openTauri);
          }}
        >
          {t("settings.general.remoteAccess.openCloudflare")}
          <ExternalArrow />
        </a>
      </Button>
    );
  }
  return (
    <Button type="button" disabled={confirmationDisabled} onClick={onConfirm}>
      {t("settings.general.remoteAccess.setupConfirmAction")}
    </Button>
  );
}

function AccessStatus({ status }: { status: RemoteAccessStatus | null }) {
  const t = useT();
  const header = remoteAccessHeaderStatus(status);
  const owner = header.owner ? t(OWNER_MESSAGE_KEYS[header.owner]) : null;
  return (
    <output
      className="flex items-center gap-1.5 text-xs text-muted-foreground"
      aria-live="polite"
    >
      <span
        className={cn("size-2 rounded-full", stateDotClass(status?.state))}
      />
      <span>
        {header.state
          ? t(STATE_MESSAGE_KEYS[header.state])
          : t("settings.general.remoteAccess.stateUnavailable")}
        {header.step ? ` · ${t(PROGRESS_MESSAGE_KEYS[header.step])}` : ""}
        {owner ? ` · ${owner}` : ""}
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
  return (
    <SegmentedControl
      value={method}
      options={[
        {
          value: "temporary" as const,
          label: t("settings.general.remoteAccess.temporaryMethod"),
          disabled,
        },
        {
          value: "custom" as const,
          label: t("settings.general.remoteAccess.customMethod"),
          disabled,
        },
      ]}
      onValueChange={onChange}
      ariaLabel={t("settings.general.remoteAccess.methodLabel")}
      size="compact"
      className="w-auto"
    />
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
  const [authorizationGrant, setAuthorizationGrant] = useState<
    number | "pending" | null
  >(null);
  const openedLoginUrl = useRef<string | null>(null);
  const pendingLoginUrl = useRef<string | null>(null);
  const authorizationAttempt = useRef(0);
  const authorizationRevisionRef = useRef<number | null>(null);
  const confirmOpenRef = useRef(confirmOpen);
  const statusRef = useRef(status);
  confirmOpenRef.current = confirmOpen;
  statusRef.current = status;
  const cancelledSetupRevision = useRef<number | null>(null);
  const disabled = remoteAccessCustomActionsDisabled(status, busy !== null);
  const waiting = busy === "provision" || status.customState === "provisioning";
  const resetAuthorization = useCallback(() => {
    authorizationAttempt.current += 1;
    authorizationRevisionRef.current = null;
    pendingLoginUrl.current = null;
    setAuthorizationGrant(null);
  }, []);
  const openAuthorizedTauriUrl = useCallback(
    (loginUrl: string, reopen = false) => {
      if (pendingLoginUrl.current === loginUrl) {
        return;
      }
      const attempt = authorizationAttempt.current;
      pendingLoginUrl.current = loginUrl;
      import("@tauri-apps/plugin-opener").then(({ openUrl }) => {
        const latestStatus = statusRef.current;
        if (
          authorizationAttempt.current !== attempt ||
          pendingLoginUrl.current !== loginUrl ||
          !remoteAccessAuthorizationShouldOpen(
            latestStatus,
            confirmOpenRef.current,
            authorizationRevisionRef.current,
            reopen ? null : openedLoginUrl.current,
            cancelledSetupRevision.current,
          )
        ) {
          if (pendingLoginUrl.current === loginUrl) {
            pendingLoginUrl.current = null;
          }
          return;
        }
        pendingLoginUrl.current = null;
        openedLoginUrl.current = loginUrl;
        openUrl(loginUrl).catch(console.error);
      });
    },
    [],
  );

  useEffect(() => {
    return () => {
      authorizationAttempt.current += 1;
      pendingLoginUrl.current = null;
    };
  }, []);

  useEffect(() => {
    if (authorizationGrant === null) {
      return;
    }
    if (authorizationGrant === "pending") {
      if (status.customState === "provisioning") {
        authorizationRevisionRef.current = status.customOperationRevision;
        setAuthorizationGrant(status.customOperationRevision);
      } else if (busy === null) {
        resetAuthorization();
      }
      return;
    }
    if (authorizationGrant === status.customOperationRevision) {
      return;
    }
    resetAuthorization();
  }, [
    authorizationGrant,
    busy,
    resetAuthorization,
    status.customState,
    status.customOperationRevision,
  ]);

  useEffect(() => {
    if (status.customState !== "provisioning") {
      if (cancelledSetupRevision.current !== null && busy === null) {
        cancelledSetupRevision.current = null;
        setConfirmOpen(false);
      }
      if (busy === null) {
        resetAuthorization();
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
  }, [busy, resetAuthorization, status]);

  useEffect(() => {
    const loginUrl = status.loginUrl;
    if (
      !remoteAccessAuthorizationShouldOpen(
        status,
        confirmOpen,
        typeof authorizationGrant === "number" ? authorizationGrant : null,
        openedLoginUrl.current,
        cancelledSetupRevision.current,
      ) ||
      loginUrl === null ||
      pendingLoginUrl.current === loginUrl
    ) {
      return;
    }
    if (isTauri) {
      openAuthorizedTauriUrl(loginUrl);
      return;
    }
    openLink(loginUrl);
    openedLoginUrl.current = loginUrl;
  }, [authorizationGrant, confirmOpen, openAuthorizedTauriUrl, status]);

  const cancelSetup = () => {
    resetAuthorization();
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

  const confirmAuthorization = () => {
    const loginUrl = status.loginUrl;
    const revision =
      status.customState === "provisioning"
        ? status.customOperationRevision
        : null;
    authorizationAttempt.current += 1;
    authorizationRevisionRef.current = revision;
    setAuthorizationGrant(revision ?? "pending");
    openedLoginUrl.current = null;
    if (!loginUrl) {
      return;
    }
    if (isTauri) {
      openAuthorizedTauriUrl(loginUrl);
      return;
    }
    openLink(loginUrl);
    openedLoginUrl.current = loginUrl;
  };
  const authorizationView = remoteAccessAuthorizationView(
    status,
    busy === "provision",
    authorizationGrant,
    cancelledSetupRevision.current,
  );
  const authorizationMessage =
    authorizationView.phase === "approval"
      ? t("settings.general.remoteAccess.setupAuthorize", {
          hostname: requestedHostname,
        })
      : t("settings.general.remoteAccess.setupPreparing");

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
            resetAuthorization();
            setConfirmOpen(true);
            onProvision(targetHostname).then((started) => {
              if (!started) {
                resetAuthorization();
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
            if (!open) {
              resetAuthorization();
            }
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
              {authorizationView.phase ? (
                <span
                  aria-atomic="true"
                  aria-live="polite"
                  className="flex items-center gap-2"
                >
                  <Loader2Icon
                    aria-hidden="true"
                    className="size-3.5 shrink-0 animate-spin"
                  />
                  {authorizationMessage}
                </span>
              ) : null}
              {requestError || status.customError ? (
                <span className="block text-destructive" role="alert">
                  {requestError ?? localizedCustomError(status, t)}
                </span>
              ) : null}
              {!requestError && status.customError === "dns_conflict" ? (
                <Button
                  asChild={true}
                  type="button"
                  size="sm"
                  variant="outline"
                  className="mt-2"
                >
                  <a href={CLOUDFLARE_DNS_URL} target="_blank" rel="noreferrer">
                    {t("settings.general.remoteAccess.openCloudflareDns")}
                    <ExternalArrow />
                  </a>
                </Button>
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
            <CustomAuthorizationAction
              authorized={authorizationGrant !== null}
              loginUrl={status.loginUrl}
              authorizationCurrent={authorizationView.current}
              confirmationDisabled={authorizationView.confirmationDisabled}
              openTauri={openAuthorizedTauriUrl}
              onConfirm={confirmAuthorization}
            />
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
              variant="outline"
              className="shrink-0 text-destructive hover:text-destructive hover:border-destructive/60"
              disabled={
                remoteAccessCustomActionsDisabled(status, busy !== null) ||
                teardownDisconnectsOrigin
              }
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5 mr-1.5" />
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
                    <ExternalArrow />
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
                    <ExternalArrow />
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
      {provisioning || !status.customError ? null : (
        <div
          className="mt-3 text-xs leading-snug text-destructive"
          role="alert"
        >
          <p>{localizedCustomError(status, t)}</p>
        </div>
      )}
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
              <AccessStatus status={status} />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {t("settings.general.remoteAccess.description")}
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
            remoteAccessPreferredKind(status) === "custom"
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
