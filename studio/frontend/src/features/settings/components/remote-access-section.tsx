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
} from "@/features/settings/api/remote-access";
import {
  type RemoteAccessBlockMessageId,
  type RemoteAccessOperation,
  type RemoteAccessRequestAxis,
  type RemoteAccessStatus,
  remoteAccessAutoStartKind,
  remoteAccessAutoStartReadOnly,
  remoteAccessBlockMessageId,
  remoteAccessDnsConflictHostname,
  remoteAccessHeaderActionDisabled,
  remoteAccessCustomActionsDisabled,
  remoteAccessCustomOperationInFlight,
  remoteAccessCustomReadiness,
  remoteAccessCustomTeardownMessageId,
  remoteAccessOperationRevision,
  remoteAccessPollDelay,
  remoteAccessPreferredKind,
  remoteAccessRequestMessageId,
  remoteAccessSelfStopPoll,
  remoteAccessShouldClearRequestError,
  remoteAccessStopDisconnectsOrigin,
  remoteAccessTeardownNeedsLocalOrigin,
  remoteAccessUsableUrl,
  remoteAccessUsesStopAction,
} from "@/features/settings/api/remote-access-state";
import { type TranslationKey, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  Copy01Icon,
  Globe02Icon,
  QrCodeIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import QRCode from "react-qr-code";
import { ChangePasswordDialog } from "./change-password-dialog";
import { SettingsRow } from "./settings-row";

type Translate = ReturnType<typeof useT>;

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
    case "tunnel_delete_failed":
    case "teardown_failed":
      return t("settings.general.remoteAccess.errorTeardownFailed");
    default:
      return t("settings.general.remoteAccess.errorUnknown");
  }
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

function stateDotClass(state?: RemoteAccessStatus["state"]): string {
  if (state === "online") {
    return "bg-emerald-500";
  }
  if (state === "starting" || state === "stopping") {
    return "animate-pulse bg-blue-500";
  }
  return state === "error" ? "bg-red-500" : "bg-muted-foreground";
}

function AccessStatus({ status }: { status: RemoteAccessStatus | null }) {
  const t = useT();
  const labels: Record<RemoteAccessStatus["state"], string> = {
    off: t("settings.general.remoteAccess.stateOff"),
    starting: t("settings.general.remoteAccess.stateStarting"),
    online: t("settings.general.remoteAccess.stateOnline"),
    stopping: t("settings.general.remoteAccess.stateStopping"),
    error: t("common.error"),
  };
  const owners = {
    launch: t("settings.general.remoteAccess.ownerLaunch"),
    settings: t("settings.general.remoteAccess.ownerSettings"),
    colab: t("settings.general.remoteAccess.ownerColab"),
  };
  const owner = status?.managedBy ? owners[status.managedBy] : null;
  return (
    <output
      className="flex items-center gap-1.5 text-xs text-muted-foreground"
      aria-live="polite"
    >
      <span
        className={cn("size-2 rounded-full", stateDotClass(status?.state))}
      />
      {status
        ? labels[status.state]
        : t("settings.general.remoteAccess.unavailable")}
      {owner ? ` · ${owner}` : ""}
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

function ReadinessItem({
  label,
  ready,
  active,
  waitingLabel,
}: {
  label: string;
  ready: boolean;
  active: boolean;
  waitingLabel?: string;
}) {
  const t = useT();
  return (
    <li className="flex items-center justify-between gap-3 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className={ready ? "text-emerald-600" : "text-muted-foreground"}>
        {ready
          ? t("settings.general.remoteAccess.ready")
          : active
            ? (waitingLabel ?? t("settings.general.remoteAccess.waiting"))
            : t("settings.general.remoteAccess.stateOff")}
      </span>
    </li>
  );
}

function CustomTunnelSetup({
  status,
  hostname,
  setHostname,
  busy,
  onProvision,
}: {
  status: RemoteAccessStatus;
  hostname: string;
  setHostname: (hostname: string) => void;
  busy: RemoteAccessOperation | null;
  onProvision: () => void;
}) {
  const t = useT();
  const disabled = remoteAccessCustomActionsDisabled(status, busy !== null);
  return (
    <form
      className="mt-3 flex flex-col gap-2 sm:flex-row"
      onSubmit={(event) => {
        event.preventDefault();
        onProvision();
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
        {busy === "provision"
          ? t("settings.general.remoteAccess.settingUp")
          : t("settings.general.remoteAccess.setupAction")}
      </Button>
    </form>
  );
}

function CustomTunnelProgress({
  status,
  busy,
  onCancel,
}: {
  status: RemoteAccessStatus;
  busy: RemoteAccessOperation | null;
  onCancel: () => void;
}) {
  const t = useT();
  return (
    <div
      className="mt-3 rounded-md border border-border bg-muted/30 p-3"
      aria-live="polite"
      aria-atomic="true"
    >
      <p className="text-xs leading-snug text-muted-foreground">
        {status.customState === "tearing_down"
          ? t("settings.general.remoteAccess.teardownProgress")
          : t("settings.general.remoteAccess.provisionProgress")}
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        {status.loginUrl ? (
          <Button asChild={true} type="button" size="sm">
            <a href={status.loginUrl} target="_blank" rel="noreferrer">
              {t("settings.general.remoteAccess.openCloudflare")}
            </a>
          </Button>
        ) : null}
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={onCancel}
          disabled={busy !== null}
        >
          {t("common.cancel")}
        </Button>
      </div>
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
  const readiness = remoteAccessCustomReadiness(status);
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
      <code className="break-all rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs text-foreground">
        {status.customHostname}
      </code>
      <ul className="flex flex-col gap-1.5">
        <ReadinessItem
          label={t("settings.general.remoteAccess.connectorReady")}
          ready={readiness.connectorReady}
          active={readiness.active}
        />
        <ReadinessItem
          label={t("settings.general.remoteAccess.tunnelReady")}
          ready={readiness.tunnelReady}
          active={readiness.active}
        />
        <ReadinessItem
          label={t("settings.general.remoteAccess.dnsReady")}
          ready={readiness.dnsReady}
          active={readiness.active}
          waitingLabel={
            status.dns === "pending"
              ? t("settings.general.remoteAccess.dnsPending")
              : t("settings.general.remoteAccess.dnsUnknown")
          }
        />
      </ul>
      {readiness.active && status.dns !== "resolved" ? (
        <p className="text-xs leading-snug text-muted-foreground">
          {t("settings.general.remoteAccess.dnsLinkWaiting")}
        </p>
      ) : null}
      <AlertDialog>
        <AlertDialogTrigger asChild={true}>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="self-start text-destructive hover:text-destructive"
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
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={onTeardown}>
              {t("settings.general.remoteAccess.removeConfirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
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
      {status.orphanedHostnames.length > 0 ? (
        <p className="mt-3 text-xs leading-snug text-muted-foreground">
          {t("settings.general.remoteAccess.orphanedRecords", {
            hostnames: status.orphanedHostnames.join(", "),
          })}
        </p>
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
}: {
  status: RemoteAccessStatus;
  hostname: string;
  setHostname: (hostname: string) => void;
  busy: RemoteAccessOperation | null;
  onProvision: () => void;
  onCancel: () => void;
  onTeardown: () => void;
}) {
  const t = useT();
  const operating =
    status.customState === "provisioning" ||
    status.customState === "tearing_down";
  const hasIdentity = status.customHostname !== null;

  return (
    <div className="border-t border-border/60 p-4">
      <div className="flex flex-col gap-1">
        <h3 className="text-sm font-medium text-foreground">
          {t("settings.general.remoteAccess.stableTitle")}
        </h3>
        <p className="text-xs leading-snug text-muted-foreground">
          {t("settings.general.remoteAccess.stableDescription")}
        </p>
      </div>

      {hasIdentity || operating ? null : (
        <CustomTunnelSetup
          status={status}
          hostname={hostname}
          setHostname={setHostname}
          busy={busy}
          onProvision={onProvision}
        />
      )}

      {operating ? (
        <CustomTunnelProgress status={status} busy={busy} onCancel={onCancel} />
      ) : null}

      {hasIdentity && !operating ? (
        <CustomTunnelIdentity
          status={status}
          busy={busy}
          onTeardown={onTeardown}
        />
      ) : null}
      <CustomTunnelMessages status={status} />
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
    } catch (error) {
      requestErrorBaseline.current = {
        operation: requestAxis,
        revision: remoteAccessOperationRevision(
          currentStatus.current,
          requestAxis,
        ),
      };
      setRequestError(localizedRequestError(error, t));
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
    return perform(`start:${kind}`, () => startRemoteAccess(kind));
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
    perform("auto", () =>
      updateRemoteAccessAutoStart(enabled, remoteAccessAutoStartKind(status)),
    );
  const provision = () =>
    perform("provision", () => provisionCustomRemoteAccess(hostname.trim()));
  const cancel = () => perform("cancel", cancelCustomRemoteAccess);
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
  const actionLabel =
    busy === "start"
      ? t("settings.general.remoteAccess.actionStarting")
      : busy === "stop"
        ? t("settings.general.remoteAccess.actionStopping")
        : stopAction
          ? t("settings.general.remoteAccess.stopAction")
          : remoteAccessPreferredKind(status) === "custom"
            ? t("settings.general.remoteAccess.connectAction")
            : t("settings.general.remoteAccess.temporaryAction");

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
          blockMessage ??
          requestError ??
          (status?.error
            ? t("settings.general.remoteAccess.runtimeFailed")
            : null)
        }
        destructive={!blockMessage}
      />
      <output className="sr-only" aria-live="polite">
        {remoteAccessUsableUrl(status)
          ? t("settings.general.remoteAccess.remoteUrlReady")
          : ""}
      </output>
      <RemoteUrlPanel status={status} />

      {status ? (
        <CustomTunnelPanel
          status={status}
          hostname={hostname}
          setHostname={setHostname}
          busy={busy}
          onProvision={provision}
          onCancel={cancel}
          onTeardown={teardown}
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
