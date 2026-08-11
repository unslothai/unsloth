// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type RemoteAccessState =
  | "off"
  | "starting"
  | "online"
  | "stopping"
  | "error";
export type RemoteAccessOwner = "launch" | "settings" | "colab" | null;
export type RemoteAccessKind = "temporary" | "custom";
export type CustomTunnelState =
  | "unconfigured"
  | "provisioning"
  | "configured"
  | "tearing_down"
  | "error";
export type RemoteAccessDns = "unknown" | "pending" | "resolved";
export type RemoteAccessProgressStepId =
  | "connecting"
  | "openingLink"
  | "checkingHostname"
  | "disconnecting";
export type RemoteAccessOperation =
  | "start"
  | "stop"
  | "auto"
  | "method"
  | "provision"
  | "cancel"
  | "teardown";
export type RemoteAccessRequestAxis =
  | Exclude<RemoteAccessOperation, "start">
  | `start:${RemoteAccessKind}`;
export type CustomTunnelErrorPhase = "provision" | "teardown";
export type CustomTunnelTeardownMessageId = "teardownFailed" | "teardownManual";
export type RemoteAccessBlockMessageId =
  | "serverStarting"
  | "passwordDesktop"
  | "passwordWeb"
  | "explicitlyDisabled"
  | "launchManaged"
  | "colabManaged"
  | "colab";
export type RemoteAccessRequestMessageId =
  | RemoteAccessBlockMessageId
  | "invalidHostname"
  | "busy"
  | "requestFailed";
const TRAILING_SLASHES_RE = /\/+$/;
const DNS_CONFLICT_DETAIL_RE =
  /^A DNS record for ([^ ]+) already exists(?: |$)/;

export type RemoteAccessStatus = {
  state: RemoteAccessState;
  url: string | null;
  error: string | null;
  autoStart: boolean;
  method: RemoteAccessKind;
  defaultAutoStart: boolean;
  available: boolean;
  managedBy: RemoteAccessOwner;
  canStart: boolean;
  canStop: boolean;
  blockReason: string | null;
  passwordPending: boolean;
  streamingSupported: boolean;
  kind: RemoteAccessKind | null;
  connectorRegistered: boolean;
  tunnelServing: boolean;
  dns: RemoteAccessDns;
  autoStartKind: RemoteAccessKind | null;
  autoStartBlockReason: string | null;
  customState: CustomTunnelState;
  customOperationRevision: number;
  customHostname: string | null;
  customTunnelName: string | null;
  customRunnable: boolean;
  loginUrl: string | null;
  customError: string | null;
  customErrorDetail: string | null;
  customErrorPhase: CustomTunnelErrorPhase | null;
  customErrorSettled: boolean;
};

export type ApiRemoteAccessStatus = {
  state: RemoteAccessState;
  url?: string | null;
  error?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_start: boolean;
  method?: RemoteAccessKind;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_auto_start: boolean;
  available: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  managed_by?: RemoteAccessOwner;
  // biome-ignore lint/style/useNamingConvention: API schema
  can_start: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  can_stop: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  block_reason?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  password_pending?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  streaming_supported: boolean;
  kind?: RemoteAccessKind | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  connector_registered?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  tunnel_serving?: boolean;
  dns?: RemoteAccessDns;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_start_kind?: RemoteAccessKind | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_start_block_reason?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_state?: CustomTunnelState;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_operation_revision?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_hostname?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_tunnel_name?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_runnable?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  login_url?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_error?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_error_detail?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_error_phase?: CustomTunnelErrorPhase | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_error_settled?: boolean;
};

export function normalizeRemoteAccessStatus(
  status: ApiRemoteAccessStatus,
): RemoteAccessStatus {
  const { method = "temporary", custom_tunnel_name: customTunnelName = null } =
    status;
  return {
    state: status.state,
    url: status.url ?? null,
    error: status.error ?? null,
    autoStart: status.auto_start,
    method,
    defaultAutoStart: status.default_auto_start,
    available: status.available,
    managedBy: status.managed_by ?? null,
    canStart: status.can_start,
    canStop: status.can_stop,
    blockReason: status.block_reason ?? null,
    passwordPending: status.password_pending === true,
    streamingSupported: status.streaming_supported,
    kind: status.kind ?? null,
    connectorRegistered: status.connector_registered === true,
    tunnelServing: status.tunnel_serving === true,
    dns: status.dns ?? "unknown",
    autoStartKind: status.auto_start_kind ?? null,
    autoStartBlockReason: status.auto_start_block_reason ?? null,
    customState: status.custom_state ?? "unconfigured",
    customOperationRevision: status.custom_operation_revision ?? 0,
    customHostname: status.custom_hostname ?? null,
    customTunnelName,
    customRunnable: status.custom_runnable === true,
    loginUrl: status.login_url ?? null,
    customError: status.custom_error ?? null,
    customErrorDetail: status.custom_error_detail ?? null,
    customErrorPhase: status.custom_error_phase ?? null,
    customErrorSettled: status.custom_error_settled === true,
  };
}

export function remoteAccessPollDelay(
  status: RemoteAccessStatus | null,
): number {
  return status?.state === "starting" ||
    status?.state === "stopping" ||
    remoteAccessCustomOperationInFlight(status)
    ? 1000
    : 5000;
}

export function remoteAccessCustomOperationInFlight(
  status: RemoteAccessStatus | null,
): boolean {
  return (
    status?.customState === "provisioning" ||
    status?.customState === "tearing_down"
  );
}

export function remoteAccessShowsCustomPanel(
  status: RemoteAccessStatus | null,
): boolean {
  return (
    status?.method === "custom" || remoteAccessCustomOperationInFlight(status)
  );
}

export function remoteAccessSetupDialogShouldOpen(
  status: RemoteAccessStatus,
  cancelledRevision: number | null,
): boolean {
  return (
    status.customState === "provisioning" &&
    status.customOperationRevision !== cancelledRevision
  );
}

export function remoteAccessHeaderStatus(status: RemoteAccessStatus | null) {
  return {
    state: status?.state ?? null,
    owner:
      status?.managedBy === "settings" ? null : (status?.managedBy ?? null),
    step:
      status?.state === "starting" ? remoteAccessProgressStep(status) : null,
  };
}

export function remoteAccessAuthorizationShouldOpen(
  status: RemoteAccessStatus,
  dialogOpen: boolean,
  confirmedRevision: number | null,
  openedLoginUrl: string | null,
  cancelledRevision: number | null,
): boolean {
  return (
    dialogOpen &&
    confirmedRevision === status.customOperationRevision &&
    status.loginUrl !== null &&
    openedLoginUrl !== status.loginUrl &&
    cancelledRevision !== status.customOperationRevision
  );
}

export function remoteAccessAuthorizationView(
  status: RemoteAccessStatus,
  provisionRequested: boolean,
  grant: number | "pending" | null,
  cancelledRevision: number | null,
): {
  confirmationDisabled: boolean;
  current: boolean;
  phase: "connecting" | "approval" | null;
} {
  const operationAvailable =
    (provisionRequested || status.customState === "provisioning") &&
    cancelledRevision !== status.customOperationRevision;
  const current =
    typeof grant === "number" &&
    grant === status.customOperationRevision &&
    cancelledRevision !== status.customOperationRevision;
  return {
    confirmationDisabled: !operationAvailable,
    current,
    phase:
      (grant === "pending" || current) && operationAvailable
        ? status.loginUrl
          ? "approval"
          : "connecting"
        : null,
  };
}

type RemoteAccessLoginWindow = { opener: unknown };

export function openRemoteAccessLoginWindow(
  loginUrl: string,
  openWindow: (url: string, target: string) => RemoteAccessLoginWindow | null,
): void {
  const cloudflareWindow = openWindow(loginUrl, "_blank");
  if (cloudflareWindow) {
    cloudflareWindow.opener = null;
  }
}

export function remoteAccessPreferredKind(
  status: RemoteAccessStatus | null,
): RemoteAccessKind {
  return status?.method ?? "temporary";
}

export function remoteAccessUsableUrl(
  status: RemoteAccessStatus | null,
): string | null {
  if (status?.kind === "custom" && status.dns !== "resolved") {
    return null;
  }
  return status?.url ?? null;
}

export function remoteAccessIsReady(
  status: RemoteAccessStatus | null,
): boolean {
  if (status?.state !== "online") {
    return false;
  }
  return (
    status.kind !== "custom" ||
    (status.connectorRegistered &&
      status.tunnelServing &&
      status.dns === "resolved")
  );
}

export function remoteAccessProgressStep(
  status: RemoteAccessStatus | null,
): RemoteAccessProgressStepId | null {
  if (status?.state === "stopping") {
    return "disconnecting";
  }
  if (
    status?.state !== "starting" &&
    !(
      status?.kind === "custom" &&
      status.state === "online" &&
      !remoteAccessIsReady(status)
    )
  ) {
    return null;
  }
  if (status.kind === "custom") {
    if (!status.connectorRegistered) {
      return "connecting";
    }
    if (!status.tunnelServing) {
      return "openingLink";
    }
    return "checkingHostname";
  }
  return status.connectorRegistered && !status.tunnelServing
    ? "openingLink"
    : "connecting";
}

export function remoteAccessOperationRevision(
  status: RemoteAccessStatus | null,
  operation: RemoteAccessRequestAxis,
): string {
  if (status === null) {
    return "unavailable";
  }
  if (operation === "auto") {
    return JSON.stringify([
      status.autoStart,
      status.autoStartKind,
      status.autoStartBlockReason,
    ]);
  }
  if (operation === "method") {
    return status.method;
  }
  if (
    operation === "provision" ||
    operation === "cancel" ||
    operation === "teardown"
  ) {
    return JSON.stringify([
      status.customOperationRevision,
      status.customState,
      status.customHostname,
      status.customTunnelName,
      status.customRunnable,
      status.loginUrl,
      status.customError,
      status.customErrorDetail,
      status.customErrorPhase,
      status.customErrorSettled,
    ]);
  }
  const lifecycle = [
    status.state,
    status.url,
    status.error,
    status.managedBy,
    status.canStart,
    status.canStop,
    status.kind,
  ];
  return JSON.stringify(
    operation === "start:custom"
      ? [...lifecycle, status.customState, status.customRunnable]
      : lifecycle,
  );
}

export function remoteAccessShouldClearRequestError(
  baseline: { operation: RemoteAccessRequestAxis; revision: string } | null,
  next: RemoteAccessStatus,
): boolean {
  return (
    baseline !== null &&
    baseline.revision !==
      remoteAccessOperationRevision(next, baseline.operation)
  );
}

export function remoteAccessUsesStopAction(
  status: RemoteAccessStatus | null,
): boolean {
  return (
    status?.canStop === true ||
    status?.state === "starting" ||
    status?.state === "online" ||
    status?.state === "stopping"
  );
}

export function remoteAccessHeaderActionDisabled(
  status: RemoteAccessStatus | null,
  busy: boolean,
): boolean {
  const stopAction = remoteAccessUsesStopAction(status);
  if (busy) {
    return true;
  }
  if (stopAction) {
    return (
      status?.canStop !== true ||
      (status.kind === "custom" && remoteAccessCustomOperationInFlight(status))
    );
  }
  return (
    status?.canStart !== true ||
    (remoteAccessPreferredKind(status) === "custom" &&
      remoteAccessCustomOperationInFlight(status))
  );
}

export function remoteAccessCustomTeardownMessageId(
  status: RemoteAccessStatus,
): CustomTunnelTeardownMessageId | null {
  if (status.customErrorPhase !== "teardown") {
    return null;
  }
  return status.customErrorSettled ? "teardownManual" : "teardownFailed";
}

export function remoteAccessCustomActionsDisabled(
  status: RemoteAccessStatus,
  busy: boolean,
): boolean {
  return busy || !status.available || status.passwordPending;
}

export function remoteAccessAutoStartKind(
  status: RemoteAccessStatus | null,
): RemoteAccessKind {
  return status?.method ?? "temporary";
}

export function remoteApiOrigin(
  remoteUrl: string | null,
  localOrigin: string,
): string {
  return remoteUrl ?? localOrigin;
}

export function remoteAccessAutoStartReadOnly(
  status: RemoteAccessStatus | null,
): boolean {
  return (
    status === null ||
    status.managedBy === "colab" ||
    status.blockReason === "colab"
  );
}

export function remoteAccessStopDisconnectsOrigin(
  remoteUrl: string | null,
  browserOrigin: string,
): boolean {
  return (
    remoteUrl?.replace(TRAILING_SLASHES_RE, "") ===
    browserOrigin.replace(TRAILING_SLASHES_RE, "")
  );
}

export function remoteAccessTeardownNeedsLocalOrigin(
  status: RemoteAccessStatus | null,
  browserOrigin: string,
): boolean {
  if (status?.kind !== "custom") {
    return false;
  }
  const customOrigin =
    status.url ??
    (status.customHostname ? `https://${status.customHostname}` : null);
  return remoteAccessStopDisconnectsOrigin(customOrigin, browserOrigin);
}

// A Stop sent from the tunnel's own origin is answered with a terminal off, then
// that origin dies. Polls landing during teardown still report "stopping", a state
// this origin can never see resolve, so they must not overwrite the terminal off.
// One that can still stop the connector proves teardown was abandoned, so the
// origin is staying up and polls lead again.
export function remoteAccessSelfStopPoll(
  next: RemoteAccessStatus,
  expectingDisconnect: boolean,
): { expectingDisconnect: boolean; apply: boolean } {
  if (!expectingDisconnect || next.canStop) {
    return { expectingDisconnect: false, apply: true };
  }
  return { expectingDisconnect: true, apply: next.state !== "stopping" };
}

export function remoteAccessBlockMessageId(
  reason: string | null,
  isDesktop: boolean,
): RemoteAccessBlockMessageId | null {
  switch (reason) {
    case "server_starting":
      return "serverStarting";
    case "admin_password_change_required":
      return isDesktop ? "passwordDesktop" : "passwordWeb";
    case "explicitly_disabled":
      return "explicitlyDisabled";
    case "launch_managed":
      return "launchManaged";
    case "colab_managed":
      return "colabManaged";
    case "colab":
      return "colab";
    default:
      return null;
  }
}

export function remoteAccessRequestMessageId(
  code: string | null,
  isDesktop: boolean,
): RemoteAccessRequestMessageId {
  const blocked = remoteAccessBlockMessageId(code, isDesktop);
  if (blocked) {
    return blocked;
  }
  if (code === "invalid_hostname") {
    return "invalidHostname";
  }
  if (
    code === "custom_operation_in_progress" ||
    code === "operation_in_progress" ||
    code === "start_kind_conflict" ||
    code === "stop_kind_conflict" ||
    code === "server_lifecycle_changed"
  ) {
    return "busy";
  }
  return "requestFailed";
}

export function remoteAccessDnsConflictHostname(
  status: RemoteAccessStatus,
): string | null {
  if (status.customError !== "dns_conflict") {
    return null;
  }
  return status.customErrorDetail?.match(DNS_CONFLICT_DETAIL_RE)?.[1] ?? null;
}
