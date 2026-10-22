// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isLoopbackHost, normalizeHost } from "../components/agent-command.ts";

export type RemoteAccessState =
  | "off"
  | "starting"
  | "online"
  | "stopping"
  | "error";
export type RemoteAccessOwner = "launch" | "settings" | "colab" | null;
export type TunnelMode = "quick" | "named";
export type NamedTunnelMethod = "automatic" | "token";
export type NamedTunnelSetupState =
  | "preparing"
  | "awaiting_login"
  | "configuring"
  | "dns_conflict"
  | "ready"
  | "removing"
  | "removed"
  | "error";

export type NamedTunnelConfiguration = {
  configurationId: string;
  method: NamedTunnelMethod;
  hostname: string;
  ready: boolean;
  tunnelId: string | null;
  dnsCleanupHostname: string | null;
};

export type NamedTunnelSetup = {
  operationId: string;
  state: NamedTunnelSetupState;
  hostname: string;
  tunnelId: string | null;
  authUrl: string | null;
  error: string | null;
  dnsCleanupHostname: string | null;
  active: boolean;
};
const TRAILING_SLASHES_RE = /\/+$/;

export type RemoteAccessStatus = {
  state: RemoteAccessState;
  url: string | null;
  error: string | null;
  autoStart: boolean;
  defaultAutoStart: boolean;
  available: boolean;
  managedBy: RemoteAccessOwner;
  canStart: boolean;
  canStop: boolean;
  blockReason: string | null;
  passwordPending: boolean;
  streamingSupported: boolean;
  mode: TunnelMode;
  named: {
    configuration: NamedTunnelConfiguration | null;
    setup: NamedTunnelSetup | null;
  };
  canConfigure: boolean;
  configurationError: string | null;
  localOrigin: string | null;
};

export type ApiRemoteAccessStatus = {
  state: RemoteAccessState;
  url?: string | null;
  error?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_start: boolean;
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
  mode?: TunnelMode;
  named?: {
    configuration?: {
      // biome-ignore lint/style/useNamingConvention: API schema
      configuration_id: string;
      method: NamedTunnelMethod;
      hostname: string;
      ready: boolean;
      // biome-ignore lint/style/useNamingConvention: API schema
      tunnel_id?: string | null;
      // biome-ignore lint/style/useNamingConvention: API schema
      dns_cleanup_hostname?: string | null;
    } | null;
    setup?: {
      // biome-ignore lint/style/useNamingConvention: API schema
      operation_id: string;
      state: NamedTunnelSetupState;
      hostname: string;
      // biome-ignore lint/style/useNamingConvention: API schema
      tunnel_id?: string | null;
      // biome-ignore lint/style/useNamingConvention: API schema
      auth_url?: string | null;
      error?: string | null;
      // biome-ignore lint/style/useNamingConvention: API schema
      dns_cleanup_hostname?: string | null;
      active: boolean;
    } | null;
  };
  // biome-ignore lint/style/useNamingConvention: API schema
  can_configure?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  configuration_error?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  local_origin?: string | null;
};

export function normalizeRemoteAccessStatus(
  status: ApiRemoteAccessStatus,
): RemoteAccessStatus {
  const configuration = status.named?.configuration;
  const setup = status.named?.setup;
  return {
    state: status.state,
    url: status.url ?? null,
    error: status.error ?? null,
    autoStart: status.auto_start,
    defaultAutoStart: status.default_auto_start,
    available: status.available,
    managedBy: status.managed_by ?? null,
    canStart: status.can_start,
    canStop: status.can_stop,
    blockReason: status.block_reason ?? null,
    passwordPending: status.password_pending === true,
    streamingSupported: status.streaming_supported,
    mode: status.mode ?? "quick",
    named: {
      configuration: configuration
        ? {
            configurationId: configuration.configuration_id,
            method: configuration.method,
            hostname: configuration.hostname,
            ready: configuration.ready,
            tunnelId: configuration.tunnel_id ?? null,
            dnsCleanupHostname: configuration.dns_cleanup_hostname ?? null,
          }
        : null,
      setup: setup
        ? {
            operationId: setup.operation_id,
            state: setup.state,
            hostname: setup.hostname,
            tunnelId: setup.tunnel_id ?? null,
            authUrl: setup.auth_url ?? null,
            error: setup.error ?? null,
            dnsCleanupHostname: setup.dns_cleanup_hostname ?? null,
            active: setup.active,
          }
        : null,
    },
    canConfigure: status.can_configure === true,
    configurationError: status.configuration_error ?? null,
    localOrigin: status.local_origin ?? null,
  };
}

export function remoteAccessPollDelay(
  status: RemoteAccessStatus | null,
): number {
  return status?.state === "starting" ||
    status?.state === "stopping" ||
    status?.named.setup?.active
    ? 1000
    : 5000;
}

export function cloudflareLoginUrl(value: string | null): string | null {
  if (!value) {
    return null;
  }
  try {
    const url = new URL(value);
    return url.origin === "https://dash.cloudflare.com" &&
      url.pathname === "/argotunnel" &&
      !url.username &&
      !url.password &&
      !url.hash
      ? url.href
      : null;
  } catch {
    return null;
  }
}

function isLoopbackOrigin(origin: string): boolean {
  try {
    return isLoopbackHost(normalizeHost(new URL(origin).hostname));
  } catch {
    return false;
  }
}

export function remoteApiOrigin(
  remoteUrl: string | null,
  localOrigin: string,
  lanUrls: string[] = [],
): string {
  const lanUrl = isLoopbackOrigin(localOrigin) ? lanUrls[0] : undefined;
  return remoteUrl ?? lanUrl ?? localOrigin;
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

// A Stop sent from the tunnel's own origin is answered with a terminal off, then that origin dies.
// Polls landing during teardown still report "stopping", a state this origin can never see resolve,
// so they must not overwrite the terminal off. One that can still stop the connector proves
// teardown was abandoned, so the origin is staying up and polls lead again.
export function remoteAccessSelfStopPoll(
  next: RemoteAccessStatus,
  expectingDisconnect: boolean,
): { expectingDisconnect: boolean; apply: boolean } {
  if (!expectingDisconnect || next.canStop) {
    return { expectingDisconnect: false, apply: true };
  }
  return { expectingDisconnect: true, apply: next.state !== "stopping" };
}

export function remoteAccessBlockMessage(
  reason: string | null,
  isDesktop: boolean,
): string | null {
  switch (reason) {
    case "server_starting":
      return "Unsloth is still starting.";
    case "admin_password_change_required":
      return isDesktop
        ? "Set a remote password before exposing this server."
        : "Change the administrator password before exposing this server. In the desktop app, run unsloth studio reset-password.";
    case "explicitly_disabled":
      return "This launch used --no-cloudflare. Restart without it to enable remote access.";
    case "launch_managed":
      return "This tunnel is managed by the launch command.";
    case "colab_managed":
      return "This tunnel is managed by the Colab runtime.";
    case "colab":
      return "Remote access settings are managed by the Colab runtime.";
    default:
      return null;
  }
}
