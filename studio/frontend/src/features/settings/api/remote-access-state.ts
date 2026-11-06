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
};

export function normalizeRemoteAccessStatus(
  status: ApiRemoteAccessStatus,
): RemoteAccessStatus {
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
  };
}

export function remoteAccessPollDelay(
  status: RemoteAccessStatus | null,
): number {
  return status?.state === "starting" || status?.state === "stopping"
    ? 1000
    : 5000;
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

// After a self-stop, polls reporting "stopping" must not overwrite the terminal off; one that can
// still stop the connector means teardown was abandoned, so polls lead again.
export function remoteAccessSelfStopPoll(
  next: RemoteAccessStatus,
  expectingDisconnect: boolean,
): { expectingDisconnect: boolean; apply: boolean } {
  if (!expectingDisconnect || next.canStop) {
    return { expectingDisconnect: false, apply: true };
  }
  return { expectingDisconnect: true, apply: next.state !== "stopping" };
}

export function remoteAccessErrorMessage(
  error: string | null | undefined,
): string | null {
  switch (error) {
    case null:
    case undefined:
      return null;
    case "cloudflared is unavailable":
      return "Couldn't download cloudflared. Check your internet connection, or install cloudflared yourself (for example brew install cloudflared) and try again.";
    case "cloudflared did not produce a URL":
      return "Cloudflare didn't issue a URL. Check your connection and try again in a minute.";
    case "cloudflared did not register a connection":
      return "Cloudflare couldn't connect. A firewall or VPN may be blocking it.";
    case "Cloudflare URL was not reachable":
      return "The URL was created but didn't respond. Try again.";
    case "cloudflared exited":
      return "The Cloudflare tunnel stopped unexpectedly. Start it again.";
    case "cloudflared could not be stopped":
      return "The previous Cloudflare tunnel could not be stopped. Stop it before starting a new one.";
    default:
      return error;
  }
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
      return "This launch turned the Cloudflare tunnel off (--no-cloudflare, which unsloth start also sets). Relaunch with unsloth studio to use remote access.";
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
