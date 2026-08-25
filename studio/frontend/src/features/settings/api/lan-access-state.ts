// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LanAccessState = "off" | "online" | "error";
export type LanAccessOwner = "launch" | "settings" | null;
export type LanKeylessScope = "off" | "inference" | "full";

export type LanAccessStatus = {
  state: LanAccessState;
  urls: string[];
  publicUrls: string[];
  error: string | null;
  autoStart: boolean;
  managedBy: LanAccessOwner;
  canStart: boolean;
  canStop: boolean;
  blockReason: string | null;
  servesWebUi: boolean;
  keylessLanEligible: boolean;
  keylessScope: LanKeylessScope;
  keylessTools: boolean;
};

export type ApiLanAccessStatus = {
  state: LanAccessState;
  urls?: string[] | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  public_urls?: string[] | null;
  error?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_start: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  managed_by?: LanAccessOwner;
  // biome-ignore lint/style/useNamingConvention: API schema
  can_start: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  can_stop: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  block_reason?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  serves_web_ui?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  keyless_lan_eligible?: boolean | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  keyless_scope?: LanKeylessScope | string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  keyless_tools?: boolean | null;
};

function normalizeKeylessScope(value: unknown): LanKeylessScope {
  return value === "inference" || value === "full" ? value : "off";
}

export function normalizeLanAccessStatus(
  status: ApiLanAccessStatus,
): LanAccessStatus {
  const keylessScope = normalizeKeylessScope(status.keyless_scope);
  return {
    state: status.state,
    urls: Array.isArray(status.urls) ? status.urls : [],
    publicUrls: Array.isArray(status.public_urls) ? status.public_urls : [],
    error: status.error ?? null,
    autoStart: status.auto_start,
    managedBy: status.managed_by ?? null,
    canStart: status.can_start,
    canStop: status.can_stop,
    blockReason: status.block_reason ?? null,
    // absent on a backend that predates the field, where the web UI is served
    servesWebUi: status.serves_web_ui !== false,
    keylessLanEligible: status.keyless_lan_eligible === true,
    keylessScope,
    keylessTools: keylessScope !== "off" && status.keyless_tools === true,
  };
}

export function keylessLanAccessDescription(
  status: LanAccessStatus | null,
): string {
  if (!status || status.keylessScope === "off") {
    return "Authentication is required for LAN API requests.";
  }
  if (status.blockReason === "colab") {
    return "Colab never receives keyless access, regardless of the saved scope.";
  }
  if (status.keylessScope === "full") {
    return "Full keyless access is localhost-only and is never granted over LAN or public URLs.";
  }
  const tools = status.keylessTools
    ? " Agent tools are separately enabled."
    : " Agent tools remain off unless separately granted.";
  if (status.keylessLanEligible && status.publicUrls.length > 0) {
    return `Inference can be keyless on localhost and an active private LAN, but never through the listed public URL.${tools}`;
  }
  if (status.keylessLanEligible) {
    return `Inference can be keyless on localhost and this active private LAN.${tools}`;
  }
  return `Inference is keyless on localhost; LAN callers require an active private listener.${tools}`;
}

// start and stop are synchronous socket work, so there is no transition to chase
export const LAN_ACCESS_POLL_MS = 5000;

export function lanAccessAutoStartReadOnly(
  status: LanAccessStatus | null,
): boolean {
  return status === null || status.blockReason === "colab";
}

export function lanAccessStopDisconnectsOrigin(
  urls: string[],
  browserOrigin: string,
): boolean {
  let origin: string;
  try {
    origin = new URL(browserOrigin).origin;
  } catch {
    return false;
  }
  return urls.some((url) => {
    try {
      return new URL(url).origin === origin;
    } catch {
      return false;
    }
  });
}

export function lanAccessBlockMessage(
  reason: string | null,
  isDesktop: boolean,
): string | null {
  switch (reason) {
    case "server_starting":
      return "Unsloth is still starting.";
    case "admin_password_change_required":
      return isDesktop
        ? "Set a remote password before putting this server on the network."
        : "Change the administrator password before putting this server on the network. In the desktop app, run unsloth studio reset-password.";
    case "launch_managed":
      return "This launch already binds every network interface (-H 0.0.0.0), so Unsloth is on the network already.";
    case "secure_launch":
      return "This launch used --secure, which serves only through the Cloudflare link and keeps the raw port closed. Relaunch without --secure to use LAN access.";
    case "colab":
      return "LAN access is managed by the Colab runtime, which has no local network to join.";
    default:
      return null;
  }
}

export function lanAccessErrorMessage(error: string | null): string | null {
  switch (error) {
    case "no_lan_address":
      return "No network address found. Connect this machine to Wi-Fi or a wired network, then try again.";
    case "bind_failed":
      return "Could not open Unsloth's port on this machine's network addresses.";
    case "listener_start_failed":
      return "The network listener did not start. Check the logs for details.";
    case "stop_timed_out":
      return "LAN access could not confirm the port closed. Check the logs, then try again.";
    case null:
    case undefined:
      return null;
    default:
      return "LAN access failed to start.";
  }
}
