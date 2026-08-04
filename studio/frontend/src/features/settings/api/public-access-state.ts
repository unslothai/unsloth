// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type PublicAccessState =
  | "off"
  | "starting"
  | "online"
  | "stopping"
  | "error";
export type PublicAccessOwner = "launch" | "settings" | "colab" | null;
const TRAILING_SLASHES_RE = /\/+$/;

export type PublicAccessStatus = {
  state: PublicAccessState;
  url: string | null;
  error: string | null;
  autoStart: boolean;
  defaultAutoStart: boolean;
  available: boolean;
  managedBy: PublicAccessOwner;
  canStart: boolean;
  canStop: boolean;
  blockReason: string | null;
  streamingSupported: boolean;
};

export type ApiPublicAccessStatus = {
  state: PublicAccessState;
  url?: string | null;
  error?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_start: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_auto_start: boolean;
  available: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  managed_by?: PublicAccessOwner;
  // biome-ignore lint/style/useNamingConvention: API schema
  can_start: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  can_stop: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  block_reason?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  streaming_supported: boolean;
};

export function normalizePublicAccessStatus(
  status: ApiPublicAccessStatus,
): PublicAccessStatus {
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
    streamingSupported: status.streaming_supported,
  };
}

export function publicAccessPollDelay(
  status: PublicAccessStatus | null,
): number {
  return status?.state === "starting" || status?.state === "stopping"
    ? 1000
    : 5000;
}

export function publicApiOrigin(
  publicUrl: string | null,
  localOrigin: string,
): string {
  return publicUrl ?? localOrigin;
}

export function publicAccessAutoStartReadOnly(
  status: PublicAccessStatus | null,
): boolean {
  return (
    status === null ||
    status.managedBy === "colab" ||
    status.blockReason === "colab"
  );
}

export function publicAccessStopDisconnectsOrigin(
  publicUrl: string | null,
  browserOrigin: string,
): boolean {
  return (
    publicUrl?.replace(TRAILING_SLASHES_RE, "") ===
    browserOrigin.replace(TRAILING_SLASHES_RE, "")
  );
}

export function publicAccessBlockMessage(reason: string | null): string | null {
  switch (reason) {
    case "server_starting":
      return "Unsloth is still starting.";
    case "admin_password_change_required":
      return "Change the administrator password before exposing this server. In the desktop app, run unsloth studio reset-password.";
    case "explicitly_disabled":
      return "This launch used --no-cloudflare. Restart without it to enable public access.";
    case "launch_managed":
      return "This tunnel is managed by the launch command.";
    case "colab_managed":
      return "This tunnel is managed by the Colab runtime.";
    case "colab":
      return "Public access settings are managed by the Colab runtime.";
    default:
      return null;
  }
}
