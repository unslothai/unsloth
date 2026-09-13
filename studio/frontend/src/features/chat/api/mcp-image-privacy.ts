// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ImageDisclosure = {
  purpose: "mcp_image_disclosure";
  previewUrl?: string;
  sizeBytes: number;
  serverName: string;
  toolName: string;
  destination: string;
  field: string;
  encoding: "base64" | "data_url";
  status?: "pending" | "expired" | "cancelled";
  expiresAt?: number;
};

export function isMcpToolOnly(value: unknown): boolean {
  return (
    !!value &&
    typeof value === "object" &&
    (value as { mcpToolOnly?: unknown }).mcpToolOnly === true
  );
}

type Part = {
  type: string;
  image?: unknown;
  text?: unknown;
  mcpToolOnly?: boolean;
};
type PrivateMessage = {
  content: readonly Part[];
  attachments?: readonly { mcpToolOnly?: boolean; content?: readonly Part[] }[];
};

/** Remove explicitly marked private content and the private attachment itself. */
export function modelVisibleMessage<T extends PrivateMessage>(message: T): T {
  const hasPrivateAttachment =
    message.attachments?.some(isMcpToolOnly) === true;
  if (!(hasPrivateAttachment || message.content.some(isMcpToolOnly))) {
    return message;
  }
  return {
    ...message,
    content: message.content.filter((part) => !isMcpToolOnly(part)),
    attachments: message.attachments?.filter(
      (attachment) => !isMcpToolOnly(attachment),
    ),
  };
}

export function disclosureExpired(
  value: ImageDisclosure,
  now = Date.now(),
): boolean {
  return (
    value.status === "expired" ||
    value.status === "cancelled" ||
    (value.expiresAt !== undefined && now >= value.expiresAt)
  );
}

export function mayAutoApproveTool(
  disclosure: ImageDisclosure | undefined,
  alwaysAllowed: boolean,
): boolean {
  return disclosure === undefined && alwaysAllowed;
}
