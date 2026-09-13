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
  attachments?: readonly {
    id?: unknown;
    mcpToolOnly?: boolean;
    content?: readonly Part[];
  }[];
};

type IdentifiedMessage = PrivateMessage & { id?: unknown; role?: unknown };
type PersistedMessage = {
  id: string;
  threadId: string;
  attachments?: readonly { id?: unknown; mcpToolOnly?: boolean }[];
};

export type McpImageAttachmentSelection = {
  message_id: string;
  attachment_id: string;
};

/** Select only a tool-only image that is already durable in this exact conversation. */
export function mcpImageAttachmentForTokenCount(
  messages: readonly IdentifiedMessage[],
  threadId: string | undefined,
  persistedMessages: readonly PersistedMessage[],
): McpImageAttachmentSelection | undefined {
  if (!threadId) return undefined;
  const latestUser = [...messages]
    .reverse()
    .find((message) => message.role === "user");
  if (!latestUser || typeof latestUser.id !== "string" || !latestUser.id)
    return undefined;
  const privateImages = latestUser.attachments?.filter(isMcpToolOnly) ?? [];
  if (privateImages.length !== 1) return undefined;
  const attachmentId = privateImages[0]?.id;
  if (typeof attachmentId !== "string" || !attachmentId) return undefined;

  const persisted = persistedMessages.find(
    (message) => message.threadId === threadId && message.id === latestUser.id,
  );
  const persistedPrivate = persisted?.attachments?.filter(isMcpToolOnly) ?? [];
  if (persistedPrivate.length !== 1 || persistedPrivate[0]?.id !== attachmentId)
    return undefined;
  return { message_id: latestUser.id, attachment_id: attachmentId };
}

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
