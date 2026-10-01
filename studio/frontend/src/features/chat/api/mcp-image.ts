// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { McpServerConfig } from "./mcp-servers-api";

/** Sent with a tool_start that would send the user's image; approval covers this one call. */
export type ImageDisclosure = {
  server: string;
  tool: string;
  size_bytes: number;
};

type Part = { type: string; image?: unknown };
type Message = {
  content: readonly Part[];
  attachments?: readonly { content?: readonly Part[] }[];
};

export function isMcpToolOnly(value: unknown): boolean {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { mcpToolOnly?: unknown }).mcpToolOnly === true
  );
}

/** Whether attached images go to MCP tools instead of the model. */
export function mcpImageMappingsEnabled(
  servers: readonly McpServerConfig[],
): boolean {
  return servers.some(
    (server) =>
      server.is_enabled && (server.image_input_mappings?.length ?? 0) > 0,
  );
}

/** The message without its tool-only images, which the model never receives. */
export function modelVisibleMessage<T extends Message>(message: T): T {
  // The part is flagged too: a reloaded thread can carry the image as message content.
  if (
    !message.attachments?.some(isMcpToolOnly) &&
    !message.content.some(isMcpToolOnly)
  ) {
    return message;
  }
  return {
    ...message,
    content: message.content.filter((part) => !isMcpToolOnly(part)),
    attachments: message.attachments?.filter((a) => !isMcpToolOnly(a)),
  };
}

/** Data URL of the message's tool-only image, sent as the request's mcp_image. */
export function toolOnlyImage(
  message: Message | undefined,
): string | undefined {
  const parts = [
    ...(message?.attachments ?? [])
      .filter(isMcpToolOnly)
      .flatMap((attachment) => attachment.content ?? []),
    ...(message?.content ?? []).filter(isMcpToolOnly),
  ];
  const part = parts.find(
    (p) => p.type === "image" && typeof p.image === "string",
  );
  return part?.image as string | undefined;
}

/** Top-level string fields a mapping can target; the backend applies the same rule. */
export function imageFieldCandidates(schema: unknown): string[] {
  const properties = (schema as { properties?: unknown } | null)?.properties;
  if (!properties || typeof properties !== "object") return [];
  return Object.entries(properties)
    .filter(
      ([, field]) => (field as { type?: unknown } | null)?.type === "string",
    )
    .map(([name]) => name);
}
