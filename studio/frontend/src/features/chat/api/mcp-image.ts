// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { McpImageInputMapping, McpServerConfig } from "./mcp-servers-api";

/** Sent with a tool_start that would send the user's image; approval covers this one call. */
export type ImageDisclosure = {
  server: string;
  tool: string;
  size_bytes: number;
  destination: string;
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

export function mcpImageMappingsEnabled(
  servers: readonly McpServerConfig[],
): boolean {
  return servers.some(
    (server) =>
      server.is_enabled &&
      (server.image_input_mappings?.length ?? 0) > 0 &&
      server.image_mappings_active !== false,
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

/** Data URLs of the message's tool-only images; the request's mcp_image carries one. */
export function toolOnlyImages(message: Message | undefined): string[] {
  const parts = [
    ...(message?.attachments ?? [])
      .filter(isMcpToolOnly)
      .flatMap((attachment) => attachment.content ?? []),
    ...(message?.content ?? []).filter(isMcpToolOnly),
  ];
  const images = parts
    .filter((p) => p.type === "image" && typeof p.image === "string")
    .map((p) => p.image as string);
  return [...new Set(images)];
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

type ImageField = { tool: string; field: string };

/** The tool / field pairs not mapped yet, so picking one always changes something. */
export function unmappedImageFields<T extends ImageField>(
  options: readonly T[],
  mappings: readonly McpImageInputMapping[],
): T[] {
  return options.filter(
    (option) =>
      !mappings.some((m) => m.tool === option.tool && m.field === option.field),
  );
}

/** Maps `option`; the backend takes one field per tool, so it replaces that tool's row in place. */
export function withImageField(
  mappings: readonly McpImageInputMapping[],
  option: ImageField,
  encoding: McpImageInputMapping["encoding"] = "base64",
): McpImageInputMapping[] {
  const next: McpImageInputMapping = {
    tool: option.tool,
    field: option.field,
    encoding,
  };
  const index = mappings.findIndex((m) => m.tool === option.tool);
  if (index < 0) return [...mappings, next];
  // A server saved through the API can hold two rows for one tool; keep only the replacement.
  return mappings.flatMap((m, i) =>
    i === index ? [next] : m.tool === option.tool ? [] : [m],
  );
}

/** The mappings an edit should send, or undefined when they are unchanged and the update can leave them alone. */
export function changedImageMappings(
  saved: readonly McpImageInputMapping[],
  current: McpImageInputMapping[],
): McpImageInputMapping[] | undefined {
  // Resending an unchanged list would fail an unrelated edit of a server saved with two fields on one tool.
  if (JSON.stringify(saved) === JSON.stringify(current)) return undefined;
  // Any real edit must pass the one-field-per-tool check, so keep the first row per tool, as image_mapping does.
  const tools = new Set<string>();
  return current.filter((m) => !tools.has(m.tool) && tools.add(m.tool));
}
