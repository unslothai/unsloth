// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  knowledgeBaseToolName,
  toolArgText,
  webSearchToolName,
} from "@/components/assistant-ui/tool-arg-text";
import { describeToolCall } from "./voice-loop-bridge";


function hostOf(url: string): string {
  try {
    return new URL(
      /^[a-z]+:\/\//i.test(url) ? url : `https://${url}`,
    ).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

export function toolLabel(toolCallId: string): string {
  const call = describeToolCall(toolCallId);
  const name = call?.toolName ?? "";
  const args = (call?.args ?? {}) as Record<string, unknown>;
  const query = toolArgText(args.query).trim();
  if (name === "web_search") {
    const url = toolArgText(args.url).trim();
    const pattern = toolArgText(args.pattern).trim();
    return webSearchToolName({
      isRunning: true,
      isFindInPage: Boolean(url && pattern),
      isUrlFetch: Boolean(url && !pattern),
      isImageOnly: false,
      foundImages: false,
      displayDomain: hostOf(url),
      pattern,
      query,
      imageLabel: "",
    });
  }
  if (name === "search_knowledge_base")
    return knowledgeBaseToolName({ isRunning: true, query });
  return toolArgText(name) || toolCallId;
}

export function toolBrief(toolCallId: string): string {
  const call = describeToolCall(toolCallId);
  const name = call?.toolName || "tool";
  const args = (call?.args ?? {}) as Record<string, unknown>;
  let gist = "";
  for (const key of ["query", "url", "pattern", "path", "command", "name"]) {
    gist = toolArgText(args[key]).trim();
    if (gist) break;
  }
  if (!gist) return name;
  return `${name} · "${gist.length > 60 ? `${gist.slice(0, 59)}…` : gist}"`;
}
