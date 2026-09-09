// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function parseVerificationCommand(input: string): "run" | "help" | null {
  const match = /^\/verify(?:\s+([\s\S]*))?$/i.exec(input.trim());
  return match ? (match[1]?.trim() ? "help" : "run") : null;
}

export function latestVerificationText(messages: readonly { role: string; content: readonly { type: string; text?: string }[] }[]): string {
  // Reload/continue after a tool result must not replay the earlier /verify.
  const message = messages.at(-1);
  if (message?.role !== "user" || message.content.some((part) => part.type !== "text")) return "";
  return message.content.map((part) => part.text ?? "").join("\n");
}

/** A compare send invokes once and mirrors the same result into both panes. */
export async function runCompareVerificationCommand<T>(
  content: T,
  panes: readonly { appendMessage: (content: T) => void; appendAssistantMessage: (text: string) => void }[],
  execute: () => Promise<string>,
): Promise<void> {
  if (panes.length !== 2) throw new Error("Wait for both comparison panes to finish opening.");
  for (const pane of panes) pane.appendMessage(content);
  const response = await execute();
  for (const pane of panes) pane.appendAssistantMessage(response);
}
