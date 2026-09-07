// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface McpStdioSnapshot {
  originalUrl: string;
  command: string;
  arguments: string[];
}

export type McpStdioUrlDecision =
  | { kind: "reuse"; url: string }
  | { kind: "encode"; command: string; arguments: string[] };

export function createMcpStdioSnapshot(
  originalUrl: string,
  command: string,
  arguments_: readonly string[] = [],
): McpStdioSnapshot {
  return {
    originalUrl,
    command,
    arguments: [...arguments_],
  };
}

function sameArguments(left: readonly string[], right: readonly string[]) {
  return (
    left.length === right.length &&
    left.every((argument, index) => argument === right[index])
  );
}

export function resolveMcpStdioUrl(
  command: string,
  arguments_: readonly string[],
  snapshot: McpStdioSnapshot | null,
): McpStdioUrlDecision {
  if (
    snapshot !== null &&
    command === snapshot.command &&
    sameArguments(arguments_, snapshot.arguments)
  ) {
    return { kind: "reuse", url: snapshot.originalUrl };
  }

  return { kind: "encode", command, arguments: [...arguments_] };
}
