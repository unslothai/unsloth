// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProjectInitResult } from "../api/project-guidance-api";

export type InitSlashCommand = {
  name: "init";
  action: "run" | "help";
};

export type ProjectSlashCommand = InitSlashCommand;

export type CompareSlashCommandPane<TContent> = {
  appendUserMessage: (content: TContent) => void;
  appendAssistantMessage: (response: string) => void;
};

const COMMAND_PATTERN = /^\/([a-z][a-z0-9_-]*)(?:\s+([\s\S]*))?$/i;

export function parseProjectSlashCommand(
  input: string,
): ProjectSlashCommand | null {
  const match = COMMAND_PATTERN.exec(input.trim());
  if (match?.[1]?.toLowerCase() !== "init") {
    return null;
  }
  return {
    name: "init",
    action: (match[2] ?? "").trim() ? "help" : "run",
  };
}

export function projectInitCommandResponse(
  command: InitSlashCommand,
  result?: ProjectInitResult,
): string {
  if (command.action === "help") {
    return "Usage: `/init`. It creates a starter `AGENTS.md` without overwriting an existing file.";
  }
  if (!result) {
    return "The `/init` command is available only inside a project.";
  }
  return result.created
    ? "Created `AGENTS.md` in this project. Future agent turns will use its instructions."
    : `\`${result.path}\` already exists, so \`/init\` left it unchanged.`;
}

/** Execute once, then mirror the same deterministic result into both panes. */
export async function interceptCompareProjectSlashCommand<TContent>(args: {
  input: string;
  userContent: TContent;
  panes: readonly [
    CompareSlashCommandPane<TContent>,
    CompareSlashCommandPane<TContent>,
  ];
  execute: (command: ProjectSlashCommand) => Promise<string>;
  onIntercept?: () => void;
}): Promise<boolean> {
  const command = parseProjectSlashCommand(args.input);
  if (!command) {
    return false;
  }

  args.onIntercept?.();
  for (const pane of args.panes) {
    pane.appendUserMessage(args.userContent);
  }
  const response = await args.execute(command);
  for (const pane of args.panes) {
    pane.appendAssistantMessage(response);
  }
  return true;
}
