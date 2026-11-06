// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { ClaudeAgentStatus } from "./api/providers-api";
import { CLAUDE_AGENT_PROVIDER_TYPE } from "./external-providers";
export const CLAUDE_CODE_INSTALL_COMMANDS = {
  unix: "curl -fsSL https://claude.ai/install.sh | bash",
  // lint-allow: AV005 shown to the user to copy, never executed
  windows: "irm https://claude.ai/install.ps1 | iex",
} as const;
export const CLAUDE_CODE_NPM_INSTALL =
  "npm install -g @anthropic-ai/claude-code";
export const CLAUDE_CODE_SIGN_IN = "claude";
export const CLAUDE_CODE_SETUP_URL = "https://code.claude.com/docs/en/setup";

const SNAPSHOT_SUFFIX = /-\d{8}$/;
const OFF_MESSAGE = /^Claude Agent is off\b/i;
const NAMES_KEY = "unsloth_claude_agent_model_names";

export type ClaudeAgentModelOption = {
  id: string;
  name: string;
  alias: boolean;
};

export function claudeAgentFallbackName(id: string): string {
  const words = id
    .trim()
    .replace(/^claude-/i, "")
    .replace(SNAPSHOT_SUFFIX, "")
    .split(/[-_\s]+/)
    .filter(Boolean);
  const out: string[] = [];
  for (const word of words) {
    const last = out[out.length - 1];
    if (/^\d+$/.test(word) && last && /^\d+(\.\d+)*$/.test(last)) {
      out[out.length - 1] = `${last}.${word}`;
    } else {
      out.push(word.charAt(0).toUpperCase() + word.slice(1));
    }
  }
  return out.join(" ") || id;
}

export function claudeAgentModelOptions(
  models: ClaudeAgentStatus["models"],
): ClaudeAgentModelOption[] {
  const seen = new Set<string>();
  const options: ClaudeAgentModelOption[] = [];
  for (const model of models) {
    const id = (typeof model === "string" ? model : model.id || "").trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const name =
      typeof model === "string" ? "" : (model.display_name || "").trim();
    const alias =
      typeof model === "string" || typeof model.alias !== "boolean"
        ? !id.startsWith("claude-")
        : model.alias;
    options.push({
      id,
      name: name && name !== id ? name : claudeAgentFallbackName(id),
      alias,
    });
  }
  return options;
}

export function claudeAgentModelGroups(options: ClaudeAgentModelOption[]) {
  return {
    main: options.filter((option) => option.alias),
    more: options.filter((option) => !option.alias),
  };
}

export function claudeAgentDefaultModels(
  options: ClaudeAgentModelOption[],
): string[] {
  const main = options
    .filter((option) => option.alias)
    .map((option) => option.id);
  return main.length ? main : options.slice(0, 1).map((option) => option.id);
}

export function claudeAgentInitialSelection(
  options: ClaudeAgentModelOption[],
  saved: readonly string[] | null,
): { selected: string[]; extra: string[] } {
  if (!saved) return { selected: claudeAgentDefaultModels(options), extra: [] };
  const known = new Set(options.map((option) => option.id));
  return {
    selected: saved.filter((id) => known.has(id)),
    extra: saved.filter((id) => !known.has(id)),
  };
}

export function claudeAgentModelsToSave(
  options: ClaudeAgentModelOption[],
  selected: readonly string[],
  extraText: string,
): string[] {
  const order = new Map(options.map((option, index) => [option.id, index]));
  const listed = selected
    .filter((id) => order.has(id))
    .sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
  const extra = extraText
    .split(/[\n,]/)
    .map((id) => id.trim())
    .filter(Boolean);
  return [...new Set([...listed, ...extra])];
}

export type ClaudeAgentSetupStep =
  | "off"
  | "forced_off"
  | "checking"
  | "sdk_missing"
  | "install_cli"
  | "sign_in"
  | "failed"
  | "ready";

export function claudeAgentSetupStep(input: {
  off: boolean;
  forcedOff: boolean;
  checking: boolean;
  status: ClaudeAgentStatus | null;
  error: string;
}): ClaudeAgentSetupStep {
  if (input.off || isClaudeAgentOffMessage(input.error)) {
    return input.forcedOff ? "forced_off" : "off";
  }
  const { status } = input;
  if (status?.ready) return "ready";
  if (!status) return input.error && !input.checking ? "failed" : "checking";
  if (status.sdk_missing || status.problem === "sdk_missing") {
    return "sdk_missing";
  }
  if (status.problem === "cli_missing") return "install_cli";
  if (status.problem === "signed_out") return "sign_in";
  return "failed";
}

export type ClaudeAgentChecklistItem = {
  id: "install" | "sign_in" | "check";
  state: "done" | "current" | "next";
};

export function claudeAgentChecklist(
  step: ClaudeAgentSetupStep,
): ClaudeAgentChecklistItem[] | null {
  if (step !== "install_cli" && step !== "sign_in") return null;
  return [
    { id: "install", state: step === "install_cli" ? "current" : "done" },
    { id: "sign_in", state: step === "sign_in" ? "current" : "next" },
    { id: "check", state: "next" },
  ];
}

export function claudeAgentRechecksOnReturn(
  step: ClaudeAgentSetupStep,
): boolean {
  return (
    step === "install_cli" ||
    step === "sign_in" ||
    step === "sdk_missing" ||
    step === "failed"
  );
}

export function claudeAgentSdkCommand(message: string): string {
  const command = message.indexOf(": ");
  return command === -1 ? message : message.slice(command + 2);
}

export function claudeAgentInstallCommand(
  hostOs: ClaudeAgentStatus["host_os"],
): string {
  return CLAUDE_CODE_INSTALL_COMMANDS[
    hostOs === "windows" ? "windows" : "unix"
  ];
}

export function claudeAgentStatusLine(status: ClaudeAgentStatus): string {
  return [
    status.account,
    status.version ? `Claude Code ${status.version}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

export function isClaudeAgentOffMessage(message: string): boolean {
  return OFF_MESSAGE.test(message.trim());
}

function readNames(): Record<string, string> {
  try {
    const raw = globalThis.localStorage?.getItem(NAMES_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    if (!parsed || typeof parsed !== "object") return {};
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      ),
    );
  } catch {
    return {};
  }
}

export const useClaudeAgentModelNames = create<{
  names: Record<string, string>;
}>(() => ({ names: readNames() }));

export function rememberClaudeAgentModels(
  models: ClaudeAgentStatus["models"],
): void {
  const learned = Object.fromEntries(
    claudeAgentModelOptions(models).map((option) => [option.id, option.name]),
  );
  if (Object.keys(learned).length === 0) return;
  const names = { ...useClaudeAgentModelNames.getState().names, ...learned };
  useClaudeAgentModelNames.setState({ names });
  try {
    globalThis.localStorage?.setItem(NAMES_KEY, JSON.stringify(names));
  } catch {
    // Names still apply for this session.
  }
}

export function connectionModelLabel(
  providerType: string | null | undefined,
  modelId: string,
  names: Record<string, string> = useClaudeAgentModelNames.getState().names,
): string {
  if (providerType !== CLAUDE_AGENT_PROVIDER_TYPE) return modelId;
  return names[modelId] || claudeAgentFallbackName(modelId);
}
