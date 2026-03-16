// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { CheckFormatResponse } from "@/features/training/types/datasets";
import { cn } from "@/lib/utils";
import { AlertCircleIcon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Loader2, Sparkles } from "lucide-react";

const CHATML_ROLES = ["system", "user", "assistant"] as const;
const ALPACA_ROLES = ["instruction", "input", "output"] as const;
const SHAREGPT_ROLES = ["system", "human", "gpt"] as const;
const VLM_ROLES = ["image", "text"] as const;
const AUDIO_ROLES = ["audio", "text", "speaker_id"] as const;

const ROLE_LABELS: Record<string, string> = {
  system: "System",
  user: "User",
  assistant: "Assistant",
  human: "Human",
  gpt: "GPT",
  instruction: "Instruction",
  input: "Input",
  output: "Output",
  image: "Image",
  text: "Text",
  audio: "Audio",
  speaker_id: "Speaker ID",
};

export function getAvailableRoles(isVlm: boolean, format?: string, isAudio?: boolean): readonly string[] {
  if (isAudio) return AUDIO_ROLES;
  if (isVlm) return VLM_ROLES;
  if (format === "alpaca") return ALPACA_ROLES;
  if (format === "sharegpt") return SHAREGPT_ROLES;
  return CHATML_ROLES;
}

export function isMappingComplete(
  mapping: Record<string, string>,
  isVlm: boolean,
  format?: string,
  isAudio?: boolean,
): boolean {
  const roles = new Set(Object.values(mapping));
  if (isAudio) return roles.has("audio") && roles.has("text");
  if (isVlm) return roles.has("image") && roles.has("text");
  if (format === "alpaca") return roles.has("instruction") && roles.has("output");
  if (format === "sharegpt") return roles.has("human") && roles.has("gpt");
  return roles.has("user") && roles.has("assistant");
}

export function HeaderRolePicker({
  currentRole,
  onRoleChange,
  availableRoles,
}: {
  currentRole: string | undefined;
  onRoleChange: (role: string | undefined) => void;
  availableRoles: readonly string[];
}) {
  return (
    <Select
      value={currentRole ?? "_none"}
      onValueChange={(v) => onRoleChange(v === "_none" ? undefined : v)}
    >
      <SelectTrigger className="h-6 w-[90px] text-[10px] px-2 py-0 border-dashed cursor-pointer">
        <SelectValue placeholder="Role..." />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="_none" className="text-[11px]">
          None
        </SelectItem>
        {availableRoles.map((role) => (
          <SelectItem key={role} value={role} className="text-[11px]">
            {ROLE_LABELS[role] ?? role}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function DatasetMappingCard({
  mapping,
  mappingOk,
  autoDetected = false,
  isVlm = false,
  isAudio = false,
  format,
  onAiAssist,
  isAiLoading = false,
  aiError,
  advisorNotification,
  advisorSystemPrompt,
}: {
  mapping: Record<string, string>;
  mappingOk: boolean;
  autoDetected?: boolean;
  isVlm?: boolean;
  isAudio?: boolean;
  format?: string;
  onAiAssist?: () => void;
  isAiLoading?: boolean;
  aiError?: string | null;
  advisorNotification?: string | null;
  advisorSystemPrompt?: string;
}) {
  const entries = Object.entries(mapping);
  const requiredLabel = isAudio
    ? "audio and text"
    : isVlm
      ? "image and text"
      : format === "alpaca"
        ? "instruction and output"
        : format === "sharegpt"
          ? "human and gpt"
          : "user and assistant";

  return (
    <div
      className={cn(
        "rounded-xl corner-squircle ring-1 px-5 py-4 mb-4",
        mappingOk
          ? "ring-emerald-200/70 bg-emerald-50/70 text-emerald-950 dark:ring-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-50"
          : "ring-amber-200/70 bg-amber-50/70 text-amber-950 dark:ring-amber-900/50 dark:bg-amber-950/30 dark:text-amber-50",
      )}
    >
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "rounded-xl corner-squircle p-2 shrink-0",
            mappingOk ? "bg-emerald-500/15" : "bg-amber-500/15",
          )}
        >
          <HugeiconsIcon
            icon={mappingOk ? CheckmarkCircle02Icon : AlertCircleIcon}
            className={cn(
              "size-4",
              mappingOk
                ? "text-emerald-700 dark:text-emerald-300"
                : "text-amber-700 dark:text-amber-300",
            )}
          />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-semibold tracking-tight">
            {mappingOk
              ? autoDetected ? "Heuristic-detected mapping" : "Mapping ready"
              : "Map dataset columns"}
          </p>
          <p
            className={cn(
              "text-xs mt-0.5",
              mappingOk
                ? "text-emerald-800/80 dark:text-emerald-200/80"
                : "text-amber-800/80 dark:text-amber-200/80",
            )}
          >
            {mappingOk
              ? autoDetected
                ? "We auto-detected the column mapping below using heuristics. Please review and adjust using the dropdowns in the column headers, or use AI Assist for a smarter mapping."
                : "Looks good. We'll convert this dataset automatically."
              : `Assign roles to columns using the dropdowns in the headers. At minimum, assign ${requiredLabel}.`}
          </p>
          {entries.length > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {entries.map(([col, role]) => (
                <Badge
                  key={col}
                  variant="outline"
                  className="h-6 text-[11px] bg-white/60 dark:bg-transparent"
                >
                  <span className="font-mono">{col}</span>
                  <span className="mx-1 text-muted-foreground/60">&rarr;</span>
                  <span>{ROLE_LABELS[role] ?? role}</span>
                </Badge>
              ))}
            </div>
          )}
          {!mappingOk && entries.length === 0 && (
            <p className="mt-2 text-xs text-amber-800/80 dark:text-amber-200/80">
              Use the dropdowns in the column headers to assign roles.
            </p>
          )}
          {onAiAssist && (
            <div className="mt-3 flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={onAiAssist}
                disabled={isAiLoading}
                className="cursor-pointer bg-white/60 dark:bg-transparent"
              >
                {isAiLoading ? (
                  <>
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    Analyzing dataset...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                    AI Assist
                    <Badge variant="outline" className="ml-1.5 text-[9px] px-1 py-0 h-4 font-medium">Beta</Badge>
                  </>
                )}
              </Button>
              {aiError && (
                <p className="text-xs text-amber-700 dark:text-amber-300">{aiError}</p>
              )}
            </div>
          )}
          {advisorNotification && (
            <div className="mt-3 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2.5 text-xs text-indigo-700 dark:border-indigo-800 dark:bg-indigo-950 dark:text-indigo-300 space-y-2">
              <div className="flex items-start gap-2">
                <Sparkles className="size-3.5 shrink-0 mt-0.5" />
                <span>{advisorNotification}</span>
              </div>
              {advisorSystemPrompt && (
                <div className="pl-5.5 text-[11px] font-mono text-indigo-600/80 dark:text-indigo-400/80">
                  <span className="font-sans font-medium text-indigo-500 dark:text-indigo-400">System:</span>{" "}
                  <span className="break-words">{advisorSystemPrompt}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function DatasetMappingFooter({
  mappingOk,
  isStarting,
  startError,
  onCancel,
  onStartTraining,
}: {
  mappingOk: boolean;
  isStarting: boolean;
  startError: string | null;
  onCancel: () => void;
  onStartTraining: () => Promise<void>;
}) {
  return (
    <div className="mt-3 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[11px] text-muted-foreground/70 leading-relaxed">
          Tip: use the role dropdowns in the column headers to assign roles.
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="cursor-pointer"
            onClick={onCancel}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            className="cursor-pointer"
            disabled={!mappingOk || isStarting}
            onClick={() => void onStartTraining()}
          >
            {isStarting ? "Starting..." : "Continue"}
          </Button>
        </div>
      </div>

      {startError && (
        <p className="text-xs text-red-500 leading-relaxed text-center">
          {startError}
        </p>
      )}
    </div>
  );
}

/** Canonical chatml role for any format-specific role name. */
const TO_CANONICAL: Record<string, string> = {
  user: "user", assistant: "assistant", system: "system",
  instruction: "user", input: "system", output: "assistant",
  human: "user", gpt: "assistant",
  image: "image", text: "text",
  audio: "audio", speaker_id: "speaker_id",
};

/** Chatml → format-specific role names (only for formats that differ). */
const FROM_CANONICAL: Record<string, Record<string, string>> = {
  alpaca: { user: "instruction", system: "input", assistant: "output" },
  sharegpt: { user: "human", assistant: "gpt", system: "system" },
};

/**
 * Remap a column→role mapping between formats.
 * Normalises every role to canonical chatml first, then maps to the target format.
 */
export function remapRolesForFormat(
  mapping: Record<string, string>,
  format?: string,
): Record<string, string> {
  const table = format ? FROM_CANONICAL[format] : undefined;
  const out: Record<string, string> = {};
  for (const [col, role] of Object.entries(mapping)) {
    const canonical = TO_CANONICAL[role] ?? role;
    out[col] = table ? (table[canonical] ?? canonical) : canonical;
  }
  return out;
}

export function deriveDefaultMapping(
  data: CheckFormatResponse,
  isVlm: boolean,
  format?: string,
  isAudio?: boolean,
): Record<string, string> {
  if (data.suggested_mapping) {
    return remapRolesForFormat({ ...data.suggested_mapping }, format);
  }
  if (isAudio) {
    const result: Record<string, string> = {};
    if (data.detected_audio_column) result[data.detected_audio_column] = "audio";
    if (data.detected_text_column) result[data.detected_text_column] = "text";
    if (data.detected_speaker_column) result[data.detected_speaker_column] = "speaker_id";
    return result;
  }
  if (isVlm) {
    const result: Record<string, string> = {};
    if (data.detected_image_column) result[data.detected_image_column] = "image";
    if (data.detected_text_column) result[data.detected_text_column] = "text";
    return result;
  }
  return {};
}
