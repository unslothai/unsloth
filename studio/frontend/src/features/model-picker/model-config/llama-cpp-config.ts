// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LlamaCppConfig =
  | { version: 1; mode: "managed" }
  | { version: 1; mode: "custom"; ini: string; section: string | null };

export interface LlamaCppConfigSummary {
  mode: "custom";
  section: string | null;
  digest: string;
  tuning: Record<string, unknown>;
  request_defaults: Record<string, unknown>;
  diagnostics: string[];
}

export const MAX_LLAMA_CPP_CONFIG_BYTES = 65_536;

/** Shape validation only. The selected server is authoritative for INI semantics. */
export function normalizeLlamaCppConfig(
  value: unknown,
): LlamaCppConfig | undefined {
  if (!value || typeof value !== "object") return undefined;
  const source = value as Record<string, unknown>;
  if (source.version !== 1) return undefined;
  if (source.mode === "managed") return { version: 1, mode: "managed" };
  if (
    source.mode !== "custom" ||
    typeof source.ini !== "string" ||
    (source.section !== null && typeof source.section !== "string") ||
    new TextEncoder().encode(source.ini).length > MAX_LLAMA_CPP_CONFIG_BYTES
  )
    return undefined;
  return {
    version: 1,
    mode: "custom",
    ini: source.ini,
    section: source.section,
  };
}

/** Suggestions for the selector, never a parser or a validation verdict. */
export function customConfigSections(ini: string): string[] {
  return [
    ...new Set(
      [...ini.matchAll(/^\s*\[([^\]\r\n]+)\]\s*(?:[#;].*)?$/gm)]
        .map((match) => match[1])
        .filter((name) => name !== "*"),
    ),
  ];
}

export function llamaCppConfigPayload(config: LlamaCppConfig | undefined) {
  return config === undefined ? {} : { llama_cpp_config: config };
}

export const SAMPLING_WIRE_FIELDS = {
  temperature: "temperature",
  topP: "top_p",
  topK: "top_k",
  minP: "min_p",
  repetitionPenalty: "repetition_penalty",
  presencePenalty: "presence_penalty",
  frequencyPenalty: "frequency_penalty",
  reasoningEnabled: "enable_thinking",
  reasoningEffort: "reasoning_effort",
  preserveThinking: "preserve_thinking",
} as const;

export function explicitSamplingFields(
  snapshot: Record<string, unknown>,
): string[] {
  if (Array.isArray(snapshot.samplingFieldsExplicit)) {
    return snapshot.samplingFieldsExplicit.filter(
      (field): field is string =>
        typeof field === "string" &&
        (Object.values(SAMPLING_WIRE_FIELDS) as string[]).includes(field),
    );
  }
  // Presence in an old saved snapshot is user intent, regardless of its numeric value.
  return Object.entries(SAMPLING_WIRE_FIELDS)
    .filter(([key]) => snapshot[key] !== undefined)
    .map(([, wire]) => wire);
}

export function inheritedSamplingFields(
  snapshot: Record<string, unknown>,
  fallback: Record<string, unknown>,
): string[] {
  const own = new Set(explicitSamplingFields(snapshot));
  const inherited = new Set(explicitSamplingFields(fallback));
  return Object.entries(SAMPLING_WIRE_FIELDS)
    .filter(([key, wire]) =>
      (snapshot[key] !== undefined ? own : inherited).has(wire),
    )
    .map(([, wire]) => wire);
}

export function markSamplingFields<
  T extends { samplingFieldsExplicit?: string[] },
>(params: T, ...fields: string[]): T {
  return {
    ...params,
    samplingFieldsExplicit: [
      ...new Set([
        ...(params.samplingFieldsExplicit ??
          Object.values(SAMPLING_WIRE_FIELDS)),
        ...fields,
      ]),
    ],
  };
}

/** Unset provenance is a legacy user snapshot; an empty list is an automatic seed. */
export function customSamplingPayload(
  config: LlamaCppConfig | null | undefined,
  explicit: readonly string[] | undefined,
) {
  return config?.mode === "custom"
    ? {
        sampling_fields_explicit:
          explicit === undefined
            ? Object.values(SAMPLING_WIRE_FIELDS)
            : [...explicit],
      }
    : {};
}
