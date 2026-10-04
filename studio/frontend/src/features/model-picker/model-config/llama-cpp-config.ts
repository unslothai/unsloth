// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LlamaCppConfig =
  | { version: 1; mode: "managed" }
  | { version: 1; mode: "custom"; ini: string; section: string | null };

export interface LlamaCppConfigSummary {
  mode: "custom";
  section: string | null;
  options: Record<string, string | boolean>;
  request_defaults: Record<string, number>;
  diagnostics: string[];
}

export const MANAGED_LLAMA_CPP_CONFIG: LlamaCppConfig = {
  version: 1,
  mode: "managed",
};

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
    source.ini.trim().length === 0 ||
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

/** The mode switch: coming back to custom restores the last source instead of a blank one. */
export function toggledLlamaCppConfig(
  value: LlamaCppConfig | undefined,
  lastCustom: { ini: string; section: string | null } | null,
): LlamaCppConfig {
  if (value?.mode === "custom") return { version: 1, mode: "managed" };
  return {
    version: 1,
    mode: "custom",
    ini: lastCustom?.ini ?? "[*]\n",
    section: lastCustom?.section ?? null,
  };
}

/** Suggestions for the selector, never a parser or a validation verdict. */
// llama.cpp preset grammar: keys above the first header form the "default" section.
export function customConfigSections(ini: string): string[] {
  const header = /^\[[ \t]*([^\]\r\n]+)\][ \t]*(?:[#;][^\r\n]*)?\r?$/gm;
  const firstHeader = ini.search(header);
  const preamble = firstHeader < 0 ? ini : ini.slice(0, firstHeader);
  const hasDefault = /^[ \t]*[A-Za-z_]/m.test(preamble);
  return [
    ...new Set([
      ...(hasDefault ? ["default"] : []),
      ...[...ini.matchAll(header)]
        .map((match) => match[1].trim())
        .filter((name) => name !== "*"),
    ]),
  ];
}

export function llamaCppConfigPayload(
  config: LlamaCppConfig | undefined,
  options: { isDiffusion?: boolean } = {},
) {
  // The diffusion runner has no llama-server; explicit managed keeps the backend from inheriting custom.
  if (options.isDiffusion)
    return { llama_cpp_config: MANAGED_LLAMA_CPP_CONFIG };
  if (config === undefined) return {};
  return { llama_cpp_config: config };
}
