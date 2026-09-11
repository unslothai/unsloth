// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useId, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
// eslint-disable-next-line no-restricted-imports -- The chat barrel imports the picker; validation needs only its API leaf.
import { validateModel } from "@/features/chat/api/chat-api";
import { consumeNativePathToken } from "@/features/native-intents";
import {
  customConfigSections,
  MAX_LLAMA_CPP_CONFIG_BYTES,
  type LlamaCppConfig,
  type LlamaCppConfigSummary,
} from "../model-config/llama-cpp-config";

export function CustomLlamaConfigEditor({
  value,
  onChange,
  modelPath,
  ggufVariant,
  hfToken,
  nativePathToken,
  onLoadableChange,
  effectiveSummary,
}: {
  value: LlamaCppConfig | undefined;
  onChange: (value: LlamaCppConfig) => void;
  modelPath: string;
  ggufVariant?: string | null;
  hfToken: string | null;
  nativePathToken?: string | null;
  onLoadableChange: (value: boolean) => void;
  effectiveSummary?: LlamaCppConfigSummary | null;
}) {
  const id = useId();
  const revision = useRef(0);
  const [validationResult, setResult] = useState<{
    inputKey: string;
    message: string;
    summary?: LlamaCppConfigSummary | null;
    valid: boolean;
  } | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const active = value?.mode === "custom";
  const ini = active ? value.ini : "";
  const section = active ? value.section : null;
  const inputKey = JSON.stringify([
    modelPath,
    ggufVariant,
    active,
    ini,
    section,
  ]);
  const [previousInputKey, setPreviousInputKey] = useState(inputKey);
  if (previousInputKey !== inputKey) {
    setPreviousInputKey(inputKey);
    setResult(null);
    setBusyKey(null);
  }
  const result =
    validationResult?.inputKey === inputKey ? validationResult : null;
  const busy = busyKey === inputKey;
  const sections = customConfigSections(ini);
  const oversized =
    new TextEncoder().encode(ini).length > MAX_LLAMA_CPP_CONFIG_BYTES;
  useEffect(() => {
    revision.current += 1;
    // Load repeats authoritative preflight. An unvalidated draft may still be submitted.
    onLoadableChange(!active || !oversized);
  }, [
    ini,
    section,
    active,
    oversized,
    modelPath,
    ggufVariant,
    onLoadableChange,
  ]);
  const validate = async () => {
    const current = ++revision.current;
    setBusyKey(inputKey);
    try {
      const lease = nativePathToken
        ? (await consumeNativePathToken(nativePathToken, "validate-model"))
            .nativePathLease
        : null;
      const response = await validateModel({
        model_path: modelPath,
        gguf_variant: ggufVariant,
        hf_token: hfToken,
        nativePathLease: lease,
        max_seq_length: 4096,
        load_in_4bit: false,
        is_lora: false,
        llama_cpp_config: value,
      });
      if (current !== revision.current) return;
      const valid =
        response.valid && response.llama_cpp_config_summary?.mode === "custom";
      setResult({
        inputKey,
        valid,
        message: valid
          ? "Configuration validated."
          : response.message || "Configuration could not be validated.",
        summary: response.llama_cpp_config_summary,
      });
      onLoadableChange(valid);
    } catch (error) {
      if (current !== revision.current) return;
      setResult({
        inputKey,
        valid: false,
        message: error instanceof Error ? error.message : "Validation failed.",
      });
      onLoadableChange(false);
    } finally {
      if (current === revision.current) setBusyKey(null);
    }
  };
  const summary = result?.summary ?? effectiveSummary;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-ui-12 font-medium">
          Custom llama.cpp configuration
        </span>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() =>
            onChange(
              active
                ? { version: 1, mode: "managed" }
                : { version: 1, mode: "custom", ini: "[*]\n", section: null },
            )
          }
        >
          {active ? "Use Studio settings" : "Use custom configuration"}
        </Button>
      </div>
      {active && (
        <>
          <p className="text-ui-11 text-muted-foreground">
            Engine tuning comes from this configuration. Chat settings you edit
            still apply.
          </p>
          <textarea
            id={id}
            aria-label="llama.cpp INI configuration"
            spellCheck={false}
            className="min-h-40 w-full resize-y rounded-md border border-input bg-transparent px-3 py-2 font-mono text-ui-11"
            value={ini}
            onChange={(event) =>
              onChange({ ...value, ini: event.target.value })
            }
          />
          {sections.length > 0 && (
            <div className="space-y-1.5">
              <label className="text-ui-11" htmlFor={`${id}-section`}>
                Section
              </label>
              <Select
                value={section ?? ""}
                onValueChange={(next) => onChange({ ...value, section: next })}
              >
                <SelectTrigger id={`${id}-section`}>
                  <SelectValue placeholder="Choose a section" />
                </SelectTrigger>
                <SelectContent>
                  {sections.map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          {oversized && (
            <p role="alert" className="text-ui-11 text-destructive">
              Configuration must fit within 64 KiB.
            </p>
          )}
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={busy || oversized}
            onClick={() => void validate()}
          >
            {busy ? "Validating…" : "Validate configuration"}
          </Button>
          {result && (
            <p
              role="status"
              className={`text-ui-11 ${result.valid ? "text-muted-foreground" : "text-destructive"}`}
            >
              {result.message}
            </p>
          )}
          {summary && (
            <details className="text-ui-11 text-muted-foreground">
              <summary>
                Effective configuration
                {summary.section ? ` · ${summary.section}` : ""}
              </summary>
              <pre className="mt-2 whitespace-pre-wrap break-all">
                {JSON.stringify(
                  {
                    tuning: summary.tuning,
                    sampling: summary.request_defaults,
                  },
                  null,
                  2,
                )}
              </pre>
              {summary.diagnostics.map((message, index) => (
                <p key={`${index}-${message}`}>{message}</p>
              ))}
            </details>
          )}
        </>
      )}
    </div>
  );
}
