// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useId } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  customConfigSections,
  MAX_LLAMA_CPP_CONFIG_BYTES,
  toggledLlamaCppConfig,
  type LlamaCppConfig,
} from "../model-config/llama-cpp-config";

// Module scope: switching back to custom restores the last INI even after a remount.
const lastCustomSource = new Map<
  string,
  { ini: string; section: string | null }
>();

export function CustomLlamaConfigEditor({
  value,
  onChange,
  sourceKey,
  onLoadableChange,
}: {
  value: LlamaCppConfig | undefined;
  onChange: (value: LlamaCppConfig) => void;
  sourceKey: string;
  onLoadableChange: (value: boolean) => void;
}) {
  const id = useId();
  const active = value?.mode === "custom";
  const ini = active ? value.ini : "";
  const section = active ? value.section : null;
  const sections = customConfigSections(ini);
  const oversized =
    new TextEncoder().encode(ini).length > MAX_LLAMA_CPP_CONFIG_BYTES;
  const blank = active && ini.trim().length === 0;
  const needsSection =
    active && section === null && sections.some((name) => name !== "default");
  useEffect(() => {
    if (active) lastCustomSource.set(sourceKey, { ini, section });
  }, [active, ini, section, sourceKey]);
  useEffect(() => {
    onLoadableChange(!active || !(oversized || blank || needsSection));
  }, [active, oversized, blank, needsSection, onLoadableChange]);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-ui-12 font-medium">
          Custom llama.cpp configuration
        </span>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() =>
            onChange(
              toggledLlamaCppConfig(
                value,
                lastCustomSource.get(sourceKey) ?? null,
              ),
            )
          }
        >
          {active ? "Use Studio settings" : "Use custom configuration"}
        </Button>
      </div>
      {active && (
        <>
          <p className="text-ui-11 text-muted-foreground">
            llama-server runs with exactly this INI (llama.cpp preset format);
            Studio adds only the model, host and port. Its sampling values
            become this model&apos;s chat defaults.
          </p>
          <textarea
            id={id}
            aria-label="llama.cpp INI configuration"
            spellCheck={false}
            className="min-h-40 w-full resize-y rounded-md border border-input bg-transparent px-3 py-2 font-mono text-ui-11"
            value={ini}
            onChange={(event) => {
              const nextIni = event.target.value;
              onChange({
                ...value,
                ini: nextIni,
                section: customConfigSections(nextIni).includes(
                  value.section ?? "",
                )
                  ? value.section
                  : null,
              });
            }}
          />
          {sections.some((name) => name !== "default") && (
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
          {(oversized || blank || needsSection) && (
            <p role="alert" className="text-ui-11 text-destructive">
              {oversized
                ? "Configuration must fit within 64 KiB."
                : blank
                  ? "Configuration is empty."
                  : "Choose which section to load."}
            </p>
          )}
        </>
      )}
    </div>
  );
}
