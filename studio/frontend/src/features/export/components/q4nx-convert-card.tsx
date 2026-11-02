// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionBox } from "@/components/section-box";
import { SegmentedControl } from "@/components/segmented-control";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { hfApiToken, useHfTokenStore } from "@/features/hub";
import { CpuIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { convertGgufToQ4nx } from "../api/export-api";

type Source = "hub" | "local";
const SOURCE_OPTIONS = [
  { value: "hub", label: "Hugging Face" },
  { value: "local", label: "Local file" },
] as const satisfies readonly { value: Source; label: string }[];

function baseRepoFor(repoId: string): string {
  return repoId.trim().replace(/-GGUF$/i, "");
}

function Field({
  id,
  label,
  ...props
}: { id: string; label: string } & React.ComponentProps<typeof Input>) {
  return (
    <div className="flex flex-col gap-2">
      <label
        htmlFor={id}
        className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground"
      >
        {label}
      </label>
      <Input id={id} {...props} />
    </div>
  );
}

export function Q4nxConvertCard() {
  const hfToken = useHfTokenStore((s) => s.token);
  const [source, setSource] = useState<Source>("hub");
  const [repoId, setRepoId] = useState("");
  const [filename, setFilename] = useState("");
  const [ggufPath, setGgufPath] = useState("");
  const [baseModel, setBaseModel] = useState<string | null>(null);
  const [saveDirectory, setSaveDirectory] = useState("q4nx");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(
    null,
  );

  const effectiveBase =
    baseModel ?? (source === "hub" ? baseRepoFor(repoId) : "");
  const sourceReady =
    source === "hub" ? repoId.trim() && filename.trim() : ggufPath.trim();
  const canConvert =
    !running && sourceReady && effectiveBase.trim() && saveDirectory.trim();

  async function handleConvert() {
    setRunning(true);
    setResult(null);
    try {
      const response = await convertGgufToQ4nx({
        save_directory: saveDirectory.trim(),
        gguf_path: source === "local" ? ggufPath.trim() : null,
        repo_id: source === "hub" ? repoId.trim() : null,
        filename: source === "hub" ? filename.trim() : null,
        base_model: effectiveBase.trim(),
        hf_token: hfApiToken(hfToken) ?? null,
      });
      setResult({
        ok: true,
        text: `Saved to ${response.details?.output_path ?? saveDirectory}`,
      });
    } catch (error) {
      setResult({
        ok: false,
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setRunning(false);
    }
  }

  return (
    <SectionBox
      icon={CpuIcon}
      chipTint="var(--chart-1)"
      title="Convert a GGUF to Q4NX (AMD NPU)"
      description="Convert a Q4_0, Q4_1 or Q4_K_M GGUF to Q4NX. Only supported on AMD NPUs."
      titleAction={
        <SegmentedControl
          value={source}
          options={SOURCE_OPTIONS}
          onValueChange={setSource}
          ariaLabel="GGUF source"
          size="compact"
          className="@md/train-card:w-[calc(220px*var(--ui-space-scale,1))]"
        />
      }
    >
      <div className="flex flex-col gap-6">
        <div className="grid gap-4 @md/train-section:grid-cols-2">
          {source === "hub" ? (
            <>
              <Field
                id="q4nx-repo"
                label="GGUF repo"
                placeholder="unsloth/Qwen3-0.6B-GGUF"
                value={repoId}
                onChange={(e) => setRepoId(e.target.value)}
              />
              <Field
                id="q4nx-file"
                label="GGUF file"
                placeholder="Qwen3-0.6B-Q4_1.gguf"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
              />
            </>
          ) : (
            <Field
              id="q4nx-path"
              label="GGUF file path"
              placeholder="/path/to/model.Q4_1.gguf"
              value={ggufPath}
              onChange={(e) => setGgufPath(e.target.value)}
            />
          )}
          <Field
            id="q4nx-base"
            label="Original model (config and tokenizer)"
            placeholder="unsloth/Qwen3-0.6B"
            value={effectiveBase}
            onChange={(e) => setBaseModel(e.target.value)}
          />
          <Field
            id="q4nx-save"
            label="Save directory"
            value={saveDirectory}
            onChange={(e) => setSaveDirectory(e.target.value)}
          />
        </div>
        <div className="flex items-center justify-between gap-3">
          <p
            className={
              result?.ok === false
                ? "text-xs text-destructive"
                : "text-xs text-muted-foreground"
            }
          >
            {result?.text ?? "No training or GPU needed."}
          </p>
          <Button variant="dark" disabled={!canConvert} onClick={handleConvert}>
            {running && <Spinner />}
            Convert
          </Button>
        </div>
      </div>
    </SectionBox>
  );
}
