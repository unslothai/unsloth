// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  type DiffusionLoraInfo,
  listDiffusionLoras,
} from "@/features/images/api";
import { ImageAdd02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { exportDiffusionLora } from "../api/export-api";

function imageLoraLabel(lora: DiffusionLoraInfo): string {
  return lora.families.length > 0
    ? `${lora.display_name} (${lora.families.join(", ")})`
    : lora.display_name;
}

export function ImageLoraExportCard() {
  const [loras, setLoras] = useState<DiffusionLoraInfo[] | null>(null);
  const [loraId, setLoraId] = useState("");
  const [saveDirectory, setSaveDirectory] = useState("image-loras");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;
    listDiffusionLoras()
      // Only adapters on disk: curated hub entries are already downloadable from their repo.
      .then((list) => {
        if (!cancelled) {
          setLoras(list.filter((l) => l.source === "local"));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLoras([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const canExport = !running && loraId !== "" && saveDirectory.trim() !== "";

  async function handleExport() {
    setRunning(true);
    setResult(null);
    try {
      const response = await exportDiffusionLora({
        lora_id: loraId,
        save_directory: saveDirectory.trim(),
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
    <SectionCard
      icon={<HugeiconsIcon icon={ImageAdd02Icon} className="size-5" />}
      title="Export an image LoRA"
      description="Save a copy of an image generation LoRA you trained (or added) on the Images page: the .safetensors adapter plus its .json metadata."
      className="mt-6"
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1.5">
          <label htmlFor="image-lora-id" className="text-sm font-medium">
            Image LoRA
          </label>
          <Select value={loraId} onValueChange={setLoraId}>
            <SelectTrigger id="image-lora-id" className="w-full">
              <SelectValue
                placeholder={
                  loras === null
                    ? "Loading…"
                    : loras.length === 0
                      ? "No image LoRAs found"
                      : "Select an image LoRA…"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {(loras ?? []).map((l) => (
                <SelectItem key={l.id} value={l.id}>
                  {imageLoraLabel(l)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <label htmlFor="image-lora-save" className="text-sm font-medium">
            Save directory
          </label>
          <Input
            id="image-lora-save"
            value={saveDirectory}
            onChange={(e) => setSaveDirectory(e.target.value)}
          />
        </div>
      </div>
      <div className="flex items-center justify-between gap-3">
        <p
          className={
            result?.ok === false
              ? "text-xs text-destructive"
              : "text-xs text-muted-foreground"
          }
        >
          {result?.text ?? "A plain file copy: no model load or GPU needed."}
        </p>
        <Button disabled={!canExport} onClick={handleExport}>
          {running && <Spinner />}
          Export
        </Button>
      </div>
    </SectionCard>
  );
}
