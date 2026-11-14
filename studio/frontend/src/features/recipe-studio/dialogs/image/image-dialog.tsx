// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ReactElement } from "react";
import { useMemo } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { ImageConfig } from "../../types";
import { isImageModelConfig } from "../../utils/config-type-guards";
import { findInvalidJinjaReferences } from "../../utils/refs";
import { getAvailableVariables } from "../../utils/variables";
import { AvailableVariables } from "../shared/available-variables";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type ImageDialogProps = {
  config: ImageConfig;
  onUpdate: (patch: Partial<ImageConfig>) => void;
};

export function ImageDialog({
  config,
  onUpdate,
}: ImageDialogProps): ReactElement {
  const configs = useRecipeStudioStore((state) => state.configs);
  const promptId = `${config.id}-prompt`;
  const modelId = `${config.id}-model`;
  const imagePresets = useMemo(
    () =>
      Object.values(configs)
        .filter(isImageModelConfig)
        .map((preset) => preset.name),
    [configs],
  );
  const validReferences = useMemo(
    () => getAvailableVariables(configs, config.id),
    [configs, config.id],
  );
  const invalidRefs = useMemo(
    () => findInvalidJinjaReferences(config.prompt, validReferences),
    [config.prompt, validReferences],
  );

  return (
    <div className="space-y-4">
      <AvailableVariables configId={config.id} />
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-1.5">
        <FieldLabel
          label="Image model preset"
          htmlFor={modelId}
          hint="A Model preset set to Image. Each row gets one image from it."
        />
        {imagePresets.length === 0 ? (
          <p className="rounded-lg border border-border/60 bg-muted/10 px-3 py-2 text-xs text-muted-foreground">
            Add a Model preset step, switch it to Image, then pick it here.
          </p>
        ) : (
          <Select
            value={config.model_alias || undefined}
            // biome-ignore lint/style/useNamingConvention: api schema
            onValueChange={(value) => onUpdate({ model_alias: value })}
          >
            <SelectTrigger className="nodrag w-full" id={modelId}>
              <SelectValue placeholder="Choose an image model preset" />
            </SelectTrigger>
            <SelectContent>
              {imagePresets.map((alias) => (
                <SelectItem key={alias} value={alias}>
                  {alias}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Prompt"
          htmlFor={promptId}
          hint="Describe the image. Other fields fill in per row."
        />
        <Textarea
          id={promptId}
          className="corner-squircle nodrag min-h-24"
          aria-invalid={invalidRefs.length > 0}
          placeholder="A {{ style }} product photo of a {{ subject }}"
          value={config.prompt}
          onChange={(event) => onUpdate({ prompt: event.target.value })}
        />
        {invalidRefs.length > 0 && (
          <p className="text-xs text-destructive">
            Unknown field:{" "}
            {invalidRefs
              .slice(0, 3)
              .map((ref) => `{{ ${ref} }}`)
              .join(", ")}
            {invalidRefs.length > 3 ? ` +${invalidRefs.length - 3} more` : ""}
          </p>
        )}
      </div>
    </div>
  );
}
