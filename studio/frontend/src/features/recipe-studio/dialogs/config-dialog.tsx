// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import type { ReactElement } from "react";
import { getBlockDefinitionForConfig } from "../blocks/definitions";
import { renderBlockDialog } from "../blocks/registry";
import type { NodeConfig, SamplerConfig } from "../types";
import { DialogShell } from "./shared/dialog-shell";
import { ValidationBanner } from "./shared/validation-banner";

type ConfigDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  config: NodeConfig | null;
  categoryOptions: SamplerConfig[];
  modelConfigAliases: string[];
  modelProviderOptions: string[];
  localProviderNames: Set<string>;
  toolProfileAliases: string[];
  datetimeOptions: string[];
  onUpdate: (id: string, patch: Partial<NodeConfig>) => void;
  container?: HTMLDivElement | null;
  readOnly?: boolean;
};

export function ConfigDialog({
  open,
  onOpenChange,
  config,
  categoryOptions,
  modelConfigAliases,
  modelProviderOptions,
  localProviderNames,
  toolProfileAliases,
  datetimeOptions,
  onUpdate,
  container,
  readOnly = false,
}: ConfigDialogProps): ReactElement {
  const blockDefinition = getBlockDefinitionForConfig(config);
  const showDropToggle =
    config?.kind === "sampler" ||
    config?.kind === "llm" ||
    config?.kind === "validator" ||
    config?.kind === "expression" ||
    (config?.kind === "seed" &&
      (config.seed_source_type ?? "hf") === "unstructured");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="corner-squircle max-h-[650px] overflow-y-auto overflow-x-hidden sm:max-w-2xl shadow-border"
      >
        <DialogShell
          title={blockDefinition ? blockDefinition.title : undefined}
          description={
            blockDefinition
              ? blockDefinition.description
              : "Choose a step to edit."
          }
        />
        {!config && (
          <div className="text-sm text-muted-foreground">
            Select a step to edit.
          </div>
        )}
        {config && (
          <div className="min-w-0 space-y-4">
            {readOnly && (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                This recipe is locked while a run is in progress.
              </div>
            )}
            <ValidationBanner config={config} />
            <div
              className={readOnly ? "pointer-events-none min-w-0 opacity-75" : "min-w-0"}
            >
              {showDropToggle && (
                <div className="mb-2 flex items-center corner-squircle justify-between gap-3 rounded-2xl border border-border/60 px-3 pt-2 pb-4">
                  <div className="min-w-0">
                    <p className="text-sm font-semibold">Keep out of final dataset</p>
                    <p className="break-words text-xs text-muted-foreground">
                      Use this step while generating, but leave it out of exported rows.
                    </p>
                  </div>
                  <Switch
                    checked={config.drop ?? false}
                    disabled={readOnly}
                    onCheckedChange={(value) => onUpdate(config.id, { drop: value })}
                  />
                </div>
              )}
              {renderBlockDialog(
                config,
                open,
                categoryOptions,
                modelConfigAliases,
                modelProviderOptions,
                localProviderNames,
                toolProfileAliases,
                datetimeOptions,
                onUpdate,
              )}
            </div>
          </div>
        )}
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Done
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
