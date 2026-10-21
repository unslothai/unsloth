// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import type { MoeInfo } from "../api/export-api";
import {
  REAP_SAMPLES,
  REAP_SPARSITIES,
  type ReapSettings,
  reapEffectiveSparsity,
  reapKeptExperts,
  reapSparsityAllowed,
} from "../lib/reap";

export function ReapCard({
  moe,
  settings,
  onChange,
  disabled,
}: {
  moe: MoeInfo;
  settings: ReapSettings;
  onChange: (next: ReapSettings) => void;
  disabled?: boolean;
}) {
  const sparsity = reapEffectiveSparsity(settings, moe) ?? settings.sparsity;
  const kept = reapKeptExperts(moe, sparsity);
  const set = (patch: Partial<ReapSettings>) =>
    onChange({ ...settings, ...patch });
  return (
    <div className="space-y-3 rounded-lg border p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="space-y-0.5">
          <div className="text-sm font-medium">Prune experts (REAP)</div>
          <div className="text-xs text-muted-foreground">
            This model routes each token to {moe.top_k} of {moe.num_experts}{" "}
            experts. REAP scores every expert on calibration chats and removes
            the least used ones, for a smaller model that needs less memory.
          </div>
        </div>
        <Switch
          aria-label="Prune experts (REAP)"
          checked={settings.enabled}
          onCheckedChange={(enabled) => set({ enabled })}
          disabled={disabled}
        />
      </div>
      {settings.enabled && (
        <div className="space-y-3">
          <div className="space-y-1.5">
            <div className="text-sm font-medium">Experts to remove</div>
            <div className="flex flex-wrap gap-2">
              {REAP_SPARSITIES.map((s) => (
                <Button
                  key={s}
                  type="button"
                  size="sm"
                  variant={sparsity === s ? "default" : "outline"}
                  onClick={() => set({ sparsity: s })}
                  disabled={disabled || !reapSparsityAllowed(moe, s)}
                  title={
                    reapSparsityAllowed(moe, s)
                      ? undefined
                      : `Would keep fewer than the ${moe.top_k} experts each token is routed to`
                  }
                >
                  {Math.round(s * 100)}%
                </Button>
              ))}
            </div>
            <div className="text-xs text-muted-foreground">
              Keeps {kept} of {moe.num_experts} experts in every MoE layer.
              Quality drops more the more you remove; 25% is the safer start.
            </div>
          </div>
          <div className="space-y-1.5">
            <div className="text-sm font-medium">Calibration</div>
            <div className="flex flex-wrap gap-2">
              {REAP_SAMPLES.map((o) => (
                <Button
                  key={o.value}
                  type="button"
                  size="sm"
                  variant={settings.samples === o.value ? "default" : "outline"}
                  onClick={() => set({ samples: o.value })}
                  title={o.hint}
                  disabled={disabled}
                >
                  {o.label}
                </Button>
              ))}
            </div>
            <Input
              aria-label="Calibration dataset"
              placeholder="mlabonne/open-perfectblend (default)"
              value={settings.dataset}
              onChange={(e) => set({ dataset: e.target.value })}
              disabled={disabled}
            />
            <div className="text-xs text-muted-foreground">
              Optional Hugging Face dataset with chat or text rows. Use data
              like what the model will see, for example your training set.
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            The model is loaded in 16-bit to score the experts, so this needs
            more memory than a 4-bit export. Your checkpoint is not changed; the
            export folder gets a reap_report.json listing the kept experts.
          </div>
        </div>
      )}
    </div>
  );
}
