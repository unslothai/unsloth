// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { MetricConfigField } from "../api/eval-api";

export function MetricConfigFields({
  fields,
  values,
  onChange,
}: {
  fields: MetricConfigField[];
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}) {
  if (fields.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        This metric has no extra options.
      </p>
    );
  }
  const set = (name: string, value: unknown) =>
    onChange({ ...values, [name]: value });

  return (
    <div className="flex flex-col gap-3">
      {fields.map((f) => {
        const current = values[f.name] ?? f.default;
        if (f.options && f.options.length > 0) {
          return (
            <div key={f.name} className="flex flex-col gap-1.5">
              <Label htmlFor={`mc-${f.name}`}>{f.label}</Label>
              <Select
                value={typeof current === "string" ? current : undefined}
                onValueChange={(v) => set(f.name, v)}
              >
                <SelectTrigger id={`mc-${f.name}`} className="w-full">
                  <SelectValue placeholder="Select…" />
                </SelectTrigger>
                <SelectContent>
                  {f.options.map((opt) => (
                    <SelectItem key={opt} value={opt}>
                      {opt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          );
        }
        if (f.type === "bool") {
          return (
            <div key={f.name} className="flex items-center justify-between gap-3">
              <Label htmlFor={`mc-${f.name}`}>{f.label}</Label>
              <Switch
                id={`mc-${f.name}`}
                checked={Boolean(current)}
                onCheckedChange={(v) => set(f.name, v)}
              />
            </div>
          );
        }
        if (f.type === "float") {
          return (
            <div key={f.name} className="flex flex-col gap-1.5">
              <Label htmlFor={`mc-${f.name}`}>{f.label}</Label>
              <Input
                id={`mc-${f.name}`}
                type="number"
                step="0.01"
                value={current === undefined || current === null ? "" : String(current)}
                onChange={(e) => {
                  if (e.target.value === "") {
                    set(f.name, null);
                    return;
                  }
                  const n = Number(e.target.value);
                  if (!Number.isNaN(n)) set(f.name, n);
                }}
              />
            </div>
          );
        }
        // json | string → textarea (json parsed at submit-time by the form)
        return (
          <div key={f.name} className="flex flex-col gap-1.5">
            <Label htmlFor={`mc-${f.name}`}>{f.label}</Label>
            <Textarea
              id={`mc-${f.name}`}
              value={typeof current === "string" ? current : current == null ? "" : JSON.stringify(current, null, 2)}
              onChange={(e) => set(f.name, e.target.value)}
              className="min-h-24 font-mono text-xs"
              spellCheck={false}
            />
          </div>
        );
      })}
    </div>
  );
}
