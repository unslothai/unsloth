// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useEffect, useState } from "react";
import { type PostExportHook, fetchPostExportHooks } from "../api/export-api";

export type SelectedPostExportHook = {
  id: string;
  label: string;
  options: Record<string, string>;
  /** The hook is the export method: its source export is temporary and removed once the hook succeeds. */
  asMethod?: boolean;
};

/** Each option's default value (or its first choice). */
export function defaultHookOptions(hook: PostExportHook): Record<string, string> {
  return Object.fromEntries(
    hook.options.map((o) => [o.name, o.default || o.choices[0] || ""]),
  );
}

/** One select per hook option; values are limited to the hook's declared choices. */
export function HookOptionFields({
  hook,
  value,
  onChange,
}: {
  hook: PostExportHook;
  value: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
}) {
  return (
    <>
      {hook.options.map((o) => (
        <label
          key={o.name}
          className="flex items-center justify-between gap-3 text-xs"
        >
          <span className="text-muted-foreground">{o.label}</span>
          <select
            className="h-8 max-w-[60%] rounded-md border bg-background px-2 text-xs"
            value={value[o.name] ?? o.default}
            onChange={(e) => onChange({ ...value, [o.name]: e.target.value })}
          >
            {o.choices.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      ))}
    </>
  );
}

/**
 * Admin-configured post-export hooks (post_export_hooks.json), one switch each -- the same row
 * pattern as "Also convert to Q4NX". Renders nothing when no hook is configured.
 */
export function PostExportHooksPicker({
  exportType,
  localOnly,
  value,
  onChange,
}: {
  exportType: "merged" | "lora";
  localOnly: boolean;
  value: SelectedPostExportHook[];
  onChange: (next: SelectedPostExportHook[]) => void;
}) {
  const [hooks, setHooks] = useState<PostExportHook[]>([]);

  useEffect(() => {
    let alive = true;
    fetchPostExportHooks(exportType)
      .then((h) => {
        if (alive) {
          // A hook configured as a method gets its own method card instead of a switch.
          setHooks(h.filter((hook) => !hook.method));
        }
      })
      .catch(() => {
        if (alive) {
          setHooks([]);
        }
      });
    return () => {
      alive = false;
    };
  }, [exportType]);

  useEffect(() => {
    const ids = new Set(hooks.map((h) => h.id));
    if (value.some((v) => !ids.has(v.id))) {
      onChange(value.filter((v) => ids.has(v.id)));
    }
  }, [hooks, value, onChange]);

  if (hooks.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2" data-testid="post-export-hooks">
      {hooks.map((hook) => {
        const selected = value.find((v) => v.id === hook.id);
        const toggle = (on: boolean) =>
          onChange(
            on
              ? [
                  ...value,
                  {
                    id: hook.id,
                    label: hook.label,
                    options: defaultHookOptions(hook),
                  },
                ]
              : value.filter((v) => v.id !== hook.id),
          );
        return (
          <div key={hook.id} className="space-y-2 rounded-lg border p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <div className="text-sm font-medium">{hook.label}</div>
                <div className="text-xs text-muted-foreground">
                  {localOnly
                    ? hook.available
                      ? hook.description
                      : "Its command is not installed on this machine."
                    : "Runs on local exports only: pick a local destination."}
                </div>
              </div>
              <Switch
                aria-label={hook.label}
                checked={Boolean(selected)}
                onCheckedChange={toggle}
                disabled={!(localOnly && hook.available)}
              />
            </div>
            {selected && (
              <HookOptionFields
                hook={hook}
                value={selected.options}
                onChange={(options) =>
                  onChange(
                    value.map((v) => (v.id === hook.id ? { ...v, options } : v)),
                  )
                }
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
