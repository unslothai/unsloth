// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { authFetch } from "@/features/auth";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

type WikiEnvKind = "bool" | "int" | "float" | "string";

type WikiEnvVariable = {
  name: string;
  kind: WikiEnvKind;
  description: string;
  default_value: string;
  current_value: string;
  source: "environment" | "default";
  has_override: boolean;
  override_value?: string | null;
  minimum?: number | null;
  maximum?: number | null;
};

type WikiEnvConfigResponse = {
  status: string;
  variables: WikiEnvVariable[];
  overrides_file: string;
  restart_supported: boolean;
};

type WikiEnvSetResponse = {
  status: string;
  updated: string[];
  cleared: string[];
  invalid: Record<string, string>;
  overrides_file: string;
  restart_supported: boolean;
  restart_scheduled: boolean;
};

interface WikiBehaviourDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const CATEGORY_ORDER = [
  "Watcher & Ingest",
  "RAG & Logging",
  "Engine Context",
  "Ranking",
  "Auto Analysis",
  "Quality & Maintenance",
  "Web Enrichment",
  "Other",
] as const;

const EXPECTED_RUNTIME_WIKI_ENV_VARS = 58;

function categoryForVariable(name: string): (typeof CATEGORY_ORDER)[number] {
  if (
    name.includes("ENGINE_ENRICH_WEB") ||
    name.includes("ENGINE_ENRICH_FILL") ||
    name.includes("ENGINE_ENRICH_LLM_SELECTOR")
  ) {
    return "Web Enrichment";
  }
  if (name.includes("AUTO_ANALYSIS")) {
    return "Auto Analysis";
  }
  if (
    name.includes("LOW_UNIQUE") ||
    name.includes("AUTO_RETRY_FALLBACK_ANALYSES") ||
    name.includes("MERGE_MAINTENANCE") ||
    name.includes("KNOWLEDGE_MAX_INCREMENTAL") ||
    name.includes("ENRICH_REFRESH_OLDEST_NON_FALLBACK") ||
    name.includes("ENRICH_REPAIR_ANSWER_LINKS")
  ) {
    return "Quality & Maintenance";
  }
  if (name.includes("ENGINE_RANKING") || name.includes("ENGINE_LLM_RERANK")) {
    return "Ranking";
  }
  if (name.includes("ENGINE_")) {
    return "Engine Context";
  }
  if (
    name.includes("_RAG_") ||
    name.includes("_INDEX_") ||
    name.includes("LOG_INJECTED") ||
    name.includes("LLM_MAX_TOKENS")
  ) {
    return "RAG & Logging";
  }
  if (
    name.includes("_WATCHER") ||
    name.includes("AUTO_QUERY") ||
    name.includes("CHAT_HISTORY") ||
    name.includes("PENDING_INGEST") ||
    name.includes("AUTO_LINT") ||
    name.includes("WIKI_VAULT")
  ) {
    return "Watcher & Ingest";
  }
  return "Other";
}

async function parseApiErrorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.clone().json()) as { detail?: unknown };
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
  } catch {
    // Ignore parse failures and fall back to status text.
  }
  return `Request failed (${response.status})`;
}

export function WikiBehaviourDialog({
  open,
  onOpenChange,
}: WikiBehaviourDialogProps) {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [query, setQuery] = useState("");
  const [variables, setVariables] = useState<WikiEnvVariable[]>([]);
  const [draftValues, setDraftValues] = useState<Record<string, string>>({});
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [restartSupported, setRestartSupported] = useState(false);
  const [overridesFile, setOverridesFile] = useState("");

  const currentValues = useMemo(
    () => Object.fromEntries(variables.map((item) => [item.name, item.current_value])),
    [variables],
  );

  const changedNames = useMemo(
    () =>
      variables
        .filter((item) => (draftValues[item.name] ?? item.current_value) !== item.current_value)
        .map((item) => item.name),
    [draftValues, variables],
  );

  const groupedVariables = useMemo(() => {
    const filtered = variables.filter((item) => {
      if (!query.trim()) return true;
      const needle = query.trim().toLowerCase();
      return (
        item.name.toLowerCase().includes(needle) ||
        item.description.toLowerCase().includes(needle)
      );
    });

    const grouped = new Map<(typeof CATEGORY_ORDER)[number], WikiEnvVariable[]>();
    for (const item of filtered) {
      const category = categoryForVariable(item.name);
      const rows = grouped.get(category) ?? [];
      rows.push(item);
      grouped.set(category, rows);
    }

    return CATEGORY_ORDER.filter((category) => grouped.has(category)).map((category) => ({
      category,
      items: (grouped.get(category) ?? []).sort((a, b) => a.name.localeCompare(b.name)),
    }));
  }, [query, variables]);

  const visibleCount = useMemo(
    () => groupedVariables.reduce((count, group) => count + group.items.length, 0),
    [groupedVariables],
  );
  const hasRuntimeCountMismatch = variables.length < EXPECTED_RUNTIME_WIKI_ENV_VARS;

  useEffect(() => {
    if (!open) return;
    setQuery("");
    void loadConfig();
  }, [open]);

  async function loadConfig(): Promise<void> {
    setLoading(true);
    setFieldErrors({});
    try {
      const response = await authFetch("/api/inference/wiki/env", {
        method: "GET",
      });
      if (!response.ok) {
        throw new Error(await parseApiErrorMessage(response));
      }

      const payload = (await response.json()) as WikiEnvConfigResponse;
      const nextVariables = Array.isArray(payload.variables) ? payload.variables : [];
      setVariables(nextVariables);
      setDraftValues(
        Object.fromEntries(nextVariables.map((item) => [item.name, item.current_value])),
      );
      setRestartSupported(Boolean(payload.restart_supported));
      setOverridesFile(payload.overrides_file ?? "");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load wiki settings";
      toast.error("Could not load wiki behaviour", { description: message });
    } finally {
      setLoading(false);
    }
  }

  function setDraftValue(name: string, value: string): void {
    setDraftValues((current) => ({ ...current, [name]: value }));
    setFieldErrors((current) => {
      if (!(name in current)) return current;
      const next = { ...current };
      delete next[name];
      return next;
    });
  }

  async function handleSave(): Promise<void> {
    if (saving) return;

    const updates: Record<string, string | null> = {};
    for (const item of variables) {
      const next = draftValues[item.name] ?? item.current_value;
      if (next === item.current_value) continue;
      updates[item.name] = next.trim() === "" ? null : next;
    }

    if (Object.keys(updates).length === 0) {
      toast.warning("No wiki behaviour changes to apply");
      return;
    }

    setSaving(true);
    try {
      const response = await authFetch("/api/inference/wiki/env", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          values: updates,
          restart_backend: true,
        }),
      });
      if (!response.ok) {
        throw new Error(await parseApiErrorMessage(response));
      }

      const payload = (await response.json()) as WikiEnvSetResponse;
      const invalidCount = Object.keys(payload.invalid ?? {}).length;
      setFieldErrors(payload.invalid ?? {});
      setOverridesFile(payload.overrides_file ?? overridesFile);
      setRestartSupported(Boolean(payload.restart_supported));

      if (invalidCount > 0) {
        toast.error("Some values were rejected", {
          description: `${invalidCount} field(s) need correction before they can be applied.`,
        });
      }

      const appliedCount =
        (Array.isArray(payload.updated) ? payload.updated.length : 0) +
        (Array.isArray(payload.cleared) ? payload.cleared.length : 0);

      if (appliedCount === 0 && invalidCount === 0) {
        toast.warning("No settings were changed");
        return;
      }

      if (payload.restart_scheduled) {
        toast.success("Wiki behaviour updated", {
          description: "Backend restart scheduled so changes take effect.",
        });
        onOpenChange(false);
        return;
      }

      if (appliedCount > 0 && !payload.restart_supported) {
        toast.warning("Wiki behaviour updated", {
          description: "Automatic restart is unavailable. Restart backend manually.",
        });
      }

      await loadConfig();
      if (invalidCount === 0) {
        onOpenChange(false);
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to apply wiki behaviour";
      toast.error("Could not apply wiki behaviour", { description: message });
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl gap-4 p-0 sm:max-w-[min(96vw,1200px)]">
        <DialogHeader className="border-b border-border/70 px-6 pt-5 pb-4">
          <DialogTitle className="font-heading text-lg">Edit Wiki Behaviour</DialogTitle>
          <DialogDescription>
            Tune wiki env variables, then apply changes with a backend restart.
          </DialogDescription>
          <div className="flex flex-wrap items-center gap-2 pt-2 text-xs text-muted-foreground">
            <Badge variant="outline">Variables: {variables.length}</Badge>
            <Badge variant="outline">Visible: {visibleCount}</Badge>
            <Badge variant="outline">Pending changes: {changedNames.length}</Badge>
            <Badge variant={restartSupported ? "secondary" : "destructive"}>
              {restartSupported ? "Auto restart available" : "Manual restart required"}
            </Badge>
            {hasRuntimeCountMismatch ? (
              <Badge variant="destructive">
                Loaded {variables.length}/{EXPECTED_RUNTIME_WIKI_ENV_VARS}
              </Badge>
            ) : null}
            {overridesFile ? (
              <span className="truncate">Overrides file: {overridesFile}</span>
            ) : null}
          </div>
          {hasRuntimeCountMismatch ? (
            <p className="pt-1 text-xs text-destructive">
              The backend returned fewer wiki runtime variables than expected for this build.
              If you recently updated code, restart backend and re-open this dialog.
            </p>
          ) : null}
          <div className="pt-2">
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter by env name or description"
              className="h-9"
            />
          </div>
        </DialogHeader>

        <ScrollArea className="h-[min(68vh,760px)] px-6 pb-2">
          {loading ? (
            <div className="py-8 text-sm text-muted-foreground">Loading wiki settings...</div>
          ) : groupedVariables.length === 0 ? (
            <div className="py-8 text-sm text-muted-foreground">
              No wiki variables match the current filter.
            </div>
          ) : (
            <div className="space-y-6 py-4">
              {groupedVariables.map(({ category, items }) => (
                <section key={category} className="space-y-3">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold tracking-wide text-foreground/90">
                      {category}
                    </h3>
                    <Badge variant="outline">{items.length}</Badge>
                  </div>

                  <div className="space-y-2.5">
                    {items.map((item) => {
                      const value = draftValues[item.name] ?? item.current_value;
                      const isChanged = value !== item.current_value;
                      const error = fieldErrors[item.name];
                      return (
                        <div
                          key={item.name}
                          className={cn(
                            "rounded-2xl border border-border/70 bg-background/70 p-3",
                            isChanged && "border-primary/40",
                            error && "border-destructive/70",
                          )}
                        >
                          <div className="flex flex-wrap items-center gap-2 pb-2">
                            <p className="text-xs font-semibold tracking-wide text-foreground/90">
                              {item.name}
                            </p>
                            <Badge variant="outline">{item.kind}</Badge>
                            <Badge variant="outline">
                              {item.source === "environment" ? "Environment" : "Default"}
                            </Badge>
                            {item.has_override ? <Badge variant="secondary">Persisted</Badge> : null}
                            {isChanged ? <Badge variant="secondary">Edited</Badge> : null}
                          </div>

                          <p className="pb-2 text-xs text-muted-foreground">{item.description}</p>

                          <div className="flex flex-wrap items-center gap-2">
                            {item.kind === "bool" ? (
                              <select
                                value={value}
                                onChange={(event) => setDraftValue(item.name, event.target.value)}
                                disabled={saving}
                                className={cn(
                                  "bg-input/30 border-input focus-visible:border-ring focus-visible:ring-ring/50 h-9 min-w-[180px] rounded-4xl border px-3 text-sm outline-none focus-visible:ring-[3px]",
                                  error && "border-destructive/70 focus-visible:border-destructive",
                                )}
                              >
                                <option value="true">true</option>
                                <option value="false">false</option>
                              </select>
                            ) : (
                              <Input
                                value={value}
                                onChange={(event) => setDraftValue(item.name, event.target.value)}
                                disabled={saving}
                                className={cn("h-9 min-w-[240px] flex-1", error && "border-destructive/70")}
                              />
                            )}

                            <Button
                              type="button"
                              variant="ghost"
                              size="xs"
                              disabled={saving}
                              onClick={() => setDraftValue(item.name, item.default_value)}
                            >
                              Default
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="xs"
                              disabled={saving}
                              onClick={() => setDraftValue(item.name, "")}
                            >
                              Clear
                            </Button>
                          </div>

                          <div className="pt-2 text-xs text-muted-foreground">
                            <span>Current: {currentValues[item.name] ?? item.current_value}</span>
                            <span className="px-2">•</span>
                            <span>Default: {item.default_value}</span>
                            {item.minimum !== null && item.minimum !== undefined ? (
                              <>
                                <span className="px-2">•</span>
                                <span>Min: {item.minimum}</span>
                              </>
                            ) : null}
                            {item.maximum !== null && item.maximum !== undefined ? (
                              <>
                                <span className="px-2">•</span>
                                <span>Max: {item.maximum}</span>
                              </>
                            ) : null}
                          </div>

                          {error ? (
                            <p className="pt-2 text-xs font-medium text-destructive">{error}</p>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </section>
              ))}
            </div>
          )}
        </ScrollArea>

        <DialogFooter className="border-t border-border/70 px-6 py-4 sm:justify-between">
          <p className="text-xs text-muted-foreground">
            Empty values clear persisted overrides and fall back to defaults.
          </p>
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="button" variant="dark" onClick={() => void handleSave()} disabled={saving || loading}>
              {saving ? "Applying..." : "Apply & Restart"}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
