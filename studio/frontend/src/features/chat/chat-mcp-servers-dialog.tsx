// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Delete02Icon, Edit03Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { RefreshCwIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import {
  type McpServerConfig,
  createMcpServer,
  deleteMcpServer,
  listMcpServers,
  refreshMcpServerTools,
  testMcpServer,
  updateMcpServer,
} from "./api/mcp-servers-api";
type HeaderRow = { id: string; key: string; value: string };

type FormState = {
  displayName: string;
  url: string;
  headers: HeaderRow[];
  useOauth: boolean;
};

const EMPTY_FORM: FormState = {
  displayName: "",
  url: "",
  headers: [],
  useOauth: false,
};

function newRowId(): string {
  return `r_${Math.random().toString(36).slice(2, 10)}`;
}

function headersFromObject(headers: Record<string, string>): HeaderRow[] {
  return Object.entries(headers).map(([k, v]) => ({
    id: newRowId(),
    key: k,
    value: v,
  }));
}

function headersToObject(rows: HeaderRow[]): Record<string, string> | undefined {
  const out: Record<string, string> = {};
  for (const row of rows) {
    const key = row.key.trim();
    if (!key) continue;
    out[key] = row.value;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function isValidUrl(url: string): boolean {
  const trimmed = url.trim();
  if (!trimmed) return false;
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function HeadersEditor({
  rows,
  onChange,
}: {
  rows: HeaderRow[];
  onChange: (rows: HeaderRow[]) => void;
}) {
  const update = (id: string, patch: Partial<HeaderRow>) =>
    onChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  const add = () =>
    onChange([...rows, { id: newRowId(), key: "", value: "" }]);
  const remove = (id: string) =>
    onChange(rows.filter((row) => row.id !== id));

  return (
    <>
      <div className="flex items-center justify-between">
        <Label className="text-sm">Custom headers</Label>
        <Button type="button" variant="ghost" size="sm" onClick={add}>
          <HugeiconsIcon icon={PlusSignIcon} size={14} />
          Add header
        </Button>
      </div>
      {rows.length === 0 ? (
        <div className="text-xs text-muted-foreground">
          Optional. Add an <code>Authorization</code> header here for servers
          that require auth.
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {rows.map((row) => (
            <div key={row.id} className="flex items-center gap-2">
              <Input
                value={row.key}
                placeholder="Header name"
                onChange={(e) => update(row.id, { key: e.target.value })}
              />
              <Input
                value={row.value}
                placeholder="Header value"
                onChange={(e) => update(row.id, { value: e.target.value })}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => remove(row.id)}
                aria-label="Remove header"
              >
                <HugeiconsIcon icon={Delete02Icon} size={14} />
              </Button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export interface ChatMcpServersDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type View =
  | { kind: "list" }
  | { kind: "create" }
  | { kind: "edit"; id: string };

export function ChatMcpServersDialog({
  open,
  onOpenChange,
}: ChatMcpServersDialogProps) {
  const [servers, setServers] = useState<McpServerConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<View>({ kind: "list" });
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [refreshingId, setRefreshingId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const rows = await listMcpServers();
      setServers(rows);
    } catch (err) {
      toast.error("Failed to load MCP servers", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    refresh();
  }, [open, refresh]);

  function startCreate() {
    setView({ kind: "create" });
    setForm(EMPTY_FORM);
  }

  function startEdit(server: McpServerConfig) {
    setView({ kind: "edit", id: server.id });
    setForm({
      displayName: server.display_name,
      url: server.url,
      headers: headersFromObject(server.headers ?? {}),
      useOauth: server.use_oauth ?? false,
    });
  }

  function cancelForm() {
    setView({ kind: "list" });
    setForm(EMPTY_FORM);
  }

  async function testConnection() {
    const trimmedUrl = form.url.trim();
    if (!isValidUrl(trimmedUrl)) {
      toast.error("Enter a valid http:// or https:// URL first");
      return;
    }
    setTesting(true);
    try {
      const result = await testMcpServer({
        url: trimmedUrl,
        headers: headersToObject(form.headers),
        useOauth: form.useOauth,
      });
      if (result.ok) {
        toast.success(
          `Connected (${result.tool_count} tool${result.tool_count === 1 ? "" : "s"})`,
        );
      } else {
        toast.error("Connection failed", {
          description: result.error ?? "Unknown error",
        });
      }
    } catch (err) {
      toast.error("Connection test failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setTesting(false);
    }
  }

  async function submitForm() {
    const trimmedName = form.displayName.trim();
    const trimmedUrl = form.url.trim();
    if (!trimmedName) {
      toast.error("Display name is required");
      return;
    }
    if (!trimmedUrl) {
      toast.error("URL is required");
      return;
    }
    if (!isValidUrl(trimmedUrl)) {
      toast.error("URL must start with http:// or https://");
      return;
    }
    setSaving(true);
    try {
      const headers = headersToObject(form.headers);
      if (view.kind === "edit") {
        await updateMcpServer(view.id, {
          displayName: trimmedName,
          url: trimmedUrl,
          headers: headers ?? null,
          useOauth: form.useOauth,
        });
        toast.success("MCP server updated");
      } else {
        await createMcpServer({
          displayName: trimmedName,
          url: trimmedUrl,
          headers: headers,
          useOauth: form.useOauth,
        });
        toast.success("MCP server added");
      }
      cancelForm();
      await refresh();
    } catch (err) {
      toast.error("Save failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setSaving(false);
    }
  }

  async function removeServer(server: McpServerConfig) {
    const ok = window.confirm(`Delete MCP server "${server.display_name}"?`);
    if (!ok) return;
    try {
      await deleteMcpServer(server.id);
      await refresh();
    } catch (err) {
      toast.error("Delete failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    }
  }

  async function toggleEnabled(server: McpServerConfig, next: boolean) {
    // Optimistic update so the switch doesn't snap back during the round-trip.
    setServers((rows) =>
      rows.map((row) =>
        row.id === server.id ? { ...row, is_enabled: next } : row,
      ),
    );
    try {
      await updateMcpServer(server.id, { isEnabled: next });
    } catch (err) {
      setServers((rows) =>
        rows.map((row) =>
          row.id === server.id ? { ...row, is_enabled: !next } : row,
        ),
      );
      toast.error("Update failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    }
  }

  async function refreshTools(server: McpServerConfig) {
    setRefreshingId(server.id);
    try {
      const result = await refreshMcpServerTools(server.id);
      if (result.ok) {
        toast.success(
          `Refreshed "${server.display_name}" (${result.tool_count} tool${result.tool_count === 1 ? "" : "s"})`,
        );
      } else {
        toast.error(`Refresh failed for "${server.display_name}"`, {
          description: result.error ?? "Unknown error",
        });
      }
    } catch (err) {
      toast.error("Refresh failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setRefreshingId(null);
    }
  }

  const showForm = view.kind !== "list";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>MCP Servers</DialogTitle>
          <DialogDescription>
            Register remote MCP servers.
          </DialogDescription>
        </DialogHeader>

        {showForm ? (
          <div className="flex flex-col gap-4">
            <div className="grid gap-2">
              <Label htmlFor="mcp-display-name">Display name</Label>
              <Input
                id="mcp-display-name"
                value={form.displayName}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, displayName: e.target.value }))
                }
                placeholder="e.g. GitHub MCP"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="mcp-url">URL</Label>
              <Input
                id="mcp-url"
                value={form.url}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, url: e.target.value }))
                }
                placeholder="https://example.com/mcp"
              />
            </div>

            <div className="flex items-start justify-between gap-3">
              <div className="flex flex-col gap-0.5">
                <Label className="text-sm" htmlFor="mcp-oauth">
                  Use OAuth sign-in
                </Label>
                <span className="text-xs text-muted-foreground">
                  For servers that require browser-based authentication
                  (GitHub, Linear, etc.). A browser window will open on first
                  connect.
                </span>
              </div>
              <Switch
                id="mcp-oauth"
                checked={form.useOauth}
                onCheckedChange={(useOauth) =>
                  setForm((prev) => ({ ...prev, useOauth }))
                }
              />
            </div>

            <HeadersEditor
              rows={form.headers}
              onChange={(headers) => setForm((prev) => ({ ...prev, headers }))}
            />

            <div className="flex items-center justify-between gap-2 pt-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={testConnection}
                disabled={testing || saving || !form.url.trim()}
              >
                {testing ? <Spinner /> : null}
                Test connection
              </Button>
              <div className="flex gap-2">
                <Button variant="ghost" onClick={cancelForm} disabled={saving}>
                  Cancel
                </Button>
                <Button onClick={submitForm} disabled={saving}>
                  {saving ? <Spinner /> : null}
                  {view.kind === "edit" ? "Save changes" : "Add server"}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="flex justify-end">
              <Button size="sm" onClick={startCreate}>
                <HugeiconsIcon icon={PlusSignIcon} size={14} />
                Add server
              </Button>
            </div>
            {loading ? (
              <div className="flex justify-center py-6">
                <Spinner />
              </div>
            ) : servers.length === 0 ? (
              <div className="rounded-md border border-dashed py-6 text-center text-sm text-muted-foreground">
                No MCP servers configured yet.
              </div>
            ) : (
              <ul className="flex flex-col divide-y rounded-md border">
                {servers.map((server) => (
                  <li
                    key={server.id}
                    className="flex items-center justify-between gap-3 px-3 py-2"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-medium">
                        {server.display_name}
                      </div>
                      <div className="truncate text-xs text-muted-foreground">
                        {server.url}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Switch
                        checked={server.is_enabled}
                        onCheckedChange={(next) => toggleEnabled(server, next)}
                        aria-label="Enable server"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => refreshTools(server)}
                        aria-label="Refresh tools"
                        title="Refresh tools from this server"
                        disabled={refreshingId === server.id}
                      >
                        {refreshingId === server.id ? (
                          <Spinner />
                        ) : (
                          <RefreshCwIcon size={14} />
                        )}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => startEdit(server)}
                        aria-label="Edit server"
                      >
                        <HugeiconsIcon icon={Edit03Icon} size={14} />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => removeServer(server)}
                        aria-label="Delete server"
                      >
                        <HugeiconsIcon icon={Delete02Icon} size={14} />
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
