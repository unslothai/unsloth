// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Delete02Icon,
  Edit03Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { RefreshCwIcon, UploadIcon } from "lucide-react";
import {
  type ChangeEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
import { subscribeToMcpServerMutationSettlements } from "./api/mcp-server-mutation-tracker";
import {
  type McpServerConfig,
  createMcpServer,
  decodeMcpStdioCommand,
  deleteMcpServer,
  encodeMcpStdioCommand,
  importMcpServers,
  listMcpServers,
  refreshMcpServerTools,
  testMcpServer,
  updateMcpServer,
} from "./api/mcp-servers-api";
import {
  type McpStdioSnapshot,
  createMcpStdioSnapshot,
  resolveMcpStdioUrl,
} from "./mcp-server-form";
type HeaderRow = { id: string; key: string; value: string };
type ArgumentRow = { id: string; value: string };
type FormTransport = "unknown" | "http" | "stdio";

type FormState = {
  displayName: string;
  url: string;
  transport: FormTransport;
  arguments: ArgumentRow[];
  stdioSnapshot: McpStdioSnapshot | null;
  headers: HeaderRow[];
  credentialTransport: Exclude<FormTransport, "unknown"> | null;
  useOauth: boolean;
};

const EMPTY_FORM: FormState = {
  displayName: "",
  url: "",
  transport: "unknown",
  arguments: [],
  stdioSnapshot: null,
  headers: [],
  credentialTransport: null,
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

function argumentsFromStrings(arguments_: readonly string[]): ArgumentRow[] {
  return arguments_.map((value) => ({ id: newRowId(), value }));
}

function argumentsToStrings(rows: readonly ArgumentRow[]): string[] {
  return rows.map((row) => row.value);
}

function headersToObject(
  rows: HeaderRow[],
): Record<string, string> | undefined {
  const out: Record<string, string> = {};
  for (const row of rows) {
    const key = row.key.trim();
    if (!key) continue;
    out[key] = row.value;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

// A non-HTTP address is a local stdio command. Case-insensitive to match the backend's is_stdio(),
// so all layers split http-vs-command identically.
function isHttpAddress(value: string): boolean {
  const trimmed = value.trim().toLowerCase();
  return trimmed.startsWith("http://") || trimmed.startsWith("https://");
}

function transportFromAddress(
  value: string,
  credentialTransport: FormState["credentialTransport"] = null,
): FormTransport {
  const trimmed = value.trim().toLowerCase();
  if (!trimmed) {
    return "unknown";
  }
  if (isHttpAddress(value)) {
    return "http";
  }
  if (
    credentialTransport === "http" &&
    ("http://".startsWith(trimmed) || "https://".startsWith(trimmed))
  ) {
    return "unknown";
  }
  return "stdio";
}

function formWithAddress(
  form: FormState,
  url: string,
  preservePartialHttp: boolean,
): FormState {
  const transport = transportFromAddress(
    url,
    preservePartialHttp ? form.credentialTransport : null,
  );
  const nextCredentialTransport =
    transport === "unknown" ? form.credentialTransport : transport;
  const transportChanged =
    transport !== "unknown" &&
    form.credentialTransport !== null &&
    form.credentialTransport !== transport;
  return {
    ...form,
    url,
    transport,
    headers: transportChanged ? [] : form.headers,
    credentialTransport: nextCredentialTransport,
    useOauth: transport === "stdio" ? false : form.useOauth,
  };
}

function isValidAddress(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed) return false;
  if (isHttpAddress(trimmed)) {
    try {
      const parsed = new URL(trimmed);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  }
  // The backend owns stdio parsing and validation. In particular, the browser must not split an
  // executable or duplicate platform-specific quoting rules.
  return true;
}

function ArgumentsEditor({
  rows,
  onChange,
  disabled,
}: {
  rows: ArgumentRow[];
  onChange: (rows: ArgumentRow[]) => void;
  disabled: boolean;
}) {
  const update = (id: string, value: string) =>
    onChange(rows.map((row) => (row.id === id ? { ...row, value } : row)));
  const add = () => onChange([...rows, { id: newRowId(), value: "" }]);
  const remove = (id: string) => onChange(rows.filter((row) => row.id !== id));

  return (
    <div className="grid gap-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm">Arguments</Label>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={add}
          disabled={disabled}
        >
          <HugeiconsIcon icon={PlusSignIcon} size={14} />
          Add argument
        </Button>
      </div>
      {rows.length === 0 ? (
        <div className="text-xs text-muted-foreground">
          Optional. Each row is one argument; row order is preserved.
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {rows.map((row, index) => (
            <div key={row.id} className="flex items-center gap-2">
              <Input
                data-reload-snapshot-sensitive={true}
                value={row.value}
                disabled={disabled}
                aria-label={`Argument ${index + 1}`}
                placeholder={index === 0 ? "e.g. -y" : undefined}
                onChange={(event) => update(row.id, event.target.value)}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => remove(row.id)}
                disabled={disabled}
                aria-label={`Remove argument ${index + 1}`}
              >
                <HugeiconsIcon icon={Delete02Icon} size={14} />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function HeadersEditor({
  rows,
  onChange,
  stdio,
  disabled,
}: {
  rows: HeaderRow[];
  onChange: (rows: HeaderRow[]) => void;
  // stdio servers reuse this editor for environment variables instead of headers.
  stdio: boolean;
  disabled: boolean;
}) {
  const update = (id: string, patch: Partial<HeaderRow>) =>
    onChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  const add = () => onChange([...rows, { id: newRowId(), key: "", value: "" }]);
  const remove = (id: string) => onChange(rows.filter((row) => row.id !== id));

  const copy = stdio
    ? {
        label: "Environment variables",
        add: "Add variable",
        keyPlaceholder: "Variable name",
        valuePlaceholder: "Variable value",
        remove: "Remove variable",
      }
    : {
        label: "Custom headers",
        add: "Add header",
        keyPlaceholder: "Header name",
        valuePlaceholder: "Header value",
        remove: "Remove header",
      };

  return (
    <>
      <div className="flex items-center justify-between">
        <Label className="text-sm">{copy.label}</Label>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={add}
          disabled={disabled}
        >
          <HugeiconsIcon icon={PlusSignIcon} size={14} />
          {copy.add}
        </Button>
      </div>
      {rows.length === 0 ? (
        <div className="text-xs text-muted-foreground">
          {stdio ? (
            "Optional. Environment variables passed to the server process."
          ) : (
            <>
              Optional. Add an <code>Authorization</code> header here for
              servers that require auth.
            </>
          )}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {rows.map((row) => (
            <div key={row.id} className="flex items-center gap-2">
              <Input
                value={row.key}
                disabled={disabled}
                placeholder={copy.keyPlaceholder}
                onChange={(e) => update(row.id, { key: e.target.value })}
              />
              <Input
                data-reload-snapshot-sensitive={true}
                value={row.value}
                disabled={disabled}
                placeholder={copy.valuePlaceholder}
                onChange={(e) => update(row.id, { value: e.target.value })}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => remove(row.id)}
                disabled={disabled}
                aria-label={copy.remove}
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
  const [codecPending, setCodecPending] = useState(false);
  const [decodingCommand, setDecodingCommand] = useState(false);
  const [codecError, setCodecError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [refreshingIds, setRefreshingIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [togglingIds, setTogglingIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [busyIds, setBusyIds] = useState<ReadonlySet<string>>(() => new Set());
  const [confirmingDelete, setConfirmingDelete] =
    useState<McpServerConfig | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const formGenerationRef = useRef(0);
  const actionGenerationRef = useRef(0);
  const importGenerationRef = useRef(0);
  const activeEditIdRef = useRef<string | null>(null);
  const listRefreshGenerationRef = useRef(0);
  const openRef = useRef(open);
  const refreshingIdsRef = useRef(new Set<string>());
  const togglingIdsRef = useRef(new Set<string>());
  const busyIdsRef = useRef(new Set<string>());
  const importingRef = useRef(false);
  openRef.current = open;

  useEffect(() => {
    return () => {
      formGenerationRef.current += 1;
      actionGenerationRef.current += 1;
      activeEditIdRef.current = null;
      openRef.current = false;
    };
  }, []);

  const refresh = useCallback(
    async (waitForPendingMutations = true, minimumMutationEpoch = 0) => {
      const generation = listRefreshGenerationRef.current + 1;
      listRefreshGenerationRef.current = generation;
      setLoading(true);
      try {
        const rows = await listMcpServers({
          waitForPendingMutations,
          minimumMutationEpoch,
        });
        if (listRefreshGenerationRef.current !== generation || !openRef.current)
          return;
        setServers((current) =>
          rows.map((row) => {
            if (!togglingIdsRef.current.has(row.id)) return row;
            const optimistic = current.find(
              (candidate) => candidate.id === row.id,
            );
            return optimistic
              ? { ...row, is_enabled: optimistic.is_enabled }
              : row;
          }),
        );
      } catch (err) {
        if (listRefreshGenerationRef.current !== generation || !openRef.current)
          return;
        toast.error("Failed to load MCP servers", {
          description: err instanceof Error ? err.message : String(err),
        });
      } finally {
        if (listRefreshGenerationRef.current === generation && openRef.current)
          setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (!open) {
      listRefreshGenerationRef.current += 1;
      return;
    }
    const unsubscribe = subscribeToMcpServerMutationSettlements((epoch) => {
      void refresh(false, epoch);
    });
    return () => {
      unsubscribe();
      listRefreshGenerationRef.current += 1;
    };
  }, [open, refresh]);

  useEffect(() => {
    formGenerationRef.current += 1;
    actionGenerationRef.current += 1;
    activeEditIdRef.current = null;
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      setSaving(false);
      setTesting(false);
      setCodecPending(false);
      setDecodingCommand(false);
      setCodecError(null);
      setImporting(importingRef.current);
      setConfirmingDelete(null);
      setRefreshingIds(new Set(refreshingIdsRef.current));
      setTogglingIds(new Set(togglingIdsRef.current));
      setBusyIds(new Set(busyIdsRef.current));
      if (!open) return;
      void refresh();
      // Reset to the list on each open, else a stale create/edit view persists.
      setView({ kind: "list" });
      setForm(EMPTY_FORM);
    });
    return () => {
      cancelled = true;
    };
  }, [open, refresh]);

  function startCreate() {
    formGenerationRef.current += 1;
    activeEditIdRef.current = null;
    setSaving(false);
    setTesting(false);
    setCodecPending(false);
    setDecodingCommand(false);
    setCodecError(null);
    setView({ kind: "create" });
    setForm(EMPTY_FORM);
  }

  async function startEdit(server: McpServerConfig) {
    const generation = formGenerationRef.current + 1;
    formGenerationRef.current = generation;
    activeEditIdRef.current = server.id;
    setSaving(false);
    setTesting(false);
    setCodecError(null);
    setView({ kind: "edit", id: server.id });
    const baseForm: FormState = {
      displayName: server.display_name,
      url: server.url,
      transport: isHttpAddress(server.url) ? "http" : "stdio",
      arguments: [],
      stdioSnapshot: null,
      headers: headersFromObject(server.headers ?? {}),
      credentialTransport: isHttpAddress(server.url) ? "http" : "stdio",
      useOauth: server.use_oauth ?? false,
    };

    if (isHttpAddress(server.url)) {
      setCodecPending(false);
      setDecodingCommand(false);
      setForm(baseForm);
      return;
    }

    setCodecPending(true);
    setDecodingCommand(true);
    setForm(baseForm);
    try {
      const decoded = await decodeMcpStdioCommand(server.url);
      if (
        formGenerationRef.current !== generation ||
        activeEditIdRef.current !== server.id
      ) {
        return;
      }
      setForm({
        ...baseForm,
        url: decoded.command,
        arguments: argumentsFromStrings(decoded.arguments ?? []),
        stdioSnapshot: createMcpStdioSnapshot(
          server.url,
          decoded.command,
          decoded.arguments ?? [],
        ),
        useOauth: false,
      });
    } catch (err) {
      if (
        formGenerationRef.current !== generation ||
        activeEditIdRef.current !== server.id
      ) {
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      setCodecError(message);
      toast.error("Failed to read local command", { description: message });
    } finally {
      if (
        formGenerationRef.current === generation &&
        activeEditIdRef.current === server.id
      ) {
        setCodecPending(false);
        setDecodingCommand(false);
      }
    }
  }

  function cancelForm() {
    formGenerationRef.current += 1;
    activeEditIdRef.current = null;
    setSaving(false);
    setTesting(false);
    setCodecPending(false);
    setDecodingCommand(false);
    setCodecError(null);
    setView({ kind: "list" });
    setForm(EMPTY_FORM);
  }

  function handleOpenChange(next: boolean) {
    // once crud starts, dismissal must wait for the authoritative refresh
    if (!next && ((saving && !codecPending) || busyIdsRef.current.size > 0))
      return;
    if (!next) {
      formGenerationRef.current += 1;
      actionGenerationRef.current += 1;
      activeEditIdRef.current = null;
      setSaving(false);
      setTesting(false);
      setCodecPending(false);
      setDecodingCommand(false);
      setCodecError(null);
      setConfirmingDelete(null);
    }
    onOpenChange(next);
  }

  async function encodeStdioForGeneration(
    generation: number,
    command: string,
    arguments_: string[],
  ): Promise<string | null> {
    setCodecPending(true);
    try {
      const encoded = await encodeMcpStdioCommand({
        command,
        arguments: arguments_,
      });
      if (formGenerationRef.current !== generation) return null;
      return encoded.url;
    } finally {
      if (formGenerationRef.current === generation) setCodecPending(false);
    }
  }

  async function testConnection() {
    if (!form.url.trim() || form.transport === "unknown") {
      toast.error("Enter an http(s):// URL or a local command first");
      return;
    }
    const stdio = form.transport === "stdio";
    if (!stdio && !isValidAddress(form.url)) {
      toast.error("Enter an http(s):// URL or a local command first");
      return;
    }
    const generation = formGenerationRef.current;
    setTesting(true);
    try {
      const url = stdio
        ? await encodeStdioForGeneration(
            generation,
            form.url,
            argumentsToStrings(form.arguments),
          )
        : form.url.trim();
      if (url === null || formGenerationRef.current !== generation) return;
      const result = await testMcpServer({
        url,
        headers: headersToObject(form.headers),
        useOauth: stdio ? false : form.useOauth,
      });
      if (formGenerationRef.current !== generation) return;
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
      if (formGenerationRef.current !== generation) return;
      toast.error("Connection test failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (formGenerationRef.current === generation) setTesting(false);
    }
  }

  async function submitForm() {
    const trimmedName = form.displayName.trim();
    if (!trimmedName) {
      toast.error("Display name is required");
      return;
    }
    if (!form.url.trim() || form.transport === "unknown") {
      toast.error("URL or command is required");
      return;
    }
    const stdio = form.transport === "stdio";
    if (!stdio && !isValidAddress(form.url)) {
      toast.error("Enter an http(s):// URL or a local command");
      return;
    }
    const generation = formGenerationRef.current;
    setSaving(true);
    try {
      const headers = headersToObject(form.headers);
      let url: string | undefined;
      if (stdio) {
        const decision = resolveMcpStdioUrl(
          form.url,
          argumentsToStrings(form.arguments),
          form.stdioSnapshot,
        );
        if (decision.kind === "reuse") {
          url = view.kind === "edit" ? undefined : decision.url;
        } else {
          const encodedUrl = await encodeStdioForGeneration(
            generation,
            decision.command,
            decision.arguments,
          );
          if (encodedUrl === null) return;
          url = encodedUrl;
        }
      } else {
        url = form.url.trim();
      }
      if (formGenerationRef.current !== generation) return;
      if (view.kind === "edit") {
        await updateMcpServer(view.id, {
          displayName: trimmedName,
          url,
          headers: headers ?? null,
          useOauth: stdio ? false : form.useOauth,
        });
        if (formGenerationRef.current !== generation) return;
        toast.success("MCP server updated");
      } else {
        if (url === undefined) return;
        await createMcpServer({
          displayName: trimmedName,
          url,
          headers: headers,
          useOauth: stdio ? false : form.useOauth,
        });
        if (formGenerationRef.current !== generation) return;
        toast.success("MCP server added");
      }
      if (formGenerationRef.current !== generation) return;
      cancelForm();
    } catch (err) {
      if (formGenerationRef.current !== generation) return;
      toast.error("Save failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (formGenerationRef.current === generation) setSaving(false);
    }
  }

  async function onImportFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // let the user re-pick the same file later
    if (!file || importingRef.current) return;
    const generation = actionGenerationRef.current;
    const importGeneration = importGenerationRef.current + 1;
    importGenerationRef.current = importGeneration;
    importingRef.current = true;
    setImporting(true);
    try {
      let config: unknown;
      try {
        config = JSON.parse(await file.text());
      } catch {
        if (actionGenerationRef.current === generation && openRef.current)
          toast.error("Invalid JSON file");
        return;
      }
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      const result = await importMcpServers(config);
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      const parts = [`${result.created.length} added`];
      if (result.skipped.length) parts.push(`${result.skipped.length} skipped`);
      if (result.errors.length) {
        parts.push(
          `${result.errors.length} error${result.errors.length === 1 ? "" : "s"}`,
        );
      }
      const summary = parts.join(", ");
      if (result.errors.length) {
        toast.warning(summary, {
          description: (
            <div className="whitespace-pre-line">
              {result.errors.slice(0, 5).join("\n")}
            </div>
          ),
        });
      } else {
        toast.success(summary);
      }
    } catch (err) {
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      toast.error("Import failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (importGenerationRef.current === importGeneration) {
        importingRef.current = false;
        if (openRef.current) setImporting(false);
      }
    }
  }

  async function removeServer(server: McpServerConfig) {
    if (busyIdsRef.current.has(server.id)) return;
    const generation = actionGenerationRef.current;
    busyIdsRef.current.add(server.id);
    setBusyIds(new Set(busyIdsRef.current));
    try {
      await deleteMcpServer(server.id);
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      setServers((rows) => rows.filter((row) => row.id !== server.id));
    } catch (err) {
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      toast.error("Delete failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      busyIdsRef.current.delete(server.id);
      if (openRef.current) setBusyIds(new Set(busyIdsRef.current));
    }
  }

  async function toggleEnabled(server: McpServerConfig, next: boolean) {
    if (busyIdsRef.current.has(server.id)) return;
    const generation = actionGenerationRef.current;
    busyIdsRef.current.add(server.id);
    togglingIdsRef.current.add(server.id);
    setBusyIds(new Set(busyIdsRef.current));
    setTogglingIds(new Set(togglingIdsRef.current));
    // Optimistic update so the switch doesn't snap back during the round-trip.
    setServers((rows) =>
      rows.map((row) =>
        row.id === server.id ? { ...row, is_enabled: next } : row,
      ),
    );
    try {
      await updateMcpServer(server.id, { isEnabled: next });
    } catch (err) {
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      setServers((rows) =>
        rows.map((row) =>
          row.id === server.id ? { ...row, is_enabled: !next } : row,
        ),
      );
      toast.error("Update failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      busyIdsRef.current.delete(server.id);
      togglingIdsRef.current.delete(server.id);
      if (openRef.current) {
        setBusyIds(new Set(busyIdsRef.current));
        setTogglingIds(new Set(togglingIdsRef.current));
      }
    }
  }

  async function refreshTools(server: McpServerConfig) {
    if (busyIdsRef.current.has(server.id)) return;
    const generation = actionGenerationRef.current;
    busyIdsRef.current.add(server.id);
    refreshingIdsRef.current.add(server.id);
    setBusyIds(new Set(busyIdsRef.current));
    setRefreshingIds(new Set(refreshingIdsRef.current));
    try {
      const result = await refreshMcpServerTools(server.id);
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
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
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      toast.error("Refresh failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      busyIdsRef.current.delete(server.id);
      refreshingIdsRef.current.delete(server.id);
      if (openRef.current) {
        setBusyIds(new Set(busyIdsRef.current));
        setRefreshingIds(new Set(refreshingIdsRef.current));
      }
    }
  }

  const showForm = view.kind !== "list";
  const formPending = importing || codecPending || testing || saving;
  // A local stdio command uses env vars, not headers or OAuth.
  const addressIsCommand = form.transport === "stdio";

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        className="max-w-2xl"
        showCloseButton={!(saving && !codecPending) && busyIds.size === 0}
        aria-busy={decodingCommand}
      >
        <DialogHeader>
          <DialogTitle>MCP Servers</DialogTitle>
          <DialogDescription>
            Register remote (HTTP) or local (stdio command) MCP servers.
          </DialogDescription>
        </DialogHeader>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={onImportFile}
          disabled={importing || formPending}
        />

        {showForm ? (
          <div className="flex flex-col gap-4">
            {view.kind === "create" && (
              <div className="flex items-center justify-between gap-3 rounded-md border border-dashed px-3 py-2">
                <span className="text-xs text-muted-foreground">
                  Import servers from a config file.
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="shrink-0"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={importing || formPending}
                  title="Import servers from a mcpServers JSON config (Claude Desktop, Cursor, VS Code…)"
                >
                  {importing ? <Spinner /> : <UploadIcon size={14} />}
                  Import config
                </Button>
              </div>
            )}
            <div className="grid gap-2">
              <Label htmlFor="mcp-display-name">Display name</Label>
              <Input
                id="mcp-display-name"
                value={form.displayName}
                disabled={formPending}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, displayName: e.target.value }))
                }
                placeholder="e.g. GitHub MCP"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="mcp-url">
                {addressIsCommand
                  ? "Executable"
                  : form.transport === "http"
                    ? "URL"
                    : "URL or executable"}
              </Label>
              <Input
                id="mcp-url"
                value={form.url}
                disabled={formPending}
                onChange={(e) => {
                  const url = e.target.value;
                  setCodecError(null);
                  setForm((prev) => formWithAddress(prev, url, true));
                }}
                onBlur={() => {
                  setForm((prev) =>
                    prev.transport === "unknown" && prev.url.trim()
                      ? formWithAddress(prev, prev.url, false)
                      : prev,
                  );
                }}
                placeholder={
                  addressIsCommand
                    ? "e.g. npx"
                    : form.transport === "http"
                      ? "https://example.com/mcp"
                      : "https://example.com/mcp or npx"
                }
              />
              <span className="text-xs text-muted-foreground">
                {addressIsCommand
                  ? "The executable for a local stdio server. Add each local argument in an Arguments row below."
                  : form.transport === "http"
                    ? "An http(s) URL for a remote server."
                    : "An http(s) URL for a remote server, or an executable for local stdio. Add local arguments in the Arguments rows."}
              </span>
              {decodingCommand && (
                <span
                  role="status"
                  aria-live="polite"
                  className="flex items-center gap-2 text-xs text-muted-foreground"
                >
                  <Spinner />
                  Reading local command…
                </span>
              )}
            </div>

            {addressIsCommand && (
              <ArgumentsEditor
                rows={form.arguments}
                disabled={formPending}
                onChange={(arguments_) =>
                  setForm((prev) => ({ ...prev, arguments: arguments_ }))
                }
              />
            )}

            {codecError && (
              <div className="flex items-center justify-between gap-3">
                <div
                  role="alert"
                  aria-live="assertive"
                  className="text-sm text-destructive"
                >
                  {codecError}
                </div>
                {view.kind === "edit" && (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={formPending}
                    onClick={() => {
                      const server = servers.find(
                        (candidate) => candidate.id === view.id,
                      );
                      if (server) void startEdit(server);
                    }}
                  >
                    Retry
                  </Button>
                )}
              </div>
            )}

            {form.transport === "http" && (
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
                  disabled={formPending}
                  onCheckedChange={(useOauth) =>
                    setForm((prev) => ({ ...prev, useOauth }))
                  }
                />
              </div>
            )}

            {form.transport !== "unknown" && (
              <HeadersEditor
                rows={form.headers}
                onChange={(headers) =>
                  setForm((prev) => ({ ...prev, headers }))
                }
                stdio={addressIsCommand}
                disabled={formPending}
              />
            )}

            <div className="flex items-center justify-between gap-2 pt-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={testConnection}
                disabled={
                  formPending ||
                  codecError !== null ||
                  form.transport === "unknown" ||
                  !form.url.trim()
                }
              >
                {testing ? <Spinner /> : null}
                Test connection
              </Button>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  onClick={cancelForm}
                  disabled={saving && !codecPending}
                >
                  Cancel
                </Button>
                <Button
                  onClick={submitForm}
                  disabled={
                    formPending ||
                    codecError !== null ||
                    form.transport === "unknown"
                  }
                >
                  {saving ? <Spinner /> : null}
                  {view.kind === "edit" ? "Save changes" : "Add server"}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex min-w-0 flex-col gap-3">
            <div className="flex justify-end gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={importing}
                title="Import servers from a mcpServers JSON config (Claude Desktop, Cursor, VS Code…)"
              >
                {importing ? <Spinner /> : <UploadIcon size={14} />}
                Import config
              </Button>
              <Button size="sm" onClick={startCreate} disabled={importing}>
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
                        disabled={importing || busyIds.has(server.id)}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => refreshTools(server)}
                        aria-label="Refresh tools"
                        title="Refresh tools from this server"
                        disabled={importing || busyIds.has(server.id)}
                      >
                        {refreshingIds.has(server.id) ? (
                          <Spinner />
                        ) : (
                          <RefreshCwIcon size={14} />
                        )}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => void startEdit(server)}
                        aria-label="Edit server"
                        disabled={importing || busyIds.has(server.id)}
                      >
                        <HugeiconsIcon icon={Edit03Icon} size={14} />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => setConfirmingDelete(server)}
                        aria-label="Delete server"
                        disabled={importing || busyIds.has(server.id)}
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
      <AlertDialog
        open={open && confirmingDelete !== null}
        onOpenChange={(next) => {
          if (!next) setConfirmingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete MCP server</AlertDialogTitle>
            <AlertDialogDescription>
              Delete{" "}
              <span className="font-medium text-foreground">
                &quot;{confirmingDelete?.display_name}&quot;
              </span>
              ? Its tools stop being available to chats. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const server = confirmingDelete;
                setConfirmingDelete(null);
                if (server) void removeServer(server);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Dialog>
  );
}
