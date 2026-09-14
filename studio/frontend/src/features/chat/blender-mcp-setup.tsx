// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";
import { openLink } from "@/lib/open-link";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { subscribeToMcpServerMutationSettlements } from "./api/mcp-server-mutation-tracker";
import {
  type McpBuiltinConfig,
  type McpServerConfig,
  listMcpBuiltins,
  testBlenderMcp,
  updateBlenderMcp,
  updateMcpServer,
} from "./api/mcp-servers-api";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

export function BlenderMcpSetup({ servers, disabled, onBusyChange }: {
  servers: McpServerConfig[];
  disabled: boolean;
  onBusyChange: (busy: boolean) => void;
}) {
  const [config, setConfig] = useState<McpBuiltinConfig | null>(null);
  const [port, setPort] = useState("9876");
  const [blenderPath, setBlenderPath] = useState("");
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [connection, setConnection] = useState<"checking" | "ready" | "partial" | "failed" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const generation = useRef(0);
  const busyRef = useRef(false);
  const initialized = useRef(false);
  const mounted = useRef(false);

  const refresh = useCallback(async (wait = true) => {
    const current = ++generation.current;
    try {
      const rows = await listMcpBuiltins(wait);
      if (generation.current !== current) return;
      const blender = rows.find((row) => row.builtin_id === "blender");
      if (!blender) throw new Error("Blender MCP is not available from this Studio backend.");
      setConfig(blender);
      setLoadError(null);
      if (!initialized.current) {
        initialized.current = true;
        setPort(String(blender.port));
        setBlenderPath(blender.blender_path);
      }
    } catch (err) {
      if (generation.current === current) setLoadError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    let cancelled = false;
    queueMicrotask(() => { if (!cancelled) void refresh(); });
    const unsubscribe = subscribeToMcpServerMutationSettlements(() => void refresh(false));
    return () => {
      cancelled = true;
      mounted.current = false;
      unsubscribe();
      generation.current += 1;
      onBusyChange(false);
    };
  }, [refresh, onBusyChange]);

  const validPort = /^\d+$/.test(port) && Number(port) >= 1 && Number(port) <= 65535;
  const locked = disabled || busy;
  const duplicate = servers.some((server) => !server.builtin_id && /blender|blmcp/i.test(`${server.display_name} ${server.url}`));

  async function act(action: "test" | "save" | "enable" | "disable") {
    if (busyRef.current || disabled) return;
    busyRef.current = true;
    setBusy(true);
    onBusyChange(true);
    setError(null);
    setMessage(null);
    setConnection(action === "test" ? "checking" : null);
    try {
      const settings = { port: Number(port), blender_path: blenderPath.trim() };
      if (action === "test") {
        const result = await testBlenderMcp({ ...settings, consent });
        if (!mounted.current) return;
        if (!result.ok) throw new Error(result.error ?? "MCP connection failed. Retry or check the Studio backend.");
        setConnection(result.blender_ready ? "ready" : "partial");
        setMessage(`MCP connected (${result.tool_count} tools). ${result.blender_ready
          ? "Blender ready."
          : result.blender_error ?? "Blender is not connected. Open Setup help below."}`);
      } else if (action === "disable" && config?.server_id) {
        await updateMcpServer(config.server_id, { isEnabled: false });
        if (!mounted.current) return;
        setMessage("Blender tools disabled. Your Blender window remains open.");
      } else {
        const updated = await updateBlenderMcp({ ...settings, is_enabled: action === "enable", consent });
        if (!mounted.current) return;
        if (updated.is_enabled) useChatRuntimeStore.getState().setMcpEnabledForChat(true);
        setConfig(updated);
        setMessage(updated.is_enabled
          ? "Blender MCP enabled for this chat. Scene tools require a connected Blender add-on."
          : "Settings saved; Blender is disabled.");
      }
    } catch (err) {
      if (mounted.current && action === "test") setConnection("failed");
      if (mounted.current) setError(err instanceof Error ? err.message : String(err));
    } finally {
      busyRef.current = false;
      if (mounted.current) {
        setBusy(false);
        onBusyChange(false);
      }
    }
  }

  return (
    <section className="space-y-4 rounded-xl border p-4" aria-label="Blender MCP">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-sm font-medium">Blender</h3>
          {connection && (
            <span role="status" className={`inline-flex items-center gap-1.5 text-xs ${
              connection === "ready" ? "text-emerald-600 dark:text-emerald-400"
                : connection === "partial" ? "text-amber-600 dark:text-amber-400"
                  : connection === "failed" ? "text-destructive" : "text-muted-foreground"
            }`}>
              <span aria-hidden="true" className={`size-2 shrink-0 rounded-full bg-current ${connection === "checking" ? "motion-safe:animate-pulse" : ""}`} />
              {connection === "ready" ? "Connected"
                : connection === "partial" ? "MCP only"
                  : connection === "failed" ? "Connection failed" : "Testing…"}
            </span>
          )}
        </div>
        <span className="text-xs text-muted-foreground">
          {!config ? "Loading…" : config.is_enabled ? "Enabled" : !config.available ? "Unavailable" : "Disabled"}
        </span>
      </div>
      {!config?.is_enabled && <p className="text-sm leading-relaxed text-muted-foreground">
        Studio downloads and sets up MCP on first use. Internet is needed once; the Blender add-on is installed separately.
      </p>}
      {duplicate && <p className="text-xs leading-relaxed text-amber-600">A custom Blender server also exists. Disable it below if you prefer Studio’s managed setup.</p>}
      {config && !config.is_enabled && (
        <div className="flex items-start gap-3">
          <Checkbox id="blender-mcp-consent" className="mt-0.5" checked={consent} disabled={locked} onCheckedChange={(checked) => setConsent(checked === true)} />
          <Label htmlFor="blender-mcp-consent" className="text-xs font-normal leading-relaxed">
            Allow unsandboxed Python, scene changes and background Blender tool processes. Results may be sent to your model provider. Cancelling may not undo changes.
          </Label>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {config && !config.is_enabled && (
          <Button size="sm" disabled={locked || !config.available || !validPort || !consent} onClick={() => void act("enable")}>Enable Blender MCP</Button>
        )}
        <Button size="sm" variant="outline" disabled={locked || !config?.available || !validPort || (!config.is_enabled && !consent)} onClick={() => void act("test")}>Test connection</Button>
        {config?.is_enabled && <Button size="sm" variant="outline" disabled={locked} onClick={() => void act("disable")}>Disable</Button>}
        {busy && <span role="status" className="inline-flex items-center gap-2 text-xs text-muted-foreground"><Spinner />Setting up / checking MCP…</span>}
      </div>
      <Collapsible className="border-t pt-3">
        <CollapsibleTrigger className="group flex items-center gap-1 rounded-md py-1 text-xs text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
          {config?.server_id ? "Setup help" : "Set up Blender"}
          <ChevronDown aria-hidden="true" className="size-4 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180 motion-reduce:transition-none" />
        </CollapsibleTrigger>
        <CollapsibleContent className="[--duration:200ms] motion-reduce:animate-none">
        <div className="space-y-4 pt-4 text-sm leading-relaxed text-muted-foreground">
          <p>Use Blender {config?.min_blender_version ?? "5.1.0"}+ on the Studio backend machine.</p>
          <Button size="sm" variant="outline" onClick={() => openLink("https://www.blender.org/lab/mcp-server/")}>Download Blender add-on</Button>
          <ol className="list-decimal space-y-3 pl-5">
            <li>In Blender, enable <strong className="font-medium text-foreground">Online Access</strong> in Preferences → System.</li>
            <li>Drag the website’s install button into Blender to add the <strong className="font-medium text-foreground">Blender Lab</strong> repository.</li>
            <li>Drag it in again and confirm the add-on installation. Or, after adding the repository, search <strong className="font-medium text-foreground">MCP</strong> in Preferences → Get Extensions.</li>
          </ol>
          <p>Enable the add-on, keep Blender open, then test the connection above.</p>
        </div>
        </CollapsibleContent>
      </Collapsible>

      <Collapsible className="border-t pt-3">
        <CollapsibleTrigger className="group flex w-full items-center justify-between rounded-md py-1 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
          Advanced settings
          <ChevronDown aria-hidden="true" className="size-4 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180 motion-reduce:transition-none" />
        </CollapsibleTrigger>
        <CollapsibleContent className="[--duration:200ms] motion-reduce:animate-none">
        <div className="space-y-4 pt-4">
          <p className="text-xs leading-relaxed text-muted-foreground">Loopback only (127.0.0.1). Match the add-on’s port. Studio needs Python 3.10+.</p>
          <div className="grid gap-3 sm:grid-cols-[100px_1fr]">
            <div className="space-y-1">
              <Label htmlFor="blender-mcp-port">Port</Label>
              <Input id="blender-mcp-port" inputMode="numeric" value={port} disabled={locked || !config} onChange={(event) => { setPort(event.target.value); setMessage(null); setConnection(null); }} />
            </div>
            <div className="space-y-1">
              <Label htmlFor="blender-mcp-path">Blender executable (optional)</Label>
              <Input id="blender-mcp-path" value={blenderPath} disabled={locked || !config} placeholder="Auto-detect Blender" onChange={(event) => { setBlenderPath(event.target.value); setMessage(null); setConnection(null); }} />
            </div>
          </div>
          <Button size="sm" variant="outline" disabled={locked || !config || !validPort} onClick={() => void act("save")}>Save &amp; disable</Button>
        </div>
        </CollapsibleContent>
      </Collapsible>
      {config?.unavailable_reason && <p className="text-xs text-muted-foreground">{config.unavailable_reason}</p>}
      {message && <p role="status" className="text-xs">{message}</p>}
      {(error || loadError) && <div role="alert" className="text-xs text-destructive">{error || loadError}{loadError && <Button size="sm" variant="ghost" onClick={() => void refresh()}>Retry</Button>}</div>}
    </section>
  );
}
