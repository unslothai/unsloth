// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DashboardLayout } from "@/components/layout/dashboard-layout";
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
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { copyToClipboardAsync } from "@/lib/copy-to-clipboard";
import {
  AlertCircleIcon,
  Copy01Icon,
  Delete02Icon,
  Key01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { authFetch } from "./api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ApiKey {
  id: number;
  name: string;
  key_prefix: string;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
  is_active: boolean;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function fetchApiKeys(): Promise<ApiKey[]> {
  const res = await authFetch("/api/auth/api-keys");
  if (!res.ok) throw new Error("Failed to load API keys");
  const data = (await res.json()) as { api_keys: ApiKey[] };
  return data.api_keys;
}

async function createApiKey(
  name: string,
  expiresInDays: number | null,
): Promise<{ key: string; api_key: ApiKey }> {
  const res = await authFetch("/api/auth/api-keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      expires_in_days: expiresInDays,
    }),
  });
  if (!res.ok) throw new Error("Failed to create API key");
  return res.json();
}

async function revokeApiKey(keyId: number): Promise<void> {
  const res = await authFetch(`/api/auth/api-keys/${keyId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to revoke API key");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso: string | null): string {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const [copying, setCopying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleCopy = async () => {
    if (copying) return;
    setCopying(true);
    try {
      const ok = await copyToClipboardAsync(text);
      if (!ok) return;
      setCopied(true);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => setCopied(false), 2000);
    } finally {
      setCopying(false);
    }
  };

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      onClick={handleCopy}
      disabled={copying}
      className={cn(
        "shrink-0 rounded-md text-muted-foreground hover:text-foreground",
        copied && "text-emerald-600 hover:text-emerald-600",
      )}
      aria-label={copied ? "Copied API key" : "Copy API key"}
      title={copied ? "Copied" : "Copy"}
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        className="size-4"
      />
    </Button>
  );
}

function RevealKeyDialog({
  open,
  rawKey,
  onClose,
}: {
  open: boolean;
  rawKey: string;
  onClose: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>API Key Created</DialogTitle>
          <DialogDescription>
            Copy this key now. It will not be shown again.
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 p-3">
          <code className="min-w-0 flex-1 break-all font-mono text-sm">
            {rawKey}
          </code>
          <CopyButton text={rawKey} />
        </div>
        <div className="flex items-start gap-2 rounded-md border border-amber-500/20 bg-amber-50 p-3 text-amber-800 dark:border-amber-400/20 dark:bg-amber-950/30 dark:text-amber-300">
          <HugeiconsIcon icon={AlertCircleIcon} className="mt-0.5 size-4 shrink-0" />
          <p className="text-xs leading-relaxed">
            Store this key securely. You will not be able to see it again after closing this dialog.
          </p>
        </div>
        <DialogFooter>
          <Button onClick={onClose}>Done</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function CreateKeyForm({ onCreated }: { onCreated: (rawKey: string) => void }) {
  const [name, setName] = useState("");
  const [expiresInDays, setExpiresInDays] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setLoading(true);
    try {
      const days = expiresInDays ? parseInt(expiresInDays, 10) : null;
      const result = await createApiKey(name.trim(), days);
      onCreated(result.key);
      setName("");
      setExpiresInDays("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 rounded-lg border border-border p-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="key-name">Key name</Label>
        <Input
          id="key-name"
          placeholder="e.g. My application"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="key-expiry">Expires in (days)</Label>
        <Input
          id="key-expiry"
          type="number"
          min={1}
          placeholder="Leave blank for no expiry"
          value={expiresInDays}
          onChange={(e) => setExpiresInDays(e.target.value)}
        />
      </div>
      <Button type="submit" disabled={loading || !name.trim()} className="self-start">
        {loading ? "Creating..." : "Create API key"}
      </Button>
    </form>
  );
}

function KeysTable({
  keys,
  onRevoke,
}: {
  keys: ApiKey[];
  onRevoke: (id: number) => void;
}) {
  if (keys.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No API keys yet. Create one above.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-muted/40">
            <th className="px-4 py-2.5 text-left font-medium text-muted-foreground">Name</th>
            <th className="px-4 py-2.5 text-left font-medium text-muted-foreground">Key</th>
            <th className="px-4 py-2.5 text-left font-medium text-muted-foreground">Created</th>
            <th className="px-4 py-2.5 text-left font-medium text-muted-foreground">Last used</th>
            <th className="px-4 py-2.5 text-left font-medium text-muted-foreground">Expires</th>
            <th className="px-4 py-2.5 text-right font-medium text-muted-foreground" />
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr
              key={k.id}
              className={cn(
                "border-b border-border last:border-b-0",
                !k.is_active && "opacity-50",
              )}
            >
              <td className="px-4 py-2.5 font-medium">{k.name}</td>
              <td className="px-4 py-2.5">
                <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
                  sk-unsloth-{k.key_prefix}...
                </code>
              </td>
              <td className="px-4 py-2.5 text-muted-foreground">{formatDate(k.created_at)}</td>
              <td className="px-4 py-2.5 text-muted-foreground">{formatDate(k.last_used_at)}</td>
              <td className="px-4 py-2.5 text-muted-foreground">{formatDate(k.expires_at)}</td>
              <td className="px-4 py-2.5 text-right">
                {k.is_active ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onRevoke(k.id)}
                    className="text-destructive hover:text-destructive"
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-3.5 mr-1" />
                    Revoke
                  </Button>
                ) : (
                  <span className="text-xs text-muted-foreground">Revoked</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function UsageExamples() {
  const base = window.location.origin;

  const curlExample = `curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer sk-unsloth-YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'`;

  const pythonExample = `from openai import OpenAI

client = OpenAI(
    base_url="${base}/v1",
    api_key="sk-unsloth-YOUR_KEY",
)

response = client.chat.completions.create(
    model="current",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")`;

  const toolsExample = `curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer sk-unsloth-YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "Search for Python 3.13 features"}],
    "stream": true,
    "enable_tools": true,
    "enabled_tools": ["web_search", "python"],
    "session_id": "my-session"
  }'`;

  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-sm font-semibold">Usage examples</h3>
      <div className="flex flex-col gap-3">
        <div>
          <p className="mb-1.5 text-xs font-medium text-muted-foreground">curl</p>
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
            {curlExample}
          </pre>
        </div>
        <div>
          <p className="mb-1.5 text-xs font-medium text-muted-foreground">Python (OpenAI SDK)</p>
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
            {pythonExample}
          </pre>
        </div>
        <div>
          <p className="mb-1.5 text-xs font-medium text-muted-foreground">With tools (web search + code execution)</p>
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
            {toolsExample}
          </pre>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [revealedKey, setRevealedKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadKeys = useCallback(async () => {
    try {
      setError(null);
      const loaded = await fetchApiKeys();
      setKeys(loaded);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load API keys");
    }
  }, []);

  useEffect(() => {
    void loadKeys();
  }, [loadKeys]);

  const handleCreated = (rawKey: string) => {
    setRevealedKey(rawKey);
    void loadKeys();
  };

  const handleRevoke = async (keyId: number) => {
    try {
      await revokeApiKey(keyId);
      void loadKeys();
    } catch {
      setError("Failed to revoke key");
    }
  };

  return (
    <DashboardLayout>
      <div className="flex flex-col gap-8">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-muted/40">
            <HugeiconsIcon icon={Key01Icon} className="size-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold font-heading">API Keys</h1>
            <p className="text-sm text-muted-foreground">
              Create keys to access Unsloth Studio programmatically via the OpenAI-compatible API.
            </p>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 rounded-md border border-destructive/20 bg-destructive/5 p-3 text-sm text-destructive">
            <HugeiconsIcon icon={AlertCircleIcon} className="size-4 shrink-0" />
            {error}
          </div>
        )}

        <CreateKeyForm onCreated={handleCreated} />
        <KeysTable keys={keys} onRevoke={handleRevoke} />
        <UsageExamples />
      </div>

      <RevealKeyDialog
        open={revealedKey !== null}
        rawKey={revealedKey ?? ""}
        onClose={() => setRevealedKey(null)}
      />
    </DashboardLayout>
  );
}
