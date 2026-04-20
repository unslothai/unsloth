// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { createApiKey } from "../api/api-keys";

const EXPIRY_PRESETS = [
  { label: "Never", value: null as number | null },
  { label: "7d", value: 7 },
  { label: "30d", value: 30 },
  { label: "90d", value: 90 },
];

export function CreateKeyForm({
  onCreated,
  onError,
}: {
  onCreated: (rawKey: string) => void;
  onError: (message: string) => void;
}) {
  const [name, setName] = useState("");
  const [expiry, setExpiry] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || loading) return;
    setLoading(true);
    try {
      const result = await createApiKey(name.trim(), expiry);
      onCreated(result.key);
      setName("");
    } catch (err) {
      onError(err instanceof Error ? err.message : "Couldn't create key.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-2 rounded-lg border border-border bg-muted/20 p-3"
    >
      <div className="flex flex-wrap items-center gap-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Key name (e.g. production)"
          className="h-8 min-w-[180px] flex-1 text-sm"
          aria-label="New key name"
        />
        <div className="inline-flex items-center rounded-md border border-border bg-background p-0.5">
          {EXPIRY_PRESETS.map((p) => {
            const active = expiry === p.value;
            return (
              <button
                key={p.label}
                type="button"
                onClick={() => setExpiry(p.value)}
                aria-pressed={active}
                className={cn(
                  "rounded px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  active
                    ? "bg-accent text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {p.label}
              </button>
            );
          })}
        </div>
        <Button type="submit" size="sm" disabled={loading || !name.trim()}>
          {loading ? "Creating…" : "Create key"}
        </Button>
      </div>
    </form>
  );
}
