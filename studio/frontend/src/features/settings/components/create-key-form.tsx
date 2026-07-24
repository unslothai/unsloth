// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useT } from "@/i18n";
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
  const t = useT();
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
    } catch {
      // api-keys.ts throws English Error messages; use the translated one so
      // zh-CN users don't see English bleed through from internal exceptions.
      onError(t("settings.apiKeys.createError"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={t("settings.apiKeys.tokenNamePlaceholder")}
          className="h-9 min-w-[200px] flex-1 text-sm"
          aria-label={t("settings.apiKeys.newAccessTokenName")}
        />
        <div className="hub-tab-toggle inline-flex h-8 items-center rounded-full">
          {EXPIRY_PRESETS.map((p) => {
            const active = expiry === p.value;
            return (
              <button
                key={p.label}
                type="button"
                onClick={() => setExpiry(p.value)}
                aria-pressed={active}
                className={cn(
                  "inline-flex h-8 items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                  active
                    ? "hub-tab-toggle-pill text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {p.value === null ? t("settings.apiKeys.never") : p.label}
              </button>
            );
          })}
        </div>
        <Button type="submit" size="sm" disabled={loading || !name.trim()}>
          {loading
            ? t("settings.apiKeys.creating")
            : t("settings.apiKeys.createToken")}
        </Button>
      </div>
    </form>
  );
}
