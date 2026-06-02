// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useEffect, useState } from "react";
import type { KBMode } from "../api/rag-api";
import { useRagStore } from "../stores/rag-store";

/** Defaults pre-fill the KB create dialog. */
export function RagDefaultsSection() {
  const defaults = useRagStore((s) => s.defaults);
  const loadDefaults = useRagStore((s) => s.loadDefaults);
  const updateDefaults = useRagStore((s) => s.updateDefaults);

  const [mode, setMode] = useState<KBMode>("text");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadDefaults();
  }, [loadDefaults]);

  useEffect(() => {
    if (defaults) {
      setMode(defaults.mode);
    }
  }, [defaults]);

  const persist = (patch: {
    mode?: KBMode;
    embedding_model?: string | null;
  }) => {
    setError(null);
    void updateDefaults(patch).catch((err) => {
      setError(err instanceof Error ? err.message : String(err));
    });
  };

  return (
    <div className="flex flex-col gap-3">
      <div>
        <h3 className="text-sm font-medium">Defaults for new knowledge bases</h3>
        <p className="text-xs text-muted-foreground">
          Pre-fills the KB create dialog. Existing KBs keep their own
          settings — use the Reconfigure button to change those.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="defaults-mode">Mode</Label>
          <Select
            value={mode}
            onValueChange={(v) => {
              const next = v as KBMode;
              setMode(next);
              persist({ mode: next });
            }}
          >
            <SelectTrigger id="defaults-mode">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="text">Text only</SelectItem>
              <SelectItem value="multimodal">Multimodal</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      {error ? <div className="text-xs text-destructive">{error}</div> : null}
    </div>
  );
}
