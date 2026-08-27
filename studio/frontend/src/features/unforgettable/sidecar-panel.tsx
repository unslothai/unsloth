// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { shortId } from "./format";
import type { AdapterRow } from "./types";

export function SidecarPanel({
  adapters,
  busy,
  onPromote,
  onRollback,
}: {
  adapters: AdapterRow[];
  busy: boolean;
  onPromote: (id: string) => void;
  onRollback: () => void;
}) {
  const t = useT();
  if (adapters.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border/70 p-6 text-sm text-muted-foreground">
        {t("unforgettable.sidecar.empty")}
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={onRollback}
        >
          {t("unforgettable.sidecar.rollback")}
        </Button>
      </div>
      <ul className="divide-y divide-border/60 rounded-xl border border-border/70">
        {adapters.map((adapter) => (
          <li
            key={adapter.id}
            className="flex items-center justify-between gap-3 px-3 py-2.5"
          >
            <div className="min-w-0">
              <div className="truncate text-sm font-medium">
                {shortId(adapter.id)} · {adapter.status}
              </div>
              <div className="truncate text-xs text-muted-foreground">
                {adapter.recipe} · {adapter.backend}
              </div>
            </div>
            {adapter.status !== "promoted" ? (
              <Button
                type="button"
                size="sm"
                disabled={busy}
                onClick={() => onPromote(adapter.id)}
              >
                {t("unforgettable.sidecar.promote")}
              </Button>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
