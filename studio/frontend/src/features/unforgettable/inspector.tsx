// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useT } from "@/i18n";
import { shortId } from "./format";
import type { MemoryRecord, WorkspaceTab } from "./types";

export function Inspector({
  record,
  tab,
  draftTitle,
  draftBody,
  force,
  busy,
  onTitle,
  onBody,
  onForce,
  onAdmit,
  onReject,
  onSave,
  onCompile,
  onUncompile,
  onDeprecate,
}: {
  record: MemoryRecord | null;
  tab: WorkspaceTab;
  draftTitle: string;
  draftBody: string;
  force: boolean;
  busy: boolean;
  onTitle: (value: string) => void;
  onBody: (value: string) => void;
  onForce: (value: boolean) => void;
  onAdmit: () => void;
  onReject: () => void;
  onSave: () => void;
  onCompile: () => void;
  onUncompile: () => void;
  onDeprecate: () => void;
}) {
  const t = useT();
  if (!record) {
    return (
      <div className="rounded-xl border border-dashed border-border/70 p-6 text-sm text-muted-foreground">
        {t("unforgettable.inspector.noSelection")}
      </div>
    );
  }
  const proposed = record.status === "proposed";
  const canAdmit =
    record.status === "proposed" || record.status === "deprecated" || force;
  const canReject = record.status === "proposed";
  const canCompile = record.kind === "procedure" && record.status === "active";
  return (
    <div className="flex min-h-0 flex-col gap-3 rounded-xl border border-border/70 p-4">
      <div className="text-xs text-muted-foreground">
        {shortId(record.id)} · {record.kind} · {record.status} ·{" "}
        {record.provenance}
        {record.source_episode_id
          ? ` · ep ${shortId(record.source_episode_id)}`
          : ""}
      </div>
      <Input
        value={draftTitle}
        onChange={(event) => onTitle(event.target.value)}
        disabled={!proposed || busy}
      />
      <Textarea
        value={draftBody}
        onChange={(event) => onBody(event.target.value)}
        disabled={!proposed || busy}
        className="min-h-40"
      />
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={force}
          onChange={(event) => onForce(event.target.checked)}
        />
        {t("unforgettable.inspector.force")}
      </label>
      <div className="flex flex-wrap gap-2">
        {proposed ? (
          <Button type="button" size="sm" disabled={busy} onClick={onSave}>
            {t("unforgettable.inspector.save")}
          </Button>
        ) : null}
        {canAdmit ? (
          <Button type="button" size="sm" disabled={busy} onClick={onAdmit}>
            {t("unforgettable.inspector.admit")}
          </Button>
        ) : null}
        {canReject ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={onReject}
          >
            {t("unforgettable.inspector.reject")}
          </Button>
        ) : null}
        {canCompile && tab !== "standing" ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={onCompile}
          >
            {t("unforgettable.inspector.compile")}
          </Button>
        ) : null}
        {tab === "standing" ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={onUncompile}
          >
            {t("unforgettable.inspector.uncompile")}
          </Button>
        ) : null}
        {record.status === "active" ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={onDeprecate}
          >
            {t("unforgettable.inspector.deprecate")}
          </Button>
        ) : null}
      </div>
    </div>
  );
}
