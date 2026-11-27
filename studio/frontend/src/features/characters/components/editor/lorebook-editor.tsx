// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { emptyLorebook, newLorebookEntry, parseKeyList } from "../../prompt/lorebook";
import type { Lorebook, LorebookEntry, LorebookPosition } from "../../types";
import { ClampedNumberInput } from "../clamped-number-input";

const PAGE = 25;

export function LorebookEditor({ book, onChange }: { book: Lorebook | null; onChange: (book: Lorebook | null) => void }) {
  const t = useT();
  const value = book ?? emptyLorebook();
  // Rows are keyed by a local id: imported cards often repeat or omit entry ids.
  const lastUid = useRef(0);
  const makeUid = () => `entry-${(lastUid.current += 1)}`;
  const [uids, setUids] = useState<string[]>(() => value.entries.map(makeUid));
  useEffect(() => {
    if (uids.length !== value.entries.length) setUids(value.entries.map(makeUid));
  }, [value.entries.length, uids.length]);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [limit, setLimit] = useState(PAGE);

  function update(index: number, patch: Partial<LorebookEntry>) {
    const entries = value.entries.map((entry, i) => (i === index ? { ...entry, ...patch } : entry));
    onChange({ ...value, entries });
  }

  function add() {
    const entry = newLorebookEntry(value.entries);
    const uid = makeUid();
    onChange({ ...value, entries: [entry, ...value.entries] });
    setUids((current) => [uid, ...current]);
    setOpen((current) => new Set(current).add(uid));
  }

  function remove(index: number) {
    const entries = value.entries.filter((_, i) => i !== index);
    setUids((current) => current.filter((_, i) => i !== index));
    onChange(entries.length || book?.name ? { ...value, entries } : null);
  }

  const shown = value.entries.slice(0, limit);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3">
        <p className="min-w-0 flex-1 text-ui-13 text-muted-foreground">{t("characters.lorebook.hint")}</p>
        <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
          {t("characters.lorebook.scanDepth")}
          <ClampedNumberInput
            min={1}
            max={100}
            value={value.scan_depth}
            className="h-8 w-24"
            onChange={(scan_depth) => onChange({ ...value, scan_depth })}
          />
        </label>
        <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
          {t("characters.lorebook.budget")}
          <ClampedNumberInput
            min={64}
            step={64}
            value={value.token_budget}
            className="h-8 w-28"
            onChange={(token_budget) => onChange({ ...value, token_budget })}
          />
        </label>
        <Button type="button" variant="outline" size="sm" onClick={add}>
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
          {t("characters.lorebook.addEntry")}
        </Button>
      </div>
      {value.entries.length === 0 ? (
        <p className="rounded-xl bg-muted/60 px-4 py-3 text-ui-14 text-muted-foreground dark:bg-card">
          {t("characters.lorebook.empty")}
        </p>
      ) : (
        <ul className="flex flex-col gap-2" data-testid="lorebook-entries">
          {shown.map((entry, index) => {
            const uid = uids[index] ?? `entry-pending-${index}`;
            return (
              <LorebookEntryRow
                key={uid}
                entry={entry}
                expanded={open.has(uid)}
                onToggle={() =>
                  setOpen((current) => {
                    const next = new Set(current);
                    if (next.has(uid)) next.delete(uid);
                    else next.add(uid);
                    return next;
                  })
                }
                onUpdate={(patch) => update(index, patch)}
                onRemove={() => remove(index)}
              />
            );
          })}
        </ul>
      )}
      {value.entries.length > limit ? (
        <Button type="button" variant="ghost" size="sm" className="self-start" onClick={() => setLimit((n) => n + PAGE * 4)}>
          {t("characters.lorebook.showMore", { count: value.entries.length - limit })}
        </Button>
      ) : null}
    </div>
  );
}

function LorebookEntryRow({
  entry,
  expanded,
  onToggle,
  onUpdate,
  onRemove,
}: {
  entry: LorebookEntry;
  expanded: boolean;
  onToggle: () => void;
  onUpdate: (patch: Partial<LorebookEntry>) => void;
  onRemove: () => void;
}) {
  const t = useT();
  const label = entry.comment || entry.keys.join(", ") || t("characters.lorebook.untitled");
  return (
    <li className="rounded-2xl bg-muted/55 dark:bg-card">
      <div className="flex items-center gap-3 px-3.5 py-2.5">
        <button
          type="button"
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
          aria-expanded={expanded}
          onClick={onToggle}
        >
          <HugeiconsIcon
            icon={ArrowDown01Icon}
            strokeWidth={1.75}
            className={cn("size-4 shrink-0 text-muted-foreground transition-transform", !expanded && "-rotate-90")}
          />
          <span className={cn("truncate text-ui-14 font-medium", !entry.enabled && "text-muted-foreground line-through")}>
            {label}
          </span>
          {entry.constant ? (
            <span className="shrink-0 rounded-full bg-primary/12 px-2 py-0.5 text-ui-11 font-semibold text-primary">
              {t("characters.lorebook.alwaysOn")}
            </span>
          ) : null}
        </button>
        <Switch
          checked={entry.enabled}
          aria-label={t("characters.lorebook.enabled")}
          onCheckedChange={(checked) => onUpdate({ enabled: checked })}
        />
      </div>
      {expanded ? (
        <div className="flex flex-col gap-3 border-t border-border/60 px-3.5 pb-3.5 pt-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
              {t("characters.lorebook.label")}
              <Input value={entry.comment} onChange={(event) => onUpdate({ comment: event.target.value })} />
            </label>
            <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
              {t("characters.lorebook.keys")}
              <Input
                defaultValue={entry.keys.join(", ")}
                onBlur={(event) => {
                  if (event.target.value === entry.keys.join(", ")) return;
                  onUpdate({ keys: parseKeyList(event.target.value) });
                }}
              />
              <span className="text-ui-11 text-muted-foreground/80">{t("characters.lorebook.keysHint")}</span>
            </label>
          </div>
          <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
            {t("characters.lorebook.content")}
            <Textarea
              value={entry.content}
              className="max-h-80 min-h-20 font-sans text-ui-14"
              onChange={(event) => onUpdate({ content: event.target.value })}
            />
          </label>
          <div className="flex flex-wrap items-end gap-4">
            <label className="flex items-center gap-2 text-ui-13">
              <Switch checked={entry.constant} onCheckedChange={(checked) => onUpdate({ constant: checked })} />
              {t("characters.lorebook.alwaysOn")}
            </label>
            <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
              {t("characters.lorebook.position")}
              <Select
                value={entry.position}
                onValueChange={(position) => onUpdate({ position: position as LorebookPosition })}
              >
                <SelectTrigger size="sm" className="h-8 w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="before_char">{t("characters.lorebook.beforeChar")}</SelectItem>
                  <SelectItem value="after_char">{t("characters.lorebook.afterChar")}</SelectItem>
                  <SelectItem value="at_depth">{t("characters.lorebook.atDepth")}</SelectItem>
                </SelectContent>
              </Select>
            </label>
            {entry.position === "at_depth" ? (
              <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
                {t("characters.lorebook.depth")}
                <ClampedNumberInput
                  min={0}
                  max={100}
                  value={entry.depth}
                  className="h-8 w-20"
                  onChange={(depth) => onUpdate({ depth })}
                />
              </label>
            ) : null}
            <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
              {t("characters.lorebook.order")}
              <ClampedNumberInput
                value={entry.insertion_order}
                className="h-8 w-20"
                onChange={(insertion_order) => onUpdate({ insertion_order })}
              />
            </label>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="ml-auto text-destructive hover:text-destructive"
              onClick={onRemove}
            >
              <HugeiconsIcon icon={TrashIcon} strokeWidth={1.75} className="size-icon" />
              {t("characters.lorebook.remove")}
            </Button>
          </div>
        </div>
      ) : null}
    </li>
  );
}
