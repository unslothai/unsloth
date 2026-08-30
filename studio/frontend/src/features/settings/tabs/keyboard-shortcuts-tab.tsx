// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import {
  Alert01Icon,
  ArrowTurnBackwardIcon,
  Delete02Icon,
  PencilEdit02Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  SHORTCUT_DEFS,
  SHORTCUT_SLOTS,
  type ShortcutDef,
  type ShortcutId,
  type ShortcutSlot,
  bindingFromEvent,
  defaultBindingFor,
  formatBindingLabel,
  formatBindingValue,
  isAcceptableBinding,
  isBrowserReservedBinding,
  isMacPlatform,
  parseBinding,
} from "../lib/keyboard-shortcuts";
import {
  findConflicts,
  isSlotOverridden,
  resolveBinding,
  shortcutOwningBinding,
  useKeyboardShortcutsStore,
} from "../stores/keyboard-shortcuts-store";

/** The ⇧⌘O key cap. Plain text when the slot carries no chord. */
function Chord({
  label,
  tone,
  note,
}: {
  label: string;
  tone: "assigned" | "unassigned" | "recording";
  /** Why the browser may swallow this chord. Hover only, so the caps stay a
   *  clean column. */
  note?: string;
}) {
  const cap = (
    <span
      className={cn(
        // Width hugs the chord, so ⌘, and ⇧⌘O share a left edge.
        "inline-flex h-7 items-center rounded-md px-2.5 text-xs font-medium tabular-nums",
        tone === "assigned" && "bg-muted text-foreground",
        tone === "unassigned" && "text-muted-foreground",
        tone === "recording" &&
          "bg-primary/10 text-primary ring-1 ring-primary/40",
      )}
    >
      {label}
    </span>
  );
  if (!note) return cap;
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{cap}</TooltipTrigger>
      <TooltipContent className="max-w-[260px] leading-snug">
        {note}
      </TooltipContent>
    </Tooltip>
  );
}

/** Borderless pencil / trash / undo. */
function RowIconButton({
  icon,
  label,
  onClick,
  className,
}: {
  icon: typeof PencilEdit02Icon;
  label: string;
  onClick: () => void;
  className?: string;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={onClick}
          className={cn(
            "inline-flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            className,
          )}
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />
        </button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

/** Which row and which of its two chords the recorder is listening for. */
interface RecordingTarget {
  id: ShortcutId;
  slot: ShortcutSlot;
}

export function KeyboardShortcutsTab() {
  const t = useT();
  const overrides = useKeyboardShortcutsStore((s) => s.overrides);
  const setBinding = useKeyboardShortcutsStore((s) => s.setBinding);
  const clearBinding = useKeyboardShortcutsStore((s) => s.clearBinding);
  const resetBinding = useKeyboardShortcutsStore((s) => s.resetBinding);
  const resetAll = useKeyboardShortcutsStore((s) => s.resetAll);

  const [query, setQuery] = useState("");
  const [recording, setRecording] = useState<RecordingTarget | null>(null);
  // Shown under the row being recorded when the pressed chord is rejected.
  const [recordingError, setRecordingError] = useState<string | null>(null);

  const mac = isMacPlatform();
  const conflicts = useMemo(() => findConflicts(overrides), [overrides]);

  // Capture phase, so the chord being recorded reaches this listener before the
  // shortcut it is replacing fires and before Radix's Escape-to-close.
  useEffect(() => {
    if (!recording) return;
    const def = SHORTCUT_DEFS.find((entry) => entry.id === recording.id);
    const onKeyDown = (event: KeyboardEvent) => {
      // Tab held bare is never an acceptable binding, so it is still what it
      // was: the way out. Left swallowed with the rest, a row that records
      // bare keys had no keyboard exit at all, since Escape is a chord there
      // and Enter or Space on the focused pencil records instead of pressing
      // it. Not prevented, so focus moves on as it would have.
      if (
        event.code === "Tab" &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.altKey
      ) {
        setRecording(null);
        setRecordingError(null);
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      // Every keydown is swallowed above, so bare Escape is the only way out of
      // recording. The exception is a row that takes bare keys: Escape is the
      // chord it ships, so recording it has to be possible, and the pencil
      // cancels there instead.
      if (
        event.code === "Escape" &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.altKey &&
        !event.shiftKey &&
        !def?.allowBareKey
      ) {
        setRecording(null);
        setRecordingError(null);
        return;
      }
      const binding = bindingFromEvent(event);
      // Modifier held on its own: the user is still assembling the chord.
      if (!binding) return;
      if (!isAcceptableBinding(binding, def?.allowBareKey)) {
        setRecordingError(t("settings.keyboardShortcuts.needsModifier"));
        return;
      }
      setBinding(recording.id, recording.slot, formatBindingValue(binding));
      setRecording(null);
      setRecordingError(null);
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () =>
      window.removeEventListener("keydown", onKeyDown, { capture: true });
  }, [recording, setBinding, t]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return null;
    return new Set(
      SHORTCUT_DEFS.filter((def) => {
        const haystack = `${t(def.labelKey)} ${t(def.descriptionKey)}`;
        if (haystack.toLowerCase().includes(q)) return true;
        return SHORTCUT_SLOTS.some((slot) => {
          const parsed = parseBinding(resolveBinding(overrides, def.id, slot));
          return parsed
            ? formatBindingLabel(parsed, mac).toLowerCase().includes(q)
            : false;
        });
      }).map((def) => def.id),
    );
  }, [query, t, overrides, mac]);

  // One list, in registry order, so the daily rows sit above the fold.
  const visible = SHORTCUT_DEFS.filter(
    // A web-only row on the desktop build would bind a chord whose handler
    // returns, so it is left out rather than shown as a key that does nothing.
    (def) => (!matches || matches.has(def.id)) && !(isTauri && def.webOnly),
  );

  const startRecording = (def: ShortcutDef, slot: ShortcutSlot) => {
    setRecordingError(null);
    setRecording(
      recording?.id === def.id && recording.slot === slot
        ? null
        : { id: def.id, slot },
    );
  };

  /** One chord line: cap and pencil left, trash right. A row with an
   *  alternate stacks two. */
  const renderSlot = (def: ShortcutDef, slot: ShortcutSlot): ReactNode => {
    const value = resolveBinding(overrides, def.id, slot);
    const parsed = parseBinding(value);
    const isRecording = recording?.id === def.id && recording.slot === slot;
    const slotName = t(
      slot === "primary"
        ? "settings.keyboardShortcuts.primarySlot"
        : "settings.keyboardShortcuts.alternateSlot",
    );
    // The warning rides the cap's tooltip, not a third line of description.
    const reserved =
      !isTauri && !isRecording && isBrowserReservedBinding(value);

    return (
      <div key={slot} className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-1">
          <Chord
            label={
              isRecording
                ? t("settings.keyboardShortcuts.recording")
                : parsed
                  ? formatBindingLabel(parsed, mac)
                  : t("settings.keyboardShortcuts.unassigned")
            }
            tone={
              isRecording ? "recording" : parsed ? "assigned" : "unassigned"
            }
            note={
              reserved
                ? t("settings.keyboardShortcuts.browserReserved")
                : undefined
            }
          />
          <RowIconButton
            icon={PencilEdit02Icon}
            label={`${t("settings.keyboardShortcuts.edit")} (${slotName})`}
            onClick={() => startRecording(def, slot)}
          />
          {isSlotOverridden(overrides, def.id, slot) ? (
            <RowIconButton
              icon={ArrowTurnBackwardIcon}
              label={`${t("settings.keyboardShortcuts.reset")} (${slotName})`}
              onClick={() => resetBinding(def.id, slot)}
            />
          ) : null}
        </div>
        {parsed && !isRecording ? (
          <RowIconButton
            icon={Delete02Icon}
            label={`${t("settings.keyboardShortcuts.clear")} (${slotName})`}
            onClick={() => clearBinding(def.id, slot)}
          />
        ) : null}
      </div>
    );
  };

  const renderRow = (def: ShortcutDef) => {
    const isRecording = recording?.id === def.id;
    const conflicted = conflicts.has(def.id) && !isRecording;
    // Say which side of a clash this row is on, since only the owner runs.
    const shadowed =
      conflicted &&
      SHORTCUT_SLOTS.every((slot) => {
        const value = resolveBinding(overrides, def.id, slot);
        return !value || shortcutOwningBinding(overrides, value) !== def.id;
      });
    // Anything this action ships an alternate for keeps its line, cleared or
    // not: hiding a cleared slot would take its restore control with it, and
    // Reset all is not a way back from one edit.
    const hasAlternate =
      defaultBindingFor(def, "alternate", mac) !== null ||
      resolveBinding(overrides, def.id, "alternate") !== null ||
      (recording?.id === def.id && recording.slot === "alternate");

    return (
      <div
        key={def.id}
        data-settings-label={t(def.labelKey)}
        className="group/row flex items-center gap-6 py-3.5"
      >
        <div className="flex min-w-0 flex-1 basis-0 flex-col gap-0.5">
          <span className="text-sm font-medium text-foreground">
            {t(def.labelKey)}
          </span>
          <span className="text-xs leading-snug text-muted-foreground">
            {isRecording ? (
              <span className="text-primary">
                {recordingError ??
                  t("settings.keyboardShortcuts.recordingHint")}
              </span>
            ) : conflicted ? (
              <span className="flex items-center gap-1 text-amber-500">
                <HugeiconsIcon
                  icon={Alert01Icon}
                  strokeWidth={1.75}
                  className="size-3.5 shrink-0"
                />
                {shadowed
                  ? t("settings.keyboardShortcuts.conflictShadowed")
                  : t("settings.keyboardShortcuts.conflict")}
              </span>
            ) : (
              t(def.descriptionKey)
            )}
          </span>
        </div>
        {/* pl-10 nudges the caps off the descriptions: the labels run long
            enough that a bare half-and-half split left them crowding. */}
        <div className="flex flex-1 basis-0 flex-col gap-1.5 pl-10">
          {renderSlot(def, "primary")}
          {/* Only the actions that ship an alternate have a second line: there
              is no affordance for adding one. */}
          {hasAlternate ? renderSlot(def, "alternate") : null}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <h2
          data-settings-label={t("settings.keyboardShortcuts.title")}
          className="font-heading text-base font-semibold text-foreground"
        >
          {t("settings.keyboardShortcuts.title")}
        </h2>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("settings.keyboardShortcuts.description")}
        </p>
      </div>

      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={2}
          className="pointer-events-none absolute left-4 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={t("settings.keyboardShortcuts.searchPlaceholder")}
          className="h-11 rounded-full pl-11"
          aria-label={t("settings.keyboardShortcuts.searchPlaceholder")}
        />
      </div>

      {visible.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {t("settings.keyboardShortcuts.noResults")}
        </p>
      ) : (
        // Rows carry the padding, so the dividers sit inset from the edge.
        <div className="rounded-xl border border-border/70 px-5">
          <div className="divide-y divide-border/60">
            {visible.map(renderRow)}
          </div>
        </div>
      )}

      <div className="flex justify-start pt-1">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={Object.keys(overrides).length === 0}
          onClick={() => {
            setRecording(null);
            setRecordingError(null);
            resetAll();
          }}
        >
          {t("settings.keyboardShortcuts.resetAll")}
        </Button>
      </div>
    </div>
  );
}
