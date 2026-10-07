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
  EnergyRectangleIcon,
  PencilEdit02Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  SHORTCUT_DEFS,
  SHORTCUT_SLOTS,
  type ShortcutBinding,
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
  keystrokeMatchesBinding,
  parseBinding,
} from "../lib/keyboard-shortcuts";
import {
  findConflicts,
  isSlotOverridden,
  resolveBinding,
  shortcutOwningBinding,
  useKeyboardShortcutsStore,
} from "../stores/keyboard-shortcuts-store";

function Chord({
  label,
  tone,
  note,
}: {
  label: string;
  tone: "assigned" | "unassigned" | "recording";
  /** Hover only, so the caps stay a clean column. */
  note?: string;
}) {
  const cap = (
    <span
      className={cn(
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
      <TooltipContent className="max-w-[calc(260px*var(--ui-space-scale,1))] leading-snug">
        {note}
      </TooltipContent>
    </Tooltip>
  );
}

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
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [byKeystroke, setByKeystroke] = useState(false);
  const [keystroke, setKeystroke] = useState<ShortcutBinding | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const mac = isMacPlatform();
  const conflicts = useMemo(() => findConflicts(overrides), [overrides]);

  // Capture phase: reach the chord before the shortcut it replaces and Radix's Escape-to-close.
  useEffect(() => {
    if (!recording) return;
    const def = SHORTCUT_DEFS.find((entry) => entry.id === recording.id);
    const onKeyDown = (event: KeyboardEvent) => {
      // Bare Tab is never bindable, so let it move focus; otherwise bare-key rows had no keyboard exit.
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
      // Bare Escape exits recording, except on rows that take bare keys, where the pencil cancels.
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

  // Capture phase, like the recorder, which owns the keyboard while it runs.
  useEffect(() => {
    if (!byKeystroke || recording) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (document.activeElement !== searchRef.current) return;
      // Through the binding so the no-code fallback covers Tab and Escape.
      const binding = bindingFromEvent(event);
      if (!binding) return;
      const bare = !binding.mod && !binding.ctrl && !binding.alt;
      // Tab still moves focus, or the keyboard would be trapped here.
      if (binding.code === "Tab" && bare) return;
      event.preventDefault();
      event.stopPropagation();
      // Escape backs out the chord, then the mode; this persistent mode cannot take it as a query or it
      // would eat the dialog's dismiss. Shift+Escape is searchable.
      if (binding.code === "Escape" && bare && !binding.shift) {
        if (keystroke) setKeystroke(null);
        else setByKeystroke(false);
        return;
      }
      setKeystroke(binding);
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () =>
      window.removeEventListener("keydown", onKeyDown, { capture: true });
  }, [byKeystroke, recording, keystroke]);

  const toggleByKeystroke = () => {
    // Both searches filter the same list, so switching clears the other.
    setByKeystroke((on) => !on);
    setQuery("");
    setKeystroke(null);
    searchRef.current?.focus();
  };

  const matches = useMemo(() => {
    if (byKeystroke) {
      if (!keystroke) return null;
      return new Set(
        SHORTCUT_DEFS.filter((def) =>
          SHORTCUT_SLOTS.some((slot) => {
            const parsed = parseBinding(resolveBinding(overrides, def.id, slot));
            return parsed ? keystrokeMatchesBinding(keystroke, parsed) : false;
          }),
        ).map((def) => def.id),
      );
    }
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
  }, [query, t, overrides, mac, byKeystroke, keystroke]);

  const visible = SHORTCUT_DEFS.filter(
    // Web-only rows would bind dead keys on desktop.
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

  const renderSlot = (def: ShortcutDef, slot: ShortcutSlot): ReactNode => {
    const value = resolveBinding(overrides, def.id, slot);
    const parsed = parseBinding(value);
    const isRecording = recording?.id === def.id && recording.slot === slot;
    const slotName = t(
      slot === "primary"
        ? "settings.keyboardShortcuts.primarySlot"
        : "settings.keyboardShortcuts.alternateSlot",
    );
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
    // Only the owner of a clash runs.
    const shadowed =
      conflicted &&
      SHORTCUT_SLOTS.every((slot) => {
        const value = resolveBinding(overrides, def.id, slot);
        return !value || shortcutOwningBinding(overrides, value) !== def.id;
      });
    // Rows with a shipped alternate keep that line even when cleared, so its restore control stays.
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
    <div className="settings-page">
      <header className="flex flex-col gap-1">
        <h1
          data-settings-label={t("settings.keyboardShortcuts.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.keyboardShortcuts.title")}
        </h1>
      </header>

      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={2}
          className="pointer-events-none absolute left-4 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          ref={searchRef}
          // Read-only in chord mode; the capture listener fills it.
          value={
            byKeystroke
              ? keystroke
                ? formatBindingLabel(keystroke, mac)
                : ""
              : query
          }
          onChange={(e) => setQuery(e.target.value)}
          readOnly={byKeystroke}
          placeholder={t(
            byKeystroke
              ? "settings.keyboardShortcuts.keystrokePlaceholder"
              : "settings.keyboardShortcuts.searchPlaceholder",
          )}
          className={cn(
            "h-11 rounded-full pl-11 pr-12",
            byKeystroke && "font-medium tabular-nums",
          )}
          aria-label={t(
            byKeystroke
              ? "settings.keyboardShortcuts.keystrokePlaceholder"
              : "settings.keyboardShortcuts.searchPlaceholder",
          )}
        />
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-pressed={byKeystroke}
              aria-label={t(
                byKeystroke
                  ? "settings.keyboardShortcuts.searchByName"
                  : "settings.keyboardShortcuts.searchByKeystrokes",
              )}
              onClick={toggleByKeystroke}
              className={cn(
                "absolute right-2 top-1/2 inline-flex size-8 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                byKeystroke && "bg-muted text-foreground",
              )}
            >
              <HugeiconsIcon
                icon={EnergyRectangleIcon}
                strokeWidth={1.75}
                className="size-4"
              />
            </button>
          </TooltipTrigger>
          <TooltipContent>
            {t(
              byKeystroke
                ? "settings.keyboardShortcuts.searchByName"
                : "settings.keyboardShortcuts.searchByKeystrokes",
            )}
          </TooltipContent>
        </Tooltip>
      </div>

      {visible.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {t("settings.keyboardShortcuts.noResults")}
        </p>
      ) : (
        <div className="divide-y divide-border/60">{visible.map(renderRow)}</div>
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
