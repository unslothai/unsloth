// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { composerShortcutLabels } from "@/features/chat/utils/composer-preferences";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import {
  EnergyRectangleIcon,
  Search01Icon,
  UndoIcon,
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
  SHORTCUT_DEF_BY_ID,
  SHORTCUT_GROUPS,
  SHORTCUT_SLOTS,
  type ShortcutBinding,
  type ShortcutDef,
  type ShortcutId,
  type ShortcutSlot,
  bindingFromEvent,
  bindingMatchesQuery,
  defaultBindingFor,
  formatBindingLabel,
  formatBindingValue,
  isAcceptableBinding,
  isBrowserReservedBinding,
  isBrowserUnreachableBinding,
  isMacPlatform,
  keystrokeMatchesBinding,
  parseBinding,
  unrecordableModifier,
} from "../lib/keyboard-shortcuts";
import {
  assignedBinding,
  findConflicts,
  isShortcutEnabled,
  resolveBinding,
  shortcutOwningBinding,
  useKeyboardShortcutsStore,
} from "../stores/keyboard-shortcuts-store";

/** Which row and which of its two chords the recorder is listening for. */
interface RecordingTarget {
  id: ShortcutId;
  slot: ShortcutSlot;
}

export function KeyboardShortcutsTab() {
  const t = useT();
  const overrides = useKeyboardShortcutsStore((s) => s.overrides);
  const setBinding = useKeyboardShortcutsStore((s) => s.setBinding);
  const setEnabled = useKeyboardShortcutsStore((s) => s.setEnabled);
  const clearBinding = useKeyboardShortcutsStore((s) => s.clearBinding);
  const resetAction = useKeyboardShortcutsStore((s) => s.resetAction);
  const resetAll = useKeyboardShortcutsStore((s) => s.resetAll);

  const [recording, setRecording] = useState<RecordingTarget | null>(null);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [justRecorded, setJustRecorded] = useState<RecordingTarget | null>(
    null,
  );
  const sendShortcut = useChatPreferencesStore((s) => s.sendShortcut);
  const [query, setQuery] = useState("");
  // Search the list by pressing a chord instead of typing its name.
  const [byKeystroke, setByKeystroke] = useState(false);
  const [keystroke, setKeystroke] = useState<ShortcutBinding | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const mac = isMacPlatform();
  const conflicts = useMemo(() => findConflicts(overrides), [overrides]);

  const stopRecording = () => {
    setRecording(null);
    setRecordingError(null);
  };

  // Capture phase, so the chord being recorded reaches this listener before the
  // shortcut it is replacing fires and before Radix's Escape-to-close.
  useEffect(() => {
    if (!recording) return;
    const def = SHORTCUT_DEFS.find((entry) => entry.id === recording.id);
    const onKeyDown = (event: KeyboardEvent) => {
      // Bare Tab is the way out where Escape records; not prevented, so the box's blur ends recording.
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
      // A row that takes bare keys ships Escape as its chord, so there Escape records instead.
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
      if (!binding) {
        // A real key with the Windows key or AltGr, which cannot be bound; else still assembling.
        const refused = unrecordableModifier(event);
        if (refused) {
          setRecordingError(
            t(
              refused === "meta"
                ? "settings.keyboardShortcuts.metaUnsupported"
                : "settings.keyboardShortcuts.altGraphUnsupported",
            ),
          );
        }
        return;
      }
      const bare =
        !binding.mod && !binding.ctrl && !binding.alt && !binding.shift;
      if (bare && (binding.code === "Backspace" || binding.code === "Delete")) {
        clearBinding(recording.id, recording.slot);
        setJustRecorded(null);
        setRecording(null);
        setRecordingError(null);
        return;
      }
      if (!isAcceptableBinding(binding, def?.allowBareKey)) {
        setRecordingError(t("settings.keyboardShortcuts.needsModifier"));
        return;
      }
      setBinding(recording.id, recording.slot, formatBindingValue(binding));
      setJustRecorded(recording);
      setRecording(null);
      setRecordingError(null);
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () =>
      window.removeEventListener("keydown", onKeyDown, { capture: true });
  }, [recording, setBinding, clearBinding, t]);

  // Capture phase like the recorder; stands down while recording, which owns the keyboard.
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
      // Escape clears the chord, then the mode, never the dialog; decline's bare Escape is found by name.
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

  // Matched on assigned chords, so a switched-off row is still found by its keys.
  const matches = useMemo(() => {
    if (byKeystroke) {
      if (!keystroke) return null;
      return new Set(
        SHORTCUT_DEFS.filter((def) =>
          SHORTCUT_SLOTS.some((slot) => {
            const parsed = parseBinding(assignedBinding(overrides, def.id, slot));
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
          const parsed = parseBinding(assignedBinding(overrides, def.id, slot));
          return parsed ? bindingMatchesQuery(parsed, q, mac) : false;
        });
      }).map((def) => def.id),
    );
  }, [query, t, overrides, mac, byKeystroke, keystroke]);

  const startRecording = (def: ShortcutDef, slot: ShortcutSlot) => {
    setRecordingError(null);
    setJustRecorded(null);
    setRecording({ id: def.id, slot });
  };

  const builtInKey = (id: ShortcutId): string | null => {
    if (id === "clearChatSelection") return "Esc";
    if (id !== "sendMessage") return null;
    const label = (draft: string) => {
      const send = composerShortcutLabels(sendShortcut, mac, draft).send;
      return mac ? send.replace("Enter", "↩") : send;
    };
    return sendShortcut === "mod-enter-multiline"
      ? `${label("")} / ${label("\n")}`
      : label("");
  };

  const conflictingLabels = (id: ShortcutId): string => {
    const values = new Set(
      SHORTCUT_SLOTS.map((slot) => resolveBinding(overrides, id, slot)).filter(
        (value): value is string => value !== null,
      ),
    );
    return SHORTCUT_DEFS.filter(
      (other) =>
        other.id !== id &&
        SHORTCUT_SLOTS.some((slot) => {
          const value = resolveBinding(overrides, other.id, slot);
          return value !== null && values.has(value);
        }),
    )
      .map((other) => t(other.labelKey))
      .join(", ");
  };

  const renderSlot = (
    def: ShortcutDef,
    slot: ShortcutSlot,
    enabled: boolean,
  ): ReactNode => {
    const label = t(def.labelKey);
    if (recording?.id === def.id && recording.slot === slot) {
      return (
        <input
          key={slot}
          readOnly={true}
          // biome-ignore lint/a11y/noAutofocus: the box exists only once the user picked this chord.
          autoFocus={true}
          onBlur={stopRecording}
          placeholder={t("settings.keyboardShortcuts.recording")}
          aria-label={`${t("settings.keyboardShortcuts.edit")}: ${label}`}
          className="h-8 w-[calc(12.5rem*var(--ui-space-scale,1))] rounded-xl border border-control-accent bg-background px-3 text-ui-13p5 text-foreground caret-transparent outline-none ring-1 ring-control-accent placeholder:text-muted-foreground"
        />
      );
    }
    const value = assignedBinding(overrides, def.id, slot);
    const parsed = parseBinding(value);
    const builtIn = slot === "primary" ? builtInKey(def.id) : null;
    const text = parsed
      ? formatBindingLabel(parsed, mac)
      : (builtIn ?? t("settings.keyboardShortcuts.unassigned"));
    const chord = (
      <button
        key={slot}
        type="button"
        onClick={() => startRecording(def, slot)}
        aria-label={`${t("settings.keyboardShortcuts.edit")}: ${label}, ${text}`}
        className={cn(
          "inline-flex h-[calc(30px*var(--ui-space-scale,1))] min-w-[calc(30px*var(--ui-space-scale,1))] items-center justify-center rounded-full px-[calc(11px*var(--ui-space-scale,1))] text-ui-13p5 tabular-nums text-muted-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          // A built-in key keeps working with the switch off; only assigned chords dim.
          parsed && !enabled && "opacity-50",
        )}
      >
        {text}
      </button>
    );
    if (isTauri || !isBrowserReservedBinding(value, mac)) return chord;
    return (
      <Tooltip key={slot}>
        <TooltipTrigger asChild={true}>{chord}</TooltipTrigger>
        <TooltipContent className="max-w-[calc(260px*var(--ui-space-scale,1))] leading-snug">
          {t("settings.keyboardShortcuts.browserReserved")}
        </TooltipContent>
      </Tooltip>
    );
  };

  const renderRow = (def: ShortcutDef) => {
    const label = t(def.labelKey);
    const enabled = isShortcutEnabled(overrides, def.id);
    const isRecording = recording?.id === def.id;
    const conflicted = conflicts.has(def.id) && !isRecording;
    // Only the owner of a clash runs.
    const shadowed =
      conflicted &&
      SHORTCUT_SLOTS.every((slot) => {
        const value = resolveBinding(overrides, def.id, slot);
        return !value || shortcutOwningBinding(overrides, value) !== def.id;
      });
    const clashesWith = conflicted ? conflictingLabels(def.id) : "";
    const inForce = SHORTCUT_SLOTS.map((slot) =>
      resolveBinding(overrides, def.id, slot),
    );
    const desktopOnly = isTauri
      ? ""
      : inForce
          .filter((value) => isBrowserUnreachableBinding(value, mac))
          .map((value) => {
            const parsed = parseBinding(value);
            return parsed ? formatBindingLabel(parsed, mac) : "";
          })
          .join(", ");
    const recordedValue =
      justRecorded?.id === def.id && !isRecording
        ? resolveBinding(overrides, def.id, justRecorded.slot)
        : null;
    const recordedReserved =
      !isTauri &&
      isBrowserReservedBinding(recordedValue, mac) &&
      !isBrowserUnreachableBinding(recordedValue, mac);
    // Anything this action ships an alternate for keeps its chord, cleared or not: hiding a cleared
    // slot would take its restore control with it.
    const hasAlternate =
      defaultBindingFor(def, "alternate", mac) !== null ||
      assignedBinding(overrides, def.id, "alternate") !== null ||
      (recording?.id === def.id && recording.slot === "alternate");

    const note: ReactNode = isRecording ? (
      recordingError ? (
        <span className="text-destructive">{recordingError}</span>
      ) : null
    ) : conflicted ? (
      <span className="text-amber-600 dark:text-amber-500">
        {shadowed
          ? t("settings.keyboardShortcuts.conflictShadowed")
          : t("settings.keyboardShortcuts.conflict")}
        {clashesWith ? `: ${clashesWith}` : null}
      </span>
    ) : recordedReserved ? (
      t("settings.keyboardShortcuts.browserReserved")
    ) : desktopOnly ? (
      `${desktopOnly}: ${t("settings.keyboardShortcuts.desktopOnly")}`
    ) : null;

    return (
      <div
        key={def.id}
        data-settings-label={label}
        className="group/row flex flex-col gap-1 py-2.5"
      >
        <div className="flex min-h-8 items-center gap-2.5">
          <Switch
            size="sm"
            checked={enabled}
            onCheckedChange={(on) => setEnabled(def.id, on)}
            aria-label={label}
          />
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <span className="min-w-0 flex-1 truncate text-ui-13p5 text-foreground">
                {label}
              </span>
            </TooltipTrigger>
            <TooltipContent side="top" align="start">
              {t(def.descriptionKey)}
            </TooltipContent>
          </Tooltip>
          {Object.hasOwn(overrides, def.id) && !isRecording ? (
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  aria-label={`${t("settings.keyboardShortcuts.reset")}: ${label}`}
                  onClick={() => {
                    setJustRecorded(null);
                    resetAction(def.id);
                  }}
                  className="inline-flex size-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center justify-center rounded-full text-muted-foreground opacity-0 transition-[color,background-color,opacity] hover:bg-accent hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring group-hover/row:opacity-100"
                >
                  <HugeiconsIcon
                    icon={UndoIcon}
                    strokeWidth={1.75}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                {t("settings.keyboardShortcuts.reset")}
              </TooltipContent>
            </Tooltip>
          ) : null}
          <div className="flex shrink-0 items-center gap-2">
            {renderSlot(def, "primary", enabled)}
            {hasAlternate ? renderSlot(def, "alternate", enabled) : null}
          </div>
        </div>
        {note ? (
          <p className="text-xs leading-snug text-muted-foreground">{note}</p>
        ) : null}
      </div>
    );
  };

  const visibleGroups = SHORTCUT_GROUPS.map((group) => ({
    group,
    defs: group.ids
      .map((id) => SHORTCUT_DEF_BY_ID[id])
      .filter(
        // A web-only row on the desktop build would bind a chord whose handler
        // returns, so it is left out rather than shown as a key that does nothing.
        (def) =>
          (!matches || matches.has(def.id)) && !(isTauri && def.webOnly),
      ),
  })).filter(({ defs }) => defs.length > 0);

  return (
    <div className="settings-page px-4">
      {/* -mx-4 undoes px-4 so the heading lines up with every other tab's. */}
      <header className="-mx-4 flex flex-col gap-1">
        <h1
          data-settings-label={t("settings.keyboardShortcuts.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.keyboardShortcuts.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.keyboardShortcuts.intro")}
        </p>
      </header>

      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={2}
          className="pointer-events-none absolute left-4 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
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
            "h-10 rounded-full pl-10 pr-12 text-ui-13p5",
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
                "absolute right-2 top-1/2 inline-flex size-7 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                byKeystroke && "bg-muted text-foreground",
              )}
            >
              <HugeiconsIcon
                icon={EnergyRectangleIcon}
                strokeWidth={1.75}
                className="size-3.5"
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

      {visibleGroups.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {t("settings.keyboardShortcuts.noResults")}
        </p>
      ) : null}

      {visibleGroups.map(({ group, defs }) => {
        return (
          <section
            key={group.id}
            data-settings-label={t(group.labelKey)}
            className="flex flex-col"
          >
            <h2 className="settings-heading mb-1 text-ui-13 font-medium">
              {t(group.labelKey)}
            </h2>
            <div className="divide-y divide-border/60">
              {defs.map(renderRow)}
            </div>
          </section>
        );
      })}

      {/* Click-through strip; -bottom-6 undoes the scroller padding sticky would inset by. */}
      <div className="pointer-events-none sticky -bottom-6 z-10 -mb-6 mt-auto flex justify-end py-4">
        <Button
          type="button"
          variant="dark"
          // Opaque disabled fill so rows never show through; dark: outranks the variant's own dark fill.
          className="pointer-events-auto text-ui-13 disabled:bg-[color-mix(in_oklab,var(--foreground)_34%,var(--background))] disabled:text-background disabled:opacity-100 dark:disabled:bg-[color-mix(in_oklab,var(--foreground)_42%,var(--background))] dark:disabled:text-background"
          disabled={Object.keys(overrides).length === 0}
          onClick={() => {
            stopRecording();
            setJustRecorded(null);
            resetAll();
          }}
        >
          {t("settings.keyboardShortcuts.resetAll")}
        </Button>
      </div>
    </div>
  );
}
