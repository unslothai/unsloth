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
import { cn } from "@/lib/utils";
import {
  Alert01Icon,
  ArrowTurnBackwardIcon,
  Delete02Icon,
  PencilEdit02Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import {
  SHORTCUT_DEFS,
  SHORTCUT_GROUPS,
  type ShortcutDef,
  type ShortcutGroup,
  type ShortcutId,
  bindingFromEvent,
  formatBindingLabel,
  formatBindingValue,
  isAcceptableBinding,
  isMacPlatform,
  parseBinding,
} from "../lib/keyboard-shortcuts";
import {
  findConflicts,
  resolveBinding,
  shortcutOwningBinding,
  useKeyboardShortcutsStore,
} from "../stores/keyboard-shortcuts-store";

const GROUP_LABEL_KEYS = {
  general: "settings.keyboardShortcuts.groups.general",
  chat: "settings.keyboardShortcuts.groups.chat",
} as const satisfies Record<ShortcutGroup, string>;

/** The ⌘⇧O chip. Renders the placeholder text when nothing is bound. */
function BindingChip({
  label,
  tone,
}: {
  label: string;
  tone: "assigned" | "unassigned" | "recording";
}) {
  return (
    <span
      className={cn(
        "inline-flex min-w-16 items-center justify-center rounded-md px-2 py-1 text-xs font-medium tabular-nums",
        tone === "assigned" && "bg-muted text-foreground",
        tone === "unassigned" && "text-muted-foreground",
        tone === "recording" &&
          "bg-primary/10 text-primary ring-1 ring-primary/40",
      )}
    >
      {label}
    </span>
  );
}

function IconButton({
  icon,
  label,
  onClick,
  disabled,
}: {
  icon: typeof PencilEdit02Icon;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="size-8 text-muted-foreground hover:text-foreground"
          aria-label={label}
          disabled={disabled}
          onClick={onClick}
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

export function KeyboardShortcutsTab() {
  const t = useT();
  const overrides = useKeyboardShortcutsStore((s) => s.overrides);
  const setBinding = useKeyboardShortcutsStore((s) => s.setBinding);
  const clearBinding = useKeyboardShortcutsStore((s) => s.clearBinding);
  const resetBinding = useKeyboardShortcutsStore((s) => s.resetBinding);
  const resetAll = useKeyboardShortcutsStore((s) => s.resetAll);

  const [query, setQuery] = useState("");
  const [recordingId, setRecordingId] = useState<ShortcutId | null>(null);
  // Shown under the row being recorded when the pressed chord is rejected.
  const [recordingError, setRecordingError] = useState<string | null>(null);

  const mac = isMacPlatform();
  const conflicts = useMemo(() => findConflicts(overrides), [overrides]);

  // Capture phase, so the chord being recorded reaches this listener before the
  // shortcut it is replacing fires and before Radix's Escape-to-close.
  useEffect(() => {
    if (!recordingId) return;
    const onKeyDown = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (
        event.code === "Escape" &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.altKey &&
        !event.shiftKey
      ) {
        setRecordingId(null);
        setRecordingError(null);
        return;
      }
      const binding = bindingFromEvent(event);
      // Modifier held on its own: the user is still assembling the chord.
      if (!binding) return;
      if (!isAcceptableBinding(binding)) {
        setRecordingError(t("settings.keyboardShortcuts.needsModifier"));
        return;
      }
      setBinding(recordingId, formatBindingValue(binding));
      setRecordingId(null);
      setRecordingError(null);
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () =>
      window.removeEventListener("keydown", onKeyDown, { capture: true });
  }, [recordingId, setBinding, t]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return null;
    return new Set(
      SHORTCUT_DEFS.filter((def) => {
        const haystack = `${t(def.labelKey)} ${t(def.descriptionKey)}`;
        if (haystack.toLowerCase().includes(q)) return true;
        const value = resolveBinding(overrides, def.id);
        const parsed = parseBinding(value);
        return parsed
          ? formatBindingLabel(parsed, mac).toLowerCase().includes(q)
          : false;
      }).map((def) => def.id),
    );
  }, [query, t, overrides, mac]);

  const groups = SHORTCUT_GROUPS.map((group) => ({
    group,
    defs: SHORTCUT_DEFS.filter(
      (def) => def.group === group && (!matches || matches.has(def.id)),
    ),
  })).filter((entry) => entry.defs.length > 0);

  // biome-ignore lint/complexity/noExcessiveCognitiveComplexity: Chip, hint, badge and buttons all read one resolved binding; splitting only passes it around.
  const renderRow = (def: ShortcutDef) => {
    const value = resolveBinding(overrides, def.id);
    const parsed = parseBinding(value);
    const recording = recordingId === def.id;
    const conflicted = conflicts.has(def.id) && !recording;
    // Say which side of a clash this row is on, since only the owner runs.
    const shadowed =
      conflicted && shortcutOwningBinding(overrides, value) !== def.id;
    const label = recording
      ? t("settings.keyboardShortcuts.recording")
      : parsed
        ? formatBindingLabel(parsed, mac)
        : t("settings.keyboardShortcuts.unassigned");
    const overridden = Object.hasOwn(overrides, def.id);

    return (
      <SettingsRow
        key={def.id}
        label={t(def.labelKey)}
        description={
          <span className="flex flex-col gap-0.5">
            <span>{t(def.descriptionKey)}</span>
            {recording ? (
              <span className="text-xs text-primary">
                {recordingError ??
                  t("settings.keyboardShortcuts.recordingHint")}
              </span>
            ) : null}
            {conflicted ? (
              <span className="flex items-center gap-1 text-xs text-amber-500">
                <HugeiconsIcon
                  icon={Alert01Icon}
                  strokeWidth={1.75}
                  className="size-3.5 shrink-0"
                />
                {shadowed
                  ? t("settings.keyboardShortcuts.conflictShadowed")
                  : t("settings.keyboardShortcuts.conflict")}
              </span>
            ) : null}
          </span>
        }
      >
        <div className="flex items-center gap-1">
          <BindingChip
            label={label}
            tone={recording ? "recording" : parsed ? "assigned" : "unassigned"}
          />
          <IconButton
            icon={PencilEdit02Icon}
            label={t("settings.keyboardShortcuts.edit")}
            onClick={() => {
              setRecordingError(null);
              setRecordingId(recording ? null : def.id);
            }}
          />
          {overridden ? (
            <IconButton
              icon={ArrowTurnBackwardIcon}
              label={t("settings.keyboardShortcuts.reset")}
              onClick={() => resetBinding(def.id)}
            />
          ) : null}
          <IconButton
            icon={Delete02Icon}
            label={t("settings.keyboardShortcuts.clear")}
            disabled={!parsed}
            onClick={() => clearBinding(def.id)}
          />
        </div>
      </SettingsRow>
    );
  };

  return (
    <div className="flex flex-col gap-6">
      <SettingsSection
        title={t("settings.keyboardShortcuts.title")}
        description={t("settings.keyboardShortcuts.description")}
      >
        <div className="relative py-3">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={2}
            className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("settings.keyboardShortcuts.searchPlaceholder")}
            className="pl-9"
            aria-label={t("settings.keyboardShortcuts.searchPlaceholder")}
          />
        </div>
      </SettingsSection>

      {groups.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {t("settings.keyboardShortcuts.noResults")}
        </p>
      ) : (
        groups.map(({ group, defs }) => (
          <SettingsSection key={group} title={t(GROUP_LABEL_KEYS[group])}>
            {defs.map(renderRow)}
          </SettingsSection>
        ))
      )}

      <div className="flex justify-start border-t border-border/60 pt-4">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={Object.keys(overrides).length === 0}
          onClick={() => {
            setRecordingId(null);
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
