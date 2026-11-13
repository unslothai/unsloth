// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useId, useMemo, useState } from "react";

import { COLOR_THEMES } from "../../lib/color-themes";
import { planThemeApply, planUndo } from "../application/plan-apply";
import type { AppearanceTheme, ThemeParseResult, ThemeWarning } from "../domain/schema";
import {
  applyAppearanceSnapshot,
  applyThemeBackground,
  portableGlass,
  readAppearanceSnapshot,
  readLoadedAppearanceSnapshot,
  undoBackground,
} from "../snapshot";

export type ParsedThemeFile = Extract<ThemeParseResult, { ok: true }>;

type T = ReturnType<typeof useT>;
type Mode = "light" | "dark";
const SWATCH_KEYS = ["background", "foreground", "accent"] as const;

function warningText(t: T, warning: ThemeWarning): string {
  switch (warning.kind) {
    case "newerVersion":
      return t("settings.appearance.themeFile.warning.newerVersion");
    case "backgroundSkipped":
      return t("settings.appearance.themeFile.warning.backgroundSkipped");
    case "more":
      return t("settings.appearance.themeFile.warning.more", {
        count: warning.count,
      });
    case "unknownPalette":
      return t("settings.appearance.themeFile.warning.unknownPalette", {
        palette: warning.palette,
      });
    case "unknown":
    case "invalid":
    case "clamped":
    case "dropped":
      return t(`settings.appearance.themeFile.warning.${warning.kind}`, {
        path: warning.path,
      });
  }
}

function ThemeSwatchRow({
  theme,
  mode,
}: {
  theme: AppearanceTheme;
  mode: Mode;
}) {
  const t = useT();
  const fallback = COLOR_THEMES[theme.palette][mode];
  const colors = theme.look.colors[mode];
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-sm text-foreground">
        {t(
          mode === "light"
            ? "settings.appearance.theme.light"
            : "settings.appearance.theme.dark",
        )}
      </span>
      <div className="flex items-center gap-1.5">
        {SWATCH_KEYS.map((key) => {
          const hex = colors[key] ?? fallback[key];
          return (
            <span
              key={key}
              role="img"
              aria-label={`${t(`settings.appearance.custom.colors.${key}`)} ${hex}`}
              title={hex}
              className="size-5 rounded-full ring-1 ring-inset ring-wash-15"
              style={{ backgroundColor: hex }}
            />
          );
        })}
      </div>
    </div>
  );
}

/** The dialog's content mounts on each open, so the snapshot and switches start fresh. */
function ImportThemeForm({
  parsed,
  busy,
  setBusy,
  onClose,
}: {
  parsed: ParsedThemeFile;
  busy: boolean;
  setBusy: (busy: boolean) => void;
  onClose: () => void;
}) {
  const t = useT();
  const id = useId();
  const { theme, warnings } = parsed;
  const hasLayout = theme.layout !== null;
  const hasAssets = Boolean(theme.fonts?.length) || theme.background !== null;
  const [snapshot] = useState(readAppearanceSnapshot);
  const [applyLayout, setApplyLayout] = useState(false);
  const [applyAssets, setApplyAssets] = useState(true);
  const plan = useMemo(
    () => planThemeApply(snapshot, theme, { applyLayout, applyAssets }),
    [snapshot, theme, applyLayout, applyAssets],
  );
  // Fonts pushed out by the 3-font cap are listed with the file's own warnings.
  const shownWarnings: ThemeWarning[] = [
    ...warnings,
    ...plan.droppedFonts.map((name): ThemeWarning => ({ kind: "dropped", path: name })),
  ];

  async function apply() {
    if (busy) return;
    setBusy(true);
    try {
      // Wait for the stored background, or Undo would think there was none and delete it.
      const prev = await readLoadedAppearanceSnapshot();
      const next = portableGlass(planThemeApply(prev, theme, { applyLayout, applyAssets }));
      applyAppearanceSnapshot(next);
      // Undo restores the background only while it is still the one the theme set.
      let appliedBackground: ReturnType<typeof readAppearanceSnapshot>["background"] = null;
      if (next.background !== "keep") {
        if (await applyThemeBackground(next.background)) {
          appliedBackground = readAppearanceSnapshot().background;
        } else {
          toast.warning(
            t("settings.appearance.themeFile.warning.backgroundSkipped"),
          );
        }
      }
      onClose();
      toast.success(
        t("settings.appearance.themeFile.applied", { name: theme.meta.name }),
        {
          action: {
            label: t("settings.appearance.themeFile.undo"),
            onClick: () => {
              applyAppearanceSnapshot(planUndo(prev, next, readAppearanceSnapshot()));
              void undoBackground(prev.background, appliedBackground).then((restored) =>
                restored
                  ? toast(t("settings.appearance.themeFile.undone"))
                  : toast.warning(t("settings.appearance.themeFile.undoneWithoutBackground")),
              );
            },
          },
        },
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <DialogHeader>
        <DialogTitle>{t("settings.appearance.themeFile.importTitle")}</DialogTitle>
        <DialogDescription className="flex flex-col gap-1">
          <span className="font-medium text-foreground">{theme.meta.name}</span>
          {theme.meta.author ? (
            <span className="text-xs">
              {t("settings.appearance.themeFile.importBy", {
                author: theme.meta.author,
              })}
            </span>
          ) : null}
          {theme.meta.description ? (
            <span className="text-xs leading-relaxed">
              {theme.meta.description}
            </span>
          ) : null}
        </DialogDescription>
      </DialogHeader>

      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2 rounded-xl bg-wash-5 px-4 py-3">
          <ThemeSwatchRow theme={theme} mode="light" />
          <ThemeSwatchRow theme={theme} mode="dark" />
        </div>

        <section aria-labelledby={`${id}-changes`} className="flex flex-col gap-2">
          <p id={`${id}-changes`} className="text-sm font-medium text-foreground">
            {t("settings.appearance.themeFile.changesTitle")}
          </p>
          {plan.changes.length > 0 ? (
            <ul className="flex flex-wrap gap-1.5">
              {plan.changes.map((change) => (
                <li key={change}>
                  <Badge variant="secondary">
                    {t(`settings.appearance.themeFile.change.${change}`)}
                  </Badge>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">
              {t("settings.appearance.themeFile.noChanges")}
            </p>
          )}
        </section>

        {shownWarnings.length > 0 ? (
          <section aria-labelledby={`${id}-warnings`} className="flex flex-col gap-2">
            <p id={`${id}-warnings`} className="text-sm font-medium text-foreground">
              {t("settings.appearance.themeFile.warningsTitle")}
            </p>
            <ul
              tabIndex={0}
              className="scroll-rounded flex max-h-40 flex-col gap-1 overflow-y-auto rounded-xl bg-wash-5 px-4 py-3 text-xs text-muted-foreground"
            >
              {shownWarnings.map((warning, index) => (
                <li key={`${warning.kind}-${index}`}>
                  {warningText(t, warning)}
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {hasLayout || hasAssets ? (
          <div className="flex flex-col gap-3">
            {hasLayout ? (
              <div className="flex items-center justify-between gap-3">
                <Label htmlFor={`${id}-layout`}>
                  {t("settings.appearance.themeFile.importApplyLayout")}
                </Label>
                <Switch
                  id={`${id}-layout`}
                  checked={applyLayout}
                  onCheckedChange={setApplyLayout}
                  disabled={busy}
                />
              </div>
            ) : null}
            {hasAssets ? (
              <div className="flex items-center justify-between gap-3">
                <Label htmlFor={`${id}-assets`}>
                  {t("settings.appearance.themeFile.importApplyAssets")}
                </Label>
                <Switch
                  id={`${id}-assets`}
                  checked={applyAssets}
                  onCheckedChange={setApplyAssets}
                  disabled={busy}
                />
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      <DialogFooter>
        <Button
          type="button"
          variant="outline"
          disabled={busy}
          onClick={onClose}
        >
          {t("common.cancel")}
        </Button>
        <Button
          type="button"
          variant="dark"
          data-initial-focus=""
          disabled={busy}
          onClick={() => void apply()}
        >
          {t("settings.appearance.themeFile.importApply")}
        </Button>
      </DialogFooter>
    </div>
  );
}

export function ImportThemeDialog({
  open,
  onOpenChange,
  onClosed,
  parsed,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** After the close animation, so the parsed file (fonts, background) can be let go. */
  onClosed: () => void;
  parsed: ParsedThemeFile | null;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (busy && !next) return;
        onOpenChange(next);
      }}
    >
      <DialogContent
        showCloseButton={!busy}
        // The warnings list is scrollable, so it is focusable; start on Apply instead.
        onOpenAutoFocus={(event) => {
          const target = (event.currentTarget as HTMLElement).querySelector<HTMLElement>(
            "[data-initial-focus]",
          );
          if (!target) return;
          event.preventDefault();
          target.focus();
        }}
        onCloseAutoFocus={onClosed}
      >
        {parsed ? (
          <ImportThemeForm
            parsed={parsed}
            busy={busy}
            setBusy={setBusy}
            onClose={() => onOpenChange(false)}
          />
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
