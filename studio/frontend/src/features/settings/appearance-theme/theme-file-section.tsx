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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useId, useMemo, useRef, useState } from "react";

import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { COLOR_THEMES } from "../lib/color-themes";
import {
  applyAppearanceSnapshot,
  applyThemeBackground,
  exportThemeFile,
  portableGlass,
  readAppearanceSnapshot,
  readLoadedAppearanceSnapshot,
  readThemeFile,
  restoreBackground,
  withLoadableFonts,
} from "./snapshot";
import {
  type AppearanceTheme,
  MAX_THEME_NAME,
  planThemeApply,
  type ThemeWarning,
  undoTarget,
} from "./theme-file";

type T = ReturnType<typeof useT>;
const K = "settings.appearance.themeFile";

function warningText(t: T, w: ThemeWarning): string {
  if (w.kind === "more") return t(`${K}.warning.more`, { count: w.count });
  if (w.kind === "unknownPalette")
    return t(`${K}.warning.unknownPalette`, { palette: w.palette });
  if (w.kind === "unknown" || w.kind === "invalid")
    return t(`${K}.warning.${w.kind}`, { path: w.path });
  return t(`${K}.warning.${w.kind}`);
}

function Toggle({
  id,
  label,
  checked,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <Label htmlFor={id}>{label}</Label>
      <Switch
        id={id}
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
      />
    </div>
  );
}

function Swatches({
  theme,
  mode,
}: { theme: AppearanceTheme; mode: "light" | "dark" }) {
  const t = useT();
  const fallback = COLOR_THEMES[theme.palette][mode];
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-sm text-foreground">
        {t(`settings.appearance.theme.${mode}`)}
      </span>
      <div className="flex items-center gap-1.5">
        {(["background", "foreground", "accent"] as const).map((key) => {
          const hex = theme.customization.colors[mode][key] ?? fallback[key];
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

function ExportForm({
  busy,
  setBusy,
  onClose,
}: { busy: boolean; setBusy: (b: boolean) => void; onClose: () => void }) {
  const t = useT();
  const id = useId();
  const [name, setName] = useState("");
  const [includeLayout, setIncludeLayout] = useState(false);
  const [includeAssets, setIncludeAssets] = useState(false);

  async function download() {
    setBusy(true);
    try {
      const result = await exportThemeFile({
        name: name.trim() || t(`${K}.exportNamePlaceholder`),
        includeLayout,
        includeAssets,
      });
      if (!result) return; // Cancelled: keep the dialog and what was typed.
      onClose();
      toast.success(t(`${K}.exported`, { name: result.fileName }));
      if (result.backgroundOmitted)
        toast.warning(t(`${K}.exportBackgroundOmitted`));
    } catch {
      toast.error(t(`${K}.exportFailed`));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="flex flex-col gap-6"
      onSubmit={(e) => {
        e.preventDefault();
        void download();
      }}
    >
      <DialogHeader>
        <DialogTitle>{t(`${K}.exportTitle`)}</DialogTitle>
      </DialogHeader>
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor={`${id}-name`}>{t(`${K}.exportName`)}</Label>
          <Input
            id={`${id}-name`}
            autoFocus={true}
            value={name}
            maxLength={MAX_THEME_NAME}
            placeholder={t(`${K}.exportNamePlaceholder`)}
            disabled={busy}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <Toggle
          id={`${id}-layout`}
          label={t(`${K}.exportIncludeLayout`)}
          checked={includeLayout}
          onChange={setIncludeLayout}
          disabled={busy}
        />
        <Toggle
          id={`${id}-assets`}
          label={t(`${K}.exportIncludeAssets`)}
          checked={includeAssets}
          onChange={setIncludeAssets}
          disabled={busy}
        />
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
        <Button type="submit" variant="dark" disabled={busy}>
          {t(`${K}.exportAction`)}
        </Button>
      </DialogFooter>
    </form>
  );
}

function ImportForm({
  theme,
  warnings,
  busy,
  setBusy,
  onClose,
}: {
  theme: AppearanceTheme;
  warnings: ThemeWarning[];
  busy: boolean;
  setBusy: (b: boolean) => void;
  onClose: () => void;
}) {
  const t = useT();
  const id = useId();
  const [snapshot] = useState(readAppearanceSnapshot);
  const [applyLayout, setApplyLayout] = useState(false);
  const [applyAssets, setApplyAssets] = useState(true);
  const plan = useMemo(
    () => planThemeApply(snapshot, theme, { applyLayout, applyAssets }),
    [snapshot, theme, applyLayout, applyAssets],
  );
  const shown = [
    ...warnings,
    ...plan.droppedFonts.map(
      (path): ThemeWarning => ({ kind: "invalid", path }),
    ),
  ];

  async function apply() {
    setBusy(true);
    try {
      const before = await readLoadedAppearanceSnapshot();
      const [loadable, badFonts] = applyAssets
        ? await withLoadableFonts(theme)
        : [theme, []];
      for (const path of badFonts)
        toast.warning(t(`${K}.warning.invalid`, { path }));
      const next = portableGlass(
        planThemeApply(before, loadable, { applyLayout, applyAssets }),
      );
      applyAppearanceSnapshot(next);
      if (
        next.background !== "keep" &&
        !(await applyThemeBackground(next.background))
      ) {
        toast.warning(t(`${K}.warning.backgroundSkipped`));
      }
      onClose();
      toast.success(
        t(`${K}.applied`, { name: theme.name || t(`${K}.importTitle`) }),
        {
          action: {
            label: t(`${K}.undo`),
            // Undo puts back the whole look from before the import.
            onClick: () => {
              applyAppearanceSnapshot(
                undoTarget(before, readAppearanceSnapshot()),
              );
              const restoring =
                next.background === "keep"
                  ? Promise.resolve(true)
                  : restoreBackground(before.background);
              void restoring.then((ok) =>
                ok
                  ? toast(t(`${K}.undone`))
                  : toast.warning(t(`${K}.undoneWithoutBackground`)),
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
        <DialogTitle>{t(`${K}.importTitle`)}</DialogTitle>
        <DialogDescription className="font-medium text-foreground">
          {theme.name}
        </DialogDescription>
      </DialogHeader>
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2 rounded-xl bg-wash-5 px-4 py-3">
          <Swatches theme={theme} mode="light" />
          <Swatches theme={theme} mode="dark" />
        </div>
        <section
          aria-labelledby={`${id}-changes`}
          className="flex flex-col gap-2"
        >
          <p
            id={`${id}-changes`}
            className="text-sm font-medium text-foreground"
          >
            {t(`${K}.changesTitle`)}
          </p>
          {plan.changes.length > 0 ? (
            <ul className="flex flex-wrap gap-1.5">
              {plan.changes.map((change) => (
                <li key={change}>
                  <Badge variant="secondary">
                    {t(`${K}.change.${change}`)}
                  </Badge>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">
              {t(`${K}.noChanges`)}
            </p>
          )}
        </section>
        {shown.length > 0 ? (
          <section
            aria-labelledby={`${id}-warnings`}
            className="flex flex-col gap-2"
          >
            <p
              id={`${id}-warnings`}
              className="text-sm font-medium text-foreground"
            >
              {t(`${K}.warningsTitle`)}
            </p>
            <ul
              tabIndex={0}
              className="scroll-rounded flex max-h-40 flex-col gap-1 overflow-y-auto rounded-xl bg-wash-5 px-4 py-3 text-xs text-muted-foreground"
            >
              {shown.map((w, i) => (
                <li key={`${w.kind}-${i}`}>{warningText(t, w)}</li>
              ))}
            </ul>
          </section>
        ) : null}
        {theme.hasLayout ? (
          <Toggle
            id={`${id}-layout`}
            label={t(`${K}.importApplyLayout`)}
            checked={applyLayout}
            onChange={setApplyLayout}
            disabled={busy}
          />
        ) : null}
        {theme.fonts?.length || theme.background ? (
          <Toggle
            id={`${id}-assets`}
            label={t(`${K}.importApplyAssets`)}
            checked={applyAssets}
            onChange={setApplyAssets}
            disabled={busy}
          />
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
          {t(`${K}.importApply`)}
        </Button>
      </DialogFooter>
    </div>
  );
}

type Parsed = { theme: AppearanceTheme; warnings: ThemeWarning[] };

export function ThemeFileSection() {
  const t = useT();
  const inputRef = useRef<HTMLInputElement>(null);
  const [parsed, setParsed] = useState<Parsed | null>(null);
  const [dialog, setDialog] = useState<"import" | "export" | null>(null);
  const [busy, setBusy] = useState(false);
  const close = () => setDialog(null);

  async function importFile(file: File) {
    const result = await readThemeFile(file).catch(() => null);
    if (!result?.ok) {
      toast.error(
        t(
          result?.error === "tooLarge"
            ? `${K}.fileTooLarge`
            : `${K}.fileInvalid`,
        ),
      );
      return;
    }
    setParsed(result);
    setDialog("import");
  }

  return (
    <SettingsSection title={t(`${K}.title`)}>
      <SettingsRow label={t(`${K}.importLabel`)}>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => inputRef.current?.click()}
        >
          {t(`${K}.import`)}
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept=".json,application/json"
          className="sr-only"
          tabIndex={-1}
          aria-label={t(`${K}.importLabel`)}
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) void importFile(file);
          }}
        />
      </SettingsRow>
      <SettingsRow label={t(`${K}.exportLabel`)}>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setDialog("export")}
        >
          {t(`${K}.export`)}
        </Button>
      </SettingsRow>
      <Dialog
        open={dialog !== null}
        onOpenChange={(open) => !open && !busy && close()}
      >
        <DialogContent
          showCloseButton={!busy}
          // The warnings list is focusable; start on Apply instead.
          onOpenAutoFocus={(e) => {
            const target = (
              e.currentTarget as HTMLElement
            ).querySelector<HTMLElement>("[data-initial-focus]");
            if (!target) return;
            e.preventDefault();
            target.focus();
          }}
          onCloseAutoFocus={() => setParsed(null)}
        >
          {dialog === "export" ? (
            <ExportForm busy={busy} setBusy={setBusy} onClose={close} />
          ) : null}
          {dialog === "import" && parsed ? (
            <ImportForm
              theme={parsed.theme}
              warnings={parsed.warnings}
              busy={busy}
              setBusy={setBusy}
              onClose={close}
            />
          ) : null}
        </DialogContent>
      </Dialog>
    </SettingsSection>
  );
}
