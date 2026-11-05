// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useRef, useState } from "react";

import { SettingsRow } from "../../components/settings-row";
import { SettingsSection } from "../../components/settings-section";
import { readThemeFile } from "../file";
import { ExportThemeDialog } from "./export-theme-dialog";
import { ImportThemeDialog, type ParsedThemeFile } from "./import-theme-dialog";

export function ThemeFileSection() {
  const t = useT();
  const inputRef = useRef<HTMLInputElement>(null);
  const [parsed, setParsed] = useState<ParsedThemeFile | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const importLabel = t("settings.appearance.themeFile.importLabel");

  async function importFile(file: File) {
    try {
      const result = await readThemeFile(file);
      if (!result.ok) {
        toast.error(
          t(
            result.error === "tooLarge"
              ? "settings.appearance.themeFile.fileTooLarge"
              : "settings.appearance.themeFile.fileInvalid",
          ),
        );
        return;
      }
      setParsed(result);
      setImportOpen(true);
    } catch {
      toast.error(t("settings.appearance.themeFile.readFailed"));
    }
  }

  return (
    <SettingsSection
      title={t("settings.appearance.themeFile.title")}
      description={t("settings.appearance.themeFile.description")}
    >
      <SettingsRow
        label={importLabel}
        description={t("settings.appearance.themeFile.importDescription")}
      >
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => inputRef.current?.click()}
        >
          {t("settings.appearance.themeFile.import")}
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept=".json,application/json"
          className="sr-only"
          tabIndex={-1}
          aria-label={importLabel}
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) void importFile(file);
          }}
        />
      </SettingsRow>
      <SettingsRow
        label={t("settings.appearance.themeFile.exportLabel")}
        description={t("settings.appearance.themeFile.exportDescription")}
      >
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setExportOpen(true)}
        >
          {t("settings.appearance.themeFile.export")}
        </Button>
      </SettingsRow>
      <ImportThemeDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onClosed={() => setParsed(null)}
        parsed={parsed}
      />
      <ExportThemeDialog open={exportOpen} onOpenChange={setExportOpen} />
    </SettingsSection>
  );
}
