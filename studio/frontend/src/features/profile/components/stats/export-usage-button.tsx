// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { exportProfileUsageCsv } from "../../api/profile-stats";

function localDateStamp(now = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

export function ExportUsageButton() {
  const t = useT();
  const [busy, setBusy] = useState(false);

  const onExport = async () => {
    setBusy(true);
    try {
      const csv = await exportProfileUsageCsv();
      await downloadFile(
        csv,
        `unsloth-token-usage-${localDateStamp()}.csv`,
        "text/csv",
      );
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error(t("settings.profile.stats.exportError"), {
          description: error instanceof Error ? error.message : undefined,
        });
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <Button
      type="button"
      size="sm"
      variant="outline"
      disabled={busy}
      onClick={() => void onExport()}
    >
      <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
      {t("settings.profile.stats.exportCsv")}
    </Button>
  );
}
