// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DocumentAttachmentIcon, DocumentCodeIcon, Plant01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import type { SeedConfig } from "../../types";
import { HfDatasetCombobox } from "../shared/hf-dataset-combobox";
import { InlineField } from "./inline-field";

type InlineSeedProps = {
  config: SeedConfig;
  onUpdate: (patch: Partial<SeedConfig>) => void;
};

export function InlineSeed({ config, onUpdate }: InlineSeedProps): ReactElement {
  const mode = config.seed_source_type ?? "hf";

  if (mode === "hf") {
    return (
      <div className="space-y-2">
        <InlineField label="数据集">
          <HfDatasetCombobox
            value={config.hf_repo_id}
            accessToken={config.hf_token?.trim() || undefined}
            onValueChange={(next) =>
              onUpdate({
                hf_repo_id: next,
                hf_path: "",
                seed_columns: [],
                seed_drop_columns: [],
                seed_preview_rows: [],
              })
            }
            placeholder="org/repo"
          />
        </InlineField>
        <p className="text-[11px] text-muted-foreground">
          请在弹窗中加载列。
        </p>
      </div>
    );
  }

  const isLocal = mode === "local";
  const fileName = isLocal
    ? config.local_file_name?.trim()
    : config.unstructured_file_names?.length
      ? `${config.unstructured_file_names.length} 个文件`
      : undefined;

  return (
    <div className="corner-squircle flex items-center gap-2 rounded-md border border-border/60 bg-muted/30 px-2 py-2">
      <div className="corner-squircle rounded-md bg-primary/10 p-1.5 text-primary">
        <HugeiconsIcon
          icon={isLocal ? DocumentCodeIcon : DocumentAttachmentIcon}
          className="size-3.5"
        />
      </div>
      <div className="min-w-0">
        <p className="truncate text-xs font-medium">
          {fileName || "未选择文件"}
        </p>
        <p className="text-[11px] text-muted-foreground">
          {isLocal ? "结构化文件" : "非结构化文档"} · 请在弹窗中配置
        </p>
      </div>
      <HugeiconsIcon icon={Plant01Icon} className="ml-auto size-3.5 text-muted-foreground/60" />
    </div>
  );
}
