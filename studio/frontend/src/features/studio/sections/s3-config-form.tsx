// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import type { S3Config } from "@/types/training";
import { useShallow } from "zustand/react/shallow";

const DEFAULT_S3_CONFIG: S3Config = {
  bucket: "",
  region: "us-east-1",
  prefix: "",
  accessKeyId: "",
  secretAccessKey: "",
  useIamRole: false,
};

/**
 * Inline S3 dataset configuration form. Shown in the dataset section when the
 * selected source is "s3"; reads and writes the shared training-config store.
 */
export function S3ConfigForm() {
  const t = useT();
  const { s3Config, setS3Config } = useTrainingConfigStore(
    useShallow((s) => ({
      s3Config: s.s3Config,
      setS3Config: s.setS3Config,
    })),
  );

  const config = s3Config ?? DEFAULT_S3_CONFIG;

  const update = (patch: Partial<S3Config>) => {
    setS3Config({ ...config, ...patch });
  };

  const handleIamRoleChange = (useIamRole: boolean) => {
    if (useIamRole) {
      update({ useIamRole, accessKeyId: "", secretAccessKey: "" });
      return;
    }
    update({ useIamRole });
  };

  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div>
        <p className="text-xs font-medium text-foreground">
          {t("studio.dataset.s3.title")}
        </p>
        <p className="text-[0.625rem] text-muted-foreground/80">
          {t("studio.dataset.s3.description")}
        </p>
      </div>

      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="s3-bucket" className="text-xs text-muted-foreground">
          {t("studio.dataset.s3.bucket")}
        </Label>
        <Input
          id="s3-bucket"
          value={config.bucket}
          onChange={(e) => update({ bucket: e.target.value })}
          placeholder={t("studio.dataset.s3.bucketPlaceholder")}
        />
      </div>

      <div className="flex min-w-0 gap-2">
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <Label htmlFor="s3-region" className="text-xs text-muted-foreground">
            {t("studio.dataset.s3.region")}
          </Label>
          <Input
            id="s3-region"
            value={config.region}
            onChange={(e) => update({ region: e.target.value })}
            placeholder={t("studio.dataset.s3.regionPlaceholder")}
          />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <Label htmlFor="s3-prefix" className="text-xs text-muted-foreground">
            {t("studio.dataset.s3.prefix")}
          </Label>
          <Input
            id="s3-prefix"
            value={config.prefix ?? ""}
            onChange={(e) => update({ prefix: e.target.value })}
            placeholder={t("studio.dataset.s3.prefixPlaceholder")}
          />
        </div>
      </div>

      <div className="flex items-center justify-between gap-2">
        <Label htmlFor="s3-iam" className="text-xs text-muted-foreground">
          {t("studio.dataset.s3.useIamRole")}
        </Label>
        <Switch
          id="s3-iam"
          checked={config.useIamRole ?? false}
          onCheckedChange={handleIamRoleChange}
        />
      </div>

      {!config.useIamRole && (
        <>
          <div className="flex min-w-0 flex-col gap-1">
            <Label
              htmlFor="s3-access-key"
              className="text-xs text-muted-foreground"
            >
              {t("studio.dataset.s3.accessKeyId")}
            </Label>
            <Input
              id="s3-access-key"
              value={config.accessKeyId ?? ""}
              onChange={(e) => update({ accessKeyId: e.target.value })}
              placeholder={t("studio.dataset.s3.accessKeyIdPlaceholder")}
            />
          </div>
          <div className="flex min-w-0 flex-col gap-1">
            <Label
              htmlFor="s3-secret-key"
              className="text-xs text-muted-foreground"
            >
              {t("studio.dataset.s3.secretAccessKey")}
            </Label>
            <Input
              id="s3-secret-key"
              type="password"
              value={config.secretAccessKey ?? ""}
              onChange={(e) => update({ secretAccessKey: e.target.value })}
              placeholder={t("studio.dataset.s3.secretAccessKeyPlaceholder")}
            />
          </div>
        </>
      )}
    </div>
  );
}
