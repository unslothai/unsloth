// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { AlertCircleIcon, ArrowRight01Icon, CheckmarkCircle02Icon, Key01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion } from "motion/react";
import { useTranslation } from "react-i18next";
import { collapseAnim } from "../anim";
import { EXPORT_METHODS, type ExportMethod } from "../constants";

type Destination = "local" | "hub";

interface ExportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  checkpoint: string | null;
  exportMethod: ExportMethod | null;
  quantLevels: string[];
  estimatedSize: string;
  baseModelName: string;
  isAdapter: boolean;
  destination: Destination;
  onDestinationChange: (v: Destination) => void;
  hfUsername: string;
  onHfUsernameChange: (v: string) => void;
  modelName: string;
  onModelNameChange: (v: string) => void;
  hfToken: string;
  onHfTokenChange: (v: string) => void;
  privateRepo: boolean;
  onPrivateRepoChange: (v: boolean) => void;
  onExport: () => void;
  exporting: boolean;
  exportError: string | null;
  exportSuccess: boolean;
}

export function ExportDialog({
  open,
  onOpenChange,
  checkpoint,
  exportMethod,
  quantLevels,
  estimatedSize: _estimatedSize,
  baseModelName,
  isAdapter,
  destination,
  onDestinationChange,
  hfUsername,
  onHfUsernameChange,
  modelName,
  onModelNameChange,
  hfToken,
  onHfTokenChange,
  privateRepo,
  onPrivateRepoChange,
  onExport,
  exporting,
  exportError,
  exportSuccess,
}: ExportDialogProps) {
  const { t } = useTranslation();
  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (exporting) return;
        onOpenChange(v);
      }}
    >
      <DialogContent className="sm:max-w-lg" onInteractOutside={(e) => { if (exporting) e.preventDefault(); }}>
        {exportSuccess ? (
          <>
            <div className="flex flex-col items-center gap-3 py-6">
              <div className="flex size-12 items-center justify-center rounded-full bg-emerald-500/10">
                <HugeiconsIcon icon={CheckmarkCircle02Icon} className="size-6 text-emerald-500" />
              </div>
              <div className="text-center">
                <h3 className="text-lg font-semibold">{t("export.exportSuccess")}</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  {destination === "hub"
                    ? t("export.saveToHuggingFace").replace(".", "")
                    : t("export.saveToLocal")}
                </p>
              </div>
            </div>
              <DialogFooter>
                <Button onClick={() => onOpenChange(false)}>{t("export.done")}</Button>
              </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>{t("export.export")}</DialogTitle>
              <DialogDescription>
                {t("export.destination").replace(".", "")}
              </DialogDescription>
            </DialogHeader>

            <div className="flex gap-2">
              <Button
                variant={destination === "local" ? "dark" : "outline"}
                onClick={() => onDestinationChange("local")}
                disabled={exporting}
                className="flex-1"
              >
                保存到本地
              </Button>
              <Button
                variant={destination === "hub" ? "dark" : "outline"}
                onClick={() => onDestinationChange("hub")}
                disabled={exporting}
                className="flex-1"
              >
                发布到 Hub
              </Button>
            </div>

            <AnimatePresence>
              {destination === "hub" && (
                <motion.div {...collapseAnim} className="overflow-hidden">
                  <div className="flex flex-col gap-4 px-0.5">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="flex flex-col gap-1.5">
                        <label htmlFor="hf-username" className="text-xs font-medium text-muted-foreground">
                          HF 用户名/组
                        </label>
                        <Input
                          id="hf-username"
                          placeholder="your-username"
                          value={hfUsername}
                          onChange={(e) => onHfUsernameChange(e.target.value)}
                          disabled={exporting}
                        />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <label htmlFor="model-name" className="text-xs font-medium text-muted-foreground">
                          模型名称
                        </label>
                        <Input
                          id="model-name"
                          placeholder="my-model-gguf"
                          value={modelName}
                          onChange={(e) => onModelNameChange(e.target.value)}
                          disabled={exporting}
                        />
                      </div>
                    </div>

                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center justify-between">
                          <label htmlFor="hf-token" className="text-xs font-medium text-muted-foreground">
                            HF 写权限令牌
                          </label>
                          <a
                            href="https://huggingface.co/settings/tokens"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 text-[11px] text-emerald-600 hover:text-emerald-700 transition-colors"
                          >
                            获取令牌
                            <HugeiconsIcon
                              icon={ArrowRight01Icon}
                              className="size-3"
                            />
                          </a>
                        </div>
                        <InputGroup>
                          <InputGroupAddon>
                            <HugeiconsIcon icon={Key01Icon} className="size-4" />
                          </InputGroupAddon>
                          <InputGroupInput
                            id="hf-token"
                            type="password"
                            autoComplete="new-password"
                            name="hf-token"
                            placeholder="hf_..."
                            value={hfToken}
                            onChange={(e) => onHfTokenChange(e.target.value)}
                            disabled={exporting}
                          />
                        </InputGroup>
                        <p className="text-[11px] text-muted-foreground/70">
                          如已登录可留空
                        </p>
                      </div>

                    <div className="flex items-center gap-3">
                      <Switch
                        id="private-repo"
                        size="sm"
                        checked={privateRepo}
                        onCheckedChange={onPrivateRepoChange}
                        disabled={exporting}
                      />
                      <label
                        htmlFor="private-repo"
                        className="text-xs font-medium cursor-pointer"
                      >
                        私有仓库
                      </label>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error banner */}
            {exportError && (
              <div className="flex items-start gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                <HugeiconsIcon icon={AlertCircleIcon} className="size-4 mt-0.5 shrink-0" />
                <span>{exportError}</span>
              </div>
            )}

            {/* Summary */}
            <div className="rounded-xl bg-muted/50 p-3 text-xs text-muted-foreground flex flex-col gap-1">
              <div className="flex justify-between">
                <span>基础模型</span>
                <span className="font-medium text-foreground">{baseModelName}</span>
              </div>
              <div className="flex justify-between">
                <span>{isAdapter ? "检查点" : "模型"}</span>
                <span className="font-medium text-foreground">{checkpoint}</span>
              </div>
              <div className="flex justify-between">
                <span>导出方法</span>
                <span className="font-medium text-foreground">
                  {EXPORT_METHODS.find((m) => m.value === exportMethod)?.title}
                </span>
              </div>
              {exportMethod === "gguf" && quantLevels.length > 0 && (
                <div className="flex justify-between">
                  <span>量化</span>
                  <span className="font-medium text-foreground">
                    {quantLevels.join(", ")}
                  </span>
                </div>
              )}
              {/* TODO: unhide once estimated size comes from the backend API */}
              {/* <div className="flex justify-between">
            <span>Est. size</span>
            <span className="font-medium text-foreground">{estimatedSize}</span>
          </div> */}
            </div>

            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={exporting}
              >
                取消
              </Button>
              <Button onClick={onExport} disabled={exporting}>
                {exporting ? (
                  <span className="flex items-center gap-2">
                    <Spinner className="size-4" />
                    正在导出...
                  </span>
                ) : (
                  "开始导出"
                )}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
