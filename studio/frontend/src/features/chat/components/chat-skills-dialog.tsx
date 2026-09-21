// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { Scroll01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { RefreshCwIcon } from "lucide-react";
import { type ReactElement, useEffect, useState } from "react";
import {
  listSkills,
  setSkillEnabled,
  useSkillsCatalog,
} from "../api/skills-api";

export function ChatSkillsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  const t = useT();
  const { skills, loading, error } = useSkillsCatalog();
  const [changing, setChanging] = useState<string | null>(null);
  const sourceLabel = (source: "agents" | "claude" | "bundled") =>
    source === "agents"
      ? t("skills.sourceAgents")
      : source === "claude"
        ? t("skills.sourceClaude")
        : t("skills.sourceBundled");
  // Skills are added by writing files, so each open re-reads the folders.
  useEffect(() => {
    if (open) void listSkills(true).catch(() => undefined);
  }, [open]);

  const toggle = async (name: string, enabled: boolean) => {
    setChanging(name);
    try {
      await setSkillEnabled(name, enabled);
    } catch (cause) {
      toast.error(t("skills.updateError"), {
        description: cause instanceof Error ? cause.message : undefined,
      });
    } finally {
      setChanging(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="rounded-xl shadow-border ring-0 [--radius:1.1rem] max-sm:flex max-sm:flex-col max-sm:overflow-hidden sm:max-w-xl">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <HugeiconsIcon
              icon={Scroll01Icon}
              strokeWidth={1.75}
              className="size-5 text-primary"
            />
            <DialogTitle>{t("skills.title")}</DialogTitle>
          </div>
          <DialogDescription>{t("skills.description")}</DialogDescription>
        </DialogHeader>

        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-muted-foreground">
            {t("skills.precedence")}
          </p>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={loading}
            onClick={() => void listSkills(true).catch(() => undefined)}
          >
            {loading ? <Spinner /> : <RefreshCwIcon />}
            {t("skills.refresh")}
          </Button>
        </div>

        <div className="hover-scrollbar min-h-0 max-h-[min(58dvh,520px)] space-y-2 overflow-y-auto pr-1 max-sm:flex-1 max-sm:max-h-none">
          {error ? (
            <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
              {error}
            </div>
          ) : null}
          {!loading && !error && skills.length === 0 ? (
            <div className="rounded-xl border border-dashed p-6 text-center text-sm text-muted-foreground">
              {t("skills.empty")}
            </div>
          ) : null}
          {skills.map((skill) => {
            const selectable = skill.valid && !skill.shadowed;
            return (
              <div
                key={`${skill.source}:${skill.name}`}
                className="rounded-xl border border-border/60 bg-muted/20 p-4 dark:border-transparent dark:bg-white/[0.06]"
              >
                <div className="flex items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="truncate font-medium">{skill.name}</span>
                      <Badge variant="outline">{sourceLabel(skill.source)}</Badge>
                      {skill.shadowed ? (
                        <Badge variant="secondary">{t("skills.shadowed")}</Badge>
                      ) : null}
                      {skill.valid ? null : (
                        <Badge variant="destructive">{t("skills.invalid")}</Badge>
                      )}
                    </div>
                    {skill.description ? (
                      <p className="mt-1 text-sm text-muted-foreground">
                        {skill.description}
                      </p>
                    ) : null}
                    {skill.compatibility ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        {t("skills.compatibility", { value: skill.compatibility })}
                      </p>
                    ) : null}
                    {skill.shadowed_by ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        {t("skills.shadowedBy", {
                          source: sourceLabel(skill.shadowed_by),
                        })}
                      </p>
                    ) : null}
                    {skill.error ? (
                      <p className="mt-2 text-xs text-destructive">
                        {skill.error}
                      </p>
                    ) : null}
                  </div>
                  <Switch
                    aria-label={t(skill.enabled ? "skills.disable" : "skills.enable", {
                      name: skill.name,
                    })}
                    checked={selectable && skill.enabled}
                    disabled={!selectable || changing === skill.name}
                    onCheckedChange={(checked) =>
                      void toggle(skill.name, checked)
                    }
                  />
                </div>
              </div>
            );
          })}
        </div>
      </DialogContent>
    </Dialog>
  );
}
