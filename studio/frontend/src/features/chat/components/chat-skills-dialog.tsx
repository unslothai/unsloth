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
import {
  listSkills,
  setSkillEnabled,
  useSkillsCatalog,
} from "../api/skills-api";
import { toast } from "@/lib/toast";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { RefreshCwIcon } from "lucide-react";
import { type ReactElement, useState } from "react";

export function ChatSkillsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  const { skills, loading, error } = useSkillsCatalog();
  const [changing, setChanging] = useState<string | null>(null);

  const toggle = async (name: string, enabled: boolean) => {
    setChanging(name);
    try {
      await setSkillEnabled(name, enabled);
    } catch (cause) {
      toast.error("Could not update Agent Skill", {
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
              icon={BookOpen01Icon}
              strokeWidth={1.75}
              className="size-5 text-primary"
            />
            <DialogTitle>Agent Skills</DialogTitle>
          </div>
          <DialogDescription>
            Skills are discovered from your standard agent folders. Enable them
            here, then type @ in chat to mention one.
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-muted-foreground">
            ~/.agents/skills takes precedence over ~/.claude/skills.
          </p>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={loading}
            onClick={() => void listSkills(true).catch(() => undefined)}
          >
            {loading ? <Spinner /> : <RefreshCwIcon />}
            Refresh
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
              No Agent Skills found. Add a SKILL.md folder under
              ~/.agents/skills or ~/.claude/skills, then refresh.
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
                      <Badge variant="outline">
                        {skill.source === "agents" ? "Agents" : "Claude"}
                      </Badge>
                      {skill.shadowed ? (
                        <Badge variant="secondary">Shadowed</Badge>
                      ) : null}
                      {!skill.valid ? (
                        <Badge variant="destructive">Invalid</Badge>
                      ) : null}
                    </div>
                    {skill.description ? (
                      <p className="mt-1 text-sm text-muted-foreground">
                        {skill.description}
                      </p>
                    ) : null}
                    {skill.compatibility ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Compatibility: {skill.compatibility}
                      </p>
                    ) : null}
                    {skill.shadowed_by ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Another{" "}
                        {skill.shadowed_by === "agents" ? "Agents" : "Claude"}{" "}
                        skill with this name takes precedence.
                      </p>
                    ) : null}
                    {skill.error ? (
                      <p className="mt-2 text-xs text-destructive">
                        {skill.error}
                      </p>
                    ) : null}
                  </div>
                  <Switch
                    aria-label={`${skill.enabled ? "Disable" : "Enable"} ${skill.name}`}
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
