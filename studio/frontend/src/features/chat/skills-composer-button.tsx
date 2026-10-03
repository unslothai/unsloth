// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Scroll01Icon, Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDownIcon } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import { MenuTickIcon } from "@/lib/tick-icon";

import {
  type SkillRecord,
  refreshSkillsCatalog,
  setSkillEnabled,
  useSkillsCatalog,
} from "./api/skills-api";
import { ChatSkillsDialog } from "./components/chat-skills-dialog";

/** Composer pill for the enabled skills, shown once any is on. Toggles them in place. */
export function SkillsComposerButton({
  side = "bottom",
}: {
  side?: "top" | "bottom";
} = {}) {
  const t = useT();
  const { skills } = useSkillsCatalog();
  const [menuOpen, setMenuOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [pending, setPending] = useState<ReadonlySet<string>>(() => new Set());

  // Runnable skills only. Yours first, then bundled ones (skill-creator) by Manage skills.
  const usable = skills.filter((skill) => skill.valid && !skill.shadowed);
  const own = usable.filter((skill) => skill.source !== "bundled");
  const bundled = usable.filter((skill) => skill.source === "bundled");
  const enabledCount = usable.filter((skill) => skill.enabled).length;

  async function toggle(name: string, enabled: boolean) {
    if (pending.has(name)) return;
    setPending((current) => new Set(current).add(name));
    try {
      await setSkillEnabled(name, enabled);
    } catch (cause) {
      toast.error(t("skills.updateError"), {
        description: cause instanceof Error ? cause.message : String(cause),
      });
    } finally {
      setPending((current) => {
        const next = new Set(current);
        next.delete(name);
        return next;
      });
    }
  }

  const renderRow = (skill: SkillRecord) => (
    <DropdownMenuItem
      key={`${skill.source}:${skill.name}`}
      disabled={pending.has(skill.name)}
      // Stays open to toggle several.
      onSelect={(event) => {
        event.preventDefault();
        void toggle(skill.name, !skill.enabled);
      }}
      aria-label={t(skill.enabled ? "skills.disable" : "skills.enable", { name: skill.name })}
      className={skill.enabled ? "text-primary font-medium" : undefined}
    >
      <span className="truncate">{skill.name}</span>
      {skill.enabled ? (
        <HugeiconsIcon icon={MenuTickIcon} strokeWidth={2} className="ml-auto" />
      ) : null}
    </DropdownMenuItem>
  );

  return (
    <>
      {/* Stays while open, so turning off the last skill does not close it. */}
      {enabledCount > 0 || menuOpen ? (
        <DropdownMenu
          open={menuOpen}
          onOpenChange={(open) => {
            setMenuOpen(open);
            if (open) refreshSkillsCatalog();
          }}
        >
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              className="composer-pill-btn"
              data-pill-label={t("skills.title")}
              data-active={enabledCount > 0 ? "true" : "false"}
              aria-label={t("skills.title")}
            >
              <span className="composer-pill-glyph">
                <HugeiconsIcon
                  icon={Scroll01Icon}
                  className="size-[calc(15px*var(--ui-space-scale,1))]"
                  strokeWidth={2}
                />
              </span>
              <span>{t("skills.title")}</span>
              <ChevronDownIcon
                strokeWidth={1.5}
                className="composer-pill-caret size-[calc(15px*var(--ui-space-scale,1))]"
              />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side={side}
            align="start"
            sideOffset={0}
            avoidCollisions={true}
            className="unsloth-plus-menu mcp-menu w-[calc(232px*var(--ui-space-scale,1))]"
          >
            {own.length > 0 ? (
              <div className="max-h-[calc(280px*var(--ui-space-scale,1))] overflow-y-auto">
                {own.map(renderRow)}
              </div>
            ) : null}
            {own.length > 0 ? <DropdownMenuSeparator /> : null}
            {bundled.map(renderRow)}
            <DropdownMenuItem
              onSelect={() => {
                setMenuOpen(false);
                setDialogOpen(true);
              }}
            >
              <HugeiconsIcon icon={Settings02Icon} strokeWidth={2} />
              {t("skills.manage")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : null}
      <ChatSkillsDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </>
  );
}
