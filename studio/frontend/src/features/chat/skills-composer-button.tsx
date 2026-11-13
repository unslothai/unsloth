// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Scroll01Icon, Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { toast } from "sonner";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import { MenuTickIcon } from "@/lib/tick-icon";

import {
  refreshSkillsCatalog,
  setSkillEnabled,
  useSkillsCatalog,
} from "./api/skills-api";
import { ChatSkillsDialog } from "./components/chat-skills-dialog";
import { skillsSwitchedOn, useSkillsSwitchStore } from "./stores/skills-switch-store";

export function useSkillsSwitch(): [boolean, (on: boolean) => void] {
  const { skills } = useSkillsCatalog();
  const on = useSkillsSwitchStore((s) => s.on);
  const setOn = useSkillsSwitchStore((s) => s.setOn);
  const enabledCount = skills.filter(
    (skill) => skill.valid && !skill.shadowed && skill.enabled,
  ).length;
  return [skillsSwitchedOn(on, enabledCount), setOn];
}

/** Composer pill for skills, shown while the Skills switch is on. Toggles them in place. */
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
  const [skillsOn, setSkillsOn] = useSkillsSwitch();

  const usable = skills.filter((skill) => skill.valid && !skill.shadowed);
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

  return (
    <>
      {/* Stays with every skill unticked: only its X or the + menu turns Skills off. */}
      {skillsOn || menuOpen ? (
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
              {/* Outside compact mode, the hover X turns Skills off without opening the menu. */}
              <span
                role="button"
                aria-label={t("skills.turnOff")}
                tabIndex={-1}
                onPointerDown={(e) => {
                  if (e.currentTarget.closest('[data-pill-compact="true"]')) {
                    return;
                  }
                  e.stopPropagation();
                }}
                onClick={(e) => {
                  if (e.currentTarget.closest('[data-pill-compact="true"]')) {
                    return;
                  }
                  e.stopPropagation();
                  setSkillsOn(false);
                }}
                className="composer-pill-glyph cursor-pointer"
              >
                <HugeiconsIcon
                  icon={Scroll01Icon}
                  className="size-[calc(15px*var(--ui-space-scale,1))]"
                  strokeWidth={2}
                />
                <HugeiconsIcon icon={Cancel01Icon} className="composer-pill-x" />
              </span>
              <span>{t("skills.title")}</span>
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
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
            <DropdownMenuLabel>{t("skills.title")}</DropdownMenuLabel>
            <div className="max-h-[calc(280px*var(--ui-space-scale,1))] overflow-y-auto">
              {usable.map((skill) => (
                <DropdownMenuItem
                  key={`${skill.source}:${skill.name}`}
                  disabled={pending.has(skill.name)}
                  onSelect={(event) => {
                    event.preventDefault();
                    void toggle(skill.name, !skill.enabled);
                  }}
                  aria-label={t(skill.enabled ? "skills.disable" : "skills.enable", {
                    name: skill.name,
                  })}
                  className={skill.enabled ? "text-primary font-medium" : undefined}
                >
                  <span className="truncate">{skill.name}</span>
                  {skill.enabled ? (
                    <HugeiconsIcon icon={MenuTickIcon} strokeWidth={2} className="ml-auto" />
                  ) : null}
                </DropdownMenuItem>
              ))}
            </div>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={() => {
                setMenuOpen(false);
                setDialogOpen(true);
              }}
            >
              {t("skills.manage")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : null}
      <ChatSkillsDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </>
  );
}
