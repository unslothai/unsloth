// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import { useT } from "@/i18n";
import { AiSpeechIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import {
  BotIcon,
  Image01Icon,
  McpServerIcon,
  TaskDone01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type SettingsScrollTarget,
  useSettingsDialogStore,
} from "../stores/settings-dialog-store";

export function ApiJumpLinks({ className }: { className?: string }) {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const links: {
    key: string;
    icon: typeof Image01Icon;
    label: string;
    target: SettingsScrollTarget;
    ownerOnly?: boolean;
  }[] = [
    {
      key: "media",
      icon: Image01Icon,
      label: t("settings.apiKeys.mediaApi.title"),
      target: "api-keys-media-api",
    },
    {
      key: "audio",
      icon: AiSpeechIcon,
      label: t("settings.apiKeys.audioApi.title"),
      target: "api-keys-audio-api",
    },
    {
      key: "decision",
      icon: TaskDone01Icon,
      label: t("settings.apiKeys.decisionApi.title"),
      target: "api-keys-decision-api",
      ownerOnly: true,
    },
    {
      key: "agents",
      icon: BotIcon,
      label: t("settings.tabs.agents"),
      target: "api-keys-agents",
    },
    {
      key: "mcp",
      icon: McpServerIcon,
      label: "MCP",
      target: "api-keys-mcp",
      ownerOnly: true,
    },
  ];
  return (
    <nav
      aria-label={t("settings.apiKeys.title")}
      className={cn("flex min-w-0 flex-wrap items-center gap-2", className)}
    >
      {links
        .filter((link) => isOwner || !link.ownerOnly)
        .map((link) => (
          <button
            key={link.key}
            type="button"
            onClick={() =>
              useSettingsDialogStore
                .getState()
                .openDialog("api-keys", { scrollTarget: link.target })
            }
            className="btn-fill group/jump inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-ui-13 font-medium text-foreground/80 outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
          >
            <HugeiconsIcon
              icon={link.icon}
              strokeWidth={1.75}
              className="size-3.5 shrink-0 text-muted-foreground transition-colors group-hover/jump:text-foreground"
            />
            {link.label}
          </button>
        ))}
    </nav>
  );
}
