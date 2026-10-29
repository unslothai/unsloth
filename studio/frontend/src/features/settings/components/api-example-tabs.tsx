// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import { type TranslationKey, useT } from "@/i18n";
// Not through the barrel: it pulls in chat, which every barrel importer would then load.
import { RAISED_SURFACE } from "@/features/library/surface";
import { AiSpeechIcon, MessageCircleIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import {
  BotIcon,
  Image01Icon,
  TaskDone01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  type ReactElement,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import { scrollIntoPane } from "../lib/scroll-into-pane";
import {
  type SettingsScrollTarget,
  useSettingsDialogStore,
} from "../stores/settings-dialog-store";
import { AudioApiExamples } from "./audio-api-examples";
import { DecisionApiSection } from "./decision-api-section";
import { MediaApiExamples } from "./media-api-examples";
import { UsageExamples } from "./usage-examples";

export type ApiExampleKind = "chat" | "media" | "audio" | "decision" | "agents";

export function useApiExampleKinds(): {
  id: ApiExampleKind;
  label: string;
  icon: IconSvgElement;
  /** Settings search target: the card is unmounted off its tab. */
  searchLabel?: TranslationKey;
}[] {
  const isOwner = useIsAccountOwner();
  return [
    { id: "chat", label: "Chat", icon: MessageCircleIcon },
    {
      id: "media",
      label: "Images / Video",
      icon: Image01Icon,
      searchLabel: "settings.apiKeys.mediaApi.title",
    },
    {
      id: "audio",
      label: "Audio",
      icon: AiSpeechIcon,
      searchLabel: "settings.apiKeys.audioApi.title",
    },
    ...(isOwner
      ? [
          {
            id: "decision" as const,
            label: "Decision",
            icon: TaskDone01Icon,
            searchLabel: "settings.apiKeys.decisionApi.title" as const,
          },
          { id: "agents" as const, label: "Agents", icon: BotIcon },
        ]
      : []),
  ];
}

export function ApiExampleTabNav({
  kind,
  onChange,
  className,
}: {
  kind: ApiExampleKind;
  onChange: (kind: ApiExampleKind) => void;
  className?: string;
}): ReactElement {
  const t = useT();
  const kinds = useApiExampleKinds();
  return (
    <nav aria-label="Example" className={cn("flex flex-wrap gap-1", className)}>
      {kinds.map((entry) => (
        <button
          key={entry.id}
          type="button"
          aria-current={entry.id === kind ? "page" : undefined}
          data-settings-label={
            entry.searchLabel ? t(entry.searchLabel) : undefined
          }
          onClick={() =>
            entry.id === "agents"
              ? useSettingsDialogStore.getState().openDialog("agents")
              : onChange(entry.id)
          }
          className={cn(
            "flex h-9 shrink-0 items-center gap-2 rounded-full px-3.5 font-heading text-ui-15 text-foreground/80 transition-colors hover:text-foreground",
            entry.id === kind &&
              cn(RAISED_SURFACE, "font-medium text-foreground dark:bg-accent"),
          )}
        >
          <HugeiconsIcon
            icon={entry.icon}
            strokeWidth={1.75}
            className="size-4 shrink-0"
          />
          {entry.label}
        </button>
      ))}
    </nav>
  );
}

const TAB_FOR_TARGET: Partial<Record<SettingsScrollTarget, ApiExampleKind>> = {
  "api-keys-media-api": "media",
  "api-keys-audio-api": "audio",
  "api-keys-decision-api": "decision",
};

function pinUntilUserScrolls(
  el: HTMLElement,
  target: SettingsScrollTarget,
): () => void {
  const observer = new ResizeObserver(() => scrollIntoPane(el));
  observer.observe(el.parentElement ?? el);
  const release = () => {
    observer.disconnect();
    useSettingsDialogStore.getState().consumeScrollTarget(target);
  };
  const timer = window.setTimeout(release, 1500);
  const events = ["wheel", "touchmove", "keydown"] as const;
  for (const name of events) {
    window.addEventListener(name, release, { passive: true });
  }
  return () => {
    observer.disconnect();
    window.clearTimeout(timer);
    for (const name of events) {
      window.removeEventListener(name, release);
    }
  };
}

export function ApiExamplesPanel({
  apiKey,
  useTunnel,
  onUseTunnelChange,
  keylessScope,
  keylessTools,
  keylessExposure,
}: {
  apiKey: string | null;
  useTunnel: boolean;
  onUseTunnelChange: (next: boolean) => void;
  keylessScope: KeylessApiAccessScope;
  keylessTools: boolean;
  keylessExposure: KeylessApiAccessExposure | null;
}): ReactElement {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const [kind, setKind] = useState<ApiExampleKind>("chat");
  const panelRef = useRef<HTMLDivElement | null>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  // Picking the tab mounts the card that takes the request.
  useEffect(() => {
    const tab = scrollTarget ? TAB_FOR_TARGET[scrollTarget] : undefined;
    const el = panelRef.current;
    if (!(scrollTarget && tab && el)) return;
    if (tab === "decision" && !isOwner) return;
    setKind(tab);
    return pinUntilUserScrolls(el, scrollTarget);
  }, [scrollTarget, isOwner]);
  const keyless = { keylessScope, keylessExposure };
  return (
    <div ref={panelRef} className="flex min-w-0 scroll-mt-6 flex-col gap-4">
      <h2 className="settings-heading text-base font-semibold font-heading">
        {t("settings.apiKeys.quickstartTitle")}
      </h2>
      <ApiExampleTabNav kind={kind} onChange={setKind} className="-ml-1" />
      {kind === "chat" ? (
        <UsageExamples
          part="chat"
          apiKey={apiKey}
          useTunnel={useTunnel}
          onUseTunnelChange={onUseTunnelChange}
          keylessScope={keylessScope}
          keylessTools={keylessTools}
          keylessExposure={keylessExposure}
        />
      ) : kind === "media" ? (
        <MediaApiExamples
          apiKey={apiKey}
          useTunnel={useTunnel}
          showHeading={false}
          {...keyless}
        />
      ) : kind === "audio" ? (
        <AudioApiExamples
          apiKey={apiKey}
          useTunnel={useTunnel}
          showHeading={false}
          {...keyless}
        />
      ) : (
        <DecisionApiSection showHeading={false} />
      )}
    </div>
  );
}
