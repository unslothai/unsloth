// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ChatSearch,
  clearNewChatDraft,
  useChatRuntimeStore,
} from "@/features/chat";
import { toast } from "@/lib/toast";
import { modelConfigDraftKey } from "../model-config/model-config-draft";
import {
  clearModelConfigHandoff,
  requestModelConfigHandoff,
} from "../model-config/model-config-handoff";
import {
  RunConfigResolutionError,
  resolveCachedRunConfigTarget,
} from "./cached-target";
import { type RunConfigRequest, runConfigInbox } from "./inbox";
import { cancelEditedRunConfigImport } from "./receive-link";
import { resolveRunConfigTarget } from "./target";

export type RunConfigNavigation = {
  id: string;
  from: string;
  started: boolean;
};
type LinkContext = {
  pending: RunConfigRequest | null;
  canOpen: boolean;
  settingsHydrated: boolean;
  currentModel: string;
  location: { href: string; pathname: string; searchStr: string };
};

function atRunConfigDestination(
  pending: RunConfigRequest,
  location: LinkContext["location"],
): boolean {
  const search = new URLSearchParams(location.searchStr);
  return (
    location.pathname === "/chat" &&
    !search.has("thread") &&
    !search.has("compare") &&
    !search.has("project") &&
    search.get("new") ===
      (pending.newChatId === undefined ? pending.id : pending.newChatId)
  );
}

export function navigateRunConfig({
  pending,
  canOpen,
  settingsHydrated,
  currentModel,
  location,
  navigation,
  navigate,
}: LinkContext & {
  navigation: { current: RunConfigNavigation | null };
  navigate: (options: {
    to: "/chat";
    search: { new?: string };
    replace: boolean;
  }) => Promise<unknown>;
}): void {
  if (
    !pending ||
    pending.draftKey ||
    !canOpen ||
    !settingsHydrated ||
    runConfigInbox.getSnapshot() !== pending ||
    !(pending.selectedModel ?? pending.value.model ?? currentModel)
  ) {
    return;
  }
  const atDestination = atRunConfigDestination(pending, location);
  if (navigation.current?.id === pending.id) {
    if (atDestination) {
      navigation.current.from = location.href;
    } else if (location.href !== navigation.current.from) {
      runConfigInbox.clear(pending.id);
    }
    if (navigation.current.started || !pending.target) {
      return;
    }
  }
  navigation.current = {
    id: pending.id,
    from: location.href,
    started: Boolean(pending.target),
  };
  if (
    !pending.target ||
    atDestination ||
    runConfigInbox.getSnapshot() !== pending
  ) {
    return;
  }
  navigate({
    to: "/chat",
    search: {
      new:
        pending.newChatId === undefined
          ? pending.id
          : (pending.newChatId ?? undefined),
    },
    replace: pending.replaceHistory === true,
  }).catch(() => {
    if (runConfigInbox.getSnapshot()?.id !== pending.id) {
      return;
    }
    clearModelConfigHandoff(pending.id);
    runConfigInbox.clear(pending.id);
    toast.error("Could not open the model’s run settings.");
  });
}

export function openRunConfigTarget({
  pending,
  chatSearch,
  canOpen,
  settingsHydrated,
  currentModel,
  location,
  routeReady,
  hfToken,
  inventoryVersion,
}: LinkContext & {
  chatSearch: ChatSearch | null;
  routeReady: boolean;
  hfToken?: string;
  inventoryVersion: number;
}): (() => void) | undefined {
  if (
    !pending ||
    pending.draftKey ||
    !canOpen ||
    !settingsHydrated ||
    !(pending.selectedModel ?? pending.value.model ?? currentModel) ||
    runConfigInbox.getSnapshot() !== pending
  ) {
    return;
  }
  if (pending.target) {
    if (!routeReady || !atRunConfigDestination(pending, location)) {
      return;
    }
    if (pending.newChatId === undefined) {
      clearNewChatDraft();
      const runtime = useChatRuntimeStore.getState();
      runtime.setActiveThreadId(null);
      runtime.setActiveProjectId(null);
      runtime.setIncognito(false);
    }
    runConfigInbox.bind(
      pending.id,
      modelConfigDraftKey(pending.target.id, pending.target.meta.ggufVariant),
    );
    requestModelConfigHandoff({
      requestId: pending.id,
      ...(pending.newChatId !== undefined && { newChatId: pending.newChatId }),
      ...pending.target,
      displayName: pending.target.id,
    });
    return;
  }
  const runtime = useChatRuntimeStore.getState();
  const preserveDraft =
    chatSearch !== null &&
    !chatSearch.thread &&
    !chatSearch.compare &&
    !chatSearch.project;
  const target = resolveRunConfigTarget(
    pending.value,
    runtime,
    pending.selectedModel,
  );
  if (!target) {
    return;
  }
  const controller = new AbortController();
  const loadingToast = toast.loading("Resolving shared model…");
  Promise.all([
    resolveCachedRunConfigTarget(target, {
      hfToken,
      inventoryVersion,
      signal: controller.signal,
      checkLocalPath: !pending.value.model || Boolean(pending.selectedModel),
    }),
    import("./runtime"),
  ])
    .then(([resolved]) => {
      if (
        controller.signal.aborted ||
        runConfigInbox.getSnapshot() !== pending
      ) {
        return;
      }
      const key = modelConfigDraftKey(resolved.id, resolved.meta.ggufVariant);
      if (runConfigInbox.wasEdited(key)) {
        cancelEditedRunConfigImport(key);
        return;
      }
      runConfigInbox.submit({
        ...pending,
        target: resolved,
        ...(preserveDraft && { newChatId: chatSearch.new ?? null }),
      });
    })
    .catch((error: unknown) => {
      if (
        controller.signal.aborted ||
        runConfigInbox.getSnapshot() !== pending
      ) {
        return;
      }
      runConfigInbox.clear(pending.id);
      toast.error(
        error instanceof RunConfigResolutionError
          ? error.message
          : "Could not resolve the shared model. Reopen the link to try again.",
      );
    })
    .finally(() => toast.dismiss(loadingToast));
  return () => {
    controller.abort();
    toast.dismiss(loadingToast);
  };
}
