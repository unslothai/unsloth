import { clearNewChatDraft, useChatRuntimeStore } from "@/features/chat";
import { toast } from "@/lib/toast";
import { modelConfigDraftKey } from "../model-picker/model-config/model-config-draft";
import {
  clearModelConfigHandoff,
  requestModelConfigHandoff,
} from "../model-picker/model-config/model-config-handoff";
import { resolveCachedRunConfigTarget } from "./cached-target";
import { type RunConfigRequest, runConfigInbox } from "./inbox";
import { resolveRunConfigTarget } from "./target";

export type RunConfigNavigation = { id: string; from: string };
type LinkContext = {
  pending: RunConfigRequest | null;
  canOpen: boolean;
  settingsHydrated: boolean;
  currentModel: string;
  location: { href: string; pathname: string; searchStr: string };
};

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
    search: { new: string };
    replace: boolean;
  }) => Promise<unknown>;
}): void {
  if (
    !pending ||
    pending.draftKey ||
    !canOpen ||
    !settingsHydrated ||
    runConfigInbox.getSnapshot() !== pending ||
    !(pending.value.model ?? currentModel)
  ) {
    return;
  }
  const atDestination =
    location.pathname === "/chat" &&
    new URLSearchParams(location.searchStr).get("new") === pending.id;
  if (navigation.current?.id === pending.id) {
    if (atDestination) {
      navigation.current.from = location.href;
    } else if (location.href !== navigation.current.from) {
      runConfigInbox.clear(pending.id);
    }
    return;
  }
  navigation.current = { id: pending.id, from: location.href };
  clearNewChatDraft();
  const runtime = useChatRuntimeStore.getState();
  runtime.setActiveThreadId(null);
  runtime.setActiveProjectId(null);
  runtime.setIncognito(false);
  navigate({
    to: "/chat",
    search: { new: pending.id },
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
  canOpen,
  settingsHydrated,
  currentModel,
  location,
  routeReady,
  hfToken,
  inventoryVersion,
}: LinkContext & {
  routeReady: boolean;
  hfToken?: string;
  inventoryVersion: number;
}): (() => void) | undefined {
  if (
    !pending ||
    pending.draftKey ||
    !canOpen ||
    !settingsHydrated ||
    !routeReady ||
    !(pending.value.model ?? currentModel) ||
    location.pathname !== "/chat" ||
    new URLSearchParams(location.searchStr).get("new") !== pending.id ||
    runConfigInbox.getSnapshot() !== pending
  ) {
    return;
  }
  const target = resolveRunConfigTarget(
    pending.value,
    useChatRuntimeStore.getState(),
  );
  if (!target) {
    return;
  }
  const controller = new AbortController();
  resolveCachedRunConfigTarget(target, {
    hfToken,
    inventoryVersion,
    signal: controller.signal,
  })
    .then((resolved) => {
      if (
        controller.signal.aborted ||
        runConfigInbox.getSnapshot() !== pending
      ) {
        return;
      }
      runConfigInbox.bind(
        pending.id,
        modelConfigDraftKey(resolved.id, resolved.meta.ggufVariant),
      );
      requestModelConfigHandoff({ requestId: pending.id, ...resolved });
    })
    .catch(() => {
      if (
        controller.signal.aborted ||
        runConfigInbox.getSnapshot() !== pending
      ) {
        return;
      }
      runConfigInbox.clear(pending.id);
      toast.error(
        "Could not check local model availability. Reopen the link to try again.",
      );
    });
  return () => controller.abort();
}
