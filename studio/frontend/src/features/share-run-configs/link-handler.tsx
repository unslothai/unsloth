// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useChatRuntimeStore } from "@/features/chat";
import { useHfTokenStore, useInventoryVersion } from "@/features/hub";
import { toast } from "@/lib/toast";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useEffect, useState, useSyncExternalStore } from "react";
import {
  AUTH_SESSION_STORED_EVENT,
  hasAuthToken,
  mustChangePassword,
} from "../auth/session";
import { isExternalModelId } from "../chat/external-providers";
import { modelConfigDraftKey } from "../model-picker/model-config/model-config-draft";
import {
  clearModelConfigHandoff,
  requestModelConfigHandoff,
} from "../model-picker/model-config/model-config-handoff";
import { resolveCachedRunConfigTarget } from "./cached-target";
import { runConfigInbox } from "./inbox";
import { isShareableModelId } from "./links";
import {
  receiveRunConfigUrl,
  receiveStartupRunConfigUrl,
} from "./receive-link";
import { resolveRunConfigTarget } from "./target";

export function SharedRunConfigLinkHandler() {
  const navigate = useNavigate();
  const location = useRouterState({ select: (state) => state.location });
  const routeReady = useRouterState({
    select: (state) =>
      !state.isLoading &&
      state.status === "idle" &&
      state.resolvedLocation?.href === state.location.href,
  });
  const pending = useSyncExternalStore(
    runConfigInbox.subscribe,
    runConfigInbox.getSnapshot,
  );
  const [, setAuthRevision] = useState(0);
  const [modelInput, setModelInput] = useState("");
  const hfToken = useHfTokenStore((state) => state.token) || undefined;
  const inventoryVersion = useInventoryVersion();
  const checkpoint = useChatRuntimeStore((state) => state.params.checkpoint);
  const settingsHydrated = useChatRuntimeStore(
    (state) => state.settingsHydrated,
  );
  const currentModel = isExternalModelId(checkpoint) ? "" : checkpoint;
  const canOpen =
    hasAuthToken() &&
    !mustChangePassword() &&
    location.pathname !== "/login" &&
    location.pathname !== "/change-password";

  useEffect(() => {
    receiveStartupRunConfigUrl(window.location.href);
    let previousUrl = window.location.href;
    const onLocation = () => {
      const currentUrl = window.location.href;
      if (currentUrl === previousUrl) {
        return;
      }
      previousUrl = currentUrl;
      receiveRunConfigUrl(currentUrl);
    };
    const onAuth = () => setAuthRevision((revision) => revision + 1);
    window.addEventListener("hashchange", onLocation);
    window.addEventListener("popstate", onLocation);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, onAuth);
    return () => {
      window.removeEventListener("hashchange", onLocation);
      window.removeEventListener("popstate", onLocation);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, onAuth);
    };
  }, []);

  useEffect(() => {
    if (!pending || pending.draftKey || !canOpen || !settingsHydrated) {
      return;
    }
    if (runConfigInbox.getSnapshot() !== pending) {
      return;
    }
    if (!(pending.value.model ?? currentModel)) {
      return;
    }
    const target = resolveRunConfigTarget(
      pending.value,
      useChatRuntimeStore.getState(),
    );
    if (!target) {
      return;
    }
    if (
      location.pathname !== "/chat" ||
      new URLSearchParams(location.searchStr).get("new") !== pending.id
    ) {
      navigate({
        to: "/chat",
        search: { new: pending.id },
        replace: true,
      }).catch(() => {
        clearModelConfigHandoff(pending.id);
        runConfigInbox.clear(pending.id);
        toast.error("Could not open the model’s run settings.");
      });
      return;
    }
    if (!routeReady) {
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
  }, [
    canOpen,
    currentModel,
    location.pathname,
    location.searchStr,
    navigate,
    pending,
    settingsHydrated,
    routeReady,
    hfToken,
    inventoryVersion,
  ]);

  const chooseModel =
    canOpen &&
    settingsHydrated &&
    pending !== null &&
    !pending.draftKey &&
    !pending.value.model &&
    !currentModel;
  return (
    <Dialog
      open={chooseModel}
      onOpenChange={(open) => {
        if (!open && pending) {
          runConfigInbox.clear(pending.id);
        }
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Choose a model</DialogTitle>
          <DialogDescription>
            This link contains settings without a model. Enter a model ID to
            open its run settings.
          </DialogDescription>
        </DialogHeader>
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (!(pending && isShareableModelId(modelInput.trim()))) {
              return;
            }
            runConfigInbox.submit({
              ...pending,
              value: { ...pending.value, model: modelInput.trim() },
            });
            setModelInput("");
          }}
        >
          <label htmlFor="shared-run-model" className="text-sm font-medium">
            Hugging Face model ID
          </label>
          <Input
            id="shared-run-model"
            autoComplete="off"
            placeholder="owner/model"
            value={modelInput}
            onChange={(event) => setModelInput(event.target.value)}
          />
          <Button
            type="submit"
            disabled={!isShareableModelId(modelInput.trim())}
          >
            Open run settings
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
