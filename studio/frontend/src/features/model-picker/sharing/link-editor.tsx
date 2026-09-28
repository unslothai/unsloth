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
import { hasAuthToken, mustChangePassword } from "@/features/auth";
import {
  type ChatSearch,
  isExternalModelId,
  useChatRuntimeStore,
} from "@/features/chat";
import { useHfTokenStore, useInventoryVersion } from "@/features/hub";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { type RunConfigRequest, runConfigInbox } from "./inbox";
import {
  type RunConfigNavigation,
  navigateRunConfig,
  openRunConfigTarget,
} from "./link-lifecycle";
import { isKnownNonGgufModel, isRunConfigModelInput } from "./target";

export function SharedRunConfigLinkEditor({
  pending,
  chatSearch,
}: { pending: RunConfigRequest; chatSearch: ChatSearch | null }) {
  const navigate = useNavigate();
  const location = useRouterState({ select: (state) => state.location });
  const routeReady = useRouterState({
    select: (state) =>
      !state.isLoading &&
      state.status === "idle" &&
      state.resolvedLocation?.href === state.location.href,
  });
  const [modelInput, setModelInput] = useState("");
  const navigation = useRef<RunConfigNavigation | null>(null);
  const hfToken = useHfTokenStore((state) => state.token) || undefined;
  const inventoryVersion = useInventoryVersion();
  const currentModel = useChatRuntimeStore((state) =>
    isExternalModelId(state.params.checkpoint) ||
    isKnownNonGgufModel(state.params.checkpoint, state)
      ? ""
      : state.params.checkpoint,
  );
  const settingsHydrated = useChatRuntimeStore(
    (state) => state.settingsHydrated,
  );
  const canOpen =
    hasAuthToken() &&
    !mustChangePassword() &&
    location.pathname !== "/login" &&
    location.pathname !== "/change-password";

  useEffect(() => {
    navigateRunConfig({
      pending,
      canOpen,
      settingsHydrated,
      currentModel,
      location,
      navigation,
      navigate,
    });
  }, [canOpen, currentModel, location, navigate, pending, settingsHydrated]);

  useEffect(
    () =>
      openRunConfigTarget({
        pending,
        chatSearch,
        canOpen,
        settingsHydrated,
        currentModel,
        location,
        routeReady,
        hfToken,
        inventoryVersion,
      }),
    [
      canOpen,
      chatSearch,
      currentModel,
      location,
      pending,
      settingsHydrated,
      routeReady,
      hfToken,
      inventoryVersion,
    ],
  );

  const chooseModel =
    canOpen &&
    settingsHydrated &&
    !pending.draftKey &&
    !pending.selectedModel &&
    !pending.value.model &&
    !currentModel;
  return (
    <Dialog
      open={chooseModel}
      onOpenChange={(open) => {
        if (!open) {
          runConfigInbox.clear(pending.id);
        }
      }}
    >
      <DialogContent className="content-start gap-5">
        <DialogHeader>
          <DialogTitle>Choose a GGUF model</DialogTitle>
          <DialogDescription>
            This link contains settings without a model. Enter a Hugging Face
            GGUF model ID, a local GGUF path on the machine running Unsloth, or
            an Ollama reference to review the settings with that model.
          </DialogDescription>
        </DialogHeader>
        <form
          className="space-y-5"
          onSubmit={(event) => {
            event.preventDefault();
            if (!isRunConfigModelInput(modelInput.trim())) {
              return;
            }
            runConfigInbox.submit({
              ...pending,
              selectedModel: modelInput.trim(),
            });
            setModelInput("");
          }}
        >
          <div className="space-y-2">
            <label
              htmlFor="shared-run-model"
              className="block text-sm font-medium"
            >
              GGUF model ID or local path
            </label>
            <Input
              id="shared-run-model"
              autoComplete="off"
              placeholder="owner/model-GGUF or local GGUF path"
              value={modelInput}
              onChange={(event) => setModelInput(event.target.value)}
            />
          </div>
          <div className="flex justify-end">
            <Button
              type="submit"
              disabled={!isRunConfigModelInput(modelInput.trim())}
            >
              Open run settings
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
