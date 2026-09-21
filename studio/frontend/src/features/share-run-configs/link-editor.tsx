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
import { useChatRuntimeStore } from "@/features/chat";
import { useHfTokenStore, useInventoryVersion } from "@/features/hub";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { isExternalModelId } from "../chat/external-providers";
import { type RunConfigRequest, runConfigInbox } from "./inbox";
import {
  type RunConfigNavigation,
  navigateRunConfig,
  openRunConfigTarget,
} from "./link-lifecycle";
import { isShareableModelId } from "./links";

export function SharedRunConfigLinkEditor({
  pending,
}: { pending: RunConfigRequest }) {
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
            if (!isShareableModelId(modelInput.trim())) {
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
