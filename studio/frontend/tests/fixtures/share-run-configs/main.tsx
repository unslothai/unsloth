// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* eslint-disable react-refresh/only-export-components -- standalone browser fixture */
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { useChatRuntimeStore } from "@/features/chat";
import { ChatRuntimeProvider } from "../../../src/features/chat/runtime-provider";
import { ComposerPrimitive, useAssistantRuntime } from "@assistant-ui/react";
import {
  DEFAULT_PER_MODEL_CONFIG,
  ModelSelector,
  SharedRunConfigLinkHandler,
  clearModelConfigHandoff,
  modelConfigHandoffForDestination,
  receiveSharedRunConfigUrls,
  useModelConfigHandoffStore,
  type ModelSelectorChangeMeta,
} from "@/features/model-picker";
import { ShareRunConfigDialog } from "../../../src/features/model-picker/sharing/share-dialog";
import { runConfigInbox } from "../../../src/features/model-picker/sharing/inbox";
import {
  createRunConfigLink,
  type SharedRunConfig,
} from "../../../src/features/model-picker/sharing/links";
import {
  Outlet,
  RouterProvider,
  createRootRoute,
  createRoute,
  createRouter,
  useRouterState,
} from "@tanstack/react-router";
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const draftMode = new URLSearchParams(window.location.search).has("draft");
if (window.location.pathname === "/login") {
  localStorage.removeItem("unsloth_auth_token");
} else {
  localStorage.setItem("unsloth_auth_token", "sharing-browser-fixture");
}
localStorage.setItem("unsloth_model_selector_section", "downloaded");
useChatRuntimeStore.setState((state) => ({
  settingsHydrated: true,
  params: {
    ...state.params,
    checkpoint: new URLSearchParams(window.location.search).has("choose-model")
      ? ""
      : "owner/Native",
  },
  activeLoadId: "/cache/native",
  loadedIsGguf: false,
  activeThreadId: draftMode ? null : "existing-thread",
  activeProjectId: draftMode ? null : "existing-project",
  incognito: true,
}));

let loaded: { id: string; meta: ModelSelectorChangeMeta } | null = null;
let composer:
  | NonNullable<ReturnType<typeof useAssistantRuntime>>["thread"]["composer"]
  | null = null;
const api = {
  stageDraft: async () => {
    if (!composer) throw new Error("Composer not mounted");
    composer.setText("Unsent draft");
    await composer.addAttachment(
      new File(["Unsent attachment"], "draft.txt", {
        type: "text/plain",
      }),
    );
  },
  draft: () => {
    const state = composer?.getState();
    return state
      ? {
          text: state.text,
          attachments: state.attachments.map(({ id, name, status }) => ({
            id,
            name,
            status,
          })),
        }
      : null;
  },
  link: (value: SharedRunConfig) =>
    createRunConfigLink(value, window.location.href),
  receive: (value: SharedRunConfig) =>
    receiveSharedRunConfigUrls([createRunConfigLink(value)]),
  snapshot: () => {
    const state = useChatRuntimeStore.getState();
    return {
      thread: state.activeThreadId,
      project: state.activeProjectId,
      incognito: state.incognito,
      pending: runConfigInbox.getSnapshot(),
      loaded,
    };
  },
};
declare global {
  interface Window {
    sharingTest: typeof api;
  }
}
window.sharingTest = api;

function DraftComposer() {
  const runtime = useAssistantRuntime();
  useEffect(() => {
    composer = runtime.thread.composer;
    return () => {
      composer = null;
    };
  }, [runtime]);
  return (
    <ComposerPrimitive.Root>
      <ComposerPrimitive.Input aria-label="Draft message" />
    </ComposerPrimitive.Root>
  );
}

function Chat() {
  const location = useRouterState({ select: (state) => state.location });
  const request = useModelConfigHandoffStore((state) =>
    modelConfigHandoffForDestination(state.request, {
      active: true,
      newChatId:
        new URLSearchParams(location.searchStr).get("new") ?? undefined,
    }),
  );
  const [open, setOpen] = useState(false);
  return (
    <>
      <ModelSelector
        models={[]}
        value="owner/Native"
        loaded={true}
        configRequest={request}
        open={open || request !== null}
        onOpenChange={setOpen}
        onConfigRequestAdopted={(id) => {
          setOpen(true);
          clearModelConfigHandoff(id);
        }}
        onValueChange={(id, meta) => {
          loaded = { id, meta };
        }}
      />
      {draftMode && (
        <ChatRuntimeProvider
          newThreadNonce={
            new URLSearchParams(location.searchStr).get("new") ?? undefined
          }
          listThreads={false}
        >
          <DraftComposer />
        </ChatRuntimeProvider>
      )}
    </>
  );
}

function LocalShare() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Share local settings
      </button>
      {open && (
        <ShareRunConfigDialog
          target={{
            id: "/models/local.gguf",
            displayName: "Local model",
            isGguf: true,
            apiLoadable: true,
            meta: { source: "local", isLora: false },
          }}
          config={{
            ...DEFAULT_PER_MODEL_CONFIG,
            reasoningBudgetMessage: "a".repeat(2500),
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

const root = createRootRoute({
  component: () => (
    <TooltipProvider>
      <SharedRunConfigLinkHandler />
      <main className="p-10">
        <Outlet />
        <LocalShare />
      </main>
      <Toaster />
    </TooltipProvider>
  ),
});
const chat = createRoute({
  getParentRoute: () => root,
  path: "/chat",
  component: Chat,
});
const hub = createRoute({
  getParentRoute: () => root,
  path: "/hub",
  component: () => <p>Hub fixture</p>,
});
const login = createRoute({
  getParentRoute: () => root,
  path: "/login",
  component: () => <p>Login fixture</p>,
});
const router = createRouter({
  routeTree: root.addChildren([chat, hub, login]),
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
