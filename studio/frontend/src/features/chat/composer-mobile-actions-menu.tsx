// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ComposerPrimitive } from "@assistant-ui/react";
import { Image03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  DownloadIcon,
  GlobeIcon,
  LightbulbIcon,
  LightbulbOffIcon,
  PlusIcon,
} from "lucide-react";
import { applyQwenThinkingParams } from "./utils/qwen-params";
import { parseExternalModelId } from "./external-providers";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

type ComposerMobileActionsMenuProps = {
  onAddAttachment?: () => void;
};

export function ComposerMobileActionsMenu({
  onAddAttachment,
}: ComposerMobileActionsMenuProps) {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const supportsBuiltinWebSearch = useChatRuntimeStore(
    (s) => s.supportsBuiltinWebSearch,
  );
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const supportsPreserveThinking = useChatRuntimeStore(
    (s) => s.supportsPreserveThinking,
  );
  const preserveThinking = useChatRuntimeStore((s) => s.preserveThinking);
  const setPreserveThinking = useChatRuntimeStore((s) => s.setPreserveThinking);
  const supportsBuiltinCodeExecution = useChatRuntimeStore(
    (s) => s.supportsBuiltinCodeExecution,
  );
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const setCodeToolsEnabled = useChatRuntimeStore((s) => s.setCodeToolsEnabled);
  const supportsBuiltinImageGeneration = useChatRuntimeStore(
    (s) => s.supportsBuiltinImageGeneration,
  );
  const imageToolsEnabled = useChatRuntimeStore((s) => s.imageToolsEnabled);
  const setImageToolsEnabled = useChatRuntimeStore(
    (s) => s.setImageToolsEnabled,
  );
  const supportsBuiltinWebFetch = useChatRuntimeStore(
    (s) => s.supportsBuiltinWebFetch,
  );
  const webFetchToolsEnabled = useChatRuntimeStore(
    (s) => s.webFetchToolsEnabled,
  );
  const setWebFetchToolsEnabled = useChatRuntimeStore(
    (s) => s.setWebFetchToolsEnabled,
  );
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const externalProvidersAll = useExternalProvidersStore((s) => s.providers);
  const externalSelection = parseExternalModelId(checkpoint);
  const selectedExternalProvider =
    externalSelection != null && connectionsEnabled
      ? externalProvidersAll.find((p) => p.id === externalSelection.providerId)
      : undefined;
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";

  const searchDisabled = !modelLoaded || !(supportsTools || supportsBuiltinWebSearch);
  const codeDisabled = !modelLoaded || !(supportsTools || supportsBuiltinCodeExecution);
  const imageDisabled = !modelLoaded;
  const webFetchDisabled = !modelLoaded || !supportsBuiltinWebFetch;

  const addAttachmentItem = (
    <DropdownMenuItem onSelect={onAddAttachment}>
      <PlusIcon className="size-4" />
      Add attachment
    </DropdownMenuItem>
  );

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          className="flex size-8.5 items-center justify-center rounded-full p-1 font-semibold text-xs hover:bg-muted-foreground/15 sm:hidden dark:hover:bg-muted-foreground/30"
          aria-label="Composer actions"
        >
          <PlusIcon className="size-5 stroke-[1.5px]" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" side="top" className="w-56">
        {onAddAttachment ? (
          addAttachmentItem
        ) : (
          <ComposerPrimitive.AddAttachment asChild={true}>
            {addAttachmentItem}
          </ComposerPrimitive.AddAttachment>
        )}
        <DropdownMenuCheckboxItem
          checked={toolsEnabled && !searchDisabled}
          disabled={searchDisabled}
          onCheckedChange={(checked) => {
            const next = Boolean(checked);
            setToolsEnabled(next);
            if (isKimiExternal) {
              setReasoningEnabled(!next);
              applyQwenThinkingParams(!next);
            }
          }}
        >
          <GlobeIcon className="size-4" />
          Search
        </DropdownMenuCheckboxItem>
        {supportsPreserveThinking && (
          <DropdownMenuCheckboxItem
            checked={preserveThinking && modelLoaded}
            disabled={!modelLoaded}
            onCheckedChange={(checked) => setPreserveThinking(Boolean(checked))}
          >
            {preserveThinking && modelLoaded ? (
              <LightbulbIcon className="size-4" />
            ) : (
              <LightbulbOffIcon className="size-4" />
            )}
            Preserve Think
          </DropdownMenuCheckboxItem>
        )}
        <DropdownMenuCheckboxItem
          checked={codeToolsEnabled && !codeDisabled}
          disabled={codeDisabled}
          onCheckedChange={(checked) => setCodeToolsEnabled(Boolean(checked))}
        >
          <CodeToggleIcon className="size-4" />
          Code
        </DropdownMenuCheckboxItem>
        {supportsBuiltinImageGeneration && (
          <DropdownMenuCheckboxItem
            checked={imageToolsEnabled && !imageDisabled}
            disabled={imageDisabled}
            onCheckedChange={(checked) => setImageToolsEnabled(Boolean(checked))}
          >
            <HugeiconsIcon icon={Image03Icon} className="size-4" strokeWidth={2} />
            Images
          </DropdownMenuCheckboxItem>
        )}
        {supportsBuiltinWebFetch && (
          <DropdownMenuCheckboxItem
            checked={webFetchToolsEnabled && !webFetchDisabled}
            disabled={webFetchDisabled}
            onCheckedChange={(checked) => setWebFetchToolsEnabled(Boolean(checked))}
          >
            <DownloadIcon className="size-4" />
            Fetch
          </DropdownMenuCheckboxItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
