// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";
import { refreshChatTitle } from "../utils/refresh-chat-title";

export function RefreshChatTitleMenuItem({ item }: { item: SidebarItem }) {
  const [pending, setPending] = useState(false);

  async function refresh() {
    setPending(true);
    const notice = toast.loading("Refreshing chat title…");
    try {
      await refreshChatTitle(item);
      toast.success("Chat title refreshed");
    } catch (error) {
      toast.error("Failed to refresh chat title", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      toast.dismiss(notice);
      setPending(false);
    }
  }

  return (
    <DropdownMenuItem disabled={pending} onSelect={() => void refresh()}>
      <RefreshCw className={pending ? "size-icon animate-spin" : "size-icon"} />
      <span>{pending ? "Refreshing chat title…" : "Refresh chat title"}</span>
    </DropdownMenuItem>
  );
}
