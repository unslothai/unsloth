// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { pickNativeProjectFolder } from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useRef, useState } from "react";

import type { ProjectRecord } from "../types";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { openChatProjectFromFolder } from "./use-chat-projects";

export function useOpenProjectFolder(): {
  openProjectFolder: () => Promise<ProjectRecord | null>;
  openingProjectFolder: boolean;
} {
  const navigate = useNavigate();
  const [openingProjectFolder, setOpeningProjectFolder] = useState(false);
  const openingRef = useRef(false);

  const openProjectFolder =
    useCallback(async (): Promise<ProjectRecord | null> => {
      if (openingRef.current) return null;
      if (!isTauri) {
        toast.error("Project folders are available in the desktop app.");
        return null;
      }

      openingRef.current = true;
      setOpeningProjectFolder(true);
      try {
        const selected = await pickNativeProjectFolder();
        if (!selected) return null;
        const project = await openChatProjectFromFolder(selected.token);
        const runtime = useChatRuntimeStore.getState();
        runtime.setActiveThreadId(null);
        runtime.setActiveProjectId(project.id);
        navigate({ to: "/chat", search: { project: project.id } });
        return project;
      } catch (error) {
        toast.error("Could not open project folder", {
          description: error instanceof Error ? error.message : String(error),
        });
        return null;
      } finally {
        openingRef.current = false;
        setOpeningProjectFolder(false);
      }
    }, [navigate]);

  return { openProjectFolder, openingProjectFolder };
}
