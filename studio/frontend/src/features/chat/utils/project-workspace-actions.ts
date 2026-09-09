// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { revealSandbox } from "@/components/assistant-ui/sandbox-reveal";
import {
  consumeNativePathToken,
  pickNativeProjectWorkspace,
} from "@/features/native-intents";
import { toast } from "@/lib/toast";

import { setChatProjectWorkspace } from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";

// The Projects list row and the project page offer the same three actions; one
// place for them so the two cannot drift. Callers gate on the desktop app.

/** Pick a folder and move the project onto it. False when nothing changed. */
export async function chooseProjectWorkspace(projectId: string): Promise<boolean> {
  try {
    const selected = await pickNativeProjectWorkspace();
    if (!selected) return false;
    const lease = await consumeNativePathToken(
      selected.token,
      "set-project-workspace",
    );
    await setChatProjectWorkspace(projectId, {
      kind: "external",
      nativePathLease: lease.nativePathLease,
    });
    toast.success("Working directory updated", { description: selected.path });
    return true;
  } catch (error) {
    toast.error("Couldn't update the working directory", {
      description: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

export async function switchToManagedWorkspace(projectId: string): Promise<void> {
  try {
    await setChatProjectWorkspace(projectId, { kind: "managed" });
    toast.success("Using an Unsloth managed folder");
  } catch (error) {
    toast.error("Couldn't update the working directory", {
      description: error instanceof Error ? error.message : String(error),
    });
  }
}

/**
 * Open the project's working directory in the OS file manager. Through the
 * workspace session rather than the stored path: the backend resolves that to
 * the folder the project's tool calls use now, and refuses one that has gone.
 */
export async function revealProjectWorkspace(project: ProjectRecord): Promise<void> {
  if (!project.workspaceSessionId) {
    toast.error("Could not read this project's folder.", {
      description: "Try again once the project list has loaded.",
    });
    return;
  }
  try {
    await revealSandbox(project.workspaceSessionId);
  } catch (error) {
    toast.error("Could not open the project folder.", {
      description: error instanceof Error ? error.message : String(error),
    });
  }
}
