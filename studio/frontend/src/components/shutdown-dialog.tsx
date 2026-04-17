// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { toastError } from "@/shared/toast";
import { useState } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface ShutdownDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called right before the shutdown API request so callers can remove the
   *  beforeunload listener — otherwise the "Server stopped" page would still
   *  trigger a "Leave site?" prompt when the user tries to close it. */
  onBeforeShutdown?: () => void;
}

export function ShutdownDialog({
  open,
  onOpenChange,
  onBeforeShutdown,
}: ShutdownDialogProps) {
  const [stopping, setStopping] = useState(false);

  const handleStop = async () => {
    setStopping(true);
    let accepted = false;
    try {
      const res = await authFetch("/api/shutdown", { method: "POST" });
      accepted = res.ok;
      if (!accepted) {
        toastError("关闭服务失败");
        setStopping(false);
        return;
      }
    } catch {
      // Network error — shutdown request never reached the server
      toastError("无法连接到服务");
      setStopping(false);
      return;
    }

    onBeforeShutdown?.();
    document.body.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;gap:12px">
        <p style="font-size:1.1rem;font-weight:600;margin:0">Unsloth Studio 已停止运行。</p>
        <p style="font-size:0.9rem;color:#888;margin:0">现在可以关闭此标签页。</p>
      </div>`;
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>要停止 Unsloth Studio 吗？</AlertDialogTitle>
          <AlertDialogDescription>
            这将关闭服务。所有正在运行的训练或推理任务都会被终止。你可以随时通过桌面快捷方式重新启动。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>取消</AlertDialogCancel>
          <AlertDialogAction
            onClick={handleStop}
            disabled={stopping}
            variant="destructive"
          >
            {stopping ? "停止中…" : "停止服务"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
