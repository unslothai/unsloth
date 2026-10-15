// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import { useEffect, useRef } from "react";
import { hasSettledAuthSession } from "@/features/auth";
import { notifyChatHistoryUpdated, useChatNavigationStore } from "@/features/chat";
import { translate } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { notifyNative } from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { type FinishedScheduledTaskRun, getScheduledTaskActivity } from "./api";
import {
  IDLE_POLL_MS,
  SCHEDULED_TASKS_CHANGED_EVENT,
  SCHEDULED_TASK_RUNS_FINISHED_EVENT,
  type SeenRuns,
  advanceSeenRuns,
  nextPollDelay,
  readSeenRuns,
  runErrorText,
} from "./results";

const SEEN_KEY = "unsloth_scheduled_tasks_seen";
// More than this at once, e.g. on reopening Studio, and one toast speaks for them all.
const INDIVIDUAL_TOASTS = 2;
const RECHECK_MS = 500;

function loadSeen(): SeenRuns | null {
  try {
    return readSeenRuns(localStorage.getItem(SEEN_KEY), Date.now());
  } catch {
    return null;
  }
}

function saveSeen(seen: SeenRuns): void {
  try {
    localStorage.setItem(SEEN_KEY, JSON.stringify(seen));
  } catch {
    // Without storage the next launch only announces runs from then on.
  }
}

function windowIsInBackground(): boolean {
  return (
    document.visibilityState !== "visible" ||
    (typeof document.hasFocus === "function" && !document.hasFocus())
  );
}

function notifySystem(runs: FinishedScheduledTaskRun[]): void {
  const [run] = runs;
  const single = runs.length === 1;
  const title = !single
    ? translate("scheduledTasks.toast.many", { count: runs.length })
    : run.status === "succeeded"
      ? translate("scheduledTasks.toast.finished", { title: run.taskTitle })
      : translate("scheduledTasks.toast.failed", { title: run.taskTitle });
  const body = !single
    ? runs.map((entry) => entry.taskTitle).join(", ")
    : run.status === "succeeded"
      ? translate("scheduledTasks.notificationBody")
      : runErrorText(run, translate);
  if (isTauri) {
    void notifyNative({
      key: `scheduled-task:${runs.map((entry) => entry.id).join(",")}`,
      title,
      body,
      requestPermission: false,
    });
    return;
  }
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  try {
    const notification = new Notification(title, { body });
    notification.onclick = () => {
      window.focus();
      notification.close();
    };
  } catch {
    // Some browsers only allow notifications from a service worker.
  }
}

/** Watches for scheduled task results while Studio is open: brings the new chats into the
 *  sidebar, marks them unread, and says so with a toast, plus a system notification when the
 *  window is in the background. Renders nothing. */
export function ScheduledTaskRunNotifier() {
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  useEffect(() => {
    navigateRef.current = navigate;
  }, [navigate]);

  useEffect(() => {
    let disposed = false;
    let polling = false;
    // A change made while a poll is in flight, whose answer may predate it.
    let recheckAfterPoll = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let seen = loadSeen();

    const openChat = (threadId: string) =>
      void navigateRef.current({ to: "/chat", search: { thread: threadId } });
    const openTasks = () => void navigateRef.current({ to: "/tasks" });

    const announce = (runs: FinishedScheduledTaskRun[]) => {
      const threadIds = runs.flatMap((run) =>
        run.status === "succeeded" && run.threadId ? [run.threadId] : [],
      );
      if (threadIds.length > 0) {
        notifyChatHistoryUpdated();
        useChatNavigationStore.getState().markThreadsUnread(threadIds);
      }
      window.dispatchEvent(new Event(SCHEDULED_TASK_RUNS_FINISHED_EVENT));
      if (runs.length > INDIVIDUAL_TOASTS) {
        toast(translate("scheduledTasks.toast.many", { count: runs.length }), {
          action: { label: translate("scheduledTasks.toast.view"), onClick: openTasks },
        });
      } else {
        for (const run of runs) {
          const threadId = run.threadId;
          if (run.status === "succeeded" && threadId) {
            toast.success(translate("scheduledTasks.toast.finished", { title: run.taskTitle }), {
              action: {
                label: translate("scheduledTasks.toast.open"),
                onClick: () => openChat(threadId),
              },
            });
          } else {
            toast.error(translate("scheduledTasks.toast.failed", { title: run.taskTitle }), {
              description: runErrorText(run, translate),
              action: { label: translate("scheduledTasks.toast.view"), onClick: openTasks },
            });
          }
        }
      }
      if (windowIsInBackground()) notifySystem(runs);
    };

    const schedule = (delay: number) => {
      if (disposed) return;
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => void poll(), delay);
    };

    const poll = async () => {
      if (disposed || polling) return;
      // Signed out, or still signing in: the request would only bounce to the login page.
      if (!hasSettledAuthSession()) {
        schedule(IDLE_POLL_MS);
        return;
      }
      polling = true;
      try {
        const activity = await getScheduledTaskActivity(seen?.since ?? null);
        if (disposed) return;
        const announced = new Set(seen?.ids);
        const fresh = activity.runs.filter((run) => !announced.has(run.id));
        seen = advanceSeenRuns(seen, activity.now, fresh.map((run) => run.id));
        saveSeen(seen);
        if (fresh.length > 0) announce(fresh);
        schedule(nextPollDelay(activity));
      } catch {
        schedule(IDLE_POLL_MS);
      } finally {
        polling = false;
        if (recheckAfterPoll) {
          recheckAfterPoll = false;
          schedule(RECHECK_MS);
        }
      }
    };

    const recheck = () => {
      if (polling) recheckAfterPoll = true;
      else schedule(RECHECK_MS);
    };
    const onVisible = () => {
      if (document.visibilityState === "visible") recheck();
    };
    window.addEventListener(SCHEDULED_TASKS_CHANGED_EVENT, recheck);
    window.addEventListener("focus", recheck);
    document.addEventListener("visibilitychange", onVisible);
    void poll();
    return () => {
      disposed = true;
      if (timer !== null) clearTimeout(timer);
      window.removeEventListener(SCHEDULED_TASKS_CHANGED_EVENT, recheck);
      window.removeEventListener("focus", recheck);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  return null;
}
