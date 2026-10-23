// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useEffect, useRef } from "react";
import { AUTH_SESSION_STORED_EVENT, getAuthSessionEpoch, hasSettledAuthSession } from "@/features/auth";
import { notifyChatHistoryUpdated, useChatNavigationStore } from "@/features/chat";
import { translate } from "@/i18n";
import { browserAccountKey } from "@/lib/account-transition";
import { isTauri } from "@/lib/api-base";
import { notifyNative } from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { type FinishedScheduledTaskRun, getScheduledTaskActivity } from "./api";
import {
  IDLE_POLL_MS,
  SCHEDULED_TASKS_CHANGED_EVENT,
  SCHEDULED_TASKS_WRITTEN_KEY,
  SCHEDULED_TASK_RUNS_FINISHED_EVENT,
  SEEN_KEY_PREFIX,
  type SeenRuns,
  claimAnnouncements,
  collectFinishedRuns,
  nextPollDelay,
  pollDelayFor,
  readSeenRuns,
  runErrorText,
  runsToAnnounce,
  runsToNotify,
  readInRecents,
  unseenChatChanges,
  shouldSaveSeen,
} from "./results";
import { useUnseenScheduledRuns } from "./unseen";

// More than this at once, e.g. on reopening Studio, and one toast speaks for them all.
const INDIVIDUAL_TOASTS = 2;
const RECHECK_MS = 500;
// Lock one tab holds while claiming what to announce (account name appended).
const ANNOUNCE_LOCK_PREFIX = "unsloth-scheduled-task-results:";

/** The signed-in account the record belongs to, by id so a recreated name starts afresh. */
function currentAccount(): string {
  return browserAccountKey();
}

function loadSeen(account: string): SeenRuns | null {
  try {
    return readSeenRuns(localStorage.getItem(SEEN_KEY_PREFIX + account));
  } catch {
    return null;
  }
}

function saveSeen(account: string, seen: SeenRuns): void {
  try {
    localStorage.setItem(SEEN_KEY_PREFIX + account, JSON.stringify(seen));
  } catch {
    // Without storage the next launch only announces runs from then on.
  }
}

/** `claim` under a Web Lock where available (secure contexts); elsewhere two tabs may both announce. */
async function exclusively<T>(name: string, claim: () => T): Promise<T> {
  const locks = (navigator as Navigator & { locks?: LockManager }).locks;
  if (!locks) return claim();
  let claimed = false;
  try {
    return await locks.request(name, () => {
      claimed = true;
      return claim();
    });
  } catch (error) {
    if (claimed) throw error;
    return claim();
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

/** Watches for scheduled results: refreshes Recents, marks chats unread, toasts and notifies. Renders nothing. */
export function ScheduledTaskRunNotifier() {
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const pathnameRef = useRef(pathname);
  const openThread = useRouterState({
    select: (state) => (state.location.search as { thread?: unknown }).thread,
  });
  useEffect(() => {
    if (typeof openThread === "string") useUnseenScheduledRuns.getState().clearThread(openThread);
  }, [openThread]);
  // Recents' unread marks follow the account's stored unseen results, so every tab and a reload
  // show them, and opening one anywhere clears it everywhere.
  useEffect(() => {
    const sync = (now: readonly string[], before: readonly string[]) => {
      const chats = useChatNavigationStore.getState();
      const { added, opened } = unseenChatChanges(now, before);
      const fresh = added.filter((id) => !chats.unreadThreadIds.has(id));
      if (fresh.length > 0) chats.markThreadsUnread(fresh);
      if (opened.length > 0) chats.clearThreadsUnread(opened);
    };
    sync(useUnseenScheduledRuns.getState().threads, []);
    const offUnseen = useUnseenScheduledRuns.subscribe((state, previous) => {
      if (state.threads !== previous.threads) sync(state.threads, previous.threads);
    });
    // Read in Recents (opened, "Mark as read", all at once) is read for good. A sign-out's reset
    // comes after the token is gone, when the record is out of reach and stays as it was.
    const offChats = useChatNavigationStore.subscribe((state, previous) => {
      if (state.unreadThreadIds === previous.unreadThreadIds) return;
      const unseen = useUnseenScheduledRuns.getState();
      for (const id of readInRecents(unseen.threads, previous.unreadThreadIds, state.unreadThreadIds)) {
        unseen.clearThread(id);
      }
    });
    return () => {
      offUnseen();
      offChats();
    };
  }, []);
  useEffect(() => {
    navigateRef.current = navigate;
    pathnameRef.current = pathname;
  }, [navigate, pathname]);

  useEffect(() => {
    let disposed = false;
    let polling = false;
    let recheckAfterPoll = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let account: string | null = null;
    let seen: SeenRuns | null = null;

    const openChat = (threadId: string) =>
      void navigateRef.current({ to: "/chat", search: { thread: threadId } });
    const openTasks = () => void navigateRef.current({ to: "/tasks" });

    const refreshFor = (finished: FinishedScheduledTaskRun[]) => {
      if (finished.some((run) => run.status === "succeeded" && run.threadId)) {
        notifyChatHistoryUpdated();
      }
      window.dispatchEvent(new Event(SCHEDULED_TASK_RUNS_FINISHED_EVENT));
    };

    // Only the claiming tab records and notifies, so it cannot revive a dot cleared in a faster tab.
    const announce = (claimed: FinishedScheduledTaskRun[]) => {
      const runs = runsToAnnounce(claimed);
      if (runs.length === 0) return;
      useUnseenScheduledRuns.getState().markFinished(
        runs.map((run) => ({
          taskId: run.taskId,
          threadId: run.status === "succeeded" ? run.threadId : null,
        })),
        pathnameRef.current === "/tasks",
      );
      const notified = runsToNotify(runs);
      if (notified.length === 0) return;
      if (notified.length > INDIVIDUAL_TOASTS) {
        toast(translate("scheduledTasks.toast.many", { count: notified.length }), {
          action: { label: translate("scheduledTasks.toast.view"), onClick: openTasks },
        });
      } else {
        for (const run of notified) {
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
      if (windowIsInBackground()) notifySystem(notified);
    };

    const schedule = (delay: number) => {
      if (disposed) return;
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => void poll(), delay);
    };

    const poll = async () => {
      if (disposed || polling) return;
      if (!hasSettledAuthSession()) {
        schedule(IDLE_POLL_MS);
        return;
      }
      const owner = currentAccount();
      if (owner !== account) {
        account = owner;
        seen = loadSeen(owner);
      }
      const epoch = getAuthSessionEpoch();
      polling = true;
      try {
        const result = await collectFinishedRuns(seen, getScheduledTaskActivity);
        const mine = await exclusively(ANNOUNCE_LOCK_PREFIX + owner, () => {
          if (disposed || getAuthSessionEpoch() !== epoch || currentAccount() !== owner) return null;
          const stored = loadSeen(owner);
          const claim = claimAnnouncements(stored, result.seen, result.fresh);
          if (shouldSaveSeen(stored, claim.record, claim.mine.length)) saveSeen(owner, claim.record);
          return claim.mine;
        });
        if (disposed) return;
        if (mine === null) {
          recheckAfterPoll = true;
          return;
        }
        seen = result.seen;
        if (result.fresh.length > 0) refreshFor(result.fresh);
        if (mine.length > 0) announce(mine);
        schedule(pollDelayFor(nextPollDelay(result.last), document.visibilityState === "hidden"));
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
    // Another tab's change: it may have made a task due sooner than this tab's next look.
    const onStorage = (event: StorageEvent) => {
      if (event.key === SCHEDULED_TASKS_WRITTEN_KEY) recheck();
    };
    window.addEventListener(SCHEDULED_TASKS_CHANGED_EVENT, recheck);
    window.addEventListener("storage", onStorage);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, recheck);
    window.addEventListener("focus", recheck);
    document.addEventListener("visibilitychange", onVisible);
    void poll();
    return () => {
      disposed = true;
      if (timer !== null) clearTimeout(timer);
      window.removeEventListener(SCHEDULED_TASKS_CHANGED_EVENT, recheck);
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, recheck);
      window.removeEventListener("focus", recheck);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  return null;
}
