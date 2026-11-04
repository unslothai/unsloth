// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { useBrowserAgentStore } from "./agent/agent-store";
import { useNativeBrowser } from "./native-view";
import { currentEntry, entryKey, type BrowserTab, useBrowserStore } from "./store";
import { WebPage } from "./tab-view";

type Slot = { tabId: string; left: number; top: number; width: number; height: number };
type Size = { width: number; height: number };
type Slots = { mounted: string[]; visible: Slot | null; sizes: Record<string, Size> };

function WebTransport({ tab, slot, size }: { tab: BrowserTab; slot: Slot | null; size?: Size }) {
  const entry = currentEntry(tab);
  if (entry.kind !== "web") return null;
  return (
    <div
      data-browser-scope-tab={tab.id}
      aria-hidden={!slot}
      inert={!slot}
      style={{ position: "fixed", left: slot?.left ?? -10000, top: slot?.top ?? 0, width: slot?.width ?? size?.width ?? 1024, height: slot?.height ?? size?.height ?? 768, zIndex: 20, pointerEvents: slot ? "auto" : "none", overflow: "hidden" }}
    >
      <WebPage key={`${entryKey(entry)}:${tab.reloadKey}`} tab={tab} entry={entry} />
    </div>
  );
}

export function WebRuntime() {
  const tabs = useBrowserStore((state) => state.tabs);
  const activeTabId = useBrowserStore((state) => state.activeTabId);
  const open = useBrowserStore((state) => state.open);
  const agents = useBrowserAgentStore((state) => state.chats);
  const native = useNativeBrowser((state) => state.enabled);
  const [slots, setSlots] = useState<Slots>({ mounted: [], visible: null, sizes: {} });
  useEffect(() => {
    let last: HTMLElement | null = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const remember = (event: FocusEvent) => {
      if (event.target instanceof HTMLElement && !event.target.closest("[inert]")) last = event.target;
    };
    const check = () => queueMicrotask(() => {
      const active = document.activeElement;
      if (active instanceof HTMLIFrameElement && active.closest("[inert]")) {
        if (last?.isConnected) last.focus({ preventScroll: true });
      } else if (active instanceof HTMLElement) last = active;
    });
    document.addEventListener("focusin", remember, true);
    window.addEventListener("blur", check);
    return () => {
      document.removeEventListener("focusin", remember, true);
      window.removeEventListener("blur", check);
    };
  }, []);
  useEffect(() => {
    let frame = 0;
    const measure = () => {
      frame = 0;
      const elements = [...document.querySelectorAll<HTMLElement>("[data-web-page]")];
      const mounted = elements.map((node) => node.dataset.webPage).filter((id): id is string => !!id);
      const element = elements.find((node) => node.offsetParent !== null);
      const box = element?.getBoundingClientRect();
      const tabId = element?.dataset.webPage;
      const visible = box && tabId && box.width > 1 && box.height > 1 ? { tabId, left: box.left, top: box.top, width: box.width, height: box.height } : null;
      setSlots((old) => {
        if (old.mounted.join("\0") === mounted.join("\0") && old.visible?.tabId === visible?.tabId && old.visible?.left === visible?.left && old.visible?.top === visible?.top && old.visible?.width === visible?.width && old.visible?.height === visible?.height) return old;
        const sizes = visible ? { ...old.sizes, [visible.tabId]: { width: visible.width, height: visible.height } } : old.sizes;
        return { mounted, visible, sizes };
      });
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(measure); };
    const observer = new MutationObserver(schedule);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["class", "style", "hidden"] });
    window.addEventListener("resize", schedule);
    window.addEventListener("scroll", schedule, true);
    const timer = window.setInterval(schedule, 150);
    schedule();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", schedule);
      window.removeEventListener("scroll", schedule, true);
      window.clearInterval(timer);
      cancelAnimationFrame(frame);
    };
  }, []);
  if (native) return null;
  const retained = new Set(slots.mounted);
  if (open && activeTabId) retained.add(activeTabId);
  for (const agent of Object.values(agents)) if (agent.tabId) retained.add(agent.tabId);
  return tabs.filter((tab) => retained.has(tab.id) && currentEntry(tab).kind === "web").map((tab) => (
    <WebTransport key={tab.id} tab={tab} size={slots.sizes[tab.id]} slot={open && activeTabId === tab.id && slots.visible?.tabId === tab.id ? slots.visible : null} />
  ));
}
