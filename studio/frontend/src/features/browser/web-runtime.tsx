// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { useBrowserAgentStore } from "./agent/agent-store";
import { FLOATING_SCALE_VAR } from "./floating-events";
import { useNativeBrowser } from "./native-view";
import { currentEntry, entryKey, type BrowserTab, useBrowserStore } from "./store";
import { WebPage } from "./tab-view";

type Edges = { left: number; top: number; right: number; bottom: number };
type Slot = { tabId: string; left: number; top: number; width: number; height: number; scale: number; readonly: boolean; clip: Edges };
type Size = { width: number; height: number };
type Slots = { mounted: string[]; visible: Slot | null; sizes: Record<string, Size> };

function WebTransport({ tab, slot, size }: { tab: BrowserTab; slot: Slot | null; size?: Size }) {
  const entry = currentEntry(tab);
  if (entry.kind !== "web") return null;
  const interactive = !!slot && !slot.readonly;
  const width = interactive ? slot.width / slot.scale : size?.width ?? 1024;
  const height = interactive ? slot.height / slot.scale : size?.height ?? 768;
  const scale = slot?.readonly ? Math.min(slot.width / width, slot.height / height) : slot?.scale ?? 1;
  const clipPath = slot ? `inset(${[
    Math.max(0, slot.clip.top - slot.top),
    Math.max(0, slot.left + width * scale - slot.clip.right),
    Math.max(0, slot.top + height * scale - slot.clip.bottom),
    Math.max(0, slot.clip.left - slot.left),
  ].map((value) => `${value / scale}px`).join(" ")})` : undefined;
  return (
    <div
      data-browser-scope-tab={tab.id}
      data-web-readonly={slot?.readonly ? tab.id : undefined}
      aria-hidden={!interactive}
      inert={!interactive}
      style={{ position: "fixed", left: slot?.left ?? -10000, top: slot?.top ?? 0, width, height, transformOrigin: "top left", transform: scale === 1 ? undefined : `scale(${scale})`, clipPath, zIndex: 20, pointerEvents: interactive ? "auto" : "none", overflow: "hidden" }}
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
      const observed = document.querySelector<HTMLElement>("[data-web-observe]");
      const readonly = !!observed && observed.offsetParent !== null;
      const element = readonly ? observed : elements.find((node) => node.offsetParent !== null);
      const box = element?.getBoundingClientRect();
      const tabId = readonly ? observed.dataset.webObserve : element?.dataset.webPage;
      const pageRoot = element?.closest<HTMLElement>("[data-browser-page]");
      const scale = pageRoot ? Number.parseFloat(getComputedStyle(pageRoot).getPropertyValue(FLOATING_SCALE_VAR)) || 1 : 1;
      const section = element?.closest("section")?.getBoundingClientRect();
      const page = element?.closest("[data-browser-page]")?.parentElement?.getBoundingClientRect() ?? element?.parentElement?.getBoundingClientRect();
      const clip = { left: Math.max(0, section?.left ?? 0, page?.left ?? 0), top: Math.max(0, section?.top ?? 0, page?.top ?? 0), right: Math.min(innerWidth, section?.right ?? innerWidth, page?.right ?? innerWidth), bottom: Math.min(innerHeight, section?.bottom ?? innerHeight, page?.bottom ?? innerHeight) };
      const visible = box && tabId && box.width > 1 && box.height > 1 ? { tabId, left: box.left, top: box.top, width: box.width, height: box.height, scale, readonly, clip } : null;
      setSlots((old) => {
        if (old.mounted.join("\0") === mounted.join("\0") && JSON.stringify(old.visible) === JSON.stringify(visible)) return old;
        const sizes = visible && !visible.readonly ? { ...old.sizes, [visible.tabId]: { width: visible.width / visible.scale, height: visible.height / visible.scale } } : old.sizes;
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
    <WebTransport key={tab.id} tab={tab} size={slots.sizes[tab.id]} slot={open && slots.visible?.tabId === tab.id ? slots.visible : null} />
  ));
}
