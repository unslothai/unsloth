// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { useT } from "@/i18n";
import { captureHiddenPage } from "./capture";
import { hasNativeView } from "./native-view";
import { requestFrameViewSnapshot, type FrameViewSnapshot } from "./page-frame";
import { observationDocument } from "./observer-state";
import { useBrowserStore } from "./store";

type Preview = { kind: "html"; content: string; viewport: FrameViewSnapshot } | { kind: "image"; content: string };

function SnapshotFrame({ snapshot, title }: { snapshot: Extract<Preview, { kind: "html" }>; title: string }) {
  const container = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const { viewportWidth: width, viewportHeight: height } = snapshot.viewport;
  const scale = Math.min(size.width / width, size.height / height);
  return (
    <div ref={container} inert={true} className="absolute inset-0 overflow-hidden">
      <iframe title={title} sandbox="allow-scripts" tabIndex={-1} srcDoc={snapshot.content}
        className="pointer-events-none absolute left-0 top-0 border-0" style={{ width, height, transformOrigin: "top left", transform: `scale(${scale})` }} />
    </div>
  );
}

export function BrowserObservation({ tabId }: { tabId: string }) {
  const t = useT();
  const [snapshot, setSnapshot] = useState<Preview | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let imageUrl: string | null = null;
    setSnapshot(null);
    setUnavailable(false);
    const update = async () => {
      let next: Preview | null = null;
      try {
        const tab = useBrowserStore.getState().tabs.find((item) => item.id === tabId);
        if (tab && hasNativeView(tabId)) {
          const blob = await captureHiddenPage(tab);
          if (blob) next = { kind: "image", content: URL.createObjectURL(blob) };
        } else {
          const view = await requestFrameViewSnapshot(tabId);
          if (view && view.viewportWidth > 0 && view.viewportHeight > 0) {
            const nonce = crypto.randomUUID().replaceAll("-", "");
            next = { kind: "html", content: observationDocument(view.html, nonce, view), viewport: view };
          }
        }
      } catch {
        next = null;
      }
      if (cancelled) {
        if (next?.kind === "image") URL.revokeObjectURL(next.content);
        return;
      }
      setUnavailable(!next);
      if (next) {
        const previous = imageUrl;
        imageUrl = next.kind === "image" ? next.content : null;
        setSnapshot(next);
        if (previous) URL.revokeObjectURL(previous);
      }
      timer = setTimeout(update, 1500);
    };
    void update();
    return () => { cancelled = true; if (timer) clearTimeout(timer); if (imageUrl) URL.revokeObjectURL(imageUrl); };
  }, [tabId]);
  return (
    <div className="relative min-h-0 flex-1 bg-background">
      {snapshot?.kind === "html" ? <SnapshotFrame title={t("browser.observer.preview")} snapshot={snapshot} /> : null}
      {snapshot?.kind === "image" ? <img inert={true} alt={t("browser.observer.preview")} src={snapshot.content} draggable={false} className="pointer-events-none h-full w-full object-contain object-top" /> : null}
      {unavailable ? <div role="status" className="absolute inset-x-0 top-0 bg-card p-3 text-ui-13 text-muted-foreground">{t("browser.observer.unavailable")}</div> : null}
    </div>
  );
}
