// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { useT } from "@/i18n";
import { captureHiddenPage } from "./capture";
import { hasNativeView, useNativeBrowser } from "./native-view";
import { useBrowserStore } from "./store";

export function BrowserObservation({ tabId }: { tabId: string }) {
  const t = useT();
  const native = useNativeBrowser((state) => state.enabled);
  const [image, setImage] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  useEffect(() => {
    if (!native) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let imageUrl: string | null = null;
    setImage(null);
    setUnavailable(false);
    const update = async () => {
      let next: string | null = null;
      try {
        const tab = useBrowserStore.getState().tabs.find((item) => item.id === tabId);
        if (tab && hasNativeView(tabId)) {
          const blob = await captureHiddenPage(tab);
          if (blob) next = URL.createObjectURL(blob);
        }
      } catch {
        next = null;
      }
      if (cancelled) {
        if (next) URL.revokeObjectURL(next);
        return;
      }
      setUnavailable(!next);
      if (next) {
        const previous = imageUrl;
        imageUrl = next;
        setImage(next);
        if (previous) URL.revokeObjectURL(previous);
      }
      timer = setTimeout(update, 1500);
    };
    void update();
    return () => { cancelled = true; if (timer) clearTimeout(timer); if (imageUrl) URL.revokeObjectURL(imageUrl); };
  }, [tabId, native]);
  if (!native) return <div data-web-observe={tabId} role="img" aria-label={t("browser.observer.preview")} className="relative min-h-0 flex-1 bg-background" />;
  return (
    <div className="relative min-h-0 flex-1 bg-background">
      {image ? <img inert={true} alt={t("browser.observer.preview")} src={image} draggable={false} className="pointer-events-none h-full w-full object-contain object-top" /> : null}
      {unavailable ? <div role="status" className="absolute inset-x-0 top-0 bg-card p-3 text-ui-13 text-muted-foreground">{t("browser.observer.unavailable")}</div> : null}
    </div>
  );
}
