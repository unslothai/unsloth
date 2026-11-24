// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import { toast } from "@/lib/toast";

let checked = false;

function sameModel(raw: string | null, wanted: string): boolean {
  if (raw === null) return false;
  if (raw === wanted) return true;
  try {
    return JSON.parse(raw) === wanted;
  } catch {
    return false;
  }
}

/** Whether ?model=<wanted> came with the document itself: the tab was opened at that address from outside
 * the app, and any site can send a signed-in tab there. Answers once per page load, since every later pick
 * is an in-app navigation. */
export function modelLinkedFromOutside(wanted: string): boolean {
  if (checked) return false;
  checked = true;
  try {
    const entry = performance.getEntriesByType("navigation")[0] as { name?: string } | undefined;
    if (!entry?.name) return false;
    const first = new URL(entry.name);
    // Opened on this same media page: a Hub page opened with ?model= hands over with Run, in the app.
    const path = (p: string) => p.replace(/\/+$/, "");
    return (
      first.origin === window.location.origin &&
      path(first.pathname) === path(window.location.pathname) &&
      sameModel(first.searchParams.get("model"), wanted)
    );
  } catch {
    return false;
  }
}

/** A linked model downloads and loads only once the user says so; an in-app pick is unchanged. */
export function confirmLinkedModel(wanted: string, start: () => void): void {
  toast(translate("picker.linkedModelTitle", { model: wanted }), {
    description: translate("picker.linkedModelBody"),
    action: { label: translate("picker.linkedModelLoad"), onClick: start },
    duration: 20_000,
  });
}

export function resetLinkedModelArrivalForTests(): void {
  checked = false;
}
