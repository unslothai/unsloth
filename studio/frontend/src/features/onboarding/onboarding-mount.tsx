// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSessionEpoch, hasSettledAuthSession } from "@/features/auth";
import { useRouterState } from "@tanstack/react-router";
import { Suspense, lazy, useEffect } from "react";
import { loadOnboarding, saveOnboarding } from "./api";
import { ONBOARDING_PAGES, decideOnboarding } from "./lib/pages";
import { useOnboardingStore } from "./store";

const OnboardingDialog = lazy(() =>
  import("./components/onboarding-dialog").then((m) => ({
    default: m.OnboardingDialog,
  })),
);

/** Decides once per auth session, after any forced password change, whether this account sees onboarding. */
export function OnboardingMount() {
  useRouterState({ select: (s) => s.location.pathname });
  const settled = hasSettledAuthSession();
  const epoch = getAuthSessionEpoch();
  const pages = useOnboardingStore((s) => s.pages);

  // biome-ignore lint/correctness/useExhaustiveDependencies: epoch is the trigger, so a switched account decides afresh.
  useEffect(() => {
    if (!settled) return;
    let cancelled = false;
    useOnboardingStore.getState().close();
    loadOnboarding()
      .then((record) => {
        if (cancelled) return;
        const decision = decideOnboarding(record);
        if (decision.kind === "adopt") {
          void saveOnboarding([...ONBOARDING_PAGES], true).catch(() => {});
        } else if (decision.kind === "show") {
          useOnboardingStore.getState().show(decision.pages);
        }
      })
      // An older backend has no route: no onboarding, and nothing to retry.
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [settled, epoch]);

  if (!pages) return null;
  return (
    <Suspense fallback={null}>
      <OnboardingDialog key={pages.join()} pages={pages} />
    </Suspense>
  );
}
