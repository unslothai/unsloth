// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LazyImportBoundary } from "@/components/lazy-import-boundary";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { type ComponentProps, Suspense, lazy, useLayoutEffect } from "react";
import { modelConfigDraftKey } from "../model-config/model-config-draft";
import { runConfigInbox } from "./inbox";

const SharedRunConfigActions = lazy(() =>
  import("./runtime").then((module) => ({
    default: module.SharedRunConfigActions,
  })),
);

const ConfigReview = lazy(() =>
  import("./runtime").then((module) => ({
    default: module.SharedRunConfigReview,
  })),
);

export function SharedRunConfigReview(
  props: ComponentProps<typeof ConfigReview>,
) {
  return (
    <LazyImportBoundary
      fallback={
        <p role="alert" className="mb-5 text-sm text-muted-foreground">
          The link settings summary could not load. Reload Studio and reopen the
          link to review it.
        </p>
      }
    >
      <Suspense fallback={null}>
        <ConfigReview {...props} />
      </Suspense>
    </LazyImportBoundary>
  );
}

export function SharedRunConfigControls(
  props: ComponentProps<typeof SharedRunConfigActions>,
) {
  const { target, canImport } = props;
  const key = modelConfigDraftKey(
    target.configId ?? target.id,
    target.ggufVariant,
  );
  useLayoutEffect(() => {
    if (canImport) {
      return runConfigInbox.retainEditor(key, (request) => {
        toast.info("Run settings import cancelled", {
          id: request.id,
          description:
            "The editor closed before the settings were imported. Reopen the link to try again.",
        });
      });
    }
  }, [canImport, key]);
  const fallback = (
    <Button
      type="button"
      size="sm"
      variant="outline"
      className={props.className}
      disabled
    >
      Share
    </Button>
  );
  return (
    <LazyImportBoundary fallback={fallback}>
      <Suspense fallback={fallback}>
        <SharedRunConfigActions {...props} />
      </Suspense>
    </LazyImportBoundary>
  );
}
