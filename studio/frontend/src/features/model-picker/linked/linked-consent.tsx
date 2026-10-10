// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { LinkedInstance } from "@/features/settings/api/linked-instances";
import { connectionKind } from "@/features/settings/components/linked-instance-format";
import { useT } from "@/i18n";

const ACCEPTED_KEY = "unsloth_linked_consent";

function accepted(): Set<string> {
  try {
    const raw = JSON.parse(localStorage.getItem(ACCEPTED_KEY) ?? "[]");
    return new Set(Array.isArray(raw) ? raw.filter((v) => typeof v === "string") : []);
  } catch {
    return new Set();
  }
}

/** Whether this machine's owner has already been told what leaves the machine for `id`. */
export function hasLinkedConsent(id: string): boolean {
  return accepted().has(id);
}

export function rememberLinkedConsent(id: string): void {
  const ids = accepted();
  ids.add(id);
  try {
    localStorage.setItem(ACCEPTED_KEY, JSON.stringify([...ids]));
  } catch {
    // Ignore unavailable storage: the dialog just asks again next time.
  }
}

/** Shown once per instance, before the first chat or image job is sent to it. */
export function LinkedConsentDialog({
  instance,
  onConfirm,
  onCancel,
}: {
  instance: LinkedInstance | null;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const t = useT();
  const name = `@${instance?.name ?? ""}`;
  const tunnel = instance ? connectionKind(instance.base_url) === "tunnel" : false;
  return (
    <Dialog
      open={instance !== null}
      onOpenChange={(open) => !open && onCancel()}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t("picker.linkedConsentTitle", { name })}</DialogTitle>
          <DialogDescription>
            {t("picker.linkedConsentBody", { name })}
            {tunnel ? ` ${t("picker.linkedConsentTunnel")}` : ""}
          </DialogDescription>
        </DialogHeader>
        <ul className="flex flex-col gap-1.5 text-ui-12 text-muted-foreground">
          <li>{t("picker.linkedConsentSends")}</li>
          <li>
            {instance?.allow_tools
              ? t("picker.linkedConsentToolsOn", { name })
              : t("picker.linkedConsentToolsOff")}
          </li>
        </ul>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            {t("common.cancel")}
          </Button>
          <Button
            onClick={() => {
              if (instance) rememberLinkedConsent(instance.id);
              onConfirm();
            }}
          >
            {t("picker.linkedConsentConfirm")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
