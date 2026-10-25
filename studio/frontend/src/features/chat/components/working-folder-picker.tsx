// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useIsAccountOwner } from "@/features/auth";
import {
  type NativeDocumentFolderSelection,
  pickNativeDocumentFolder,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { useEffect, useRef, useState } from "react";

export function WorkingFolderPicker({
  value,
  onChange,
  disabled,
  needsReselect = false,
  onPendingChange,
}: {
  value: string | null;
  onChange: (
    value: NativeDocumentFolderSelection | null,
  ) => void | Promise<void>;
  disabled?: boolean;
  needsReselect?: boolean;
  onPendingChange?: (pending: boolean) => void;
}) {
  const isOwner = useIsAccountOwner();
  const supported = useNativePathLeasesSupported();
  const mounted = useRef(true);
  const pending = useRef(false);
  const [picking, setPicking] = useState(false);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const available = isTauri && supported && isOwner;

  async function change(choose: boolean) {
    if ((choose && !available) || disabled || pending.current) return;
    pending.current = true;
    setPicking(true);
    onPendingChange?.(true);
    try {
      const selected = choose ? await pickNativeDocumentFolder() : null;
      if (mounted.current && (!choose || selected)) await onChange(selected);
    } catch (error) {
      if (mounted.current) {
        toast.error("Could not change the working folder", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      pending.current = false;
      if (mounted.current) {
        setPicking(false);
        onPendingChange?.(false);
      }
    }
  }

  return (
    <section className="min-w-0 space-y-2" aria-label="Working folder">
      <h3 className="text-ui-15 font-medium text-foreground">Working folder</h3>
      <p className="text-ui-13 text-muted-foreground">
        Where code and file tools run for every chat in this project.
      </p>
      <div className="flex min-w-0 items-center gap-2 rounded-full border border-border py-2 pr-2 pl-5">
        <span
          className="min-w-0 flex-1 truncate text-sm"
          title={value ?? undefined}
        >
          {value ?? "Unsloth-managed folder"}
        </span>
        {value && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="shrink-0"
            disabled={disabled || picking}
            onClick={() => void change(false)}
          >
            Use managed folder
          </Button>
        )}
        <Button
          type="button"
          variant="dark"
          size="sm"
          className="shrink-0"
          disabled={!available || disabled || picking}
          onClick={() => void change(true)}
        >
          {picking ? "Choosing…" : value ? "Change folder" : "Choose folder"}
        </Button>
      </div>
      {(needsReselect || !available) && (
        <p
          className="text-ui-11 text-muted-foreground"
          role={needsReselect ? "alert" : undefined}
        >
          {needsReselect
            ? "Choose the folder again to retry, or use the managed folder."
            : !isTauri
              ? "Choose an existing folder in Unsloth Desktop, or use the managed folder."
              : !isOwner
                ? "Only the owner can choose a local working folder."
                : "Folder selection is unavailable with this connection."}
        </p>
      )}
    </section>
  );
}
