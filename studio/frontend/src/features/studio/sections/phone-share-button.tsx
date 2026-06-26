// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  PhoneNotReachableError,
  type PhoneShareResponse,
  shareTrainingToPhone,
} from "@/features/training";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon, WifiDisconnected02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { QRCodeSVG } from "qrcode.react";
import { type ReactElement, useState } from "react";

function PhoneGlyph(): ReactElement {
  return (
    <svg
      viewBox="0 0 24 24"
      className="size-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="7" y="2" width="10" height="20" rx="2" />
      <path d="M11 18h2" />
    </svg>
  );
}

/** Loopback bind: guide the user to relaunch Studio. */
function NotReachablePanel({ command }: { command: string }): ReactElement {
  const t = useT();
  const [copied, setCopied] = useState(false);

  const copy = async (): Promise<void> => {
    if (await copyToClipboard(command)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 px-1 text-center">
      <div className="text-muted-foreground bg-muted flex size-12 items-center justify-center rounded-full">
        <HugeiconsIcon icon={WifiDisconnected02Icon} className="size-6" />
      </div>
      <p className="text-muted-foreground text-sm">
        {t("studio.phoneShare.notReachableBody")}
      </p>
      <button
        type="button"
        onClick={copy}
        className={cn(
          "border-border bg-muted/50 hover:bg-muted text-foreground",
          "focus-visible:ring-ring flex w-full items-center justify-between gap-2",
          "cursor-pointer rounded-lg border px-3 py-2 text-left font-mono text-xs",
          "transition-colors focus-visible:outline-none focus-visible:ring-2",
        )}
      >
        <span className="truncate">{command}</span>
        <span className="text-muted-foreground flex shrink-0 items-center gap-1 font-sans text-[11px]">
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            className={cn("size-3.5", copied && "text-emerald-600")}
          />
          {copied
            ? t("studio.phoneShare.commandCopied")
            : t("studio.phoneShare.copyCommand")}
        </span>
      </button>
      <p className="text-muted-foreground/80 text-xs">
        {t("studio.phoneShare.notReachableHint")}
      </p>
    </div>
  );
}

function QrPanel({
  url,
  copied,
  onCopy,
}: {
  url: string;
  copied: boolean;
  onCopy: () => void;
}): ReactElement {
  const t = useT();
  return (
    <>
      <div className="border-border rounded-2xl border bg-white p-4 shadow-sm">
        <QRCodeSVG
          value={url}
          size={196}
          marginSize={0}
          bgColor="#ffffff"
          fgColor="#0b0b0d"
        />
      </div>
      <button
        type="button"
        onClick={onCopy}
        title={url}
        className={cn(
          "text-muted-foreground hover:text-foreground max-w-[16rem] truncate text-xs",
          "cursor-pointer underline-offset-2 hover:underline",
        )}
      >
        {copied ? t("studio.phoneShare.copied") : url}
      </button>
    </>
  );
}

export function PhoneShareButton(): ReactElement {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<PhoneShareResponse | null>(null);
  const [notReachable, setNotReachable] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const openAndShare = async (): Promise<void> => {
    setOpen(true);
    setError(null);
    setNotReachable(null);
    setData(null);
    setCopied(false);
    setLoading(true);
    try {
      setData(await shareTrainingToPhone());
    } catch (e) {
      if (e instanceof PhoneNotReachableError) {
        setNotReachable(e.command);
      } else {
        setError(e instanceof Error ? e.message : t("studio.phoneShare.error"));
      }
    } finally {
      setLoading(false);
    }
  };

  const copy = async (): Promise<void> => {
    if (data && (await copyToClipboard(data.page_url))) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className="h-8 cursor-pointer rounded-full px-3.5 text-xs shadow-sm"
        onClick={openAndShare}
      >
        <PhoneGlyph />
        {t("studio.phoneShare.button")}
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader className="items-center text-center">
            <DialogTitle>
              {notReachable
                ? t("studio.phoneShare.notReachableTitle")
                : t("studio.phoneShare.dialogTitle")}
            </DialogTitle>
            <DialogDescription>
              {notReachable
                ? t("studio.phoneShare.notReachableDescription")
                : t("studio.phoneShare.dialogDescription")}
            </DialogDescription>
          </DialogHeader>

          <div className="flex min-h-60 flex-col items-center justify-center gap-4">
            {loading ? (
              <div className="text-muted-foreground text-sm">
                {t("studio.phoneShare.creatingLink")}
              </div>
            ) : notReachable ? (
              <NotReachablePanel command={notReachable} />
            ) : error ? (
              <div className="text-destructive px-2 text-center text-sm">
                {error}
              </div>
            ) : data ? (
              <QrPanel url={data.page_url} copied={copied} onCopy={copy} />
            ) : null}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
