// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState, type ReactElement } from "react";
import { QRCodeSVG } from "qrcode.react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import {
  shareTrainingToPhone,
  type PhoneShareResponse,
} from "@/features/training";

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

// "View on phone": QR to a read-only training dashboard (routes/phone.py).
export function PhoneShareButton(): ReactElement {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<PhoneShareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const openAndShare = async (): Promise<void> => {
    setOpen(true);
    setError(null);
    setData(null);
    setCopied(false);
    setLoading(true);
    try {
      setData(await shareTrainingToPhone());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create a phone link.");
    } finally {
      setLoading(false);
    }
  };

  const copy = async (): Promise<void> => {
    if (!data) return;
    try {
      await navigator.clipboard.writeText(data.page_url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard may be unavailable
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
        View on phone
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader className="items-center text-center">
            <DialogTitle>Watch on your phone</DialogTitle>
            <DialogDescription>
              Scan with your phone camera. Same Wi-Fi only · view-only · the link
              expires automatically.
            </DialogDescription>
          </DialogHeader>

          <div className="flex h-60 flex-col items-center justify-center gap-4">
            {loading ? (
              <div className="text-muted-foreground text-sm">Creating link…</div>
            ) : error ? (
              <div className="text-destructive px-2 text-center text-sm">
                {error}
              </div>
            ) : data ? (
              <>
                <div className="rounded-2xl border border-border bg-white p-4 shadow-sm">
                  <QRCodeSVG
                    value={data.page_url}
                    size={196}
                    marginSize={0}
                    bgColor="#ffffff"
                    fgColor="#0b0b0d"
                  />
                </div>
                <button
                  type="button"
                  onClick={copy}
                  title={data.page_url}
                  className={cn(
                    "text-muted-foreground hover:text-foreground max-w-[16rem] truncate text-xs",
                    "cursor-pointer underline-offset-2 hover:underline",
                  )}
                >
                  {copied ? "Copied ✓" : data.page_url}
                </button>
              </>
            ) : null}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
