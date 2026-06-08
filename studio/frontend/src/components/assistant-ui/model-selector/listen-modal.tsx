// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { useTtsPlayer } from "@/features/chat/hooks/use-tts-player";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { SquareIcon, Volume2Icon } from "lucide-react";
import { useEffect, useRef, useState, type FC } from "react";
import type { LoraModelOption } from "./types";

interface ListenModalProps {
  adapter: LoraModelOption;
  onClose: () => void;
}

export const ListenModal: FC<ListenModalProps> = ({ adapter, onClose }) => {
  const [text, setText] = useState("Hello, this is a test.");
  const inputRef = useRef<HTMLInputElement>(null);

  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const isReady = checkpoint === adapter.id && !modelLoading;

  const { isSpeaking, speak, stop } = useTtsPlayer(adapter.audioType ?? null);

  // Focus the input once the model finishes loading.
  useEffect(() => {
    if (isReady) inputRef.current?.focus();
  }, [isReady]);

  const handleClose = () => {
    stop();
    onClose();
  };

  const handlePlay = () => {
    if (isSpeaking) {
      stop();
      return;
    }
    const trimmed = text.trim();
    if (trimmed) speak(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && isReady && !isSpeaking) handlePlay();
  };

  return (
    <Dialog open onOpenChange={(open) => { if (!open) handleClose(); }}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="truncate pr-6">
            Listen: {adapter.name}
          </DialogTitle>
        </DialogHeader>

        {!isReady ? (
          <div className="flex items-center gap-3 py-4 text-sm text-muted-foreground">
            <Spinner className="size-4 shrink-0" />
            <span>Loading model…</span>
          </div>
        ) : (
          <div className="flex gap-2">
            <Input
              ref={inputRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Hello, this is a test."
              className="flex-1"
              disabled={isSpeaking}
            />
            <Button
              type="button"
              variant={isSpeaking ? "destructive" : "default"}
              size="icon"
              onClick={handlePlay}
              disabled={!text.trim() && !isSpeaking}
              aria-label={isSpeaking ? "Stop" : "Play"}
            >
              {isSpeaking ? (
                <SquareIcon className="size-4 fill-current" />
              ) : (
                <Volume2Icon className="size-4" />
              )}
            </Button>
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="ghost" onClick={handleClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
