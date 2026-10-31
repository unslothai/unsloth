// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { useAui } from "@assistant-ui/react";
import { toast } from "sonner";
import { MicIcon } from "@/lib/mic-icon";
import { isTauri } from "@/lib/api-base";
import { MAX_AUDIO_SIZE } from "@/lib/audio-utils";
// eslint-disable-next-line no-restricted-imports -- The recorder is a leaf module shared with Audio.
import { PcmRecorder } from "@/features/chat/adapters/pcm-recorder";
// eslint-disable-next-line no-restricted-imports -- Reuse dictation's device fallback policy.
import { isMissingDeviceError } from "@/features/chat/adapters/studio-web-speech-dictation-adapter";
// eslint-disable-next-line no-restricted-imports -- These microphone helpers have no feature-barrel export.
import {
  micAudioConstraints,
  micErrorMessage,
} from "@/features/audio/hooks/audio-source-state";
import { useVoiceSettingsStore } from "@/features/settings";
import { useChatRuntimeStore } from "@/features/chat";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

/** Offer recording for the same models that accept audio attachments. */
export function ChatAudioRecording({ disabled }: { disabled?: boolean }) {
  const audioEnabled = useChatRuntimeStore((state) => {
    const checkpoint = state.params.checkpoint;
    return (
      !checkpoint ||
      state.modelLoading ||
      Boolean(
        state.models.find((model) => model.id === checkpoint)?.hasAudioInput,
      )
    );
  });
  if (
    !audioEnabled ||
    typeof navigator === "undefined" ||
    typeof navigator.mediaDevices?.getUserMedia !== "function"
  )
    return null;
  return (
    <ChatAudioRecorder
      key={disabled ? "disabled" : "enabled"}
      disabled={disabled}
    />
  );
}

/** Record a WAV for the normal attachment path, without starting speech transcription. */
function ChatAudioRecorder({ disabled }: { disabled?: boolean }) {
  const aui = useAui();
  const [open, setOpen] = useState(false);
  const [phase, setPhase] = useState<
    "idle" | "acquiring" | "recording" | "attaching"
  >("idle");
  const recorderRef = useRef<PcmRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const ticketRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const discardRef = useRef(false);
  const mountedRef = useRef(true);

  /** Release the microphone and cancel any in-flight permission request. */
  const discard = useCallback(() => {
    ticketRef.current += 1;
    discardRef.current = true;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    const recorder = recorderRef.current;
    recorderRef.current = null;
    if (recorder?.state === "recording") recorder.stop();
    for (const track of streamRef.current?.getTracks() ?? []) track.stop();
    streamRef.current = null;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      discard();
    };
  }, [discard]);

  /** Start the selected microphone, falling back if the saved device is unavailable. */
  const start = async () => {
    if (phase !== "idle") return;
    const ticket = ++ticketRef.current;
    setPhase("acquiring");
    let stream: MediaStream;
    try {
      const deviceId = useVoiceSettingsStore.getState().micDeviceId;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: micAudioConstraints(deviceId),
        });
      } catch (error) {
        if (deviceId === "default" || !isMissingDeviceError(error)) throw error;
        stream = await navigator.mediaDevices.getUserMedia({
          audio: micAudioConstraints(null),
        });
      }
    } catch (error) {
      if (ticket !== ticketRef.current || !mountedRef.current) return;
      setPhase("idle");
      toast.error(micErrorMessage(error, isTauri));
      return;
    }
    if (ticket !== ticketRef.current || !mountedRef.current) {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    streamRef.current = stream;
    discardRef.current = false;
    try {
      const recorder = new PcmRecorder(stream);
      recorderRef.current = recorder;
      const chunks: Blob[] = [];
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size) chunks.push(event.data);
      });
      recorder.addEventListener("stop", () => {
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = null;
        recorderRef.current = null;
        for (const track of stream.getTracks()) track.stop();
        streamRef.current = null;
        if (discardRef.current || !mountedRef.current) return;
        const audio = new File(chunks, "Recording.wav", { type: "audio/wav" });
        if (!audio.size) {
          setPhase("idle");
          toast.error("No audio was recorded. Try again.");
          return;
        }
        setPhase("attaching");
        const composer = aui.composer();
        void composer
          .addAttachment(audio)
          .then(async () => {
            // A chat change or cancel cannot abort addAttachment, so remove a late result.
            if (!mountedRef.current || ticket !== ticketRef.current) {
              const added = composer
                .getState()
                .attachments.find((item) => item.file === audio);
              if (added) await composer.attachment({ id: added.id }).remove();
              return;
            }
            setPhase("idle");
            setOpen(false);
          })
          .catch(() => {
            if (mountedRef.current && ticket === ticketRef.current)
              setPhase("idle");
          });
      });
      recorder.start();
      setPhase("recording");
      // Stop before the attachment limit, including the recorder's in-flight buffer.
      timerRef.current = setTimeout(
        () => recorder.stop(),
        Math.min(300, recorder.secondsWithin(MAX_AUDIO_SIZE)) * 1000,
      );
    } catch {
      discard();
      setPhase("idle");
      toast.error("This browser cannot record audio. Upload a file instead.");
    }
  };

  /** Closing or changing chats discards a recording and releases the device. */
  const changeOpen = (next: boolean) => {
    if (!next) {
      discard();
      setPhase("idle");
    }
    setOpen(next);
  };

  return (
    <>
      <TooltipIconButton
        tooltip="Record audio for model"
        aria-label="Record audio for model"
        type="button"
        disabled={disabled}
        onClick={() => setOpen(true)}
        className="size-9 rounded-full text-foreground"
      >
        <MicIcon className="size-5" />
      </TooltipIconButton>
      <Dialog open={open} onOpenChange={changeOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Record audio for model</DialogTitle>
            <DialogDescription>
              Record an audio clip to attach to your message. It will be sent to
              the model without transcription.
            </DialogDescription>
          </DialogHeader>
          <p role="status">
            {phase === "recording"
              ? "Recording…"
              : phase === "acquiring"
                ? "Waiting for microphone…"
                : phase === "attaching"
                  ? "Attaching recording…"
                  : "Start recording when you are ready."}
          </p>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => changeOpen(false)}
            >
              Cancel
            </Button>
            {phase === "recording" ? (
              <Button type="button" onClick={() => recorderRef.current?.stop()}>
                Stop and attach
              </Button>
            ) : (
              <Button
                type="button"
                disabled={phase !== "idle"}
                onClick={() => void start()}
              >
                Start recording
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
