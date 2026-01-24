/**
 * Audio utilities for recording and playback
 */

export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private stream: MediaStream | null = null;
  private onDataCallback: ((data: ArrayBuffer) => void) | null = null;

  async start(onData: (data: ArrayBuffer) => void): Promise<boolean> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      this.onDataCallback = onData;
      this.audioChunks = [];

      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      this.mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);

          // Convert to ArrayBuffer and send
          const arrayBuffer = await event.data.arrayBuffer();
          if (this.onDataCallback) {
            this.onDataCallback(arrayBuffer);
          }
        }
      };

      // Record in chunks
      this.mediaRecorder.start(500); // 500ms chunks

      return true;
    } catch (error) {
      console.error("Error starting audio recording:", error);
      return false;
    }
  }

  stop(): Blob | null {
    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.audioChunks.length > 0) {
      return new Blob(this.audioChunks, { type: "audio/webm" });
    }

    return null;
  }

  isRecording(): boolean {
    return this.mediaRecorder?.state === "recording";
  }
}

export class AudioPlayer {
  private audioContext: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private isPlaying = false;
  private queue: ArrayBuffer[] = [];

  constructor() {
    if (typeof window !== "undefined") {
      this.audioContext = new AudioContext();
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);
    }
  }

  async play(audioData: ArrayBuffer): Promise<void> {
    if (!this.audioContext || !this.gainNode) return;

    // Resume context if suspended
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }

    try {
      // Decode audio data
      const audioBuffer = await this.audioContext.decodeAudioData(audioData.slice(0));

      // Create source
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);

      this.isPlaying = true;
      source.onended = () => {
        this.isPlaying = false;
        this.playNextInQueue();
      };

      source.start();
    } catch (error) {
      console.error("Error playing audio:", error);
      this.isPlaying = false;
    }
  }

  queueAudio(audioData: ArrayBuffer): void {
    this.queue.push(audioData);
    if (!this.isPlaying) {
      this.playNextInQueue();
    }
  }

  private async playNextInQueue(): Promise<void> {
    if (this.queue.length > 0) {
      const next = this.queue.shift();
      if (next) {
        await this.play(next);
      }
    }
  }

  setVolume(volume: number): void {
    if (this.gainNode) {
      this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
    }
  }

  stop(): void {
    this.queue = [];
    this.isPlaying = false;
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = new AudioContext();
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);
    }
  }
}

// Convert base64 to ArrayBuffer
export function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

// Convert ArrayBuffer to base64
export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
