"use client";

/**
 * Waveform Component - ElevenLabs UI Style
 * Real-time audio visualization using Canvas API
 * Based on https://github.com/elevenlabs/ui
 *
 * NOTE: For the official ElevenLabs waveform, run:
 * npx shadcn@latest add https://ui.elevenlabs.io/r/waveform.json
 * npx shadcn@latest add https://ui.elevenlabs.io/r/live-waveform.json
 */

import { useRef, useEffect, useMemo } from "react";
import { cn } from "@/lib/utils";

interface WaveformProps {
  /** Whether the waveform is actively animating */
  isActive?: boolean;
  /** Number of bars to display */
  barCount?: number;
  /** Color of the bars (CSS color string) */
  color?: string;
  /** Height of the waveform container */
  height?: number;
  /** Gap between bars */
  gap?: number;
  /** Audio analyser data (optional) */
  frequencyData?: Uint8Array;
  className?: string;
}

export function Waveform({
  isActive = false,
  barCount = 24,
  color = "hsl(var(--primary))",
  height = 64,
  gap = 2,
  frequencyData,
  className,
}: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const barsRef = useRef<number[]>([]);

  // Initialize bar heights
  useMemo(() => {
    barsRef.current = Array(barCount).fill(0.1);
  }, [barCount]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas resolution
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const barWidth = (rect.width - gap * (barCount - 1)) / barCount;
    const maxHeight = rect.height * 0.9;
    const minHeight = rect.height * 0.1;

    const animate = () => {
      ctx.clearRect(0, 0, rect.width, rect.height);

      barsRef.current.forEach((_, i) => {
        let targetHeight: number;

        if (frequencyData && frequencyData.length > 0) {
          // Use actual frequency data
          const dataIndex = Math.floor((i / barCount) * frequencyData.length);
          targetHeight = (frequencyData[dataIndex] / 255) * maxHeight;
        } else if (isActive) {
          // Generate random heights for demo
          targetHeight = minHeight + Math.random() * (maxHeight - minHeight);
        } else {
          // Idle state - small bars
          targetHeight = minHeight;
        }

        // Smooth transition
        barsRef.current[i] += (targetHeight - barsRef.current[i]) * 0.2;

        const x = i * (barWidth + gap);
        const barHeight = Math.max(barsRef.current[i], minHeight);
        const y = (rect.height - barHeight) / 2;

        // Draw bar with rounded corners
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, barWidth / 2);
        ctx.fill();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, barCount, color, gap, frequencyData]);

  return (
    <canvas
      ref={canvasRef}
      className={cn("w-full", className)}
      style={{ height }}
    />
  );
}

/**
 * LiveWaveform - Real-time audio visualization
 * Connects to an audio stream and visualizes frequency data
 */
interface LiveWaveformProps extends Omit<WaveformProps, "frequencyData"> {
  /** MediaStream to analyze */
  stream?: MediaStream;
}

export function LiveWaveform({ stream, ...props }: LiveWaveformProps) {
  const analyserRef = useRef<AnalyserNode>();
  const dataArrayRef = useRef<Uint8Array>();

  useEffect(() => {
    if (!stream) return;

    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();

    analyser.fftSize = 64;
    source.connect(analyser);

    analyserRef.current = analyser;
    dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

    return () => {
      audioContext.close();
    };
  }, [stream]);

  useEffect(() => {
    if (!analyserRef.current || !dataArrayRef.current) return;

    const update = () => {
      if (analyserRef.current && dataArrayRef.current) {
        analyserRef.current.getByteFrequencyData(dataArrayRef.current);
      }
      requestAnimationFrame(update);
    };

    update();
  }, [stream]);

  return (
    <Waveform
      {...props}
      frequencyData={dataArrayRef.current}
      isActive={!!stream}
    />
  );
}
