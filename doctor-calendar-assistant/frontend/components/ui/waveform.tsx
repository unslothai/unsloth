"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface WaveformProps {
  isActive: boolean;
  barCount?: number;
  className?: string;
}

export function Waveform({ isActive, barCount = 12, className }: WaveformProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-center gap-1 h-16",
        className
      )}
    >
      {[...Array(barCount)].map((_, i) => (
        <motion.div
          key={i}
          className="w-1.5 rounded-full bg-gradient-to-t from-primary/50 to-primary"
          animate={
            isActive
              ? {
                  height: [8, Math.random() * 40 + 16, 8],
                }
              : { height: 8 }
          }
          transition={
            isActive
              ? {
                  duration: 0.4 + Math.random() * 0.3,
                  repeat: Infinity,
                  ease: "easeInOut",
                  delay: i * 0.05,
                }
              : { duration: 0.3 }
          }
        />
      ))}
    </div>
  );
}

interface AudioVisualizerProps {
  audioData?: Uint8Array;
  className?: string;
}

export function AudioVisualizer({ audioData, className }: AudioVisualizerProps) {
  const barCount = 32;
  const bars = audioData
    ? Array.from(audioData).slice(0, barCount)
    : new Array(barCount).fill(0);

  return (
    <div
      className={cn(
        "flex items-end justify-center gap-0.5 h-24",
        className
      )}
    >
      {bars.map((value, i) => (
        <motion.div
          key={i}
          className="w-1 rounded-full bg-gradient-to-t from-primary/30 to-primary"
          style={{ height: `${Math.max(4, (value / 255) * 96)}px` }}
          transition={{ duration: 0.05 }}
        />
      ))}
    </div>
  );
}
