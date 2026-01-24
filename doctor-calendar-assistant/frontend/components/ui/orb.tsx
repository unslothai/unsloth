"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface OrbProps {
  isActive: boolean;
  isListening: boolean;
  isSpeaking: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function Orb({
  isActive,
  isListening,
  isSpeaking,
  size = "lg",
  className,
}: OrbProps) {
  const sizeClasses = {
    sm: "w-24 h-24",
    md: "w-32 h-32",
    lg: "w-48 h-48",
  };

  const getGradient = () => {
    if (isSpeaking) {
      return "from-green-400 via-emerald-500 to-teal-500";
    }
    if (isListening) {
      return "from-blue-400 via-primary to-indigo-500";
    }
    if (isActive) {
      return "from-blue-500 via-primary to-blue-400";
    }
    return "from-gray-600 via-gray-500 to-gray-600";
  };

  return (
    <div className={cn("relative", sizeClasses[size], className)}>
      {/* Pulse rings */}
      {isActive && (
        <>
          <motion.div
            className={cn(
              "absolute inset-0 rounded-full bg-gradient-to-r opacity-30",
              getGradient()
            )}
            animate={{
              scale: [1, 1.4],
              opacity: [0.3, 0],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeOut",
            }}
          />
          <motion.div
            className={cn(
              "absolute inset-0 rounded-full bg-gradient-to-r opacity-30",
              getGradient()
            )}
            animate={{
              scale: [1, 1.4],
              opacity: [0.3, 0],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeOut",
              delay: 0.5,
            }}
          />
        </>
      )}

      {/* Main orb */}
      <motion.div
        className={cn(
          "absolute inset-0 rounded-full bg-gradient-to-br shadow-2xl",
          getGradient()
        )}
        animate={
          isActive
            ? {
                scale: isSpeaking ? [1, 1.05, 1] : isListening ? [1, 1.02, 1] : 1,
              }
            : { scale: 1 }
        }
        transition={{
          duration: isSpeaking ? 0.3 : 1.5,
          repeat: isActive ? Infinity : 0,
          ease: "easeInOut",
        }}
      >
        {/* Inner glow */}
        <div className="absolute inset-4 rounded-full bg-gradient-to-br from-white/20 to-transparent" />

        {/* Shine effect */}
        <div className="absolute top-6 left-6 w-12 h-12 rounded-full bg-white/30 blur-xl" />
      </motion.div>

      {/* Activity indicator */}
      {isActive && (
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="flex gap-1">
            {[...Array(3)].map((_, i) => (
              <motion.div
                key={i}
                className="w-2 h-2 rounded-full bg-white"
                animate={
                  isListening || isSpeaking
                    ? {
                        y: [0, -8, 0],
                        opacity: [0.5, 1, 0.5],
                      }
                    : { y: 0, opacity: 0.5 }
                }
                transition={{
                  duration: 0.6,
                  repeat: Infinity,
                  delay: i * 0.15,
                }}
              />
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
