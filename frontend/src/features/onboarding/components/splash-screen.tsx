import { Button } from "@/components/ui/button";
import { motion } from "motion/react";

interface SplashScreenProps {
  onStartOnboarding: () => void;
  onGoToStudio: () => void;
}

export function SplashScreen({
  onStartOnboarding,
  onGoToStudio,
}: SplashScreenProps) {
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-b from-background via-background to-primary/5">
      {/* Mascot */}
      <motion.img
        src="/Sloth emojis/Sloth loca pc.png"
        alt="Sloth mascot"
        className="size-30"
        initial={{ opacity: 0, y: 40, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{
          type: "spring",
          duration: 0.7,
          bounce: 0.3,
          delay: 0.1,
        }}
      />

      {/* Brand text */}
      <motion.div
        className="flex flex-col items-center gap-1 mt-4"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{
          duration: 0.4,
          ease: [0.165, 0.84, 0.44, 1],
          delay: 0.4,
        }}
      >
        <h1 className="text-2xl font-semibold tracking-tight">
          Unsloth Studio
        </h1>
        <p className="text-sm text-muted-foreground">Fine-tune LLMs faster</p>
      </motion.div>

      {/* Buttons */}
      <motion.div
        className="flex flex-col gap-3 mt-8"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{
          duration: 0.4,
          ease: [0.165, 0.84, 0.44, 1],
          delay: 0.8,
        }}
      >
        <Button size="lg" onClick={onStartOnboarding}>
          Start Onboarding
        </Button>
        <Button size="lg" variant="outline" onClick={onGoToStudio}>
          Skip Onboarding
        </Button>
      </motion.div>
    </div>
  );
}
