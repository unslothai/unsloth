import {
  AnimatedSpan,
  Terminal,
  TypingAnimation,
} from "@/components/ui/terminal"
import type { ReactElement } from "react"

type TrainingStartOverlayProps = {
  message: string
  currentStep: number
}

export function TrainingStartOverlay({
  message,
  currentStep,
}: TrainingStartOverlayProps): ReactElement {
  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center rounded-2xl bg-background/45 backdrop-blur-[1px]">
      <div className="flex w-[860px] max-w-[calc(100%-2rem)] flex-col items-center gap-4">
        <img
          src="/Sloth emojis/large sloth wave.png"
          alt="Unsloth mascot"
          className="size-24 animate-bounce object-contain"
        />
        <Terminal
          className="w-full min-h-[390px] rounded-2xl px-7 py-6 text-left"
          startOnView={false}
        >
          <TypingAnimation
            duration={36}
            className="bg-gradient-to-r from-emerald-300 via-lime-300 to-teal-300 bg-clip-text font-semibold text-transparent"
          >
            {"> unsloth training starts..."}
          </TypingAnimation>
          <AnimatedSpan className="my-2">
            <pre className="whitespace-pre text-left text-muted-foreground">{`==((====))==
\\\\   /|
O^O/ \\_/ \\
\\        /
 "-____-"`}</pre>
          </AnimatedSpan>
          <TypingAnimation duration={44}>
            {"> Preparing model and dataset..."}
          </TypingAnimation>
          <TypingAnimation duration={44}>
            {"> We are getting everything ready for your run..."}
          </TypingAnimation>
          <TypingAnimation duration={44}>
            {"> Did you know, Mugi is actually short for \"Mugiwara\" xd"}
          </TypingAnimation>
          <AnimatedSpan className="mt-2 text-muted-foreground">
            {`> ${message || "starting training..."} | waiting for first step... (${currentStep})`}
          </AnimatedSpan>
        </Terminal>
      </div>
    </div>
  )
}
