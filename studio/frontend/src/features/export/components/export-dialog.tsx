import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Switch } from "@/components/ui/switch";
import { ArrowRight01Icon, Key01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion } from "motion/react";
import { EXPORT_METHODS, type ExportMethod } from "../constants";

type Destination = "local" | "hub";

const anim = {
  initial: { height: 0, opacity: 0 },
  animate: { height: "auto" as const, opacity: 1 },
  exit: { height: 0, opacity: 0 },
  transition: { duration: 0.3, ease: [0.25, 0.1, 0.25, 1] as const },
};

interface ExportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  checkpoint: string | null;
  exportMethod: ExportMethod | null;
  quantLevels: string[];
  estimatedSize: string;
  baseModelName: string;
  isAdapter: boolean;
  destination: Destination;
  onDestinationChange: (v: Destination) => void;
  hfUsername: string;
  onHfUsernameChange: (v: string) => void;
  modelName: string;
  onModelNameChange: (v: string) => void;
  hfToken: string;
  onHfTokenChange: (v: string) => void;
  privateRepo: boolean;
  onPrivateRepoChange: (v: boolean) => void;
}

export function ExportDialog({
  open,
  onOpenChange,
  checkpoint,
  exportMethod,
  quantLevels,
  estimatedSize,
  baseModelName,
  isAdapter,
  destination,
  onDestinationChange,
  hfUsername,
  onHfUsernameChange,
  modelName,
  onModelNameChange,
  hfToken,
  onHfTokenChange,
  privateRepo,
  onPrivateRepoChange,
}: ExportDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Export Model</DialogTitle>
          <DialogDescription>Choose where to save your exported model.</DialogDescription>
        </DialogHeader>

        <div className="flex gap-2">
          <Button
            variant={destination === "local" ? "dark" : "outline"}
            onClick={() => onDestinationChange("local")}
            className="flex-1"
          >
            Save Locally
          </Button>
          <Button
            variant={destination === "hub" ? "dark" : "outline"}
            onClick={() => onDestinationChange("hub")}
            className="flex-1"
          >
            Push to Hub
          </Button>
        </div>

        <AnimatePresence>
          {destination === "hub" && (
            <motion.div {...anim} className="overflow-hidden">
              <div className="flex flex-col gap-4 px-0.5">
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Username / Org</label>
                    <Input placeholder="your-username" value={hfUsername} onChange={(e) => onHfUsernameChange(e.target.value)} />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Model Name</label>
                    <Input placeholder="my-model-gguf" value={modelName} onChange={(e) => onModelNameChange(e.target.value)} />
                  </div>
                </div>

                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-muted-foreground">HF Write Token</label>
                    <a
                      href="https://huggingface.co/settings/tokens"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-[11px] text-emerald-600 hover:text-emerald-700 transition-colors"
                    >
                      Get token
                      <HugeiconsIcon icon={ArrowRight01Icon} className="size-3" />
                    </a>
                  </div>
                  <InputGroup>
                    <InputGroupAddon>
                      <HugeiconsIcon icon={Key01Icon} className="size-4" />
                    </InputGroupAddon>
                    <InputGroupInput type="password" placeholder="hf_..." value={hfToken} onChange={(e) => onHfTokenChange(e.target.value)} />
                  </InputGroup>
                  <p className="text-[11px] text-muted-foreground/70">Leave empty if already logged in via CLI.</p>
                </div>

                <div className="flex items-center gap-3">
                  <Switch id="private-repo" size="sm" checked={privateRepo} onCheckedChange={onPrivateRepoChange} />
                  <label htmlFor="private-repo" className="text-xs font-medium cursor-pointer">Private Repository</label>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Summary */}
        <div className="rounded-xl bg-muted/50 p-3 text-xs text-muted-foreground flex flex-col gap-1">
          <div className="flex justify-between">
            <span>Base Model</span>
            <span className="font-medium text-foreground">{baseModelName}</span>
          </div>
          <div className="flex justify-between">
            <span>{isAdapter ? "Checkpoint" : "Model"}</span>
            <span className="font-medium text-foreground">{checkpoint}</span>
          </div>
          <div className="flex justify-between">
            <span>Export Method</span>
            <span className="font-medium text-foreground">
              {EXPORT_METHODS.find((m) => m.value === exportMethod)?.title}
            </span>
          </div>
          {exportMethod === "gguf" && quantLevels.length > 0 && (
            <div className="flex justify-between">
              <span>Quantizations</span>
              <span className="font-medium text-foreground">{quantLevels.join(", ")}</span>
            </div>
          )}
          <div className="flex justify-between">
            <span>Est. size</span>
            <span className="font-medium text-foreground">{estimatedSize}</span>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={() => onOpenChange(false)}>Start Export</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
