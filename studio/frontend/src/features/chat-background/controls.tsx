// SPDX-License-Identifier: AGPL-3.0-only
import { useRef } from "react";
import { Cancel01Icon, Upload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { useT, type TranslationKey } from "@/i18n";
// Shared settings primitives: avoid importing the settings barrel into this feature.
// eslint-disable-next-line no-restricted-imports
import { SettingsSection } from "@/features/settings/components/settings-section";
// eslint-disable-next-line no-restricted-imports
import { SettingsRow } from "@/features/settings/components/settings-row";
import { ChatBackground } from "./index";
import {
	DEFAULT_ADJUSTMENTS,
	EFFECTS,
	IMAGE_SIZING,
	type ImageSizing,
	POSITIONS,
	normalizedSize,
	validUpload,
	type BackgroundEffect,
} from "./model";
import {
	beginBackgroundUpload,
	failBackgroundUpload,
	finishBackgroundUpload,
	patchBackground,
	resetChatBackground,
	useChatBackground,
} from "./storage";
import { canvasBlob, decodeImage } from "./image";

const labels: Record<BackgroundEffect, TranslationKey> = {
	None: "chatBackground.none",
	Dither: "chatBackground.dither",
	ASCII: "chatBackground.ascii",
	Halftone: "chatBackground.halftone",
	Scanlines: "chatBackground.scanlines",
};
const sizingLabels: Record<ImageSizing, TranslationKey> = {
	cover: "chatBackground.fill",
	contain: "chatBackground.fit",
	stretch: "chatBackground.stretch",
	original: "chatBackground.original",
};
const errors = {
	storage: "chatBackground.storageError",
	file: "chatBackground.fileError",
	dimensions: "chatBackground.dimensionsError",
	decode: "chatBackground.decodeError",
	effect: "chatBackground.effectError",
} as const;
export function ChatBackgroundControls() {
	const t = useT();
	const { value, error, ready, busy } = useChatBackground();
	const input = useRef<HTMLInputElement>(null);
	const adjustments = value ?? DEFAULT_ADJUSTMENTS;
	return (
		<SettingsSection
			title={t("chatBackground.title")}
			action={
				<div className="flex max-w-full flex-wrap items-center justify-end gap-1">
					{value && (
						<Button
							// Same hover wash as the outline (btn-fill) Replace button beside it.
							className="text-foreground hover:bg-[color-mix(in_srgb,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] dark:hover:bg-[color-mix(in_srgb,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)]"
							type="button"
							variant="ghost"
							size="sm"
							disabled={busy}
							onClick={() => void resetChatBackground()}
						>
							<HugeiconsIcon
								icon={Cancel01Icon}
								strokeWidth={1.75}
								className="size-4"
								aria-hidden="true"
							/>
							{t("chatBackground.remove")}
						</Button>
					)}

					<Button
						className="text-foreground"
						type="button"
						variant="outline"
						size="sm"
						disabled={!ready || busy}
						onClick={() => input.current?.click()}
					>
						<HugeiconsIcon
							icon={Upload01Icon}
							strokeWidth={1.75}
							className="size-4"
							aria-hidden="true"
						/>
						{busy
							? t("common.loading")
							: t(value ? "chatBackground.replace" : "chatBackground.upload")}
					</Button>
				</div>
			}
		>
			<input
				ref={input}
				className="sr-only"
				tabIndex={-1}
				aria-label={t("chatBackground.upload")}
				type="file"
				accept="image/jpeg,image/png,image/webp"
				disabled={!ready || busy}
				onChange={async (event) => {
					const file = event.target.files?.[0];
					event.target.value = "";
					if (!file) return;
					const ticket = beginBackgroundUpload();
					if (!validUpload(file)) {
						failBackgroundUpload(ticket, "file");
						return;
					}
					try {
						const decoded = await decodeImage(file);
						try {
							const [width, height] = normalizedSize(
								decoded.width,
								decoded.height,
							);
							const canvas = document.createElement("canvas");
							canvas.width = width;
							canvas.height = height;
							const ctx = canvas.getContext("2d");
							if (!ctx) throw new Error("decode");
							ctx.drawImage(decoded.source, 0, 0, width, height);
							await finishBackgroundUpload(ticket, {
								...DEFAULT_ADJUSTMENTS,
								blob: await canvasBlob(canvas),
							});
						} finally {
							decoded.close();
						}
					} catch (error) {
						failBackgroundUpload(
							ticket,
							error instanceof Error && error.message === "dimensions"
								? "dimensions"
								: "decode",
						);
					}
				}}
			/>
			{error && (
				<p role="alert" className="mt-3 mb-3 text-sm text-destructive">
					{t(errors[error])}
				</p>
			)}
			<div
				className={`${error ? "" : "mt-3 "}grid grid-cols-1 gap-3 sm:grid-cols-2`}
				data-wallpaper-previews=""
			>
				{[false, true].map((conversation) => (
					<div key={String(conversation)} className="min-w-0">
						<p className="mb-2 text-xs text-muted-foreground">
							{t(
								conversation
									? "chatBackground.conversation"
									: "chatBackground.newChat",
							)}
						</p>
						<div
							className="wallpaper-preview relative h-48 overflow-hidden rounded-xl border border-border bg-background"
							data-wallpaper-conversation={conversation || undefined}
						>
							<ChatBackground />
							{!value && (
								<p className="absolute inset-x-4 top-4 text-center text-xs text-muted-foreground">
									{t("chatBackground.noBackground")}
								</p>
							)}
							<div className="wallpaper-preview-content absolute inset-x-4 bottom-4 space-y-3 text-xs">
								{conversation && (
									<p className="wallpaper-preview-prose rounded-lg p-3">
										{t("chatBackground.response")}
									</p>
								)}
								<div className="wallpaper-preview-composer rounded-xl p-3">
									{t("chatBackground.prompt")}
								</div>
							</div>
						</div>
					</div>
				))}
			</div>
			{value && (
				<>
					<SettingsRow label={t("chatBackground.effect")} className="mt-3 items-start">
						<Select
							disabled={!value || busy}
							value={adjustments.effect}
							onValueChange={(effect) =>
								patchBackground({ effect: effect as BackgroundEffect })
							}
						>
							<SelectTrigger
								aria-label={t("chatBackground.effect")}
								className="w-52"
							>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								{EFFECTS.map((effect) => (
									<SelectItem key={effect} value={effect}>
										{t(labels[effect])}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</SettingsRow>
					<SettingsRow label={t("chatBackground.sizing")} className="items-start">
						<Select
							disabled={!value || busy}
							value={adjustments.sizing}
							onValueChange={(sizing) =>
								patchBackground({ sizing: sizing as ImageSizing })
							}
						>
							<SelectTrigger
								aria-label={t("chatBackground.sizing")}
								className="w-52"
							>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								{IMAGE_SIZING.map((sizing) => (
									<SelectItem key={sizing} value={sizing}>
										{t(sizingLabels[sizing])}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</SettingsRow>
					{(["visibility", "blur", "zoom"] as const).map((key) => (
						<SettingsRow
							key={key}
							className="items-start"
							label={t(
								key === "zoom"
									? "chatBackground.zoom"
									: key === "blur"
										? "chatBackground.blur"
										: "chatBackground.visibility",
							)}
						>
							<div className="flex w-52 items-center gap-3">
								<Slider
									disabled={!value || busy}
									aria-label={t(
										key === "zoom"
											? "chatBackground.zoom"
											: key === "blur"
												? "chatBackground.blur"
												: "chatBackground.visibility",
									)}
									min={key === "zoom" ? 100 : 0}
									max={key === "zoom" ? 300 : key === "blur" ? 20 : 100}
									step={1}
									value={[adjustments[key]]}
									onValueChange={([v]) => patchBackground({ [key]: v })}
								/>
								<span className="w-12 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
									{adjustments[key]}
									{key === "blur" ? "px" : "%"}
								</span>
							</div>
						</SettingsRow>
					))}
					<SettingsRow label={t("chatBackground.outline")} className="items-start">
						<Switch
							aria-label={t("chatBackground.outline")}
							checked={adjustments.outline}
							disabled={!value || busy}
							onCheckedChange={(outline) => patchBackground({ outline })}
						/>
					</SettingsRow>
					<SettingsRow label={t("chatBackground.fade")} className="items-start">
						<Switch
							aria-label={t("chatBackground.fade")}
							checked={adjustments.fade}
							disabled={!value || busy}
							onCheckedChange={(fade) => patchBackground({ fade })}
						/>
					</SettingsRow>
					<SettingsRow
						label={t("chatBackground.position")}
						className="items-start"
					>
						<div className="flex w-52 justify-end">
							<div
								className="grid w-fit grid-cols-3 gap-1"
								role="group"
								aria-label={t("chatBackground.position")}
							>
								{POSITIONS.map((position) => {
									const [x, y] = position
										.split(" ")
										.map((v) => Number.parseInt(v));
									return (
										<button
											key={position}
											type="button"
											disabled={!value || busy}
											aria-pressed={adjustments.position === position}
											aria-label={t("chatBackground.positionValue", { x, y })}
											title={t("chatBackground.positionValue", { x, y })}
											className="flex size-6 items-center justify-center rounded-full border border-border text-muted-foreground hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring aria-pressed:border-primary aria-pressed:bg-primary/10 aria-pressed:text-primary disabled:opacity-40"
											onClick={() => patchBackground({ position })}
										>
											<span className="size-1 rounded-full bg-current" />
										</button>
									);
								})}
							</div>
						</div>
					</SettingsRow>
				</>
			)}
		</SettingsSection>
	);
}
