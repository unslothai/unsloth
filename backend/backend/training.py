"""
Training backend and UI integration
"""
import gradio as gr
import matplotlib.pyplot as plt
from typing import Dict, Any, Generator, Tuple
import logging

from .trainer import get_trainer, TrainingProgress

logger = logging.getLogger(__name__)

# Plot styling constants
PLOT_WIDTH = 8  # Inches
PLOT_HEIGHT = 3.5  # Inches


class TrainingBackend:
    """
    Training orchestration and UI integration.
    Handles both text and vision models, LoRA and full finetuning.
    """

    def __init__(self):
        self.trainer = get_trainer()

        # Training Metrics
        self.loss_history = []
        self.lr_history = []
        self.step_history = []
        self.current_theme = "light"

        self.trainer.add_progress_callback(self._on_progress_update)

        logger.info("TrainingBackend initialized")

    def _on_progress_update(self, progress: TrainingProgress):
        """Callback for progress updates"""
        if progress.step > 0 and progress.loss > 0:
            self.loss_history.append(progress.loss)
            self.lr_history.append(progress.learning_rate)
            self.step_history.append(progress.step)

    def start_training(self,
                       # Model parameters
                       model_name: str,
                       training_type: str,  # NEW: "LoRA/QLoRA" or "Full Finetuning"
                       hf_token: str,
                       load_in_4bit: bool,
                       max_seq_length: int,

                       # Dataset parameters
                       hf_dataset: str,
                       local_datasets: list,
                       format_type: str,  # CHANGED: was data_template

                       # Training parameters
                       num_epochs: int,
                       learning_rate: str,
                       batch_size: int,
                       gradient_accumulation_steps: int,
                       warmup_steps: int,          # May be None even without default
                       warmup_ratio: float,        # May be None even without default
                       max_steps: int,
                       save_steps: int,
                       weight_decay: float,
                       random_seed: int,
                       packing: bool,

                       # LoRA parameters
                       use_lora: bool,  # Should be derived from training_type
                       lora_r: int,
                       lora_alpha: int,
                       lora_dropout: float,
                       target_modules: list,
                       gradient_checkpointing: str,
                       use_rslora: bool,
                       use_loftq: bool,
                       train_on_completions: bool,

                       # NEW: Vision-specific LoRA parameters
                       finetune_vision_layers: bool,
                       finetune_language_layers: bool,
                       finetune_attention_modules: bool,
                       finetune_mlp_modules: bool,

                       # Logging parameters
                       enable_wandb: bool,
                       wandb_token: str,
                       wandb_project: str,
                       enable_tensorboard: bool,
                       tensorboard_dir: str,
                       optim: str = "adamw_8bit",
                       lr_scheduler_type: str = "linear") -> Generator[Tuple, None, None]:
        """
        Start training - yields UI updates as generator.

        Yields:
            Tuple of (start_btn_update, stop_btn_update, progress_visible, config_visible)
        """
        try:
            # Reset stop flag and clear history
            self.trainer.should_stop = False
            self.loss_history = []
            self.lr_history = []
            self.step_history = []
            import time
            output_dir = f"./outputs/{model_name.replace('/', '_')}_{int(time.time())}"

            # NEW: Derive use_lora from training_type
            use_lora_actual = (training_type == "LoRA/QLoRA")
            if use_lora_actual: print("using Lora")
            else: print("using full finetuning")
            logger.info(f"Starting training - Type: {training_type}, Model: {model_name}")

            # Yield initial status - buttons toggle immediately
            yield (
                gr.update(interactive=False),  # Start button disabled
                gr.update(interactive=True),   # Stop button enabled
                gr.update(visible=True),       # Training progress visible
                #gr.update(visible=False)       # Config selection hidden
            )

            # ========== LOAD MODEL ==========
            logger.info("Loading model...")
            success = self.trainer.load_model(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit if use_lora_actual else False,  # Only 4bit for LoRA
                hf_token=hf_token if hf_token.strip() else None
            )

            if not success or self.trainer.should_stop:
                logger.error("Failed to load model or stopped by user")
                return

            # Capture if this is a vision model
            #self.current_training_session['is_vlm'] = self.trainer.is_vlm

            yield (
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(visible=True),
                #gr.update(visible=False)
            )

            # ========== PREPARE MODEL FOR TRAINING ==========
            if use_lora_actual:
                logger.info("Preparing model with LoRA...")
                success = self.trainer.prepare_model_for_training(
                    use_lora=True,
                    # Vision-specific parameters
                    finetune_vision_layers=finetune_vision_layers,
                    finetune_language_layers=finetune_language_layers,
                    finetune_attention_modules=finetune_attention_modules,
                    finetune_mlp_modules=finetune_mlp_modules,
                    # Standard LoRA parameters
                    target_modules=target_modules,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    use_gradient_checkpointing=gradient_checkpointing,
                    use_rslora=use_rslora,
                    use_loftq=use_loftq
                )
            else:
                logger.info("Preparing model for full finetuning...")
                success = self.trainer.prepare_model_for_training(
                    use_lora=False  # Full finetuning
                )

            if not success or self.trainer.should_stop:
                logger.error("Failed to prepare model or stopped by user")
                return

            yield (
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(visible=True),
                #gr.update(visible=False)
            )

            # ========== LOAD DATASET ==========
            logger.info("Loading dataset...")
            #breakpoint()
            dataset = self.trainer.load_and_format_dataset(
                dataset_source=hf_dataset if hf_dataset.strip() else None,
                format_type=format_type,
                local_datasets=local_datasets if local_datasets else None
            )

            if dataset is None or self.trainer.should_stop:
                logger.error("Failed to load dataset or stopped by user")
                return

            yield (
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(visible=True),
                #gr.update(visible=False)
            )

            # ========== START TRAINING ==========
            # Convert learning rate string to float
            try:
                lr_value = float(learning_rate)
            except ValueError:
                logger.error(f"Invalid learning rate: {learning_rate}")
                self.trainer._update_progress(
                    error=f"Invalid learning rate: {learning_rate}",
                    is_training=False
                )
                return

            logger.info("Starting training worker thread...")
            success = self.trainer.start_training(
                dataset=dataset,
                #output_dir=f"./outputs/{model_name.replace('/', '_')}_{int(__import__('time').time())}",
                output_dir=output_dir,
                num_epochs=num_epochs,
                learning_rate=lr_value,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                warmup_ratio=warmup_ratio,
                max_steps=max_steps if max_steps > 0 else 0,
                save_steps=save_steps if save_steps > 0 else 0,
                weight_decay=weight_decay,
                random_seed=random_seed,
                packing=packing,
                train_on_completions=train_on_completions,
                enable_wandb=enable_wandb,
                wandb_project=wandb_project,
                wandb_token=wandb_token if wandb_token.strip() else None,
                enable_tensorboard=enable_tensorboard,
                tensorboard_dir=tensorboard_dir,
                max_seq_length=max_seq_length,  # Pass through for config
                optim=optim,
                lr_scheduler_type=lr_scheduler_type,
            )

            if not success:
                logger.error("Failed to start training")
                yield (
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update(visible=False),
                    #gr.update(visible=True)
                )

        except Exception as e:
            logger.error(f"Error in start_training: {e}", exc_info=True)
            self.trainer._update_progress(
                error=str(e),
                is_training=False
            )
            yield (
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(visible=False),
                #gr.update(visible=True)
            )

    def stop_training(self) -> Tuple:
        """
        Stop ongoing training.

        Returns:
            Tuple of (start_btn_update, stop_btn_update, progress_visible, config_visible)
        """
        try:
            logger.info("Stopping training...")
            self.trainer.stop_training()

            return (
                gr.update(interactive=True),   # Start button enabled
                gr.update(interactive=False),  # Stop button disabled
                gr.update(visible=False),      # Training progress hidden
                #gr.update(visible=True)        # Config selection visible
            )
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return (
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(visible=False),
                #gr.update(visible=True)
            )

    def get_training_status(self, theme: str = "light") -> Tuple[plt.Figure, gr.update, gr.update, gr.update]:
        """
        Get current training status and loss plot.

        Args:
            theme: "light" or "dark" for plot styling

        Returns:
            Tuple of (plot, start_btn, stop_btn, progress_visible)
        """

        try:
            progress = self.trainer.get_training_progress()

            # If not training and not completed, return no updates
            if not (progress.is_training or progress.is_completed or progress.error):
                return (None, gr.update(), gr.update(), gr.update())

            # Generate plot
            plot = self._create_loss_plot(progress, theme)

            # If completed or error, enable start button
            if progress.is_completed or progress.error:
                return (
                    plot,
                    gr.update(interactive=True),   # Start button enabled
                    gr.update(interactive=False),  # Stop button disabled
                    gr.update(visible=True),       # Training progress visible
                )

            # Still training - no button updates
            return (plot, gr.update(), gr.update(), gr.update())

        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return (None, gr.update(), gr.update(), gr.update())

    def refresh_plot_for_theme(self, theme: str) -> plt.Figure:
        """
        Refresh plot with new theme.

        Args:
            theme: "light" or "dark"

        Returns:
            Updated matplotlib figure
        """
        if theme and isinstance(theme, str) and theme in ['light', 'dark']:
            self.current_theme = theme

        # Always generate plot if we have loss history
        if self.loss_history:
            progress = self.trainer.get_training_progress()
            return self._create_loss_plot(progress, self.current_theme)

        return None

    def is_training_active(self) -> bool:
        """
        Check if training is currently active (from load_model start to completion/error).
        
        Returns:
            True if training is in progress, False otherwise
        """
        try:
            progress = self.trainer.get_training_progress()
            # Training is active if is_training is True
            # Also check if we're in loading/preparation phase (status_message indicates activity)
            is_active = progress.is_training
            # Also consider it active if we have a status message indicating loading/preparation
            # but haven't completed or errored yet
            if not is_active and not progress.is_completed and not progress.error:
                status = progress.status_message or ""
                if any(keyword in status.lower() for keyword in ["loading", "preparing", "training"]):
                    is_active = True
            return is_active
        except Exception as e:
            logger.error(f"Error checking training state: {e}")
            return False

    def _create_loss_plot(self, progress: TrainingProgress, theme: str = "light") -> plt.Figure:
            """
            Create training loss plot with theme-aware styling.

            Args:
                progress: Current training progress
                theme: "light" or "dark"

            Returns:
                Matplotlib figure
            """
            plt.close('all')

            # Theme-specific styling
            LIGHT_STYLE = {
                "facecolor": "#ffffff",
                "grid_color": "#d1d5db",
                "line": "#16b88a",
                "text": "#1f2937",
                "empty_text": "#6b7280"
            }
            DARK_STYLE = {
                "facecolor": "#292929",
                "grid_color": "#404040",
                "line": "#4ade80",
                "text": "#e5e7eb",
                "empty_text": "#9ca3af"
            }

            style = LIGHT_STYLE if theme == "light" else DARK_STYLE

            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
            fig.patch.set_facecolor(style["facecolor"])
            ax.set_facecolor(style["facecolor"])

            if self.loss_history:
                steps = self.step_history
                losses = self.loss_history
                scatter_color = "#60a5fa"
                # Scatter plot for raw loss points
                ax.scatter(
                    steps,
                    losses,
                    s=16,
                    alpha=0.6,
                    color=scatter_color,
                    linewidths=0,
                    label="Training Loss (raw)",
                )

                # Moving average line overlay (trailing window)
                MA_WINDOW = 20  # adjust smoothing aggressiveness
                window = min(MA_WINDOW, len(losses))

                if window >= 2:
                    cumsum = [0.0]
                    for v in losses:
                        cumsum.append(cumsum[-1] + float(v))

                    ma = []
                    for i in range(len(losses)):
                        start = max(0, i - window + 1)
                        denom = i - start + 1
                        ma.append((cumsum[i + 1] - cumsum[start]) / denom)

                    ax.plot(
                        steps,
                        ma,
                        color=style["line"],
                        linewidth=2.5,
                        alpha=0.95,
                        label=f"Moving Avg ({ma[-1]:.4f})",
                    )

                    leg = ax.legend(frameon=False, fontsize=9)
                    for t in leg.get_texts():
                        t.set_color(style["text"])

                ax.set_xlabel('Steps', fontsize=10, color=style["text"])
                ax.set_ylabel('Loss', fontsize=10, color=style["text"])

                # Build status message for title
                if progress.error:
                    title = f"Error: {progress.error}"
                elif progress.is_completed:
                    title = f"Training completed! Final loss: {progress.loss:.4f}"
                elif progress.status_message:
                    title = progress.status_message
                elif progress.step > 0:
                    title = f"Epoch: {progress.epoch} | Step: {progress.step}/{progress.total_steps} | Loss: {progress.loss:.4f}"
                else:
                    title = "Training Loss"

                ax.set_title(title, fontsize=11, fontweight='bold',
                            pad=10, color=style["text"])

                # Style grid and spines
                ax.grid(True, alpha=0.4, linestyle='--', color=style["grid_color"])
                ax.tick_params(colors=style["text"], which='both')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color(style["text"])
                ax.spines['left'].set_color(style["text"])
            else:
                display_msg = progress.status_message if progress.status_message else 'Waiting for training data...'
                ax.text(0.5, 0.5, display_msg,
                       ha='center', va='center', fontsize=16,
                       color=style["empty_text"],
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            fig.tight_layout()
            return fig

    def _transfer_to_inference_backend(self) -> bool:
        """
        Transfer the trained model to InferenceBackend.
        Called automatically when training completes.
        """
        print("=" * 60)
        print("DEBUG: _transfer_to_inference_backend() CALLED")
        print("=" * 60)

        try:
            from .inference import get_inference_backend

            session = self.current_training_session

            # Check if already transferred
            if session.get('transferred', False):
                print("DEBUG: Already transferred, returning True")
                logger.info("Model already transferred, skipping")
                return True

            # Validate session data
            if not session.get('base_model_name') or not session.get('output_dir'):
                logger.warning("Training session incomplete, cannot transfer")
                logger.warning(f"Session data: {session}")
                return False

            inference_backend = get_inference_backend()

            base_model_name = session['base_model_name']
            output_dir = session['output_dir']
            is_lora = session['is_lora']
            is_vlm = session['is_vlm']

            logger.info(f"=" * 60)
            logger.info(f"TRANSFERRING MODEL TO INFERENCE BACKEND")
            logger.info(f"=" * 60)
            logger.info(f"  Base model: {base_model_name}")
            logger.info(f"  Output dir: {output_dir}")
            logger.info(f"  Is LoRA: {is_lora}")
            logger.info(f"  Is VLM: {is_vlm}")

            # Transfer the model object directly from trainer memory.
            # If is_lora is True, self.trainer.model is a PeftModel (Base + Adapter).
            # If is_lora is False, it is the finetuned Base Model.
            inference_backend.models[base_model_name] = {
                "model": self.trainer.model,
                "tokenizer": self.trainer.tokenizer,
                "is_vision": is_vlm,
                "is_lora": is_lora,
                "model_path": base_model_name,
                "base_model": None,
                "loaded_adapters": {},
                # Unsloth/PEFT training keeps the active adapter named 'default' in memory
                "active_adapter": "default" if is_lora else None,
            }

            # For vision models, also transfer processor
            if is_vlm:
                if hasattr(self.trainer, 'tokenizer'):
                    inference_backend.models[base_model_name]["processor"] = self.trainer.tokenizer
                    logger.info("  Transferred processor for vision model")

            # Load chat template info
            inference_backend._load_chat_template_info(base_model_name)

            # If it was LoRA, register the output path.
            # This ensures the Eval UI dropdown (which lists files) knows that
            # the model currently in memory corresponds to this specific output directory.
            if is_lora:
                inference_backend.models[base_model_name]["last_trained_adapter"] = output_dir
                logger.info(f"Marked trained LoRA adapter: {output_dir}")

            # Set as active model
            inference_backend.active_model_name = base_model_name
            logger.info(f"Set active model: {base_model_name}")

            return True

        except Exception as e:
            logger.error(f"Error transferring model to inference backend: {e}")
            import traceback
            traceback.print_exc()
            return False


# ========== GLOBAL INSTANCE ==========
_training_backend = None

def get_training_backend() -> TrainingBackend:
    """Get global training backend instance"""
    global _training_backend
    if _training_backend is None:
        _training_backend = TrainingBackend()
    return _training_backend


# ========== UI HANDLER CREATION ==========
def create_training_handlers(train_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create training event handlers for Gradio UI components.

    Args:
        train_components: Dictionary of Gradio components from train page

    Returns:
        Dictionary of handler functions
    """
    backend = get_training_backend()

    def start_training_handler(*args):
        """Handler for start training button - yields status updates"""
        try:
            # Extract parameters in the order they're passed from the UI
            (model_name, training_type, hf_token, load_4bit, max_seq_length,
             hf_dataset, local_datasets, format_type,
             num_epochs, learning_rate, batch_size, gradient_accumulation_steps,
             warmup_steps, warmup_ratio, max_steps, save_steps, weight_decay, random_seed, packing,
             use_lora, lora_r, lora_alpha, lora_dropout, target_modules,
             gradient_checkpointing, use_rslora, use_loftq, train_on_completions,
             finetune_vision_layers, finetune_language_layers,
             finetune_attention_modules, finetune_mlp_modules,
             enable_wandb, wandb_token, wandb_project,
             enable_tensorboard, tensorboard_dir, optim, lr_scheduler_type) = args

            # Start training with correctly named parameters - this is a generator
            for update_tuple in backend.start_training(
                model_name=model_name,
                training_type=training_type,
                hf_token=hf_token,
                load_in_4bit=load_4bit,
                max_seq_length=max_seq_length,
                hf_dataset=hf_dataset,
                local_datasets=local_datasets,
                format_type=format_type,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                warmup_ratio=warmup_ratio,
                max_steps=max_steps,
                save_steps=save_steps,
                weight_decay=weight_decay,
                random_seed=random_seed,
                packing=packing,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                gradient_checkpointing=gradient_checkpointing,
                use_rslora=use_rslora,
                use_loftq=use_loftq,
                train_on_completions=train_on_completions,
                finetune_vision_layers=finetune_vision_layers,
                finetune_language_layers=finetune_language_layers,
                finetune_attention_modules=finetune_attention_modules,
                finetune_mlp_modules=finetune_mlp_modules,
                enable_wandb=enable_wandb,
                wandb_token=wandb_token,
                wandb_project=wandb_project,
                enable_tensorboard=enable_tensorboard,
                tensorboard_dir=tensorboard_dir
            ):
                # Yield each status update to Gradio
                yield update_tuple

        except Exception as e:
            logger.error(f"Error in start_training_handler: {e}", exc_info=True)
            yield (
                gr.update(interactive=True),   # Start button
                gr.update(interactive=False),  # Stop button
                gr.update(visible=False),      # Training progress
                #gr.update(visible=True)        # Config selection
            )

    def stop_training_handler():
        """Handler for stop training button"""
        return backend.stop_training()

    def update_training_status():
        """Periodic update of training status and plot"""
        return backend.get_training_status(backend.current_theme)

    def refresh_plot_for_theme(theme):
        """Refresh plot with new theme"""
        return backend.refresh_plot_for_theme(theme)

    return {
        'start_training': start_training_handler,
        'stop_training': stop_training_handler,
        'update_status': update_training_status,
        'refresh_plot': refresh_plot_for_theme
    }
