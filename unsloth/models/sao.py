# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-Rollout Asynchronous Optimization (SAO), from arXiv:2607.07508.

Three algorithmic pieces, all of which are independent of the paper's
data-centre rollout infrastructure and are implemented here to run inside one
synchronous training process:

  1. Direct double-sided Importance Sampling (DIS, paper Sec 3.1). The rollout
     engine's own log-probabilities stand in for the behaviour policy, so no
     `pi_theta_old` snapshot is kept, and the importance ratio is calibrated by
     a hard mask to zero outside `(1 - eps_low, 1 + eps_high)` rather than by
     PPO's clamp.
  2. Single rollout per prompt with a stabilised critic (Sec 3.2): `K` value
     gradient steps for every policy step, and a critic whose attention
     parameters are frozen so only its MLP/projection stack is optimised.
  3. Skip-observation token-level GAE (Sec 3.2, Eqs. 4-5): advantage propagates
     from action token to action token, stepping over environment/tool
     observation spans that the policy did not generate.

The paper's asynchronous rollout scheduler (disaggregated rollout workers
streaming trajectories into a trainer) is deliberately not implemented; it is
cluster infrastructure orthogonal to Unsloth's single-process training model.
Rollouts here are produced synchronously, one per prompt per step.
"""

__all__ = [
    "SAOConfig",
    "SAOTrainer",
    "dis_calibrate",
    "sao_policy_loss",
    "sao_value_loss",
    "skip_observation_gae",
    "freeze_critic_attention",
]

import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

# Both bases are resolved at import so `SAOConfig`/`SAOTrainer` stay ordinary
# classes, and both are optional so the pure-math helpers below remain importable
# (and unit-testable) on a machine with nothing but torch installed.
try:
    from trl import PPOConfig as _SAOConfigBase
except Exception:
    try:
        from transformers import TrainingArguments as _SAOConfigBase
    except Exception:

        @dataclass
        class _SAOConfigBase:
            output_dir: str = "sao_output"
            learning_rate: float = 1e-6


try:
    from transformers import Trainer as _SAOTrainerBase
except Exception:
    _SAOTrainerBase = object


# gamma/lambda are not restated by the paper because they are standard PPO
# hyperparameters; these match trl.PPOConfig's own defaults (no discounting
# across tokens of one episode, GAE trace decay 0.95).
DEFAULT_GAMMA = 1.0
DEFAULT_LAM = 0.95


def dis_calibrate(
    ratios: torch.Tensor,
    eps_low: float = 0.3,
    eps_high: float = 5.0,
) -> torch.Tensor:
    """Paper Eq. 3: `f(x) = x` inside `(1 - eps_low, 1 + eps_high)`, else `0`.

    Note this is a mask, not PPO's clamp: an out-of-range token contributes
    nothing to the gradient instead of contributing at the clipped ratio.
    """
    if eps_low <= 0.0 or eps_high <= 0.0:
        raise ValueError(
            f"Unsloth: SAO needs eps_low and eps_high > 0, got "
            f"eps_low = {eps_low}, eps_high = {eps_high}."
        )
    keep = (ratios > (1.0 - eps_low)) & (ratios < (1.0 + eps_high))
    return torch.where(keep, ratios, torch.zeros_like(ratios))


def sao_policy_loss(
    policy_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    eps_low: float = 0.3,
    eps_high: float = 5.0,
) -> torch.Tensor:
    """Paper Eqs. 1-2: `-E[f(r) * A * log pi_theta]` over action tokens.

    `r = exp(log pi_theta - log pi_rollout)` is treated as a coefficient and
    detached. Differentiating through it as well would add a
    `grad(r) * A * log pi` term that is not the policy-gradient estimator the
    objective stands for.
    """
    ratios = torch.exp(policy_logprobs.detach() - rollout_logprobs)
    coefficients = dis_calibrate(ratios, eps_low, eps_high) * advantages
    mask = action_mask.to(policy_logprobs.dtype)
    per_token = -coefficients.detach() * policy_logprobs * mask
    denominator = mask.sum().clamp(min = 1.0)
    return per_token.sum() / denominator


def sao_value_loss(
    values: torch.Tensor, returns: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """`L_VF = E[(V_phi(q, y_<t) - R)^2]` over action tokens."""
    mask = action_mask.to(values.dtype)
    squared = (values - returns) ** 2 * mask
    return squared.sum() / mask.sum().clamp(min = 1.0)


def skip_observation_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float = DEFAULT_GAMMA,
    lam: float = DEFAULT_LAM,
):
    """Skip-observation token-level GAE (paper Eqs. 4-5).

    All tensors are `(batch, sequence)`. `action_mask` marks tokens the policy
    generated; every other position (tool output, environment observation,
    padding) is stepped over, so the recursion only ever bridges the last token
    of one action to the first token of the next:

        delta = r + gamma * V(a_{i+1,0}) - V(a_{i,N})
        A(a_{i,N}) = delta + gamma * lam * A(a_{i+1,0})

    With no skipped spans this is exactly standard token-level GAE, so a
    single-turn rollout needs no separate code path. Rewards landing on skipped
    positions are carried back onto the preceding action token rather than
    dropped, which keeps an environment reward emitted at an observation
    position inside the estimator.

    Returns `(advantages, returns)`, both zero outside the action mask.
    """
    if rewards.shape != values.shape or rewards.shape != action_mask.shape:
        raise ValueError(
            f"Unsloth: SAO GAE needs matching shapes, got rewards "
            f"{tuple(rewards.shape)}, values {tuple(values.shape)}, "
            f"action_mask {tuple(action_mask.shape)}."
        )
    mask = action_mask.to(values.dtype)
    advantages = torch.zeros_like(values)
    batch, length = values.shape
    for b in range(batch):
        next_value = torch.zeros((), dtype = values.dtype, device = values.device)
        next_advantage = torch.zeros((), dtype = values.dtype, device = values.device)
        carried_reward = torch.zeros((), dtype = values.dtype, device = values.device)
        for t in range(length - 1, -1, -1):
            if mask[b, t] == 0:
                carried_reward = carried_reward + rewards[b, t]
                continue
            delta = rewards[b, t] + carried_reward + gamma * next_value - values[b, t]
            advantage = delta + gamma * lam * next_advantage
            advantages[b, t] = advantage
            next_value = values[b, t]
            next_advantage = advantage
            carried_reward = torch.zeros_like(carried_reward)
    returns = (advantages + values) * mask
    return advantages * mask, returns


def freeze_critic_attention(
    model: nn.Module, attention_keywords: Sequence[str] = ("attention", "attn")
) -> int:
    """Freeze the critic's attention parameters (paper Sec 3.2).

    The paper trains only the critic's MLP/MoE projection stack, on the grounds
    that attention gradient norms are the unstable part and the pretrained
    attention weights already carry the semantics the value head needs. Modules
    are matched by class name first so a dense model behaves the same as the
    paper's MoE one; the parameter-name pass is the fallback for architectures
    whose attention block is named something else.

    Returns the number of frozen parameter tensors.
    """
    frozen = 0
    for name, module in model.named_modules():
        class_name = type(module).__name__.lower()
        if not any(keyword in class_name for keyword in attention_keywords):
            continue
        for parameter in module.parameters(recurse = True):
            if parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen += 1
    if frozen != 0:
        return frozen
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(keyword in lowered for keyword in attention_keywords):
            if parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen += 1
    return frozen


@dataclass
class SAOConfig(_SAOConfigBase):
    """Hyperparameters for `SAOTrainer` (arXiv:2607.07508).

    Extends the installed TRL `PPOConfig` where available, so learning rate,
    batch size and every other `TrainingArguments` field behave as usual.

    Args:
        eps_low, eps_high: DIS calibration window (paper Eq. 3). The asymmetry
            of the paper's fitted values (0.3 / 5.0) is deliberate - far more
            tolerance above 1 than below it.
        value_updates_per_policy_update: `K` value gradient steps per policy
            step (paper's "faster value update", K = 2).
        freeze_critic_attention: train only the critic's non-attention
            parameters.
        gamma, lam: GAE discount and trace decay.
        max_completion_length, temperature, top_p, top_k: rollout sampling.
        observation_mask_column: dataset column holding a per-completion-token
            mask (1 = observation/tool token to skip in GAE). Absent means a
            single-turn rollout, which reduces to standard GAE.
    """

    eps_low: float = 0.3
    eps_high: float = 5.0
    value_updates_per_policy_update: int = 2
    freeze_critic_attention: bool = True
    gamma: float = DEFAULT_GAMMA
    lam: float = DEFAULT_LAM
    value_learning_rate: Optional[float] = None
    max_completion_length: int = 256
    max_prompt_length: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    prompt_column: str = "prompt"
    observation_mask_column: str = "observation_mask"

    def __post_init__(self):
        # SAOTrainer.training_step reads dataset columns (prompt_column,
        # observation_mask_column) directly - none of them match the model's
        # forward signature, so the base Trainer's default column-pruning
        # dataloader strips them before training_step ever runs, and
        # `trainer.train()` fails immediately with "No columns in the dataset
        # match the model's forward method signature". Force this off the way
        # TRL's GRPOConfig does, regardless of what was passed in.
        self.remove_unused_columns = False
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        if self.value_updates_per_policy_update < 1:
            raise ValueError(
                f"Unsloth: SAO needs value_updates_per_policy_update >= 1, got "
                f"{self.value_updates_per_policy_update}."
            )
        if self.eps_low <= 0.0 or self.eps_high <= 0.0:
            raise ValueError(
                f"Unsloth: SAO needs eps_low and eps_high > 0, got "
                f"eps_low = {self.eps_low}, eps_high = {self.eps_high}."
            )


class SAOValueModel(nn.Module):
    """A backbone plus a scalar value head, one value per token position."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone
        if hidden_size is None:
            config = getattr(backbone, "config", None)
            hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Unsloth: SAO could not read hidden_size off the value model's "
                "config; pass hidden_size = ... explicitly."
            )
        self.value_head = nn.Linear(hidden_size, 1, bias = False)

    def forward(
        self,
        input_ids,
        attention_mask = None,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            return_dict = True,
            **kwargs,
        )
        # Clone: Unsloth's patched fast-path kernels reuse activation buffers
        # in place across forward calls on the same backbone. SAOTrainer calls
        # this forward K times per training_step (once per critic gradient
        # step), each producing its own backward - without cloning, a later
        # forward's in-place op can mutate a tensor an earlier call's backward
        # still needs, raising "modified by an inplace operation ... expected
        # version 0 instead".
        hidden_states = outputs.hidden_states[-1].clone()
        return self.value_head(hidden_states.to(self.value_head.weight.dtype)).squeeze(-1)


class SAOTrainer(_SAOTrainerBase):
    """Single-Rollout Asynchronous Optimization trainer (arXiv:2607.07508).

    Usage mirrors `GRPOTrainer`, with a value model added and exactly one
    rollout per prompt - the single-rollout sampling is the algorithm's
    identity, so it is not configurable:

    ```python
    from unsloth import FastLanguageModel
    from unsloth.models.sao import SAOTrainer, SAOConfig

    trainer = SAOTrainer(
        model = model,
        value_model = value_model,
        reward_funcs = [my_reward_fn],
        args = SAOConfig(output_dir = "sao", eps_low = 0.3, eps_high = 5.0),
        train_dataset = dataset,
        processing_class = tokenizer,
    )
    trainer.train()
    ```

    Reward functions take `prompts` and `completions` keyword lists (plus any
    remaining dataset columns) and return one float per completion, exactly as
    TRL's GRPO reward functions do.
    """

    def __init__(
        self,
        model,
        value_model,
        reward_funcs: Union[Callable, List[Callable]],
        args: Optional[SAOConfig] = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        **kwargs,
    ):
        if _SAOTrainerBase is object:
            raise ImportError(
                "Unsloth: SAOTrainer needs `transformers` installed. The SAO "
                "algorithm helpers (skip_observation_gae, dis_calibrate) work "
                "without it."
            )
        if args is None:
            args = SAOConfig(output_dir = "sao_output")
        if not isinstance(reward_funcs, (list, tuple)):
            reward_funcs = [reward_funcs]
        if len(reward_funcs) == 0:
            raise ValueError("Unsloth: SAOTrainer needs at least one reward function.")

        super().__init__(
            model = model,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            data_collator = lambda features: features,
            **kwargs,
        )
        # SAOConfig is a trl.PPOConfig (via OnPolicyConfig), which redeclares
        # `world_size` as a plain dataclass field defaulting to None - it
        # deliberately shadows transformers.TrainingArguments' computed
        # `world_size` property, because the real PPOTrainer sets it manually
        # during its own bespoke setup (see trl's PPOTrainer.__init__). SAOTrainer
        # instead drives training through the generic Trainer.train() inner loop,
        # which reads args.world_size expecting it to be populated; left at None
        # this crashes immediately in get_total_train_batch_size. Set it the same
        # way PPOTrainer does, right after self.accelerator exists.
        if getattr(args, "world_size", "unset") is None:
            try:
                args.world_size = self.accelerator.num_processes
            except AttributeError:
                # world_size is a read-only property on plain
                # transformers.TrainingArguments (no trl installed); nothing
                # to patch in that case.
                pass
        self.reward_funcs = list(reward_funcs)
        self.sao_args = args

        if not isinstance(value_model, SAOValueModel):
            value_model = SAOValueModel(value_model)
        self.value_model = value_model.to(self.args.device)
        if args.freeze_critic_attention:
            frozen = freeze_critic_attention(self.value_model)
            if frozen == 0:
                raise ValueError(
                    "Unsloth: freeze_critic_attention = True but no attention "
                    "modules or parameters were found on the value model. Set it "
                    "to False, or pass a value model whose attention blocks are "
                    "discoverable by name."
                )
        value_parameters = [p for p in self.value_model.parameters() if p.requires_grad]
        value_optimizer = torch.optim.AdamW(
            value_parameters,
            lr = args.value_learning_rate
            if args.value_learning_rate is not None
            else args.learning_rate,
        )
        # Must go through accelerator.prepare like the main optimizer: under fp16
        # mixed precision this is what makes `optimizer.step()` unscale gradients
        # (and skip the step on an inf/NaN) after `accelerator.backward()` scaled
        # the loss. An unprepared optimizer would apply the still-scaled
        # gradients directly.
        self.value_optimizer = self.accelerator.prepare(value_optimizer)

    def _generate(self, prompt_texts: List[str]):
        """One rollout per prompt, keeping the sampler's own token logprobs."""
        tokenizer = self.processing_class
        args = self.sao_args
        encoded = tokenizer(
            prompt_texts,
            return_tensors = "pt",
            padding = True,
            truncation = args.max_prompt_length is not None,
            max_length = args.max_prompt_length,
        ).to(self.args.device)

        model = self.model
        was_training = model.training
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens = args.max_completion_length,
                do_sample = True,
                temperature = args.temperature,
                top_p = args.top_p,
                top_k = args.top_k,
                num_return_sequences = 1,
                return_dict_in_generate = True,
                output_scores = True,
                pad_token_id = tokenizer.pad_token_id,
            )
        if was_training:
            model.train()

        prompt_length = encoded["input_ids"].shape[1]
        # generate() runs under torch.inference_mode() internally (regardless
        # of the torch.no_grad() above), so `generated.sequences` comes back as
        # an inference tensor. `sequences`/`completion_ids` are threaded into
        # `_policy_logprobs`' forward pass below, which needs to be part of the
        # backward graph for the policy loss - an inference tensor there raises
        # "Inference tensors cannot be saved for backward". Clone to get plain
        # tensors, the same fix TRL's own generation-based trainers apply.
        sequences = generated.sequences.clone()
        completion_ids = sequences[:, prompt_length:]
        # The sampler's own distribution is the behaviour policy (paper Sec 3.1);
        # no separate pi_theta_old snapshot is ever taken.
        rollout_logprobs = torch.stack(
            [
                torch.log_softmax(score.float(), dim = -1)
                .gather(1, completion_ids[:, step : step + 1])
                .squeeze(1)
                for step, score in enumerate(generated.scores)
            ],
            dim = 1,
        )
        completion_mask = (completion_ids != tokenizer.pad_token_id).to(torch.long)
        if tokenizer.eos_token_id is not None:
            is_eos = completion_ids == tokenizer.eos_token_id
            completion_mask = torch.logical_or(completion_mask.bool(), is_eos).to(torch.long)
        return (
            sequences,
            encoded["attention_mask"],
            completion_ids,
            completion_mask,
            rollout_logprobs,
        )

    def _policy_logprobs(self, sequences, attention_mask, completion_ids):
        prompt_length = sequences.shape[1] - completion_ids.shape[1]
        full_mask = torch.cat([attention_mask, torch.ones_like(completion_ids)], dim = 1)
        logits = self.model(input_ids = sequences, attention_mask = full_mask, return_dict = True).logits
        logits = logits[:, prompt_length - 1 : -1, :] / max(self.sao_args.temperature, 1e-7)
        logprobs = torch.log_softmax(logits.float(), dim = -1)
        return logprobs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1), full_mask

    def _rewards(self, prompt_texts, completion_texts, batch) -> torch.Tensor:
        totals = torch.zeros(len(prompt_texts), dtype = torch.float32, device = self.args.device)
        extra_columns = {
            key: [example.get(key) for example in batch]
            for key in (batch[0].keys() if len(batch) != 0 else [])
        }
        for reward_func in self.reward_funcs:
            accepted = inspect.signature(reward_func).parameters
            has_var_kw = any(p.kind == p.VAR_KEYWORD for p in accepted.values())
            passed = {
                key: value for key, value in extra_columns.items() if has_var_kw or key in accepted
            }
            values = reward_func(prompts = prompt_texts, completions = completion_texts, **passed)
            totals = totals + torch.tensor(values, dtype = torch.float32, device = self.args.device)
        return totals

    def _observation_mask(self, batch, completion_mask) -> torch.Tensor:
        """Action mask: completion tokens minus any observation span.

        Absent an observation column every completion token is an action token,
        and skip-observation GAE degenerates to standard token-level GAE.
        """
        column = self.sao_args.observation_mask_column
        if len(batch) == 0 or column not in batch[0]:
            return completion_mask
        action_mask = completion_mask.clone()
        width = completion_mask.shape[1]
        for row, example in enumerate(batch):
            observations = example.get(column) or []
            for position, flag in enumerate(observations[:width]):
                if flag:
                    action_mask[row, position] = 0
        return action_mask

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch = None,
    ):
        batch = list(inputs)
        args = self.sao_args
        tokenizer = self.processing_class
        prompt_texts = [example[args.prompt_column] for example in batch]

        (
            sequences,
            prompt_mask,
            completion_ids,
            completion_mask,
            rollout_logprobs,
        ) = self._generate(prompt_texts)
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens = True)
        action_mask = self._observation_mask(batch, completion_mask)

        rewards = self._rewards(prompt_texts, completion_texts, batch)
        token_rewards = torch.zeros_like(rollout_logprobs)
        last_action = action_mask.sum(dim = 1).clamp(min = 1) - 1
        token_rewards[torch.arange(token_rewards.shape[0]), last_action] = rewards

        full_mask = torch.cat([prompt_mask, torch.ones_like(completion_ids)], dim = 1)
        prompt_length = sequences.shape[1] - completion_ids.shape[1]
        with torch.no_grad():
            values = self.value_model(input_ids = sequences, attention_mask = full_mask)[
                :, prompt_length - 1 : -1
            ].float()
        advantages, returns = skip_observation_gae(
            token_rewards, values, action_mask, gamma = args.gamma, lam = args.lam
        )

        policy_logprobs, _ = self._policy_logprobs(sequences, prompt_mask, completion_ids)
        loss = sao_policy_loss(
            policy_logprobs,
            rollout_logprobs,
            advantages,
            action_mask,
            eps_low = args.eps_low,
            eps_high = args.eps_high,
        )
        self.accelerator.backward(loss)

        # K critic steps per policy step ("faster value update", paper Sec 3.2).
        value_losses = []
        for _ in range(args.value_updates_per_policy_update):
            predicted = self.value_model(input_ids = sequences, attention_mask = full_mask)[
                :, prompt_length - 1 : -1
            ].float()
            value_loss = sao_value_loss(predicted, returns, action_mask)
            self.value_optimizer.zero_grad(set_to_none = True)
            # Route through the accelerator like the policy backward above, not a
            # raw `.backward()` / `.step()`: under fp16 mixed precision the
            # accelerator's GradScaler must scale this loss and unscale the
            # gradients before the optimizer step, or gradients silently
            # underflow/overflow and the critic diverges within a few steps.
            self.accelerator.backward(value_loss)
            self.value_optimizer.step()
            value_losses.append(value_loss.detach())

        ratios = torch.exp(policy_logprobs.detach() - rollout_logprobs)
        kept = dis_calibrate(ratios, args.eps_low, args.eps_high) != 0
        self._sao_metrics = {
            "reward": rewards.mean().item(),
            "value_loss": torch.stack(value_losses).mean().item(),
            "dis_kept_fraction": (kept & action_mask.bool()).sum().item()
            / max(action_mask.sum().item(), 1),
        }
        return loss.detach()

    def log(
        self,
        logs,
        start_time = None,
    ):
        metrics = getattr(self, "_sao_metrics", None)
        if metrics is not None:
            logs = {**logs, **metrics}
            self._sao_metrics = None
        try:
            return super().log(logs, start_time)
        except TypeError:
            return super().log(logs)
