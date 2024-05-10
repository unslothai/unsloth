from torch.nn import CrossEntropyLoss

from unsloth.kernels.fused_cel import fused_cel_linear


def CEL_only_forward(model, labels, hidden_states):
    self = model

    if model.config.use_fused_cel:
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.to(shift_hidden_states.device)

        loss = fused_cel_linear(
            shift_hidden_states,
            self.lm_head.weight,
            shift_labels,
            n_loop_iters=self.config.fused_cel_n_loop_iters,
            ignore_index=self.config.fused_cel_ignore_index,
            reduction=self.config.fused_cel_reduction,
        )
        logits = None
    else:
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
    return loss
