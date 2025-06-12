# import torch
# import torch.nn as nn
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoConfig,
#     PreTrainedModel,
#     MllamaForConditionalGeneration,
#     MllamaConfig,
#     LlavaNextForConditionalGeneration,
#     LlavaNextConfig,
#     AutoTokenizer
# )
# from transformers.modeling_outputs import SequenceClassifierOutput
# from typing import Optional, Union, Tuple
# import warnings


# class SequenceClassificationMixin:
#     """
#     Mixin class containing common methods for sequence classification models.
#     """
    
#     @staticmethod
#     def compute_classification_loss(logits, labels, num_labels, config):
#         """Compute loss based on problem type."""
#         if labels is None:
#             return None
        
#         # Determine problem type if not set
#         if config.problem_type is None:
#             if num_labels == 1:
#                 config.problem_type = "regression"
#             elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                 config.problem_type = "single_label_classification"
#             else:
#                 config.problem_type = "multi_label_classification"
        
#         # Compute loss based on problem type
#         if config.problem_type == "regression":
#             loss_fct = nn.MSELoss()
#             if num_labels == 1:
#                 loss = loss_fct(logits.squeeze(), labels.squeeze())
#             else:
#                 loss = loss_fct(logits, labels)
#         elif config.problem_type == "single_label_classification":
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
#         elif config.problem_type == "multi_label_classification":
#             loss_fct = nn.BCEWithLogitsLoss()
#             loss = loss_fct(logits, labels)
#         else:
#             raise ValueError(f"Unknown problem type: {config.problem_type}")
        
#         return loss
    
#     @staticmethod
#     def pool_sequence(last_hidden_state, attention_mask=None):
#         """Pool the sequence representation using the last non-padded token."""
#         if attention_mask is not None:
#             # Find the last non-padded token for each sequence
#             batch_size = last_hidden_state.shape[0]
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             pooled_output = last_hidden_state[torch.arange(batch_size), sequence_lengths]
#         else:
#             # Use the last token
#             pooled_output = last_hidden_state[:, -1, :]
        
#         return pooled_output


# class MllamaForSequenceClassification(PreTrainedModel, SequenceClassificationMixin):
#     """
#     Mllama model with a sequence classification head on top (a linear layer on top of the pooled output).
#     """
#     config_class = MllamaConfig
    
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
        
#         # Load the base vision model
#         self.mllama = MllamaForConditionalGeneration(config)
        
#         # Get the hidden size from the language model
#         if hasattr(config, 'text_config') and config.text_config is not None:
#             hidden_size = config.text_config.hidden_size
#         elif hasattr(config, 'hidden_size'):
#             hidden_size = config.hidden_size
#         else:
#             # Fallback - get from the actual model
#             hidden_size = self.mllama.language_model.config.hidden_size
        
#         # Classification head
#         self.score = nn.Linear(hidden_size, config.num_labels)
#         self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        
#         # Initialize weights
#         self.post_init()

#     def enable_input_require_grads(self):
#         """
#         Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
#         the model weights fixed.
#         """
#         def make_inputs_require_grads(module, input, output):
#             output.requires_grad_(True)
        
#         # Access embeddings through the language model
#         embedding_layer = self.mllama.model.language_model.embed_tokens
#         self._require_grads_hook = embedding_layer.register_forward_hook(make_inputs_require_grads)

#     def disable_input_require_grads(self):
#         """
#         Removes the `_require_grads_hook`.
#         """
#         if hasattr(self, '_require_grads_hook'):
#             self._require_grads_hook.remove()
    
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         aspect_ratio_ids: Optional[torch.LongTensor] = None,
#         aspect_ratio_mask: Optional[torch.LongTensor] = None,
#         cross_attention_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs
#     ) -> Union[Tuple, SequenceClassifierOutput]:
        
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         # Get outputs from the language model part only (ignore vision for sequence classification)
#         language_model_outputs = self.mllama.language_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=True,
#             return_dict=return_dict
#         )
        
#         # Get the last hidden state and pool it
#         last_hidden_state = language_model_outputs.last_hidden_state
#         pooled_output = self.pool_sequence(last_hidden_state, attention_mask)
        
#         # Apply dropout and classification
#         pooled_output = self.dropout(pooled_output)
#         logits = self.score(pooled_output)
        
#         # Compute loss using the mixin method
#         loss = self.compute_classification_loss(logits, labels, self.num_labels, self.config)
        
#         if not return_dict:
#             output = (logits,) + language_model_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=language_model_outputs.hidden_states,
#             attentions=language_model_outputs.attentions,
#         )


# class LlavaNextForSequenceClassification(PreTrainedModel, SequenceClassificationMixin):
#     """
#     LlavaNext model with a sequence classification head on top (a linear layer on top of the pooled output).
#     """
#     config_class = LlavaNextConfig
    
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
        
#         # Load the base vision model
#         self.llava_next = LlavaNextForConditionalGeneration(config)
        
#         # Get the hidden size from the language model
#         if hasattr(config, 'text_config') and config.text_config is not None:
#             hidden_size = config.text_config.hidden_size
#         elif hasattr(config, 'hidden_size'):
#             hidden_size = config.hidden_size
#         else:
#             # Fallback - get from the actual model
#             hidden_size = self.llava_next.language_model.config.hidden_size
        
#         # Classification head - handle quantization
#         self.score = self._create_classification_head(hidden_size, config.num_labels)
#         self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        
#         # Initialize weights
#         self.post_init()
    
#     def _create_classification_head(self, hidden_size, num_labels):
#         """Create classification head with quantization support"""
#         try:
#             import bitsandbytes as bnb
#             from transformers.utils import is_bitsandbytes_available
#             if is_bitsandbytes_available() and hasattr(self.llava_next, 'language_model'):
#                 # Check if the base model is quantized
#                 if hasattr(self.llava_next.language_model, 'model'):
#                     first_layer = next(iter(self.llava_next.language_model.model.layers))
#                     if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
#                         if hasattr(first_layer.self_attn.q_proj, 'quant_state'):
#                             # Model is quantized, use Linear8bitLt for the classification head
#                             return bnb.nn.Linear8bitLt(hidden_size, num_labels, has_fp16_weights=False)
#         except (ImportError, AttributeError, StopIteration):
#             pass
        
#         # Default to regular Linear layer
#         return nn.Linear(hidden_size, num_labels)

#     def enable_input_require_grads(self):
#         """
#         Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
#         the model weights fixed.
#         """
#         def make_inputs_require_grads(module, input, output):
#             output.requires_grad_(True)
        
#         # Access embeddings through the language model
#         embedding_layer = self.llava_next.model.language_model.embed_tokens
#         self._require_grads_hook = embedding_layer.register_forward_hook(make_inputs_require_grads)

#     def disable_input_require_grads(self):
#         """
#         Removes the `_require_grads_hook`.
#         """
#         if hasattr(self, '_require_grads_hook'):
#             self._require_grads_hook.remove()
    
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         image_sizes: Optional[torch.LongTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs
#     ) -> Union[Tuple, SequenceClassifierOutput]:
        
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         # Get outputs from the language model part only (ignore vision for sequence classification)
#         language_model_outputs = self.llava_next.language_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=True,
#             return_dict=return_dict
#         )
        
#         # Get the last hidden state and pool it
#         last_hidden_state = language_model_outputs.last_hidden_state
#         pooled_output = self.pool_sequence(last_hidden_state, attention_mask)
        
#         # Apply dropout and classification
#         pooled_output = self.dropout(pooled_output)
#         logits = self.score(pooled_output)
        
#         # Compute loss using the mixin method
#         loss = self.compute_classification_loss(logits, labels, self.num_labels, self.config)
        
#         if not return_dict:
#             output = (logits,) + language_model_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=language_model_outputs.hidden_states,
#             attentions=language_model_outputs.attentions,
#         )

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    PreTrainedModel,
    MllamaForConditionalGeneration,
    MllamaConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextConfig,
    AutoTokenizer
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Tuple
import warnings


class SequenceClassificationMixin:
    """
    Mixin class containing common methods for sequence classification models.
    """
    
    @staticmethod
    def compute_classification_loss(logits, labels, num_labels, config):
        """Compute loss based on problem type."""
        if labels is None:
            return None
        
        # Determine problem type if not set
        if config.problem_type is None:
            if num_labels == 1:
                config.problem_type = "regression"
            elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                config.problem_type = "single_label_classification"
            else:
                config.problem_type = "multi_label_classification"
        
        # Compute loss based on problem type
        if config.problem_type == "regression":
            loss_fct = nn.MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif config.problem_type == "single_label_classification":
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        elif config.problem_type == "multi_label_classification":
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        else:
            raise ValueError(f"Unknown problem type: {config.problem_type}")
        
        return loss
    
    @staticmethod
    def pool_sequence(last_hidden_state, attention_mask=None):
        """Pool the sequence representation using the last non-padded token."""
        if attention_mask is not None:
            # Find the last non-padded token for each sequence
            batch_size = last_hidden_state.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            pooled_output = last_hidden_state[torch.arange(batch_size), sequence_lengths]
        else:
            # Use the last token
            pooled_output = last_hidden_state[:, -1, :]
        
        return pooled_output
    


class MllamaForSequenceClassification(PreTrainedModel, SequenceClassificationMixin):
    """
    Mllama model with a sequence classification head on top (a linear layer on top of the pooled output).
    """
    config_class = MllamaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Load the base vision model
        self.mllama = MllamaForConditionalGeneration(config)
        
        # Get the hidden size from the language model
        if hasattr(config, 'text_config') and config.text_config is not None:
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        else:
            # Fallback - get from the actual model
            hidden_size = self.mllama.language_model.config.hidden_size
        
        # Classification head
        self.score = nn.Linear(hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        
        # Initialize weights
        self.post_init()


    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        
        # Access embeddings through the language model
        embedding_layer = self.mllama.model.language_model.embed_tokens
        self._require_grads_hook = embedding_layer.register_forward_hook(make_inputs_require_grads)

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_ids: Optional[torch.LongTensor] = None,
        aspect_ratio_mask: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get outputs from the language model part only (ignore vision for sequence classification)
        language_model_outputs = self.mllama.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
        
        # Get the last hidden state and pool it
        last_hidden_state = language_model_outputs.last_hidden_state
        pooled_output = self.pool_sequence(last_hidden_state, attention_mask)
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.score(pooled_output)
        
        # Compute loss using the mixin method
        loss = self.compute_classification_loss(logits, labels, self.num_labels, self.config)
        
        if not return_dict:
            output = (logits,) + language_model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=language_model_outputs.hidden_states,
            attentions=language_model_outputs.attentions,
        )


class LlavaNextForSequenceClassification(PreTrainedModel, SequenceClassificationMixin):
    """
    LlavaNext model with a sequence classification head on top (a linear layer on top of the pooled output).
    """
    config_class = LlavaNextConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Load the base vision model
        self.llava_next = LlavaNextForConditionalGeneration(config)
        
        # Get the hidden size from the language model
        if hasattr(config, 'text_config') and config.text_config is not None:
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        else:
            # Fallback - get from the actual model
            hidden_size = self.llava_next.language_model.config.hidden_size
        
        # Classification head - handle quantization
        self.score = self._create_classification_head(hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        
        # Initialize weights
        self.post_init()
    
    def _create_classification_head(self, hidden_size, num_labels):
        """Create classification head with quantization support"""
        try:
            import bitsandbytes as bnb
            from transformers.utils import is_bitsandbytes_available
            if is_bitsandbytes_available() and hasattr(self.llava_next, 'language_model'):
                # Check if the base model is quantized
                if hasattr(self.llava_next.language_model, 'model'):
                    first_layer = next(iter(self.llava_next.language_model.model.layers))
                    if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                        if hasattr(first_layer.self_attn.q_proj, 'quant_state'):
                            # Model is quantized, use Linear8bitLt for the classification head
                            return bnb.nn.Linear8bitLt(hidden_size, num_labels, has_fp16_weights=False)
        except (ImportError, AttributeError, StopIteration):
            pass
        
        # Default to regular Linear layer
        return nn.Linear(hidden_size, num_labels)

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        
        # Access embeddings through the language model
        embedding_layer = self.llava_next.model.language_model.embed_tokens
        self._require_grads_hook = embedding_layer.register_forward_hook(make_inputs_require_grads)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get outputs from the language model part only (ignore vision for sequence classification)
        language_model_outputs = self.llava_next.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
        
        # Get the last hidden state and pool it
        last_hidden_state = language_model_outputs.last_hidden_state
        pooled_output = self.pool_sequence(last_hidden_state, attention_mask)
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.score(pooled_output)
        
        # Compute loss using the mixin method
        loss = self.compute_classification_loss(logits, labels, self.num_labels, self.config)
        
        if not return_dict:
            output = (logits,) + language_model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=language_model_outputs.hidden_states,
            attentions=language_model_outputs.attentions,
        )