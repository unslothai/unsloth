# Unsloth Slowbie
# Copyright (C) 2024-present the Unsloth AI team. All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch

__all__ = [
	"mean_of_trained_tokens",
	"add_new_tokens",
]


@torch.inference_mode
def mean_of_trained_tokens(model, eps = 1e-16):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    embedding_matrix = model.get_input_embeddings ().weight.clone()
    lm_head_matrix   = model.get_output_embeddings().weight.clone()

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis = 1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    # if n_untrained != 0:
    #     print(
    #         f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"\
    #         "We shall set them to the mean of the other trained tokens."
    #     )
    # pass

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
    sum_lm_head   = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

    # Remove bad tokens
    sum_embedding -= torch.sum(embedding_matrix[where_untrained], dtype = torch.float32, axis = 0)
    sum_lm_head   -= torch.sum(lm_head_matrix  [where_untrained], dtype = torch.float32, axis = 0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = (sum_embedding / n_trained)
    mean_lm_head   = (sum_lm_head   / n_trained)

    return mean_embedding, mean_lm_head
pass


@torch.inference_mode
def add_new_tokens(
    model,
    tokenizer,
    new_tokens = [],
    method = "mean",
    interpolation = 0.5,
):
    """
    Smartly resizes the tokenizer and adds new tokens to the model.
    We also disregard untrained tokens by removing them from the mean calculation.
    """
    assert(isinstance(new_tokens, (list, tuple)))
    assert(len(new_tokens) > 0)
    assert(method == "mean" or method == "interpolation")
    assert(interpolation >= 0 and interpolation <= 1)

    # Check if tokens already exist
    overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
    if len(overlapping_tokens) != 0:
        print(
            f"Unsloth: You're adding new_tokens = {new_tokens}\n"\
            f"There are tokens which are overlapping = {list(overlapping_tokens)}\n"\
            f"We shall safely ignore these overlapping tokens."
        )
        new_tokens = [x for x in new_tokens if x not in overlapping_tokens]
    pass

    # Get mean of trained tokens
    # mean_embedding, mean_lm_head = fix_untrained_tokens(model)

    # Weirdly be careful reserved tokens can pop out
    mean_embedding, mean_lm_head = mean_of_trained_tokens(model)
    mean_embedding = mean_embedding.to(torch.float32)
    mean_lm_head   = mean_lm_head  .to(torch.float32)

    # Get old lengths
    old_input_length  = model.get_input_embeddings ().weight.shape[0]
    old_output_length = model.get_output_embeddings().weight.shape[0]
    old_config_size   = model.config.vocab_size

    # Add tokens!
    old_length = len(tokenizer)
    tokenizer.add_tokens(new_tokens)
    # Also resizes lm_head as well!
    model.resize_token_embeddings(len(tokenizer))

    # If we use interpolation, we interpolate between the mean embeddings and
    # the Word2Vec sum of the other vectors
    embedding_matrix = model.get_input_embeddings ().weight
    lm_head_matrix   = model.get_output_embeddings().weight

    # Confirm sizes are correct
    if embedding_matrix.shape[0] != (old_input_length  + len(new_tokens)):
        raise RuntimeError(
            "Unsloth: Embedding matrix size did not get resized properly. Please file a bug report!"
        )
    if lm_head_matrix.shape[0]   != (old_output_length + len(new_tokens)):
        raise RuntimeError(
            "Unsloth: LM Head matrix size did not get resized properly. Please file a bug report!"
        )
    if model.config.vocab_size   != (old_config_size   + len(new_tokens)):
        raise RuntimeError(
            "Unsloth: Model's config vocab_size did not get resized properly. Please file a bug report!"
        )
    pass

    if method == "interpolation":
        print(
            "Unsloth: You are using interpolation to add new tokens.\n"\
            f"We shall set new tokens = mean(embeddings)*{1-interpolation} + mean(new_tokens)*{interpolation}"
        )
        for j, token in enumerate(new_tokens):
            input_ids = tokenizer(token, add_special_tokens = False).input_ids
            mean_embedding_token = embedding_matrix[input_ids].mean(axis = 0, dtype = torch.float32)
            mean_lm_head_token   = lm_head_matrix  [input_ids].mean(axis = 0, dtype = torch.float32)

            # Interpolate
            mean_embedding_token = mean_embedding*(1-interpolation) + mean_embedding_token*interpolation
            mean_lm_head_token   = mean_lm_head  *(1-interpolation) + mean_lm_head_token  *interpolation

            # Set the new vector
            embedding_matrix[old_length+j] = mean_embedding_token
            lm_head_matrix  [old_length+j] = mean_lm_head_token
        pass
    else:
        # Now set the new tokens to the mean!
        embedding_matrix[old_length:] = mean_embedding
        lm_head_matrix  [old_length:] = mean_lm_head
    pass

    # We set a flag to say we need to train embeddings
    internal_model = model
    while hasattr(internal_model, "model"):
        internal_model._need_to_train_embeddings = True
        internal_model = internal_model.model
    pass
    internal_model._need_to_train_embeddings = True
    
    return
pass
