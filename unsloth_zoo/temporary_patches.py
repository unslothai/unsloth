# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import re
from typing import Union, List, Any, Tuple, Dict, Callable
import inspect

global TEMPORARY_PATCHES
TEMPORARY_PATCHES = []

def patch_gemma3_processor():
    try:
        import transformers.models.gemma3.processing_gemma3
    except:
        return
    from transformers.models.gemma3.processing_gemma3 import (
        ImageInput,
        PreTokenizedInput,
        Unpack,
        Gemma3ProcessorKwargs,
        make_nested_list_of_images,
        TextInput,
        BatchFeature,
        to_py_obj,
    )
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos=None,
        audio=None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        if images is not None:
            batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(batched_images, **output_kwargs["images_kwargs"])

            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.boi_token] * len(images)) for images in batched_images]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Replace image tokens by the full expanded sequence
            batch_num_crops = to_py_obj(image_inputs.pop("num_crops"))
            text_with_crops = text
            for batch_idx, (prompt, images, num_crops) in enumerate(zip(text, batched_images, batch_num_crops)):
                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images)} images."
                    )

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(num_crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len(self.boi_token) :]
                        text_with_crops[batch_idx] = prompt

            # Expand placeholder image tokens to the full image token sequence
            text = [prompt.replace(self.boi_token, self.full_image_sequence) for prompt in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        # text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"], return_tensors="np")

        # Fix double BOS tokens
        bos = self.tokenizer.bos_token
        n = len(bos)
        text = [x[i + n:] if (i := x.find(bos)) != -1 else x for x in text]

        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

        # Add token type ids manually, as tokenizer can't do arbitrary position token types
        # [TODO] FAILS for batched tokens since text_inputs["input_ids"] is a list of lists, so np.array creates an object!
        # array_ids = np.array(text_inputs["input_ids"])
        # mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
        # mm_token_type_ids[array_ids == self.image_token_id] = 1
        # text_inputs = {k: v.tolist() for k, v in text_inputs.items()}  # in case user requested list inputs
        # text_inputs["token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)
    
    old_keys = inspect.signature(transformers.models.gemma3.processing_gemma3.Gemma3Processor.__call__).parameters
    new_keys = inspect.signature(__call__).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3Processor.")
    else:
        transformers.models.gemma3.processing_gemma3.Gemma3Processor.__call__ = __call__
    return
pass
TEMPORARY_PATCHES.append(patch_gemma3_processor)
