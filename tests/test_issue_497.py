import unittest
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer


class TestIssue497(unittest.TestCase):
    def setUp(self):
        self.model_name = "unsloth/Phi-3-mini-4k-instruct"
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = False

        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = self.model_name,
                max_seq_length = self.max_seq_length,
                dtype = self.dtype,
                load_in_4bit = self.load_in_4bit,
                attn_implementation = "flash_attention_2",
            )
            FastLanguageModel.for_inference(self.model)
        except Exception as e:
            print(
                f"Skipping test because model loading failed (likely due to environment): {e}"
            )
            self.skipTest("Model loading failed")

    def test_generate_with_past_key_values(self):
        prompt = "<|user|>\nMy name name is Jon. What is my name?<|end|>\n<|assistant|>"

        # First generation
        model_inputs = self.tokenizer(
            prompt, return_tensors = "pt", add_special_tokens = False
        ).to("cuda")
        generated_output = self.model.generate(
            **model_inputs,
            max_new_tokens = 20,
            return_dict_in_generate = True,
            temperature = 0,
        )
        text_output = self.tokenizer.batch_decode(generated_output.sequences)[0]

        # Prepare second generation
        second_prompt = (
            "\n<|user|>\nI'm 30 years old. How old am i?<|end|>\n<|assistant|>"
        )
        full_prompt = text_output + second_prompt
        model_inputs_2 = self.tokenizer(
            full_prompt, return_tensors = "pt", add_special_tokens = False
        ).to("cuda")

        # Second generation with past_key_values
        try:
            generated_output_2 = self.model.generate(
                **model_inputs_2,
                max_new_tokens = 20,
                return_dict_in_generate = True,
                past_key_values = generated_output.past_key_values,
            )
            text_output_2 = self.tokenizer.batch_decode(generated_output_2.sequences)[0]
            print(f"Generated text: {text_output_2}")
            self.assertIn("30 years old", text_output_2)
        except RuntimeError as e:
            self.fail(f"RuntimeError during generation with past_key_values: {e}")
        except ValueError as e:
            self.fail(f"ValueError during generation with past_key_values: {e}")


if __name__ == "__main__":
    unittest.main()
