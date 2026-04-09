from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.utils.cleanup_utils import safe_remove_directory


print(f"\n{'='*80}")
print("🔍 PHASE 1: Loading Base Model")
print(f"{'='*80}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)


print("✅ Base model loaded successfully!")

### Attemtping save merge


print(f"\n{'='*80}")
print("🔍 PHASE 2: Attempting save_pretrained_merged (Should Warn)")
print(f"{'='*80}")

with warnings.catch_warnings(record = True) as w:
    warnings.simplefilter("always")
    model.save_pretrained_merged("test_output", tokenizer)

    # Verify warning
    assert len(w) >= 1, "Expected warning but none raised"
    warning_msg = str(w[0].message)
    expected_msg = "Model is not a PeftModel (no Lora adapters detected). Skipping Merge. Please use save_pretrained() or push_to_hub() instead!"
    assert expected_msg in warning_msg, f"Unexpected warning: {warning_msg}"
    assert expected_msg in warning_msg, f"Unexpected warning: {warning_msg}"

print("✅ Correct warning detected for non-PeftModel merge attempt!")


print(f"\n{'='*80}")
print("🔍 PHASE 3: Using save_pretrained (Should Succeed)")
print(f"{'='*80}")


try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Treat warnings as errors here
        model.save_pretrained("test_output")
        print("✅ Standard save_pretrained completed successfully!")
except Exception as e:
    assert False, f"Phase 3 failed: {e}"

safe_remove_directory("./test_output")
safe_remove_directory("./unsloth_compiled_cache")
