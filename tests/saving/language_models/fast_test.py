from unsloth import FastLanguageModel


merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
        model_name="./unsloth_out/merged_llama_text_model",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
    )
