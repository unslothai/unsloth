from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained("unsloth/gemma-3-1B-it", load_in_4bit=True, load_in_8bit=False)
