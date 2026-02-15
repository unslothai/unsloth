from unsloth import FastLanguageModel
from peft import PeftModel, PeftModelForCausalLM

lora_path="/home/support/new-ui-prototype/outputs/meta-llama_Llama-3.1-8B-Instruct_1771048481"
adapter_name_to_load="test"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/support/new-ui-prototype/outputs/meta-llama_Llama-3.1-8B-Instruct_1771048481",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Quick sanity check
FastLanguageModel.for_inference(model)
print("Model loaded successfully!")
print(f"Model type before unloading: {type(model)}")
print(f"Tokenizer: {type(tokenizer)}")
print(f"Model class before unloading: {model.__class__.__name__}")
# Test generation
#inputs = tokenizer("What is the capital of France?", return_tensors="pt").to(model.device)
#outputs = model.generate(**inputs, max_new_tokens=32)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))


print("unloading base model")
if isinstance(model, (PeftModel, PeftModelForCausalLM)):
        print("Model is a PeftModel. Unloading adapters...")
        unwrapped_base_model = model.unload()
        model = unwrapped_base_model
        #if hasattr(model, 'peft_config') and model.peft_config:
        #        print("Found lingering adapter configurations. Deleting them now...")
        #        # Create a static list of keys before iterating and deleting
        #        for name in list(model.peft_config.keys()):
        #            if name == "default":
        #                continue
        #            print(f"Deleting adapter config: '{name}'")
        #            mode=model.delete_adapter(name)
        #model.disable_adapters()
        if hasattr(model, 'peft_config'):
            del model.peft_config
print(f"Model type post unloading{type(model)}")
print(f"Model class post unloading: {model.__class__.__name__}")
#print(f"model: {model}")

#print("generating using base model")
#inputs = tokenizer("What is the capital of Lebanon?", return_tensors="pt").to(model.device)
#outputs = model.generate(**inputs, max_new_tokens=32)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#print(f"model config: {model.config}")
#print(f"model: {model}")
#print(f"model.peft_config: {model.peft_config}")

#print("loading lora dapter")
#model.load_adapter(lora_path, adapter_name=adapter_name_to_load)
#model.enable_adapters
#model.set_adapter(adapter_name_to_load)
model = PeftModel.from_pretrained(model, lora_path, adapter_name=adapter_name_to_load)
print(f"Model type {type(model)}")
print(f"Model class: {model.__class__.__name__}")
#print(f"model: {model}")
#print("generating using peft model hopefully")
#inputs = tokenizer("What is the capital of Brasil?", return_tensors="pt").to(model.device)
#outputs = model.generate(**inputs, max_new_tokens=32)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("unloading base model")
if isinstance(model, (PeftModel, PeftModelForCausalLM)):
        print("Model is a PeftModel. Unloading adapters...")
        unwrapped_base_model = model.unload()
        model = unwrapped_base_model
        #if hasattr(model, 'peft_config') and model.peft_config:
        #        print("Found lingering adapter configurations. Deleting them now...")
        #        # Create a static list of keys before iterating and deleting
        #        for name in list(model.peft_config.keys()):
        #            if name == "default":
        #                continue
        #            print(f"Deleting adapter config: '{name}'")
        #            mode=model.delete_adapter(name)
        #model.disable_adapters()
        if hasattr(model, 'peft_config'):
            del model.peft_config
print(f"Model type post unloading{type(model)}")
print(f"Model class post unloading: {model.__class__.__name__}")
