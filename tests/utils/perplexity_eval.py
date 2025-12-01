from tqdm import tqdm
import torch
import pandas as pd

model_comparison_results = {}
# return the perplexity of the model on the dataset
# The perplexity is computed on each example, individually, with a sliding window for examples longer than 512 tokens.


def ppl_model(model, tokenizer, dataset):
    nlls = []
    max_length = 2048
    stride = 512
    for s in tqdm(range(len(dataset["text"]))):
        encodings = tokenizer(dataset["text"][s], return_tensors = "pt")
        seq_len = encodings.input_ids.size(1)
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            # Create attention mask based on pad token id
            pad_token_id = (
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
            attention_mask = (input_ids != pad_token_id).long()
            with torch.no_grad():
                outputs = model(
                    input_ids, labels = target_ids, attention_mask = attention_mask
                )
                neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


# --------------------------------------------------------------------


## ----------- Reporting helper function ----------- ##


# Create a simple function to add results to the comparison
def add_to_comparison(model_name, ppl):
    """Add model results to the comparison tracker"""
    model_comparison_results[model_name] = {"ppl": ppl}
    # return model_comparison_results


# Create a function to print the comparison report whenever needed
def print_model_comparison():
    """Print a comparison of all models evaluated so far"""
    if not model_comparison_results:
        print("No model results available for comparison")
        return

    print("\n==== MODEL COMPARISON REPORT ====")

    # Create a comparison dataframe
    comparison_df = pd.DataFrame(
        {
            "Model": list(model_comparison_results.keys()),
            # "Perplexity": [results["ppl"] for results in model_comparison_results.values()],
            "Perplexity": [
                # Convert tensors to CPU and then to float if needed
                results["ppl"].cpu().item()
                if torch.is_tensor(results["ppl"])
                else results["ppl"]
                for results in model_comparison_results.values()
            ],
        }
    )

    # Display the comparison table
    print("\nComparison Table:")
    print(comparison_df.to_string(index = False))
