# GGUF Tool Calling Benchmark Results

Prompt: "List and categorize all the songs that charted #3 on the Billboard Hot 100 in 2015."
10 runs per configuration, web search + code execution + thinking enabled.
GPU: NVIDIA B200, CUDA_VISIBLE_DEVICES=2.

Ground truth: 4 songs peaked at #3 in 2015 -- "Love Me like You Do" (Ellie Goulding), "Earned It" (The Weeknd), "Watch Me" (Silento), "Drag Me Down" (One Direction).

## Cartesian Grid: Model x Quant x KV Cache

| Model | Quant | KV Cache | OK/10 | Avg Time | Avg Tools | XML Leaks | URL Fetch | Peak3 Avg | All 4/4 | Best Songs |
|-------|-------|----------|-------|----------|-----------|-----------|-----------|-----------|---------|------------|
| 4B | UD-Q4_K_XL | f16 | 10/10 | 9.8s | 3.5 | 0/10 | 4/10 | 0.8/4 | 2/10 | 9 |
| 4B | UD-Q4_K_XL | bf16 | 10/10 | 10.6s | 4.5 | 0/10 | 4/10 | 0.4/4 | 1/10 | 5 |
| 4B | Q8_0 | f16 | 10/10 | 4.9s | 2.4 | 0/10 | 8/10 | 0.4/4 | 1/10 | 5 |
| 4B | Q8_0 | bf16 | 10/10 | 8.0s | 3.0 | 0/10 | 5/10 | 0.0/4 | 0/10 | 0 |
| 9B | UD-Q4_K_XL | f16 | 10/10 | 6.7s | 2.0 | 0/10 | 5/10 | 0.0/4 | 0/10 | 3 |
| 9B | UD-Q4_K_XL | bf16 | 9/10 | 49.5s | 2.4 | 0/10 | 5/10 | 0.0/4 | 0/10 | 1 |
| 9B | Q8_0 | f16 | 10/10 | 7.4s | 2.5 | 0/10 | 5/10 | 0.0/4 | 0/10 | 2 |
| 9B | Q8_0 | bf16 | 10/10 | 10.4s | 2.7 | 0/10 | 6/10 | 1.0/4 | 2/10 | 15 |
| **27B** | **UD-Q4_K_XL** | **bf16** | **9/10** | **131.1s** | **13.8** | **0/10** | **7/10** | **2.7/4** | **6/10** | **27** |
| 27B | UD-Q4_K_XL | f16 | 7/10 | 201.6s | 14.1 | 0/10 | 8/10 | 2.0/4 | 5/10 | 26 |
| 27B | Q8_0 | f16 | 4/10 | 312.5s | 16.0 | 1/10 | 10/10 | 2.4/4 | 6/10 | 28 |
| 27B | Q8_0 | bf16 | 5/10 | 258.4s | 16.5 | 2/10 | 10/10 | 0.9/4 | 1/10 | 27 |
| 35B-A3B | UD-Q4_K_XL | f16 | 3/10 | 353.6s | 14.7 | 1/10 | 6/10 | 1.2/4 | 3/10 | 27 |
| 35B-A3B | UD-Q4_K_XL | bf16 | 3/10 | 356.2s | 17.2 | 1/10 | 8/10 | 1.6/4 | 4/10 | 27 |
| 35B-A3B | Q8_0 | f16 | 2/10 | 372.1s | 17.6 | 1/10 | 7/10 | 1.2/4 | 3/10 | 26 |
| 35B-A3B | Q8_0 | bf16 | 6/10 | 267.7s | 17.5 | 1/10 | 8/10 | 2.4/4 | 6/10 | 27 |

**Column definitions:**
- **Peak3 Avg**: Average number of correct peak-#3 songs found per run (out of 4)
- **All 4/4**: Runs where all 4 correct songs were identified
- **Best Songs**: Maximum number of Billboard 2015 songs mentioned in any single run (out of 31 tracked)
- **URL Fetch**: Runs where the model used web_search with `url` parameter to fetch full page content

## Key Findings

1. **27B UD-Q4_K_XL + bf16 KV is the sweet spot.** 6/10 runs found all 4 correct songs, 0 XML leaks, 131s average. Best balance of accuracy, speed, and reliability.

2. **Larger models use tools more effectively.** 27B and 35B-A3B models used 13-17 tool calls per query (vs 2-4 for 4B/9B), performing multiple searches and URL fetches to find the answer.

3. **27B Q8_0 had the highest raw accuracy (6/10 all-4/4) but lower reliability** -- only 4/10 OK runs due to timeouts on long agentic chains. The UD-Q4_K_XL quant is more practical.

4. **4B models were fastest (5-10s) but least accurate.** They occasionally found all 4 songs (2/10 best case) when they happened to fetch the right Wikipedia page.

5. **9B was surprisingly weaker than 4B on this task.** It used fewer tool calls and rarely extracted song data from fetched pages. The 9B model may need higher temperature or different prompting for this specific task type.

6. **35B-A3B had reliability issues.** Most runs timed out or errored due to slow per-token generation with many tool iterations. When it completed (2-6/10 OK), accuracy was comparable to 27B.

7. **bf16 KV cache had mixed effects.** For 27B it improved both speed (131s vs 202s) and accuracy (6/10 vs 5/10 all-4/4). For smaller models it had no consistent benefit.

8. **XML leaks are nearly eliminated.** 0/10 for all 4B and 9B configs, and only 1-2/10 for the largest models (which generate much more text in complex agentic loops).

## Before vs After (4B UD-Q4_K_XL, f16 KV)

| Metric | Before Changes | After Changes |
|--------|---------------|---------------|
| XML leaks | 10/10 | 0/10 |
| URL fetches | 0/10 | 4/10 |
| Peak3 accuracy | 0.0/4 | 0.8/4 |
| Runs with all 4 songs | 0/10 | 2/10 |
| Avg time | 12.3s | 9.8s |
