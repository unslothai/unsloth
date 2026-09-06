# Windows account path budget

Example root: `C:\Users\Research Engineer\AppData\Local\UnslothStudio`. MAX_PATH is 260 UTF-16 units including the trailing NUL.

| Relative leaf | Owner path units | Account path units | Maximum root units |
| --- | ---: | ---: | ---: |
| `studio.db` | 64 | 106 | 207 |
| `outputs\Llama-3.1-8B-Instruct\checkpoint-10000\adapter_model.safetensors` | 127 | 169 | 144 |
| `exports\Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 96 | 138 | 175 |
| `rag\documents\0123456789abcdef0123456789abcdef\quarterly-research-report.pdf` | 131 | 173 | 140 |
| `sandbox\0123456789abcdef0123456789abcdef\workspace\analysis\results.csv` | 126 | 168 | 145 |
| `assets\datasets\0123456789abcdef0123456789abcdef\training-data.jsonl` | 123 | 165 | 148 |

All sample roots/leaves are ASCII, so displayed character and UTF-16-unit counts coincide. The test also checks non-BMP and decomposed Unicode unit counts. Account UUIDs add exactly 42 units; longer configured roots may require Windows long-path support. No new owner path restriction is added.
