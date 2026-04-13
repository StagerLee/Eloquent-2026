Data Results: https://huggingface.co/datasets/StagerLee/eloquent-2026

# Eloquent 2026
 
Mechanistic interpretability study of culturally grounded open-ended question answering in large language models. Built on the ELOQUENT cultural robustness benchmark.

 
## Experimental Setup
 
### Compute Infrastructure
All open-weight model inference was conducted on the Snellius HPC cluster (SURF, University of Amsterdam), using NVIDIA A100-SXM4-40GB GPUs via the `gpu_a100` partition. Each model was run as an independent SLURM batch job, allowing parallel execution across nodes.
 
### Dataset
We use the ELOQUENT cultural robustness benchmark, consisting of open-ended cultural question answering prompts across five languages and two prompt types. The dataset comprises 10 files in total:
 
- **Languages**: English (en), German (de), French (fr), Italian (it), Swedish (sv)
- **Prompt types**: specific (culturally grounded) and unspecific (culturally underspecified)
 
### Models
We evaluate four open-weight models alongside two closed API models:
 
| Model | Size | Provider | Purpose |
|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | 7B | Mistral AI | General baseline |
| Llama-3.1-8B-Instruct | 8B | Meta | Mechanistic interpretability |
| Qwen2.5-7B-Instruct | 7B | Alibaba | Multilingual/cultural strength |
| Aya-Expanse-8B | 8B | Cohere | Purpose-built multilingual |
| GPT-4.1-mini | — | OpenAI | Closed model reference |
| Gemini | — | Google | Closed model reference |
 
All open-weight models were run in float16 precision on a single A100 GPU using HuggingFace Transformers with `device_map="auto"`.
 
### Sampling Protocol
Each prompt was run **15 independent times** under a sterile session protocol:
 
- Each generation is fully independent with no shared state, KV cache, or conversation history across runs
- A unique deterministic seed is assigned to each `(model, prompt_id, run_index)` triple, ensuring reproducibility while guaranteeing diversity across runs
- Sampling parameters: `temperature=0.7`, `do_sample=True`, `max_new_tokens=200`
- GPU cache is explicitly cleared between runs via `torch.cuda.empty_cache()`
 
### Output Format
Results are stored as `.jsonl` files, one line per `(prompt, run)` pair, following the schema:
```json
{"id": "1-1", "run": 3, "prompt": "...", "answer": "..."}
```
 
Output files are named `{model_tag}_{lang}_{type}.jsonl` (e.g. `mistral7b_en_specific.jsonl`) and organized into per-model subdirectories under `results/`. The pipeline supports resumption — previously completed `(id, run)` pairs are skipped on resubmission.
 
### Repository Structure
```
Eloquent-2026/
├── data/                        # Input prompt files (10 x .jsonl)
│   ├── en_specific.jsonl
│   ├── en_unspecific.jsonl
│   └── ...
├── results/                     # Output files organized by model
│   ├── mistral7b/
│   ├── llama31-8b/
│   ├── qwen25-7b/
│   └── aya-expanse-8b/
├── logs/                        # SLURM job logs
├── run_llm.py                   # Main inference script
├── job_mistral.sh               # SLURM job script — Mistral
├── job_llama.sh                 # SLURM job script — Llama
├── job_qwen.sh                  # SLURM job script — Qwen
└── job_aya.sh                   # SLURM job script — Aya
```
 
### Reproducibility
All scripts, job configurations, and results are version-controlled in this repository. The HuggingFace model cache is stored on Snellius scratch storage (`/scratch-shared/$USER/hf_cache`) to avoid home directory quota limits.
