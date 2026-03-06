# local-lora-cookbook

Fine-tune a local LLM on your app's own data — runs entirely on your device,
zero cloud after training.

**Apple Silicon (M-series Mac)** uses mlx-lm.
**Linux / NVIDIA GPU** uses Unsloth + TRL.

## The idea

Most AI apps call a cloud LLM and hope for the best. This cookbook shows a
different pattern:

1. Use your app's existing RAG pipeline to generate hundreds of (question, data, answer) examples automatically
2. Use Claude **once** to annotate gold-standard responses in your app's exact style
3. Fine-tune a 4B model locally with LoRA — takes 15-40 min on an M4 Mac mini
4. Fuse and serve the fine-tuned model on-device — forever, for free

The result is a model that speaks your app's language, knows your data schema,
and runs at ~60 tokens/second on consumer hardware.

## Included example: Finance Coach

A working end-to-end example that fine-tunes a personal finance coaching model
on your transaction history. See [`examples/finance_coach/`](examples/finance_coach/).

## Quick start

```bash
git clone https://github.com/sandseb123/local-lora-cookbook
cd local-lora-cookbook

# Apple Silicon
pip install -r requirements-apple-silicon.txt

# Linux / CUDA
pip install -r requirements-linux-cuda.txt

# Generate synthetic finance data to test with
python3 examples/finance_coach/generate_sample_db.py

# Start Ollama and pull a base model
ollama serve
ollama pull qwen3.5:4b

# Phase 1: collect baseline examples (~20 min, no API key)
python3 scripts/generate_training_data.py \
  --domain examples/finance_coach \
  --db ~/finance.db

# Phase 2: annotate with Claude (~5 min, costs ~$2-5)
python3 scripts/generate_training_data.py \
  --domain examples/finance_coach \
  --db ~/finance.db \
  --phase annotate \
  --key sk-ant-...

# Train the LoRA adapter (~15-40 min on M4, ~10-25 min on RTX 3090)
python3 scripts/train_lora.py \
  --data examples/finance_coach/training_data/training_data.jsonl \
  --system-prompt "You are a private finance coach."

# Fuse and serve (Apple Silicon)
python3 scripts/train_lora.py --fuse
python3 scripts/train_lora.py --serve

# Or export to GGUF + Ollama (any platform)
python3 scripts/train_lora.py --gguf --ollama-model finance-coach
ollama run finance-coach "How much did I spend on restaurants last month?"
```

## Architecture

```text
Phase 1 - COLLECT (local, free, ~20 min)

  your.db --> domain.py/rag.py --> Ollama LLM --> raw_examples.jsonl
                                                        |
----------------------------------------------------------------
Phase 2 - ANNOTATE (cloud, one-time, ~$2-5)             |
                                                        v
                                                  Claude API
                                              (rewrites each answer)
                                                        |
----------------------------------------------------------------
Phase 3 - TRAIN (local, free, 15-40 min)                |
                                                        v
                                              training_data.jsonl
                                                        |
                                                  train_lora.py
                                                 (LoRA fine-tune)
                                                        |
                                                   LoRA adapter
                                                        |
----------------------------------------------------------------
Phase 4 - SERVE (local, free, forever)                  |
                                                   +----+----+
                                                   |         |
                                             --fuse/serve  --gguf
                                             mlx-lm :8080  Ollama
                                            (Apple Silicon) (any)
```

**Phase 1 (collect)**: Your domain's `answer()` function runs each question
through your RAG pipeline and records the SQL, data rows, and baseline answer.

**Phase 2 (annotate)**: Claude rewrites each baseline answer to gold-standard
quality in your app's voice. This is the only cloud call — you pay once.

**Train**: `train_lora.py` fine-tunes Qwen3.5-4B (or any HuggingFace model)
on the (prompt, gold_answer) pairs using LoRA.

**Serve**: The fused model is served via mlx-lm (port 8080) on Mac or
exported to GGUF and registered with Ollama on Linux.

## Adapting to your own domain

Create a directory under `examples/` with three files:

### `examples/my_domain/domain.py`

```python
SYSTEM_PROMPT = "You are a private [X] coach..."

QUESTIONS = [
    "Question 1?",
    "Question 2?",
    ...
]

ANNOTATION_PROMPT_TEMPLATE = """
You are writing gold-standard responses for [X]...

Question: {question}
SQL: {sql}
Data: {results}
{context_section}
Baseline answer: {llama_answer}

Rewrite to gold-standard quality...
"""

def answer(question: str, db_path: str) -> dict:
    # Call your RAG pipeline
    # Return: {sql, rows, answer, error, context_section, full_prompt}
    ...
```

### `examples/my_domain/rag.py`

Implements your SQL RAG pipeline. Needs:
- Schema description (for the SQL generation prompt)
- `answer(question, db_path) -> dict`

See `examples/finance_coach/rag.py` for a complete reference implementation.

### `examples/my_domain/schema.sql`

Your SQLite schema. Used as documentation and for generating sample data.

Then run:

```bash
python3 scripts/generate_training_data.py --domain examples/my_domain --db /path/to/db
```

## Hardware requirements

| Platform | RAM | Training time (default) | Training time (overnight) |
|---|---|---|---|
| M4 Mac mini | 16 GB | 15-40 min | 60-120 min |
| M3 MacBook Pro | 18 GB | 20-50 min | 90-150 min |
| RTX 3090 (24 GB) | — | 10-25 min | 30-60 min |
| RTX 4090 (24 GB) | — | 8-20 min | 25-50 min |

For tighter memory: `python3 scripts/train_lora.py --low-memory`

## Model

Default base model: **Qwen/Qwen3.5-4B-Instruct**
(MLX: `mlx-community/Qwen3.5-4B-Instruct-4bit`)

Any HuggingFace causal LM works. Pass `--model your/model` to override.

## Cost

- **Phase 1 (collect)**: free — runs entirely locally via Ollama
- **Phase 2 (annotate)**: ~$2-5 for ~100 examples via Claude Sonnet
- **Training**: free — runs on your local GPU/CPU
- **Inference**: free — serves from your device indefinitely

## License

MIT
