# local_lora.md — local-lora-cookbook

This file is a brief for Claude Code. Read it before making any changes.

## What this repo is

A standalone, public cookbook for fine-tuning a local LLM on any data-aware
app's own data using LoRA. It was extracted and generalized from a private
health app project. The finance coach is the reference example.

## Architecture

### Domain adapter pattern

Everything domain-specific lives in `examples/<domain_name>/`:


examples/finance_coach/
domain.py <- required: QUESTIONS, SYSTEM_PROMPT, ANNOTATION_PROMPT_TEMPLATE, answer()
rag.py <- SQL RAG pipeline for this domain
schema.sql <- SQLite schema documentation
generate_sample_db.py <- creates synthetic data for testing
training_data/ <- gitignored output directory

The generic scripts (`generate_training_data.py`, `train_lora.py`) import
nothing from a specific domain — they load `domain.py` dynamically via
`importlib`.

### Data flow


Phase 1 (collect): questions -> domain.answer() -> Ollama -> raw_examples.jsonl
Phase 2 (annotate): raw_examples.jsonl -> Claude API -> training_data.jsonl
Training: training_data.jsonl -> train_lora.py -> LoRA adapter
Post-train: adapter -> fuse -> GGUF -> Ollama (or mlx-lm serve on Mac)

### Key files

| File | Purpose |
|---|---|
| `lora_cookbook/llm/ollama_client.py` | Generic LLM client — mlx-lm (port 8080) + Ollama fallback |
| `scripts/generate_training_data.py` | Two-phase pipeline: collect then annotate |
| `scripts/train_lora.py` | LoRA fine-tuning: mlx-lm (Mac) or Unsloth (Linux) |
| `examples/finance_coach/rag.py` | Finance SQL RAG reference implementation |
| `examples/finance_coach/domain.py` | Domain adapter: questions, prompts, answer() |

## What to work on next

1. `examples/finance_coach/import_csv.py` — import real bank CSV exports into finance.db
2. Test the end-to-end pipeline with generate_sample_db.py
3. A second domain example (e.g. reading_tracker or workout_log)
4. CI smoke test — tests/test_domain_interface.py
5. scripts/quickstart.py — interactive setup script

## Important conventions

- **Never commit personal data.** `.gitignore` excludes all `*.db`, `*.jsonl` training files.
- **No hardcoded API keys.** Use `--key` CLI arg or `ANTHROPIC_API_KEY` env var.
- **Both phases are resumable.** Re-running skips already-done items.
- **Zero external dependencies** in `ollama_client.py` — stdlib only.