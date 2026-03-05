# Finance Coach Example

This is the reference implementation of the local-lora-cookbook domain adapter.
It fine-tunes a local LLM to answer personal finance questions from your own
transaction data — entirely on-device.

## What it does

1. **Phase 1 — collect**: runs 90+ finance questions through a local SQL RAG
   pipeline against your SQLite transaction database, capturing question/SQL/data/answer
2. **Phase 2 — annotate**: sends each raw example to Claude once to get a
   gold-standard coaching response
3. **Train**: fine-tunes Qwen3.5-4B with LoRA on those gold-standard pairs
4. **Fuse + Serve**: merges the adapter and serves the fine-tuned model locally

## Quick start

```bash
# 1. Create synthetic test data (or import your own bank export)
python3 examples/finance_coach/generate_sample_db.py --out ~/finance.db

# 2. Make sure Ollama is running with a base model
ollama serve
ollama pull qwen3.5:4b

# 3. Phase 1: collect baseline answers (no API key needed)
python3 scripts/generate_training_data.py \
  --domain examples/finance_coach \
  --db ~/finance.db

# 4. Phase 2: annotate with Claude (one-time API cost ~$2-5 for 90 examples)
python3 scripts/generate_training_data.py \
  --domain examples/finance_coach \
  --db ~/finance.db \
  --phase annotate \
  --key sk-ant-...

# 5. Train
python3 scripts/train_lora.py \
  --data examples/finance_coach/training_data/training_data.jsonl \
  --system-prompt "You are a private finance coach. All data is processed locally."

# 6. Fuse and serve (Apple Silicon)
python3 scripts/train_lora.py --fuse
python3 scripts/train_lora.py --serve

# 7. Or export to GGUF for Ollama (any platform)
python3 scripts/train_lora.py --gguf --ollama-model finance-coach
ollama run finance-coach "How much did I spend on restaurants last month?"
