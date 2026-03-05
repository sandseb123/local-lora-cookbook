# local-lora-cookbook

Fine-tune a local LLM on your app's own data — runs entirely on your device,
zero cloud after training.

**Apple Silicon (M-series Mac)** uses mlx-lm.
**Linux / NVIDIA GPU** uses Unsloth + TRL.

## The idea

1. Use your app's existing RAG pipeline to generate hundreds of (question, data, answer) examples automatically
2. Use Claude **once** to annotate gold-standard responses in your app's exact style
3. Fine-tune a 4B model locally with LoRA — takes 15-40 min on an M4 Mac mini
4. Fuse and serve the fine-tuned model on-device — forever, for free

## Included example: Finance Coach

See [`examples/finance_coach/`](examples/finance_coach/).

## Quick start

```bash
git clone https://github.com/sandseb123/local-lora-cookbook
cd local-lora-cookbook

# Apple Silicon
pip install -r requirements-apple-silicon.txt

# Linux / CUDA
pip install -r requirements-linux-cuda.txt

# Generate synthetic finance data
python3 examples/finance_coach/generate_sample_db.py

# Start Ollama
ollama serve
ollama pull qwen3.5:4b

# Phase 1: collect baseline examples (~20 min, no API key)
python3 scripts/generate_training_data.py \
  --domain examples/finance_coach \
  --db ~/finance.db

# Phase 2: annotate with Claude (~5 min, ~$2-5)
python3 scripts/generate_training_data.py \
  --domain examples/finance_coach \
  --db ~/finance.db \
  --phase annotate \
  --key sk-ant-...

# Train
python3 scripts/train_lora.py \
  --data examples/finance_coach/training_data/training_data.jsonl \
  --system-prompt "You are a private finance coach."

# Fuse and serve (Apple Silicon)
python3 scripts/train_lora.py --fuse
python3 scripts/train_lora.py --serve

# Or export to GGUF + Ollama
python3 scripts/train_lora.py --gguf --ollama-model finance-coach
ollama run finance-coach "How much did I spend on restaurants last month?"

## License
MIT