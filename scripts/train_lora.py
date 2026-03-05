#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script
=======================
Fine-tunes any Qwen/Llama-family model on your gold-standard coaching examples.

Backend is chosen automatically:
  Apple Silicon (M-series Mac)  ->  mlx-lm   (pip install mlx-lm)
  Linux / NVIDIA GPU            ->  Unsloth  (pip install unsloth trl datasets)

Usage
-----
  python3 scripts/train_lora.py --data examples/finance_coach/training_data/training_data.jsonl
  python3 scripts/train_lora.py --data ... --overnight
  python3 scripts/train_lora.py --fuse
  python3 scripts/train_lora.py --serve
  python3 scripts/train_lora.py --gguf --ollama-model finance-coach
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


BACKEND = "mlx" if _is_apple_silicon() else "unsloth"

MLX_MODEL     = "mlx-community/Qwen3.5-4B-Instruct-4bit"
HF_MODEL      = "Qwen/Qwen3.5-4B-Instruct"
DEFAULT_MODEL = MLX_MODEL if BACKEND == "mlx" else HF_MODEL

DEFAULTS = {
    "iters": 400, "batch_size": 4, "lora_layers": 8, "lora_rank": 8,
    "lora_alpha": 16, "learning_rate": 1e-4, "max_seq_length": 1024,
    "grad_accum": 4, "steps_per_report": 10, "steps_per_eval": 50,
    "save_every": 100, "seed": 42, "val_split": 0.15,
}

OVERNIGHT = {
    "iters": 2000, "batch_size": 4, "lora_layers": 16, "lora_rank": 16,
    "lora_alpha": 32, "learning_rate": 5e-5, "max_seq_length": 2048,
    "grad_accum": 4, "steps_per_report": 20, "steps_per_eval": 100,
    "save_every": 200, "seed": 42, "val_split": 0.15,
}

LOW_MEMORY = {
    "iters": 400, "batch_size": 1, "lora_layers": 4, "lora_rank": 4,
    "lora_alpha": 8, "learning_rate": 1e-4, "max_seq_length": 512,
    "grad_accum": 4, "steps_per_report": 10, "steps_per_eval": 50,
    "save_every": 100, "seed": 42, "val_split": 0.15,
}


def _load_examples(data_path: Path) -> list[dict]:
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cleaned = [
                {k: v for k, v in msg.items() if k != "_question"}
                for msg in obj.get("messages", [])
            ]
            if len(cleaned) >= 2:
                examples.append({"messages": cleaned})
    return examples


def prepare_data(data_path: Path, out_dir: Path, val_split: float,
                 seed: int) -> tuple[dict, dict, Path]:
    examples = _load_examples(data_path)
    if not examples:
        print(f"ERROR: No valid examples found in {data_path}")
        sys.exit(1)

    random.seed(seed)
    random.shuffle(examples)

    n_val  = max(1, int(len(examples) * val_split))
    n_test = max(1, min(5, n_val))

    if len(examples) - n_val - n_test < 5:
        splits = {
            "train": examples,
            "valid": examples[:min(3, len(examples))],
            "test":  examples[:min(2, len(examples))],
        }
    else:
        splits = {
            "train": examples[n_val + n_test:],
            "valid": examples[:n_val],
            "test":  examples[n_val:n_val + n_test],
        }

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(data_dir / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return {k: len(v) for k, v in splits.items()}, splits, data_dir


# ── MLX backend ───────────────────────────────────────────────────────────────

def run_training_mlx(model: str, data_dir: Path, adapter_dir: Path, cfg: dict) -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("mlx-lm not installed.  Run:  pip install mlx-lm")
        sys.exit(1)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = adapter_dir / "lora_config.yaml"
    with open(cfg_path, "w") as f:
        f.write(f"lora_parameters:\n  rank: {cfg['lora_rank']}\n"
                f"  alpha: {cfg['lora_alpha']}\n  dropout: 0.0\n  scale: 10.0\n")

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model, "--data", str(data_dir), "--train",
        "--iters", str(cfg["iters"]),
        "--batch-size", str(cfg["batch_size"]),
        "--lora-layers", str(cfg["lora_layers"]),
        "--learning-rate", str(cfg["learning_rate"]),
        "--adapter-path", str(adapter_dir),
        "--max-seq-length", str(cfg["max_seq_length"]),
        "--steps-per-report", str(cfg["steps_per_report"]),
        "--steps-per-eval", str(cfg["steps_per_eval"]),
        "--save-every", str(cfg["save_every"]),
        "--seed", str(cfg["seed"]),
        "--config", str(cfg_path),
    ]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: mlx_lm.lora failed (exit {result.returncode})")
        sys.exit(result.returncode)


def run_fuse_mlx(model: str, adapter_dir: Path, fused_dir: Path) -> None:
    fused_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm.fuse",
           "--model", model, "--adapter-path", str(adapter_dir),
           "--save-path", str(fused_dir)]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\nFused model: {fused_dir}")


def run_serve_mlx(fused_dir: Path) -> None:
    if not fused_dir.exists():
        print(f"ERROR: Fused model not found at {fused_dir}. Run --fuse first.")
        sys.exit(1)
    cmd = [sys.executable, "-m", "mlx_lm.server",
           "--model", str(fused_dir), "--port", "8080"]
    print("$", " ".join(cmd))
    subprocess.run(cmd)


def run_gguf_mlx(model: str, adapter_dir: Path, gguf_path: Path,
                 fused_dir: Path, system_prompt: str,
                 ollama_model: str = "my-coach") -> None:
    if not fused_dir.exists():
        run_fuse_mlx(model, adapter_dir, fused_dir)
    converter = shutil.which("llama-convert-hf-to-gguf") or \
                shutil.which("convert-hf-to-gguf")
    if not converter:
        print("ERROR: llama.cpp not found. Install with: brew install llama.cpp")
        sys.exit(1)
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([converter, str(fused_dir),
                             "--outfile", str(gguf_path), "--outtype", "q4_k_m"])
    if result.returncode != 0:
        sys.exit(result.returncode)
    _register_with_ollama(gguf_path, ollama_model, system_prompt)


# ── Unsloth backend ───────────────────────────────────────────────────────────

def run_training_unsloth(model: str, splits: dict, adapter_dir: Path,
                         cfg: dict) -> None:
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError:
        print("Run: pip install unsloth trl datasets transformers accelerate bitsandbytes")
        sys.exit(1)

    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model, max_seq_length=cfg["max_seq_length"],
        dtype=None, load_in_4bit=True,
    )
    model_obj = FastLanguageModel.get_peft_model(
        model_obj, r=cfg["lora_rank"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    def _fmt(batch):
        return {"text": [tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]]}

    train_ds = Dataset.from_list(splits["train"]).map(_fmt, batched=True)
    valid_ds = Dataset.from_list(splits["valid"]).map(_fmt, batched=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model_obj, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=valid_ds,
        dataset_text_field="text", max_seq_length=cfg["max_seq_length"],
        args=SFTConfig(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            max_steps=cfg["iters"], learning_rate=cfg["learning_rate"],
            warmup_steps=min(20, cfg["iters"] // 10),
            optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="linear", seed=cfg["seed"],
            logging_steps=cfg["steps_per_report"],
            eval_steps=cfg["steps_per_eval"], eval_strategy="steps",
            save_steps=cfg["save_every"], save_total_limit=2,
            bf16=True, report_to="none",
        ),
    )
    trainer.train()
    model_obj.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nAdapter saved to: {adapter_dir}")


def run_fuse_unsloth(model: str, adapter_dir: Path, fused_dir: Path) -> None:
    from unsloth import FastLanguageModel
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir), max_seq_length=2048,
        dtype=None, load_in_4bit=False,
    )
    fused_dir.mkdir(parents=True, exist_ok=True)
    model_obj.save_pretrained_merged(str(fused_dir), tokenizer,
                                     save_method="merged_16bit")
    print(f"\nFused model: {fused_dir}")


def run_gguf_unsloth(model: str, adapter_dir: Path, gguf_path: Path,
                     system_prompt: str, ollama_model: str = "my-coach") -> None:
    from unsloth import FastLanguageModel
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir), max_seq_length=2048,
        dtype=None, load_in_4bit=True,
    )
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    gguf_base = str(gguf_path.parent / ollama_model)
    model_obj.save_pretrained_gguf(gguf_base, tokenizer,
                                   quantization_method="q4_k_m")
    actual_gguf = Path(gguf_base + ".gguf")
    if not actual_gguf.exists():
        candidates = list(Path(gguf_base).parent.glob(f"{ollama_model}*.gguf"))
        actual_gguf = candidates[0] if candidates else actual_gguf
    _register_with_ollama(actual_gguf, ollama_model, system_prompt)


# ── Ollama registration ───────────────────────────────────────────────────────

def _register_with_ollama(gguf_path: Path, ollama_model: str,
                           system_prompt: str) -> None:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        print("ERROR: ollama not found on PATH.")
        sys.exit(1)
    modelfile = gguf_path.parent / "Modelfile"
    system_line = f'SYSTEM "{system_prompt}"\n' if system_prompt else ""
    modelfile.write_text(
        f"FROM {gguf_path.resolve()}\n{system_line}"
        f"PARAMETER temperature 0.7\nPARAMETER top_p 0.9\n"
        f"PARAMETER num_ctx 16384\n"
    )
    result = subprocess.run([ollama_bin, "create", ollama_model,
                             "-f", str(modelfile)])
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\nModel ready:  ollama run {ollama_model}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a local LLM with LoRA")
    parser.add_argument("--data",    default="training_data/training_data.jsonl")
    parser.add_argument("--output",  default="training_data/lora-output")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--iters",          type=int,   default=DEFAULTS["iters"])
    parser.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lora-layers",    type=int,   default=DEFAULTS["lora_layers"])
    parser.add_argument("--rank",           type=int,   default=DEFAULTS["lora_rank"])
    parser.add_argument("--lr",             type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--val-split",      type=float, default=DEFAULTS["val_split"])
    parser.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--max-seq-length", type=int,   default=None)
    parser.add_argument("--overnight",  action="store_true")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--test",       action="store_true")
    parser.add_argument("--fuse",       action="store_true")
    parser.add_argument("--serve",      action="store_true")
    parser.add_argument("--gguf",       action="store_true")
    parser.add_argument("--gguf-path",    default="training_data/lora-output/my-coach.gguf")
    parser.add_argument("--ollama-model", default="my-coach")

    args = parser.parse_args()
    out_dir     = Path(args.output)
    data_path   = Path(args.data)
    adapter_dir = out_dir / "adapter"
    fused_dir   = out_dir / "fused"

    print(f"Backend: {'mlx-lm (Apple Silicon)' if BACKEND == 'mlx' else 'Unsloth (Linux/CUDA)'}")

    if args.test:
        if BACKEND == "mlx":
            cmd = [sys.executable, "-m", "mlx_lm.generate",
                   "--model", args.model, "--adapter-path", str(adapter_dir),
                   "--max-tokens", "300",
                   "--prompt", "How much did I spend last month overall?"]
            subprocess.run(cmd)
        return

    if args.fuse:
        if BACKEND == "mlx":
            run_fuse_mlx(args.model, adapter_dir, fused_dir)
        else:
            run_fuse_unsloth(args.model, adapter_dir, fused_dir)
        return

    if args.serve:
        if BACKEND != "mlx":
            print("--serve is Apple Silicon only. On Linux, use --gguf.")
            sys.exit(1)
        run_serve_mlx(fused_dir)
        return

    if args.gguf:
        if BACKEND == "mlx":
            run_gguf_mlx(args.model, adapter_dir, Path(args.gguf_path),
                         fused_dir, args.system_prompt, args.ollama_model)
        else:
            run_gguf_unsloth(args.model, adapter_dir, Path(args.gguf_path),
                             args.system_prompt, args.ollama_model)
        return

    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        sys.exit(1)

    base = OVERNIGHT if args.overnight else LOW_MEMORY if args.low_memory else DEFAULTS
    cfg = {
        "iters":            args.iters       if args.iters       != DEFAULTS["iters"]         else base["iters"],
        "batch_size":       args.batch_size  if args.batch_size  != DEFAULTS["batch_size"]    else base["batch_size"],
        "lora_layers":      args.lora_layers if args.lora_layers != DEFAULTS["lora_layers"]   else base["lora_layers"],
        "lora_rank":        args.rank        if args.rank        != DEFAULTS["lora_rank"]     else base["lora_rank"],
        "lora_alpha":       base["lora_alpha"],
        "learning_rate":    args.lr          if args.lr          != DEFAULTS["learning_rate"] else base["learning_rate"],
        "max_seq_length":   args.max_seq_length if args.max_seq_length else base["max_seq_length"],
        "grad_accum":       base["grad_accum"],
        "steps_per_report": base["steps_per_report"],
        "steps_per_eval":   base["steps_per_eval"],
        "save_every":       base["save_every"],
        "seed":             args.seed,
    }

    counts, splits, data_dir = prepare_data(data_path, out_dir, args.val_split, args.seed)
    print(f"train: {counts['train']}  valid: {counts['valid']}  test: {counts['test']}")

    if args.dry_run:
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        return

    if BACKEND == "mlx":
        run_training_mlx(args.model, data_dir, adapter_dir, cfg)
    else:
        run_training_unsloth(args.model, splits, adapter_dir, cfg)

    print("\nTraining complete!")
    print(f"  Test:  python3 scripts/train_lora.py --test")
    print(f"  Fuse:  python3 scripts/train_lora.py --fuse")
    print(f"  GGUF:  python3 scripts/train_lora.py --gguf")


if __name__ == "__main__":
    main()
