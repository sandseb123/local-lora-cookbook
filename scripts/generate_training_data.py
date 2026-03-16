#!/usr/bin/env python3
"""
LoRA Training Data Generation Pipeline
=======================================
A two-phase pipeline for generating fine-tuning data for any data-aware
AI coach application.

Phase 1 — collect
  Runs a question bank through your app's SQL RAG pipeline (via Ollama).
  Captures: question, SQL, raw rows, baseline model answer.
  Output → training_data/raw_examples.jsonl

Phase 2 — annotate
  Sends each raw example to a cloud LLM with the full data context.
  The LLM writes a gold-standard coaching answer in your app's style.
  Supports Anthropic (default), MiniMax, and OpenAI as annotation providers.
  Output → training_data/training_data.jsonl  (TRL / Unsloth chat format)

Usage
-----
  # Phase 1 only (Ollama must be running, no API key needed):
  python3 scripts/generate_training_data.py --domain examples/finance_coach --db ~/finance.db

  # Phase 1 then 2 with Claude (annotate immediately after):
  python3 scripts/generate_training_data.py --domain examples/finance_coach --db ~/finance.db --phase both --key sk-ant-...

  # Phase 2 with MiniMax (often cheaper, 204K context):
  python3 scripts/generate_training_data.py --domain examples/finance_coach --phase annotate --provider minimax --key <minimax-key>

  # Phase 2 only (raw_examples.jsonl already exists):
  python3 scripts/generate_training_data.py --domain examples/finance_coach --phase annotate --key sk-ant-...

After data generation
---------------------
  python3 scripts/train_lora.py --data examples/finance_coach/training_data/training_data.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── Domain loader ─────────────────────────────────────────────────────────────

def _load_domain(domain_dir: str) -> types.ModuleType:
    """
    Dynamically load a domain adapter from <domain_dir>/domain.py.

    The domain module must export:
      QUESTIONS: list[str]
      SYSTEM_PROMPT: str
      ANNOTATION_PROMPT_TEMPLATE: str  (with {question}, {sql}, {results},
                                         {context_section}, {llama_answer})
      answer(question: str, db_path: str) -> dict
        returns: {sql, rows, answer, error, context_section, full_prompt}
    """
    path = Path(domain_dir) / "domain.py"
    if not path.exists():
        print(f"ERROR: domain.py not found at {path}")
        print(f"  Point --domain to a directory containing domain.py")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("domain", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for attr in ("QUESTIONS", "SYSTEM_PROMPT", "ANNOTATION_PROMPT_TEMPLATE", "answer"):
        if not hasattr(mod, attr):
            print(f"ERROR: domain.py is missing required export: {attr}")
            sys.exit(1)
    return mod


# ── Annotator backends ───────────────────────────────────────────────────────

# Provider presets: (default_model, base_url, env_var)
_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "anthropic": ("claude-sonnet-4-6", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "minimax":   ("MiniMax-M2.5",      "https://api.minimax.io/v1", "MINIMAX_API_KEY"),
    "openai":    ("gpt-4o-mini",       "https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def _call_anthropic(prompt: str, api_key: str,
                    model: str = "claude-sonnet-4-6") -> str:
    """Call the Anthropic Messages API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except ImportError:
        pass

    payload = json.dumps({
        "model":      model,
        "max_tokens": 600,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key":          api_key,
            "anthropic-version":  "2024-06-01",
            "content-type":       "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    return body["content"][0]["text"].strip()


def _call_openai_compatible(prompt: str, api_key: str,
                            model: str, base_url: str) -> str:
    """
    Call any OpenAI-compatible chat completions API.

    Works with OpenAI, MiniMax, and other providers that implement
    the /v1/chat/completions (or /chat/completions) endpoint.
    """
    # Normalise: ensure base_url ends without trailing slash
    base_url = base_url.rstrip("/")
    # Build endpoint — append /chat/completions if the URL doesn't already
    # contain the full path (e.g. bare "https://api.minimax.io/v1").
    if base_url.endswith("/v1"):
        endpoint = f"{base_url}/chat/completions"
    elif base_url.endswith("/chat/completions"):
        endpoint = base_url
    else:
        endpoint = f"{base_url}/v1/chat/completions"

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        pass

    payload = json.dumps({
        "model":      model,
        "max_tokens": 600,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"].strip()


def _call_annotator(prompt: str, api_key: str,
                    model: str = "claude-sonnet-4-6",
                    provider: str = "anthropic",
                    base_url: str = "") -> str:
    """
    Route annotation requests to the chosen provider.

    Supported providers:
      anthropic  — Anthropic Messages API (default)
      minimax    — MiniMax OpenAI-compatible API
      openai     — OpenAI Chat Completions API
    """
    if provider == "anthropic":
        return _call_anthropic(prompt, api_key, model=model)
    return _call_openai_compatible(prompt, api_key, model=model,
                                   base_url=base_url)


# ── Phase 1: collect ─────────────────────────────────────────────────────────

def run_collect(domain, db_path: str, out_dir: Path) -> Path:
    """
    Run each question through the domain's RAG pipeline and save raw examples.
    Skips questions that error or return no rows.
    Resumable — already-collected questions are skipped.
    """
    out_path = out_dir / "raw_examples.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["question"])
                except Exception:
                    pass

    # Warm up Ollama
    from lora_cookbook.llm import ollama_client
    model = ollama_client.best_model()
    print(f"  Warming up model '{model}' (first load may take up to 2 min) ...",
          flush=True)
    try:
        ollama_client.chat_sql("SELECT 1", model=model, timeout=180)
        print("  Model ready.\n")
    except Exception as e:
        print(f"\nERROR: Could not reach Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        sys.exit(1)

    questions: list[str] = domain.QUESTIONS
    new_count = skip_count = error_count = 0

    with open(out_path, "a") as f:
        for i, question in enumerate(questions, 1):
            if question in already_done:
                print(f"  [{i}/{len(questions)}] skip (done): {question[:60]}")
                skip_count += 1
                continue

            print(f"  [{i}/{len(questions)}] {question[:70]}", end=" ", flush=True)

            result = domain.answer(question, db_path)

            if result.get("error"):
                print(f"-> ERROR: {str(result['error'])[:60]}")
                error_count += 1
                continue

            if not result.get("rows"):
                print("-> no data (skipped)")
                skip_count += 1
                continue

            record = {
                "question":        question,
                "sql":             result["sql"],
                "rows":            result["rows"][:10],
                "context_section": result.get("context_section", ""),
                "full_prompt":     result.get("full_prompt", ""),
                "llama_answer":    result["answer"],
                "gold_answer":     "",
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            new_count += 1
            print(f"-> {len(result['rows'])} rows")

    print(f"\nPhase 1 done — {new_count} new, {skip_count} skipped, "
          f"{error_count} errors")
    print(f"Raw examples: {out_path}")
    return out_path


# ── Phase 2: annotate ─────────────────────────────────────────────────────────

_MAX_RETRIES = 3

def run_annotate(domain, raw_path: Path, out_dir: Path,
                 api_key: str, model: str = "claude-sonnet-4-6",
                 delay: float = 0.5, provider: str = "anthropic",
                 base_url: str = "") -> Path:
    """
    Read raw_examples.jsonl, call Claude to write gold-standard answers,
    and write training_data.jsonl in TRL/Unsloth chat format.
    Resumable — already-annotated questions are skipped.
    """
    training_path = out_dir / "training_data.jsonl"

    already_annotated: set[str] = set()
    if training_path.exists():
        with open(training_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    q = obj["messages"][0].get("_question", "")
                    if q:
                        already_annotated.add(q)
                except Exception:
                    pass

    raw_examples: list[dict] = []
    with open(raw_path) as f:
        for line in f:
            try:
                raw_examples.append(json.loads(line.strip()))
            except Exception:
                pass

    print(f"\nPhase 2: annotating {len(raw_examples)} examples with {model}")
    success = skipped = errors = 0

    with open(training_path, "a") as out_f:
        for i, ex in enumerate(raw_examples, 1):
            question = ex["question"]

            if question in already_annotated:
                print(f"  [{i}/{len(raw_examples)}] skip (done): {question[:60]}")
                skipped += 1
                continue

            print(f"  [{i}/{len(raw_examples)}] annotating: {question[:60]}",
                  end=" ", flush=True)

            annotation_prompt = domain.ANNOTATION_PROMPT_TEMPLATE.format(
                question=question,
                sql=ex["sql"],
                results=json.dumps(ex["rows"], indent=2),
                context_section=ex.get("context_section", ""),
                llama_answer=ex["llama_answer"],
            )

            gold_answer = None
            for attempt in range(_MAX_RETRIES):
                try:
                    gold_answer = _call_annotator(annotation_prompt, api_key,
                                                  model=model,
                                                  provider=provider,
                                                  base_url=base_url)
                    break
                except Exception as e:
                    wait = 2 ** (attempt + 1)
                    if attempt < _MAX_RETRIES - 1:
                        print(f"-> retry {attempt + 1}/{_MAX_RETRIES} "
                              f"(waiting {wait}s): {e}", end=" ", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"-> FAILED after {_MAX_RETRIES} attempts: {e}")

            if gold_answer is None:
                errors += 1
                continue

            training_record = {
                "messages": [
                    {
                        "role":      "system",
                        "content":   domain.SYSTEM_PROMPT,
                        "_question": question,
                    },
                    {
                        "role":    "user",
                        "content": ex["full_prompt"],
                    },
                    {
                        "role":    "assistant",
                        "content": gold_answer,
                    },
                ],
                "metadata": {
                    "question": question,
                    "sql":      ex["sql"],
                },
            }

            out_f.write(json.dumps(training_record) + "\n")
            out_f.flush()
            success += 1
            print(f"-> done ({len(gold_answer)} chars)")

            if delay > 0:
                time.sleep(delay)

    print(f"\nPhase 2 done — {success} annotated, {skipped} skipped, "
          f"{errors} errors")
    print(f"Training data: {training_path}")
    print(f"\nNext step:")
    print(f"  python3 scripts/train_lora.py --data {training_path}")
    return training_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LoRA training data from your app's RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Path to domain directory containing domain.py (e.g. examples/finance_coach)",
    )
    parser.add_argument(
        "--db",
        required=False,
        default="",
        help="Path to your SQLite database",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output directory (default: <domain>/training_data/)",
    )
    parser.add_argument(
        "--phase",
        choices=["collect", "annotate", "both"],
        default="collect",
        help="Which phase to run (default: collect)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Shorthand for --phase both",
    )
    parser.add_argument(
        "--provider",
        choices=list(_PROVIDER_DEFAULTS.keys()),
        default="anthropic",
        help="LLM provider for annotation (default: anthropic)",
    )
    parser.add_argument(
        "--key",
        default="",
        help="API key for the annotation provider "
             "(or set ANTHROPIC_API_KEY / MINIMAX_API_KEY / OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name for annotation (defaults per provider: "
             "claude-sonnet-4-6, MiniMax-M2.5, gpt-4o-mini)",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Custom base URL for OpenAI-compatible providers",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between annotation API calls (default: 0.5)",
    )

    args = parser.parse_args()
    phase   = "both" if args.annotate else args.phase
    domain  = _load_domain(args.domain)
    out_dir = Path(args.out) if args.out else Path(args.domain) / "training_data"

    # Resolve provider defaults
    provider = args.provider
    default_model, default_base_url, env_var = _PROVIDER_DEFAULTS[provider]
    ann_model   = args.model    or default_model
    ann_key     = args.key      or os.environ.get(env_var, "")
    ann_base_url = args.base_url or default_base_url

    print("=" * 60)
    print("LoRA Training Data Generator")
    print("=" * 60)
    print(f"Domain:    {args.domain}")
    print(f"DB:        {args.db or '(not required for annotate phase)'}")
    print(f"Output:    {out_dir.resolve()}")
    print(f"Phase:     {phase}")
    print(f"Questions: {len(domain.QUESTIONS)}")
    if phase in ("annotate", "both"):
        print(f"Provider:  {provider} ({ann_model})")
    print()

    raw_path = out_dir / "raw_examples.jsonl"

    if phase in ("collect", "both"):
        if not args.db:
            print("ERROR: --db is required for the collect phase.")
            sys.exit(1)
        if not os.path.exists(args.db):
            print(f"ERROR: Database not found at {args.db}")
            sys.exit(1)
        print("Phase 1 — Collecting raw examples via domain RAG pipeline...")
        raw_path = run_collect(domain, args.db, out_dir)

    if phase in ("annotate", "both"):
        if not ann_key:
            print(f"ERROR: API key required for annotation ({provider}).")
            print(f"  Pass --key <your-key> or set {env_var}")
            sys.exit(1)
        if not raw_path.exists():
            print(f"ERROR: {raw_path} not found. Run Phase 1 first.")
            sys.exit(1)
        run_annotate(domain, raw_path, out_dir, ann_key,
                     model=ann_model, delay=args.delay,
                     provider=provider, base_url=ann_base_url)


if __name__ == "__main__":
    main()
