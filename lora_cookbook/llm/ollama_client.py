from __future__ import annotations

"""
Local LLM Client
──────────────────────────────────────────────────────────────────────────────
Supports two local backends, checked in priority order:

  1. mlx_lm server (port 8080) — fine-tuned model served via MLX directly.
     Apple Silicon only.  Start with:  python scripts/train_lora.py --serve
     Uses OpenAI-compatible /v1/chat/completions endpoint.

  2. Ollama (port 11434) — primary backend on Linux; fallback on Apple Silicon.
     Start with:  ollama serve
     Uses Ollama's native /api/chat endpoint.

Zero external dependencies — uses urllib (stdlib only).
Zero outbound network calls — both backends run entirely on your device.
"""

import json
import socket
import urllib.request
import urllib.error

MLX_LM_BASE   = "http://127.0.0.1:8080"
OLLAMA_BASE   = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:4b"


# ── Backend detection ─────────────────────────────────────────────────────────

def _mlx_is_running() -> bool:
    try:
        urllib.request.urlopen(f"{MLX_LM_BASE}/v1/models", timeout=1)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def is_running() -> bool:
    return _mlx_is_running() or _ollama_is_running()


def _ollama_is_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def list_models() -> list[str]:
    if _mlx_is_running():
        return ["my-coach"]
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def best_model(fine_tuned_name: str | None = None) -> str:
    """
    Return the best available local model name.
    mlx_lm server takes priority (serves the fine-tuned model).
    Falls back to Ollama preference order.

    Pass fine_tuned_name to prioritise your domain-specific model,
    e.g. best_model("finance-coach").
    """
    if _mlx_is_running():
        return fine_tuned_name or "my-coach"
    models = list_models()
    if not models:
        return DEFAULT_MODEL
    preference = []
    if fine_tuned_name:
        preference.append(fine_tuned_name)
    preference += ["qwen3.5:4b", "qwen3.5:2b", "qwen3.5",
                   "llama3.2", "llama3", "llama2", "mistral", "gemma"]
    for pref in preference:
        for m in models:
            if m.startswith(pref):
                return m
    return models[0]


# ── mlx_lm backend (OpenAI-compatible) ───────────────────────────────────────

def _mlx_chat(messages: list[dict], timeout: int = 120,
              max_tokens: int = 350, temperature: float | None = None) -> str:
    body: dict = {
        "model":      "default",
        "messages":   messages,
        "stream":     False,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    payload = json.dumps(body).encode()

    req = urllib.request.Request(
        f"{MLX_LM_BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            return data["choices"][0]["message"]["content"].strip()
    except (socket.timeout, TimeoutError) as e:
        raise RuntimeError(f"mlx_lm request timed out after {timeout}s: {e}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"mlx_lm server stopped. Restart with:\n"
            f"  python scripts/train_lora.py --serve\n"
            f"(detail: {e})"
        )


# ── Ollama backend ────────────────────────────────────────────────────────────

def _ollama_chat(messages: list[dict], model: str, timeout: int,
                 num_predict: int = 350, num_ctx: int = 16384,
                 temperature: float | None = None) -> str:
    opts: dict = {"num_ctx": num_ctx}
    if temperature is not None:
        opts["temperature"] = temperature
    payload = json.dumps({
        "model":       model,
        "messages":    messages,
        "stream":      False,
        "num_predict": num_predict,
        "options":     opts,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            return data.get("message", {}).get("content", "").strip()
    except (socket.timeout, TimeoutError) as e:
        raise RuntimeError(f"Ollama request timed out after {timeout}s: {e}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"No LLM backend running. Start one of:\n"
            f"  Fine-tuned model: python scripts/train_lora.py --serve\n"
            f"  Ollama:           ollama serve\n"
            f"(detail: {e})"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def chat(prompt: str, model: str = DEFAULT_MODEL, system: str = "",
         timeout: int = 90) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if _mlx_is_running():
        try:
            return _mlx_chat(messages, timeout=timeout)
        except RuntimeError:
            pass

    payload = json.dumps({
        "model":       model,
        "prompt":      prompt,
        "system":      system,
        "stream":      False,
        "num_predict": 350,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            return data.get("response", "").strip()
    except (socket.timeout, TimeoutError) as e:
        raise RuntimeError(f"Ollama request timed out after {timeout}s: {e}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"No LLM backend running. Start one of:\n"
            f"  Fine-tuned model: python scripts/train_lora.py --serve\n"
            f"  Ollama:           ollama serve\n"
            f"(detail: {e})"
        )


def chat_with_history(messages: list[dict], model: str = DEFAULT_MODEL,
                      system: str = "", timeout: int = 120) -> str:
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    if _mlx_is_running():
        try:
            return _mlx_chat(full_messages, timeout=timeout)
        except RuntimeError:
            pass

    return _ollama_chat(full_messages, model=model, timeout=timeout)


def chat_sql(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 180) -> str:
    """Lightweight call for SQL generation — moderate token budget, temperature=0."""
    messages = [{"role": "user", "content": prompt}]

    if _mlx_is_running():
        try:
            return _mlx_chat(messages, timeout=timeout, max_tokens=300, temperature=0)
        except RuntimeError:
            pass

    payload = json.dumps({
        "model":       model,
        "prompt":      prompt,
        "stream":      False,
        "num_predict": 300,
        "options":     {"num_ctx": 4096, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            return data.get("response", "").strip()
    except (socket.timeout, TimeoutError) as e:
        raise RuntimeError(f"Ollama request timed out after {timeout}s: {e}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"No LLM backend running. Start one of:\n"
            f"  Fine-tuned model: python scripts/train_lora.py --serve\n"
            f"  Ollama:           ollama serve\n"
            f"(detail: {e})"
        )


# ── Streaming helpers ─────────────────────────────────────────────────────────

def _mlx_stream(messages: list[dict], timeout: int = 120, max_tokens: int = 350):
    payload = json.dumps({
        "model":      "default",
        "messages":   messages,
        "stream":     True,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{MLX_LM_BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw_line in r:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError):
                    pass
    except (socket.timeout, TimeoutError, urllib.error.URLError):
        pass


def _ollama_stream(messages: list[dict], model: str, timeout: int,
                   num_predict: int = 350, num_ctx: int = 16384):
    payload = json.dumps({
        "model":       model,
        "messages":    messages,
        "stream":      True,
        "num_predict": num_predict,
        "options":     {"num_ctx": num_ctx},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw_line in r:
                try:
                    chunk = json.loads(raw_line.decode("utf-8", errors="replace"))
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except (json.JSONDecodeError, KeyError):
                    pass
    except (socket.timeout, TimeoutError, urllib.error.URLError):
        pass


def stream_chat(messages: list[dict], model: str = DEFAULT_MODEL,
                system: str = "", timeout: int = 120):
    """Generator that yields response tokens one at a time."""
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    if _mlx_is_running():
        had_tokens = False
        for token in _mlx_stream(full_messages, timeout=timeout):
            had_tokens = True
            yield token
        if had_tokens:
            return

    yield from _ollama_stream(full_messages, model=model, timeout=timeout)
