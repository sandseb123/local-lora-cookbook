"""
Finance Coach — SQL RAG pipeline
=================================
Generates SQL from a natural language question, runs it against the
transactions SQLite database, and produces a baseline coaching answer
using the local LLM (Ollama or mlx-lm).

Used by domain.py as the answer() backend.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lora_cookbook.llm import ollama_client

# ── Schema (used in SQL generation prompt) ───────────────────────────────────

_SCHEMA = """
Table: transactions
  id          INTEGER PRIMARY KEY
  date        TEXT              -- YYYY-MM-DD
  description TEXT              -- merchant / payee name
  amount      REAL              -- positive = expense, negative = income/credit
  category    TEXT              -- Groceries, Restaurants, Rent, Utilities,
                                --   Transport, Entertainment, Shopping, Health,
                                --   Subscriptions, Income, Other
  account     TEXT              -- Checking, Savings, Credit Card

Table: accounts
  name    TEXT PRIMARY KEY      -- Checking, Savings, Credit Card
  type    TEXT
  balance REAL

Table: budgets
  category      TEXT PRIMARY KEY
  monthly_limit REAL            -- monthly spending limit in dollars
"""

_SQL_PROMPT = """\
You are a SQLite expert. Write one SQL query to answer the question below.

Schema:
{schema}

Question: "{question}"

Rules:
- Use strftime('%Y-%m', date) to group by month
- "last month"  -> WHERE date >= date('now','start of month','-1 month')
                   AND date  <  date('now','start of month')
- "this month"  -> WHERE date >= date('now','start of month')
- "this year"   -> WHERE strftime('%Y', date) = strftime('%Y', 'now')
- "last N days" -> WHERE date >= date('now', '-N days')
- Positive amounts are expenses; negative amounts are income / credits
- For spending totals: SUM(amount) WHERE amount > 0
- Always include ORDER BY date DESC or ORDER BY total DESC where useful
- Return ONLY the SQL query, no explanation, no markdown fences
"""

_ANSWER_PROMPT = """\
The user asked: "{question}"

SQL used:
{sql}

Data returned:
{results}
{budget_section}
Give a brief, specific finance coaching response using only the numbers above.
"""

SYSTEM_PROMPT = (
    "You are a private finance coach. "
    "All data is processed locally on the user's device."
)


def _generate_sql(question: str, model: str, timeout: int = 60) -> str:
    prompt = _SQL_PROMPT.format(schema=_SCHEMA, question=question)
    raw = ollama_client.chat_sql(prompt, model=model, timeout=timeout)
    # Strip Qwen3 <think>...</think> reasoning blocks before extracting SQL
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:sql)?\s*", "", raw).strip().rstrip("`").strip()
    return raw


def _run_sql(sql: str, db_path: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        return [dict(r) for r in cursor.fetchmany(200)]
    finally:
        conn.close()


def _get_budget_context(categories: list[str], db_path: str) -> str:
    """Fetch relevant budget limits to include as context."""
    if not categories:
        return ""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" * len(categories))
            rows = conn.execute(
                f"SELECT category, monthly_limit FROM budgets "
                f"WHERE category IN ({placeholders})",
                categories,
            ).fetchall()
            if not rows:
                return ""
            lines = ["Budget limits:"] + [f"  {r[0]}: ${r[1]:.0f}/month" for r in rows]
            return "\n".join(lines) + "\n\n"
        finally:
            conn.close()
    except Exception:
        return ""


def answer(question: str, db_path: str, timeout: int = 300) -> dict:
    """
    Run a finance question through the RAG pipeline.

    Returns dict with keys:
      sql, rows, answer, error, context_section, full_prompt
    """
    model = ollama_client.best_model()

    # Step 1: Generate SQL
    try:
        sql = _generate_sql(question, model=model)
    except Exception as e:
        return {"sql": "", "rows": [], "answer": "", "error": str(e),
                "context_section": "", "full_prompt": ""}

    if not sql:
        return {"sql": "", "rows": [], "answer": "",
                "error": "LLM returned no SQL",
                "context_section": "", "full_prompt": ""}

    # Step 2: Execute SQL
    try:
        rows = _run_sql(sql, db_path)
    except Exception as e:
        return {"sql": sql, "rows": [], "answer": "",
                "error": f"SQL error: {e}",
                "context_section": "", "full_prompt": ""}

    if not rows:
        return {"sql": sql, "rows": [], "answer": "", "error": "",
                "context_section": "", "full_prompt": ""}

    # Step 3: Build budget context
    categories = list({r.get("category", "") for r in rows if r.get("category")})
    budget_section = _get_budget_context(categories, db_path)

    # Step 4: Generate baseline answer
    results_str = json.dumps(rows[:15], indent=2)
    full_prompt = _ANSWER_PROMPT.format(
        question=question,
        sql=sql,
        results=results_str,
        budget_section=budget_section,
    )
    try:
        ans = ollama_client.chat_with_history(
            [{"role": "user", "content": full_prompt}],
            model=model,
            system=SYSTEM_PROMPT,
            timeout=timeout,
        )
    except Exception as e:
        return {"sql": sql, "rows": rows, "answer": "", "error": str(e),
                "context_section": budget_section, "full_prompt": full_prompt}

    return {
        "sql":             sql,
        "rows":            rows,
        "answer":          ans,
        "error":           "",
        "context_section": budget_section,
        "full_prompt":     full_prompt,
    }
