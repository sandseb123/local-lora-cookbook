"""
Finance Coach — Domain Adapter
================================
Plugs the finance-coach example into the generic generate_training_data.py
pipeline.

Exports required by generate_training_data.py:
  QUESTIONS                  - question bank
  SYSTEM_PROMPT              - coach persona
  ANNOTATION_PROMPT_TEMPLATE - sent to Claude for gold-standard annotation
  answer(question, db_path)  - calls rag.py
"""

from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from examples.finance_coach import rag  # noqa: E402

SYSTEM_PROMPT = rag.SYSTEM_PROMPT

QUESTIONS: list[str] = [
    # ── Spending summaries ────────────────────────────────────────────────────
    "How much did I spend last month in total?",
    "What were my top 5 spending categories last month?",
    "How much did I spend on groceries last month?",
    "How much did I spend on restaurants last month?",
    "How much did I spend on entertainment this month?",
    "What was my total spending this month so far?",
    "How does my spending this month compare to last month?",
    "What were my biggest purchases last month?",
    "How much have I spent on subscriptions this month?",
    "What did I spend the most on last week?",
    "How much did I spend on transport this month?",
    "How much did I spend on health and wellness this month?",
    "What was my total spending in the last 30 days?",
    "How much did I spend on shopping last month?",
    "What are my top 3 spending categories over the past 3 months?",
    "How much did I spend on utilities last month?",
    "How much have I spent on rent this year?",
    "What is my average monthly spending over the past 3 months?",
    "How much did I spend last week in total?",
    "What is my daily average spending this month?",
    # ── Budget tracking ───────────────────────────────────────────────────────
    "Am I over budget on restaurants this month?",
    "How close am I to my grocery budget this month?",
    "Which categories am I over budget in this month?",
    "How much do I have left in my entertainment budget?",
    "Am I on track with my overall spending this month?",
    "How does my restaurant spending compare to my budget?",
    "How much have I spent vs my transport budget this month?",
    "Which spending category is furthest over budget?",
    "Am I under budget in any categories this month?",
    "How much of my subscription budget have I used?",
    # ── Income and savings ────────────────────────────────────────────────────
    "How much did I earn last month?",
    "What is my income vs spending this month?",
    "Am I saving money this month or spending more than I earn?",
    "What is my savings rate this month?",
    "How much have I saved this year?",
    "What is my average monthly income over the past 3 months?",
    "How does my income this month compare to last month?",
    "How much went into savings this month?",
    # ── Trends ────────────────────────────────────────────────────────────────
    "Has my spending been going up or down over the past 3 months?",
    "How has my restaurant spending trended over the past 3 months?",
    "Is my grocery spending increasing or decreasing?",
    "How has my total monthly spending changed over the past 3 months?",
    "Which category has grown the most in spending over the past 3 months?",
    "Am I spending more on subscriptions than I was 3 months ago?",
    "How has my savings trend looked over the past 3 months?",
    "Is my transport spending going up or down?",
    "What is my 3-month trend for total spending?",
    # ── Merchant lookups ──────────────────────────────────────────────────────
    "How much have I spent at coffee shops this month?",
    "How many times did I order food delivery last month?",
    "What was my most expensive single purchase last month?",
    "How much have I spent on Amazon this month?",
    "What are my recurring subscriptions and how much do they cost?",
    "How much did I spend at restaurants this week?",
    "What was my largest transaction last month?",
    "How many transactions did I make last month?",
    "How much have I spent at grocery stores this week?",
    # ── Time comparisons ──────────────────────────────────────────────────────
    "How does this month's spending compare to the same month last year?",
    "What was my highest spending month this year?",
    "What was my lowest spending month this year?",
    "How does my spending this week compare to last week?",
    "Which month this year did I save the most?",
    "How much did I spend in the first half vs second half of last month?",
    # ── Account-specific ──────────────────────────────────────────────────────
    "How much is on my credit card right now?",
    "How much have I spent on my credit card this month?",
    "What is my checking account balance?",
    "How much did I transfer to savings this month?",
    "What transactions hit my checking account this week?",
    "How does my credit card spending this month compare to last month?",
    # ── Coaching ─────────────────────────────────────────────────────────────
    "Where am I wasting the most money?",
    "What is one spending habit I should change based on my data?",
    "Am I living within my means based on the last 3 months?",
    "What category should I try to cut back on?",
    "Based on my spending patterns, how much could I realistically save extra per month?",
    "Am I spending too much on eating out?",
    "What does my data say about my financial health?",
    "What are my biggest financial risks based on my spending?",
    "How can I improve my savings rate based on my data?",
    "What is the one change that would have the biggest positive impact on my finances?",
    # ── Conversational ────────────────────────────────────────────────────────
    "Give me a summary of my finances this month.",
    "How am I doing financially this month?",
    "Is my spending normal this month?",
    "Walk me through where my money went last month.",
    "How have I been spending lately?",
    "Am I on track financially?",
    "What does my financial data look like recently?",
    "Give me a financial health check.",
    "How does my month look so far?",
    "What should I be watching in my spending right now?",
]

ANNOTATION_PROMPT_TEMPLATE = """\
You are writing gold-standard coaching responses for a private finance AI coach.

## Examples of perfect responses

Q: "How much did I spend on restaurants last month?"
Data: total $342.50, budget $300/month
GOOD: "You spent $342.50 on restaurants last month, which puts you $42.50 over \
your $300 budget — about 14% above target. The data shows most of the overage \
came from weekend dinners, so trying to cook at home on one of those weekend \
nights would likely bring you back on track next month."

Q: "Am I saving money this month?"
Data: income $8,400, spending $6,180, net +$2,220
GOOD: "You're ahead this month — $8,400 in income against $6,180 in spending \
leaves a $2,220 surplus, a savings rate of about 26%. If you want to put that \
surplus to work, moving it to a savings account before the end of the month \
prevents it from disappearing into discretionary spending."

---

## Context for this example

Question: {question}

SQL executed:
{sql}

Data returned:
{results}
{context_section}
## Baseline model answer
{llama_answer}

---

## Your task

Rewrite the baseline answer to gold-standard quality:
1. Lead with the key number(s) directly from the data.
2. Add context: compare to budget, prior period, or a benchmark.
3. Close with one concrete, actionable next step.

Rules:
- 3-5 sentences. Flowing prose — no bullet points, no headers.
- Warm and specific, like a knowledgeable friend.
- Only use numbers visible in the data above. Never invent values.
- Never give tax, investment, or legal advice.

Write ONLY the response text (no preamble, no sign-off):"""


def answer(question: str, db_path: str) -> dict:
    """Delegate to the finance rag pipeline."""
    return rag.answer(question, db_path)


