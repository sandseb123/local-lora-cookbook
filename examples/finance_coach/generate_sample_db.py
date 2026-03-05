#!/usr/bin/env python3
"""
Generate a synthetic finance.db for testing the finance-coach pipeline.
Creates 3 months of realistic-looking transaction data.

Usage:
  python3 examples/finance_coach/generate_sample_db.py
  python3 examples/finance_coach/generate_sample_db.py --out /tmp/finance.db
"""

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = Path(__file__).parent / "schema.sql"


def _create_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA.read_text())
    return conn


def _random_date(start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def generate(db_path: str, seed: int = 42) -> None:
    random.seed(seed)
    conn = _create_db(db_path)

    # ── Accounts ─────────────────────────────────────────────────────────────
    conn.executemany(
        "INSERT OR REPLACE INTO accounts VALUES (?, ?, ?)",
        [
            ("Checking",     "Checking",    4_230.55),
            ("Savings",      "Savings",     12_800.00),
            ("Credit Card",  "Credit Card",  -1_420.30),
        ],
    )

    # ── Budgets ──────────────────────────────────────────────────────────────
    conn.executemany(
        "INSERT OR REPLACE INTO budgets VALUES (?, ?)",
        [
            ("Groceries",     600.00),
            ("Restaurants",   300.00),
            ("Entertainment", 150.00),
            ("Shopping",      200.00),
            ("Transport",     180.00),
            ("Subscriptions", 100.00),
            ("Health",        150.00),
            ("Utilities",     250.00),
        ],
    )

    # ── Transactions (3 months) ───────────────────────────────────────────────
    today   = date.today()
    start   = date(today.year, today.month, 1) - timedelta(days=90)
    end     = today

    templates = [
        # (description, category, account, min_amt, max_amt, freq_per_month)
        ("Whole Foods",          "Groceries",     "Credit Card", 45,  140,  6),
        ("Trader Joe's",         "Groceries",     "Credit Card", 30,   90,  4),
        ("Costco",               "Groceries",     "Credit Card", 80,  200,  2),
        ("Chipotle",             "Restaurants",   "Credit Card", 12,   18,  5),
        ("Starbucks",            "Restaurants",   "Credit Card",  5,    9, 10),
        ("Local Diner",          "Restaurants",   "Credit Card", 18,   45,  4),
        ("Sushi Place",          "Restaurants",   "Credit Card", 35,   80,  2),
        ("Netflix",              "Subscriptions", "Credit Card", 15,   16,  1),
        ("Spotify",              "Subscriptions", "Credit Card", 10,   11,  1),
        ("ChatGPT Plus",         "Subscriptions", "Credit Card", 20,   20,  1),
        ("Apple iCloud",         "Subscriptions", "Credit Card",  3,    3,  1),
        ("Gym Membership",       "Health",        "Credit Card", 45,   55,  1),
        ("Pharmacy",             "Health",        "Credit Card", 12,   60,  2),
        ("Uber",                 "Transport",     "Credit Card",  8,   25,  6),
        ("Gas Station",          "Transport",     "Checking",    40,   70,  3),
        ("Parking",              "Transport",     "Credit Card",  5,   20,  4),
        ("Amazon",               "Shopping",      "Credit Card", 20,  120,  4),
        ("Target",               "Shopping",      "Credit Card", 30,   90,  2),
        ("Movies",               "Entertainment", "Credit Card", 14,   25,  2),
        ("Concert Tickets",      "Entertainment", "Credit Card", 45,  120,  1),
        ("Electric Bill",        "Utilities",     "Checking",    80,  130,  1),
        ("Internet",             "Utilities",     "Checking",    65,   75,  1),
        ("Phone Bill",           "Utilities",     "Checking",    55,   80,  1),
        ("Rent",                 "Rent",          "Checking", 1800, 1800,   1),
    ]

    rows = []
    # Salary twice a month
    for month_offset in range(3):
        m = (start.month + month_offset - 1) % 12 + 1
        y = start.year + (start.month + month_offset - 1) // 12
        rows.append((f"{y:04d}-{m:02d}-01", "Payroll Direct Deposit",
                     -4_200.00, "Income", "Checking"))
        rows.append((f"{y:04d}-{m:02d}-15", "Payroll Direct Deposit",
                     -4_200.00, "Income", "Checking"))

    for desc, cat, acct, lo, hi, freq in templates:
        for _ in range(freq * 3):  # 3 months
            d = _random_date(start, end)
            amt = round(random.uniform(lo, hi), 2)
            rows.append((d, desc, amt, cat, acct))

    # Small random one-offs
    one_offs = [
        ("Doctor Visit",   "Health",        "Checking",  120, 250),
        ("Hair Cut",       "Health",        "Checking",   35,  80),
        ("Book Store",     "Entertainment", "Credit Card", 15,  45),
        ("Clothing Store", "Shopping",      "Credit Card", 60, 180),
        ("Airbnb",         "Entertainment", "Credit Card",150, 400),
        ("Flight",         "Transport",     "Credit Card",200, 600),
    ]
    for desc, cat, acct, lo, hi in one_offs:
        if random.random() > 0.4:
            d = _random_date(start, end)
            amt = round(random.uniform(lo, hi), 2)
            rows.append((d, desc, amt, cat, acct))

    conn.executemany(
        "INSERT INTO transactions (date, description, amount, category, account)"
        " VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()

    total = len(rows)
    print(f"Created {db_path} with {total} transactions "
          f"({start} — {end})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="~/finance.db")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db_path = str(Path(args.out).expanduser())
    generate(db_path, seed=args.seed)
    print(f"\nNext: run Phase 1 with --db {db_path}")


if __name__ == "__main__":
    main()
