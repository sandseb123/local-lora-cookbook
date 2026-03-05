-- Finance Coach — SQLite schema
-- Import your bank exports into this structure.

CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,   -- YYYY-MM-DD
    description TEXT    NOT NULL,   -- merchant / payee name
    amount      REAL    NOT NULL,   -- positive = expense, negative = income/credit
    category    TEXT    NOT NULL,   -- see categories below
    account     TEXT    NOT NULL    -- Checking, Savings, Credit Card
);

-- Standard categories:
--   Groceries, Restaurants, Rent, Utilities, Transport,
--   Entertainment, Shopping, Health, Subscriptions, Income, Other

CREATE TABLE IF NOT EXISTS accounts (
    name    TEXT PRIMARY KEY,       -- Checking, Savings, Credit Card
    type    TEXT NOT NULL,
    balance REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS budgets (
    category      TEXT PRIMARY KEY,
    monthly_limit REAL NOT NULL     -- monthly spending cap in dollars
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_tx_date     ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_tx_category ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_tx_account  ON transactions(account);
