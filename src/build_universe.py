"""Run sql/universe.sql against data/factors.db (no sqlite3 CLI needed)."""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    con = sqlite3.connect(ROOT / "data" / "factors.db")
    con.executescript((ROOT / "sql" / "universe.sql").read_text())
    n, d = con.execute("SELECT COUNT(*), COUNT(DISTINCT rebal_date) FROM universe").fetchone()
    print(f"universe: {n:,} rows over {d} rebalance dates")
