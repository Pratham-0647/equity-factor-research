"""Run the whole pipeline end to end: data -> universe -> factors -> evaluation
-> model -> backtest -> sensitivity -> risk -> Power BI export -> report."""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["build_db.py", "build_universe.py", "factors.py", "preprocess.py", "evaluate.py",
         "model.py", "backtest.py", "sensitivity.py", "risk.py", "export_powerbi.py", "report.py"]

if __name__ == "__main__":
    for step in STEPS:
        t = time.time()
        print(f"==> {step}", flush=True)
        subprocess.run([sys.executable, str(SRC / step)], check=True, stdout=subprocess.DEVNULL)
        print(f"    done in {time.time() - t:.0f}s", flush=True)
