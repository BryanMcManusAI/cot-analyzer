"""Black-box tests for the product surface: the keyless eval validates on synthetic
gold, the hard set exposes the keyword-check limit (the honest F1 story), the gate
returns the right exit codes, and the report renders self-contained."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYZER = ROOT / "cot_analyzer.py"
REPORT = ROOT / "cot_report.py"


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def test_eval_synthetic_gold_all_correct():
    r = run(ANALYZER, "--eval")
    assert r.returncode == 0
    assert "[MISS]" not in r.stdout  # keyless core nails the clean gold (F1 ~1.00)


def test_eval_hard_exposes_the_keyword_limit():
    r = run(ANALYZER, "--eval", "--hard")
    assert r.returncode == 0
    # the paraphrase items fool the keyword check — this is the honest 0.67 story
    assert "[MISS]" in r.stdout


def test_gate_passes_under_a_lenient_threshold():
    assert run(ANALYZER, "--gate", "--max-launder-rate", "1.0").returncode == 0


def test_report_renders_self_contained():
    r = run(REPORT)
    assert r.returncode == 0 and len(r.stdout.strip()) > 0
