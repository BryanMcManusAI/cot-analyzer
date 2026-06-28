#!/usr/bin/env python3
"""
cot_score_human.py — score the human gold against the intended labels and the analyzer (stdlib + --llm).
The HUMAN is the ground truth; we ask how well each analyzer mode matches it, and flag the divergences.

  python3 cot_score_human.py                 # human vs gold + stdlib
  python3 cot_score_human.py --llm           # also re-run the --llm analyzer and compare
"""
import sys, json, argparse
from pathlib import Path
from cot_analyzer import gold, gold_hard, analyze_record

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(HERE / "cot_human_gold.json"))
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()

    recs = {r["id"]: r for r in gold() + gold_hard()}
    human = {it["id"]: it["human"] for it in json.load(open(a.export))["items"]}

    rows = []
    for rid, hv in human.items():
        rec = recs[rid]
        rows.append({"id": rid, "gold": rec["label"], "human": hv,
                     "stdlib": analyze_record(rec)["verdict"],
                     "llm": analyze_record(rec, use_llm=True)["verdict"] if a.llm else None,
                     "output": rec.get("output", "")})

    def agree(key):
        ok = sum(r[key] == r["human"] for r in rows)
        return ok, len(rows)

    print(f"HUMAN GOLD — {len(rows)} items\n")
    for label, key in [("human vs intended gold", "gold"), ("analyzer (stdlib) vs human", "stdlib")] + \
                      ([("analyzer (--llm) vs human", "llm")] if a.llm else []):
        ok, n = agree(key)
        print(f"  {label:28s} {ok}/{n} = {ok/n:.0%}")

    print("\nDIVERGENCES (where the human differs from a column):")
    for r in rows:
        diffs = [k for k in ("gold", "stdlib", "llm") if r.get(k) and r[k] != r["human"]]
        if diffs:
            print(f"  [{r['id']:16s}] human={r['human']:9s} "
                  + " ".join(f"{k}={r[k]}" for k in ("gold", "stdlib", "llm") if r.get(k)))
            print(f"       output: {r['output'][:110]}")

if __name__ == "__main__":
    main()
