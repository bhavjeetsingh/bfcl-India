"""rescore.py — Re-score existing prediction files with the current eval.py.

No API cost: reads reports/*_predictions.jsonl, rescores against dev+test
gold using the fixed scoring (loopholes 2-4 closed), and rewrites each
reports/{model}_report.json. Use this whenever scoring logic changes so old
predictions reflect the new rules without re-querying any model.

Usage:
    uv run python rescore.py
    uv run python rescore.py --lenient
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import eval as E


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lenient", action="store_true")
    args = ap.parse_args()

    tools = E.load_tools()
    allex: dict[str, dict] = {}
    for sp in ("dev", "test"):
        for e in E.load_examples(split=sp):
            allex[e["id"]] = e

    for pf in sorted(glob.glob("reports/*_predictions.jsonl")):
        preds: dict[str, list | None] = {}
        for line in open(pf, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            preds[o["id"]] = o.get("predicted_calls")

        rows = []
        for i, c in preds.items():
            if i in allex:
                r = E.score_example(allex[i], c, tools, lenient=args.lenient)
                r["id"] = i
                rows.append(r)

        base = os.path.basename(pf).replace("_predictions.jsonl", "")
        if not rows:
            print(f"{base}: 0 scorable (ids not in dev/test split) — skipped")
            continue

        agg = E.aggregate(rows)
        agg["model"] = base
        mode = "lenient" if args.lenient else "strict"
        agg["note"] = (
            f"RE-SCORED ({mode}) with fixed eval (multi_turn multi-call, "
            f"wrong_call_count penalty, numeric-string coercion). "
            f"Partial run: {len(rows)} of 321 dev examples."
        )
        out = f"reports/{base}_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
        print(f"{base}: weighted={agg['overall_weighted']:.3f} "
              f"unweighted={agg['overall_unweighted']:.3f} n={agg['n_examples']} -> {out}")


if __name__ == "__main__":
    main()
