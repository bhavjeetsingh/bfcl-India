"""
quick_score.py — Score whatever predictions exist for a given model on disk.

Useful when an eval run was interrupted by quota — gives you the partial baseline
without re-running anything. Reuses eval.py's score_example so numbers match a
full-run report.

Usage:
    uv run python quick_score.py llama-3.3-70b-versatile
    uv run python quick_score.py gemini-2.5-flash-lite
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from eval import (
    DATA_DIR,
    REPORTS_DIR,
    aggregate,
    load_tools,
    score_example,
)


def load_all_examples() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out[obj["id"]] = obj
            except json.JSONDecodeError:
                continue
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python quick_score.py <model-label>")
        print("       (matches the prefix of <label>_predictions.jsonl in reports/)")
        sys.exit(1)

    label = sys.argv[1].replace("/", "_").replace(":", "_")
    pred_path = REPORTS_DIR / f"{label}_predictions.jsonl"

    if not pred_path.exists():
        print(f"No predictions file at {pred_path}")
        sys.exit(1)

    tools_idx = load_tools()
    examples = load_all_examples()

    preds: dict[str, list[dict]] = {}
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            preds[rec["id"]] = rec.get("predicted_calls")
        except Exception:
            continue

    rows = []
    for ex_id, calls in preds.items():
        ex = examples.get(ex_id)
        if not ex:
            continue
        row = score_example(ex, calls, tools_idx)
        row["id"] = ex_id
        rows.append(row)

    if not rows:
        print("No scoreable rows.")
        return

    summary = aggregate(rows)
    summary["model_label"] = label
    summary["note"] = f"Partial run — scored {len(rows)} of 421 examples"

    print(json.dumps(summary, indent=2))

    out_path = REPORTS_DIR / f"{label}_partial_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
