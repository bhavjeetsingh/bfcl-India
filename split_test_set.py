"""
split_test_set.py — One-time deterministic split of the 421-example BFCL-India
test set into a public dev split (used during HP tuning) and a secret test
split (run exactly once after final HP selection).

Reads:  data/generated/*.jsonl   (the 421 examples)
Writes: data/eval/dev.jsonl      (~321 examples, public, used freely)
        data/eval/test.jsonl     (~100 examples, secret, run once at end)
        data/eval/SPLIT.md       (audit log: which IDs went where, seed used)

This addresses the integrity loophole flagged in SPEC.md §3.

Reproducibility: deterministic shuffle with seed=20260601. Re-running this
script always produces the same split, so anyone reproducing the work gets
the same dev/test partition. The split is COMMITTED to git; the test file
itself is what you should NOT look at during model development.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "generated"
OUT_DIR = ROOT / "data" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260601
TEST_SIZE = 100  # secret split

# Per-category quotas in the test split (proportional to total mix, rounded).
# Ensures every category has *some* held-out signal, not just whatever the
# random shuffle hands us.
PER_CAT_TEST = {
    "simple": 47,
    "multiple": 24,
    "parallel": 12,
    "multi_turn": 13,
    "irrelevance": 4,
}
# Totals to 100. Remaining go to dev.


def load_all() -> list[dict]:
    out = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> None:
    examples = load_all()
    print(f"Loaded {len(examples)} examples")

    rng = random.Random(SEED)
    by_cat: dict[str, list[dict]] = {}
    for ex in examples:
        by_cat.setdefault(ex["category"], []).append(ex)

    test_set: list[dict] = []
    dev_set: list[dict] = []

    for cat, items in by_cat.items():
        rng.shuffle(items)
        n_test = min(PER_CAT_TEST.get(cat, 0), len(items))
        test_set.extend(items[:n_test])
        dev_set.extend(items[n_test:])

    rng.shuffle(test_set)
    rng.shuffle(dev_set)

    test_path = OUT_DIR / "test.jsonl"
    dev_path = OUT_DIR / "dev.jsonl"
    test_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in test_set) + "\n",
        encoding="utf-8",
    )
    dev_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in dev_set) + "\n",
        encoding="utf-8",
    )

    # Audit log so reviewers can verify reproducibility.
    test_cats = Counter(e["category"] for e in test_set)
    dev_cats = Counter(e["category"] for e in dev_set)
    audit = [
        "# BFCL-India eval split audit\n",
        f"Seed: {SEED}",
        f"Total examples: {len(examples)}",
        f"Dev split (public, used during HP tuning): {len(dev_set)} examples",
        f"Test split (secret, run once after final HP): {len(test_set)} examples\n",
        "## Per-category counts",
        "| Category | Dev | Test | Total |",
        "|---|---|---|---|",
    ]
    for cat in sorted(by_cat):
        audit.append(f"| {cat} | {dev_cats.get(cat, 0)} | {test_cats.get(cat, 0)} | {len(by_cat[cat])} |")
    audit.extend([
        "",
        "## Test-split IDs (locked — DO NOT inspect during development)",
        "",
        *[f"- {e['id']}" for e in sorted(test_set, key=lambda x: x['id'])],
    ])
    (OUT_DIR / "SPLIT.md").write_text("\n".join(audit), encoding="utf-8")

    print(f"Wrote {dev_path.relative_to(ROOT)} ({len(dev_set)} examples)")
    print(f"Wrote {test_path.relative_to(ROOT)} ({len(test_set)} examples)")
    print(f"Wrote {(OUT_DIR / 'SPLIT.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
