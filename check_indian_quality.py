"""
Random quality check on data/train_indian.jsonl.
Samples 20 random records, validates each against the tool registry,
flags any issues. Also reports aggregate stats.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent
TOOLS = {t["name"]: t for t in json.loads((ROOT / "tools.json").read_text(encoding="utf-8"))}

records = []
with open(ROOT / "data" / "train_indian.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Total records: {len(records)}\n")

# --- Aggregate stats ---
n_calls = [len(r["calls"]) for r in records]
print("DISTRIBUTION:")
print(f"  Refusals (calls=[]):  {sum(1 for n in n_calls if n == 0):4d}  ({sum(1 for n in n_calls if n == 0)/len(records):.0%})")
print(f"  Single-call:          {sum(1 for n in n_calls if n == 1):4d}  ({sum(1 for n in n_calls if n == 1)/len(records):.0%})")
print(f"  Multi-call (parallel):{sum(1 for n in n_calls if n >= 2):4d}  ({sum(1 for n in n_calls if n >= 2)/len(records):.0%})")

# --- Language ---
def lang_guess(s):
    if re.search(r"[ऀ-ॿ]", s):
        return "hindi_devanagari"
    if re.search(r"[஀-௿]", s):
        return "tamil_native"
    if re.search(r"[ঀ-৿]", s):
        return "bengali_native"
    if re.search(r"\b(hai|bhai|kya|kar|mera|mujhe|paaji|de na|chahiye|karke|likh|bhej|chal|mangwa|wapas|dikhao|theke|amar|tumhi|aache|valo|kothay|kemon|enge|romba|nalla|ille|seyya|venum|panren|panniyirukken)\b", s.lower()):
        return "hinglish_or_transliterated"
    return "english"

langs = Counter(lang_guess(r["messages"][0]["content"]) for r in records)
print("\nLANGUAGE MIX:")
for k, v in langs.most_common():
    print(f"  {k:30s}: {v:4d}  ({v/len(records):.0%})")

# --- Schema re-validation ---
schema_pass = 0
schema_fail = 0
fail_reasons = Counter()
unknown_tools = Counter()
for r in records:
    for c in r["calls"]:
        tool = c.get("tool")
        if tool not in TOOLS:
            unknown_tools[tool] += 1
            schema_fail += 1
            continue
        try:
            jsonschema.validate(c.get("args", {}), TOOLS[tool]["parameters"])
            schema_pass += 1
        except jsonschema.ValidationError as e:
            schema_fail += 1
            reason = e.message[:60]
            fail_reasons[reason] += 1

total_calls = schema_pass + schema_fail
print(f"\nSCHEMA RE-VALIDATION:")
print(f"  Pass: {schema_pass}/{total_calls}  ({schema_pass/total_calls:.0%})" if total_calls else "  No calls")
print(f"  Fail: {schema_fail}/{total_calls}")
if unknown_tools:
    print(f"  Unknown tools called: {dict(unknown_tools)}")
if fail_reasons:
    print(f"  Top fail reasons:")
    for r, n in fail_reasons.most_common(5):
        print(f"    {n}x: {r}")

# --- Tool coverage ---
tool_freq = Counter()
for r in records:
    for c in r["calls"]:
        tool_freq[c["tool"]] += 1
print(f"\nTOOL COVERAGE: {len(tool_freq)} of {len(TOOLS)} tools used")
print("Most-used:")
for t, n in tool_freq.most_common(10):
    print(f"  {n:3d}  {t}")
unused = sorted(set(TOOLS.keys()) - set(tool_freq.keys()))
if unused:
    print(f"\nUnused tools ({len(unused)}): {unused[:10]}{'...' if len(unused)>10 else ''}")

# --- Random sample of 20 ---
print("\n" + "=" * 70)
print("20 RANDOM RECORDS")
print("=" * 70)
random.seed(42)
sample = random.sample(records, min(20, len(records)))
for i, r in enumerate(sample, 1):
    lang = lang_guess(r["messages"][0]["content"])
    print(f"\n[{i}] id={r['id']}  lang={lang}  calls={len(r['calls'])}")
    print(f"  TOOLS available: {[t['name'] for t in r['tools']]}")
    print(f"  USER: {r['messages'][0]['content'][:280]}")
    print(f"  CALLS: {json.dumps(r['calls'], ensure_ascii=False)[:400]}")
    # per-call validation
    issues = []
    for c in r["calls"]:
        tool = c.get("tool")
        if tool not in TOOLS:
            issues.append(f"unknown tool {tool}")
            continue
        try:
            jsonschema.validate(c.get("args", {}), TOOLS[tool]["parameters"])
        except jsonschema.ValidationError as e:
            issues.append(f"{tool}: {e.message[:80]}")
    if issues:
        print(f"  [FAIL] {issues}")
    else:
        print(f"  [OK] valid")
