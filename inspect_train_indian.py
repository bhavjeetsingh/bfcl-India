import json

print("=" * 70)
print("INSPECTING data/train_indian.jsonl")
print("=" * 70)

records = []
with open("data/train_indian.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"\nTotal records: {len(records)}\n")

# Show first 3 in detail
for i, e in enumerate(records[:3], 1):
    print(f"--- Record {i}: id={e['id']}, source={e['source']}")
    tool_names = [t["name"] for t in e["tools"]]
    print(f"  Available tools: {tool_names}")
    print(f"  USER: {e['messages'][0]['content'][:300]}")
    print(f"  CALLS: {json.dumps(e['calls'], ensure_ascii=False, indent=2)[:600]}")
    print()

# Aggregate stats
print("=" * 70)
print("STATS")
print("=" * 70)
n_calls = [len(e["calls"]) for e in records]
print(f"Empty (refusal): {sum(1 for n in n_calls if n == 0)}")
print(f"Single call:     {sum(1 for n in n_calls if n == 1)}")
print(f"Multiple calls:  {sum(1 for n in n_calls if n >= 2)}")

# Tool frequency in calls
from collections import Counter
tool_freq = Counter()
for e in records:
    for c in e["calls"]:
        tool_freq[c["tool"]] += 1
print(f"\nMost-called tools (top 10):")
for t, c in tool_freq.most_common(10):
    print(f"  {t}: {c}")

# Language guess (rough — look for devanagari)
import re
def lang_guess(s):
    if re.search(r"[ऀ-ॿ]", s):
        return "hindi_devanagari"
    if re.search(r"\b(hai|bhai|kya|kar|mera|mujhe|paaji|de na|chahiye)\b", s.lower()):
        return "hinglish"
    return "english"

langs = Counter(lang_guess(e["messages"][0]["content"]) for e in records)
print(f"\nLanguage mix:")
for k, v in langs.most_common():
    print(f"  {k}: {v} ({v/len(records):.0%})")
