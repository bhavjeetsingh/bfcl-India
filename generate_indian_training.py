"""
generate_indian_training.py — Generate Indian-context TRAINING examples.

This is the missing piece between Phase 5 and Phase 7. The mixed training set
(xLAM + Glaive) teaches the model generic function-calling but not Indian
patterns (UPI VPAs, IRCTC station codes, IFSC, PIN codes, Hinglish phrasing).

Reads:  tools.json (the 50-tool registry)
        data/seeds.json (10 hand-written examples for few-shot conditioning)
Writes: data/train_indian.jsonl

Output shape matches the {messages, tools, calls, source} record that
prepare_training_data.py's main loop consumes — so re-running prepare merges
this file into the final train/val splits with no extra parser work.

CRITICAL — train/test separation:
  - Test set: data/generated/*.jsonl with ids like bfcl_india_simple_001
  - This file: data/train_indian.jsonl with ids like bfcl_train_*
  - No id collision possible. Phrasings sampled at temperature 0.95 with
    rotating tool subsets so distribution doesn't trivially overlap test.

Usage:
    uv run python generate_indian_training.py --target 1000
    uv run python generate_indian_training.py --target 2000 --model gemini-2.0-flash-lite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
import jsonschema
from dotenv import load_dotenv
from tqdm import tqdm

from utils import (
    strip_fences, compact_field, compact_tool, pick_tool_subset,
    existing_ids, example_id, validate_call, load_tools_idx,
)

ROOT = Path(__file__).resolve().parent
TOOLS_PATH = ROOT / "tools.json"
SEEDS_PATH = ROOT / "data" / "seeds.json"
OUT_PATH = ROOT / "data" / "train_indian.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)

LANGUAGES = ["english", "hindi", "hinglish", "tamil_transliterated", "bengali_transliterated"]
LANGUAGE_WEIGHTS = [0.30, 0.25, 0.30, 0.075, 0.075]

# Distribution of training-example types. Heavier on simple to match real
# agent traffic; some parallel + refusal so the model learns those edges.
EXAMPLE_TYPE_WEIGHTS = {"simple": 0.65, "parallel": 0.20, "refusal": 0.15}

AS_OF_DATE = "2026-05-30"

SYSTEM_PROMPT = """You generate TRAINING examples for an Indian-context tool-calling fine-tune.

You will be given:
  1. A subset of tools (compact JSON Schema).
  2. One demonstration example to match the format (do NOT copy it verbatim).
  3. The TYPE of example to produce: simple / parallel / refusal.

Output a JSON array of N training examples. Each example must have exactly:
  user_query (string), calls (array of {{tool, args}}).
For 'refusal' type, calls MUST be an empty array [].

Use realistic Indian names, places, phone numbers, VPAs, PINs, station codes,
PAN numbers, etc. Vary phrasing and language as instructed.

Treat AS_OF_DATE = {as_of_date} as today. Resolve "tomorrow" / "next Friday" /
"in 3 days" to absolute YYYY-MM-DD against this anchor and put the resolved
date in args. Never put words like "tomorrow" inside args.

For Hinglish use Roman script with code-switching. For Hindi use Devanagari.

Every call MUST validate against the JSON Schema:
- Honour all regex patterns, enums, required fields.
- Use 'object' fields with their declared sub-keys, not strings.
- Do NOT invent argument keys not in the schema.

Output STRICT JSON. No markdown. No prose."""


TYPE_PROMPTS = {
    "simple": """Generate {n} SIMPLE examples. Each has ONE tool call.
Pick from these tools (mix randomly): {tool_subset}.
Languages to sample from: {languages}.""",

    "parallel": """Generate {n} PARALLEL examples. Each has 2-3 tool calls.
The user query must compose multiple actions in one sentence
('block my card AND order chequebook', 'pay electricity bill AND water bill').
Pick from: {tool_subset}.
Languages: {languages}.""",

    "refusal": """Generate {n} REFUSAL examples. Each has calls=[].
The user query should superficially relate to the available tools but actually
require capabilities none of them have (translation, weather, recipes, math,
opinion, knowledge questions, code generation).
Available tools (decoys): {tool_subset}.
Languages: {languages}.""",
}


def load_tools() -> list[dict[str, Any]]:
    return json.loads(TOOLS_PATH.read_text(encoding="utf-8"))


def load_seeds() -> list[dict[str, Any]]:
    return json.loads(SEEDS_PATH.read_text(encoding="utf-8"))


def schema_index(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {t["name"]: t for t in tools}


def generate_batch(
    model: Any,
    tools_idx: dict[str, dict[str, Any]],
    all_tools: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    ex_type: str,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    subset = pick_tool_subset(all_tools, k=4 if ex_type != "parallel" else 6, rng=rng)
    languages = rng.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS, k=n)

    seed_pool = [s for s in seeds if s["category"] in ({"parallel"} if ex_type == "parallel"
                                                       else {"irrelevance"} if ex_type == "refusal"
                                                       else {"simple", "multiple"})]
    if not seed_pool:
        seed_pool = seeds
    demo = rng.choice(seed_pool)
    demo_compact = {
        "user_query": demo["messages"][0]["content"],
        "calls": demo["ground_truth"]["predicted_calls"],
    }

    compact = [compact_tool(tools_idx[name]) for name in subset if name in tools_idx]

    prompt = "\n\n".join([
        SYSTEM_PROMPT.format(as_of_date=AS_OF_DATE),
        "TOOLS:\n" + json.dumps(compact, indent=1),
        "DEMO:\n" + json.dumps(demo_compact, indent=1, ensure_ascii=False),
        TYPE_PROMPTS[ex_type].format(n=n, tool_subset=", ".join(subset),
                                     languages=", ".join(languages)),
        'Output: {"examples": [{"user_query": "...", "calls": [...]}]}',
    ])

    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.95,
            "max_output_tokens": 4000,
        },
    )
    raw = strip_fences(response.text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    items = parsed.get("examples") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        return []

    valid: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        query = it.get("user_query")
        calls = it.get("calls")
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(calls, list):
            continue
        if ex_type == "refusal" and calls:
            continue
        if ex_type != "refusal" and not calls:
            continue
        if all(validate_call(c, tools_idx)[0] for c in calls):
            tools_for_this_example = [tools_idx[name] for name in subset]
            valid.append({
                "id": example_id(query, calls),
                "tools": tools_for_this_example,
                "messages": [{"role": "user", "content": query}],
                "calls": calls,
                "source": "bfcl_indian_train",
            })
    return valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1500,
                        help="Total Indian training examples to produce.")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.0-flash-lite",
                        help="Gemini model. Try gemini-2.0-flash-lite first (1500 RPD), "
                             "fall back to gemini-3.1-flash-lite (500 RPD).")
    parser.add_argument("--sleep", type=float, default=4.5)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    all_tools = load_tools()
    tools_idx = schema_index(all_tools)
    seeds = load_seeds()
    rng = random.Random(20260530)

    seen = existing_ids(OUT_PATH)
    print(f"[indian-train] model={args.model} target={args.target} have={len(seen)}")

    written = len(seen)
    pbar = tqdm(total=args.target, initial=written, desc="bfcl_indian_train")
    consecutive_skips = 0

    with OUT_PATH.open("a", encoding="utf-8") as fout:
        while written < args.target:
            ex_type = rng.choices(
                list(EXAMPLE_TYPE_WEIGHTS.keys()),
                weights=list(EXAMPLE_TYPE_WEIGHTS.values()),
                k=1,
            )[0]
            n = min(args.batch_size, args.target - written)
            try:
                batch = generate_batch(model, tools_idx, all_tools, seeds, ex_type, n, rng)
            except Exception as e:
                msg = str(e)
                if any(s in msg.lower() for s in ["tokens per day", "requests per day", "rpd", "quota_id\":\"generaterequestsperday"]):
                    print(f"\n  [stop] Daily quota exhausted on {args.model}. {written} examples saved.")
                    break
                m = re.search(r"retry_delay.*?seconds:\s*(\d+)", msg, re.DOTALL) or \
                    re.search(r"retry in ([\d.]+)\s*s", msg, re.IGNORECASE)
                if m:
                    wait = float(m.group(1))
                    if wait > 120:
                        print(f"\n  [stop] Long backoff ({int(wait/60)} min). {written} saved.")
                        break
                    print(f"\n  [wait] {wait:.0f}s")
                    time.sleep(wait + 1)
                    continue
                print(f"\n  [error] {type(e).__name__}: {msg[:150]}")
                consecutive_skips += 1
                if consecutive_skips >= 5:
                    print(f"\n  [stop] 5 consecutive errors. {written} saved.")
                    break
                time.sleep(8)
                continue
            consecutive_skips = 0
            for rec in batch:
                if rec["id"] in seen:
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                seen.add(rec["id"])
                written += 1
                pbar.update(1)
                if written >= args.target:
                    break
            fout.flush()
            time.sleep(args.sleep)
    pbar.close()
    print(f"\nWrote {written} examples to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
