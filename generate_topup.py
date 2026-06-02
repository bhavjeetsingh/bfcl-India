"""
generate_topup.py — Top up data/train_indian.jsonl for tools with zero coverage.

After generate_indian_training.py finished at 1399 records, 10 flagship tools
had ZERO examples (UPI, IRCTC, Swiggy, Dunzo-send, BMS-search). These are
the centerpiece of the BFCL-India story; the fine-tuned model would
underperform on them without targeted training data.

This script forces a specific list of tools to appear, producing ~30 examples
per tool. Output appends to data/train_indian.jsonl so the existing file
stays the single source of truth for Phase 5c.

Usage:
    uv run python generate_topup.py --per-tool 30
    uv run python generate_topup.py --per-tool 30 --model gemini-2.0-flash-lite
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
    strip_fences, compact_field, compact_tool,
    existing_ids, example_id, validate_call, load_tools_idx,
)

ROOT = Path(__file__).resolve().parent
TOOLS_PATH = ROOT / "tools.json"
SEEDS_PATH = ROOT / "data" / "seeds.json"
OUT_PATH = ROOT / "data" / "train_indian.jsonl"

# The 10 tools that ended up with 0 examples after the main run.
TARGET_TOOLS = [
    "upi_send",
    "upi_collect",
    "upi_mandate_create",
    "check_upi_balance",
    "irctc_search_trains",
    "irctc_book_ticket",
    "swiggy_search_restaurant",
    "swiggy_track_delivery",
    "dunzo_send",
    "bms_search_movie",
]

# Each missing tool gets a hand-picked distractor set so the model learns
# to disambiguate, not just memorize.
DISTRACTORS = {
    "upi_send": ["upi_collect", "check_upi_balance", "upi_mandate_create"],
    "upi_collect": ["upi_send", "upi_transaction_history"],
    "upi_mandate_create": ["upi_send", "fd_create"],
    "check_upi_balance": ["upi_transaction_history", "account_balance"],
    "irctc_search_trains": ["irctc_book_ticket", "irctc_pnr_status", "redbus_search"],
    "irctc_book_ticket": ["irctc_search_trains", "irctc_pnr_status"],
    "swiggy_search_restaurant": ["zomato_place_order", "swiggy_track_delivery"],
    "swiggy_track_delivery": ["zomato_place_order", "cancel_food_order", "amazon_track_order"],
    "dunzo_send": ["dunzo_pickup", "cab_book"],
    "bms_search_movie": ["bms_book_seats", "hotstar_remind", "prime_resume"],
}

LANGUAGES = ["english", "hindi", "hinglish", "tamil_transliterated", "bengali_transliterated"]
LANGUAGE_WEIGHTS = [0.30, 0.25, 0.30, 0.075, 0.075]

AS_OF_DATE = "2026-05-30"

SYSTEM_PROMPT = """You generate TRAINING examples for an Indian-context tool-calling fine-tune.

You will be given:
  1. ONE TARGET tool the user query MUST require.
  2. Distractor tools also visible to the model.
  3. A demo example for format.

Output a JSON array of N training examples. Each example must have exactly:
  user_query (string), calls (array of {{tool, args}}).

Rules:
- The TARGET tool MUST be the one called in every example.
- The user query must clearly justify calling the target tool, not the distractors.
- Use realistic Indian names, places, mobile numbers, VPAs, PINs, station codes,
  PAN numbers, IFSC, etc.
- Treat AS_OF_DATE = {as_of_date} as today. Resolve relative dates to YYYY-MM-DD.
- Vary phrasing and language: {languages}.
- For Hinglish use Roman script with code-switching. For Hindi use Devanagari.
- Every call MUST validate against the JSON Schema:
  - Honour all regex patterns, enums, required fields.
  - Use 'object' fields with their declared sub-keys, not strings.
  - Do NOT invent argument keys not in the schema.
- Output STRICT JSON. No markdown. No prose."""


def load_seeds() -> list[dict[str, Any]]:
    return json.loads(SEEDS_PATH.read_text(encoding="utf-8"))


def generate_for_tool(
    model: Any,
    tools_idx: dict[str, dict[str, Any]],
    seeds: list[dict[str, Any]],
    target: str,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    distractors = DISTRACTORS.get(target, [])
    visible_tools = [target] + distractors

    target_compact = compact_tool(tools_idx[target])
    distractor_compact = [compact_tool(tools_idx[d]) for d in distractors if d in tools_idx]

    # Use a simple-category seed as the format demo (most common shape).
    demo_pool = [s for s in seeds if s["category"] in ("simple", "multiple")]
    demo = rng.choice(demo_pool) if demo_pool else seeds[0]
    demo_compact = {
        "user_query": demo["messages"][0]["content"],
        "calls": demo["ground_truth"]["predicted_calls"],
    }

    languages = rng.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS, k=n)

    prompt = "\n\n".join([
        SYSTEM_PROMPT.format(as_of_date=AS_OF_DATE, languages=", ".join(languages)),
        f"TARGET TOOL (must be called):\n{json.dumps(target_compact, indent=1)}",
        f"DISTRACTOR TOOLS (visible to model, must NOT be called):\n{json.dumps(distractor_compact, indent=1)}",
        f"DEMO (format only — do not copy):\n{json.dumps(demo_compact, indent=1, ensure_ascii=False)}",
        f'Generate {n} examples. Output: {{"examples": [{{"user_query": "...", "calls": [{{"tool": "{target}", "args": {{...}}}}]}}]}}',
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
        if not isinstance(query, str) or not query.strip() or not isinstance(calls, list) or not calls:
            continue
        # Enforce target tool was called.
        if calls[0].get("tool") != target:
            continue
        if not all(validate_call(c, tools_idx) for c in calls):
            continue
        # The TOOLS available to the model in training must include the target + distractors.
        tools_for_record = [tools_idx[t] for t in visible_tools if t in tools_idx]
        valid.append({
            "id": example_id(query, calls),
            "tools": tools_for_record,
            "messages": [{"role": "user", "content": query}],
            "calls": calls,
            "source": "bfcl_indian_train_topup",
        })
    return valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-tool", type=int, default=30,
                        help="Target examples per missing tool.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Examples per Gemini request.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--sleep", type=float, default=4.5)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    tools_idx = load_tools_idx()
    seeds = load_seeds()
    rng = random.Random(20260601)

    seen = existing_ids(OUT_PATH)
    print(f"[topup] model={args.model} per_tool={args.per_tool}")
    print(f"[topup] existing records on disk: {len(seen)}")
    print(f"[topup] missing tools to top up: {len(TARGET_TOOLS)}")
    print(f"[topup] target additions: ~{len(TARGET_TOOLS) * args.per_tool}\n")

    consecutive_skips = 0

    with OUT_PATH.open("a", encoding="utf-8") as fout:
        for tool in TARGET_TOOLS:
            if tool not in tools_idx:
                print(f"[skip] {tool} not in registry")
                continue
            written_for_tool = 0
            pbar = tqdm(total=args.per_tool, desc=tool[:30])
            while written_for_tool < args.per_tool:
                n = min(args.batch_size, args.per_tool - written_for_tool)
                try:
                    batch = generate_for_tool(model, tools_idx, seeds, tool, n, rng)
                except Exception as e:
                    msg = str(e)
                    if any(s in msg.lower() for s in
                           ["tokens per day", "requests per day", "rpd",
                            "quota_id\":\"generaterequestsperday"]):
                        print(f"\n[stop] Daily quota on {args.model}. "
                              f"Made it through {tool} partially. Re-run tomorrow.")
                        pbar.close()
                        return
                    m = re.search(r"retry_delay.*?seconds:\s*(\d+)", msg, re.DOTALL) or \
                        re.search(r"retry in ([\d.]+)\s*s", msg, re.IGNORECASE)
                    if m:
                        wait = float(m.group(1))
                        if wait > 120:
                            print(f"\n[stop] Long backoff ({int(wait/60)} min). Saved what we have.")
                            pbar.close()
                            return
                        print(f"\n[wait] {wait:.0f}s")
                        time.sleep(wait + 1)
                        continue
                    print(f"\n[error] {type(e).__name__}: {msg[:150]}")
                    consecutive_skips += 1
                    if consecutive_skips >= 5:
                        print("[stop] 5 consecutive errors. Bailing.")
                        pbar.close()
                        return
                    time.sleep(8)
                    continue
                consecutive_skips = 0
                added = 0
                for rec in batch:
                    if rec["id"] in seen:
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    seen.add(rec["id"])
                    written_for_tool += 1
                    added += 1
                    pbar.update(1)
                    if written_for_tool >= args.per_tool:
                        break
                fout.flush()
                if added == 0:
                    # All dupes or all dropped; move on rather than spin.
                    break
                time.sleep(args.sleep)
            pbar.close()

    print(f"\n[done] Total records on disk now: {len(seen)}")


if __name__ == "__main__":
    main()
