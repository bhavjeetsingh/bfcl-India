"""
generate_examples.py — Expand BFCL-India seeds into the full 500-example test set.

Reads:  data/seeds.json (your hand-written examples)
        tools.json      (the 50-tool registry)
Writes: data/generated/{category}.jsonl

Run:    uv run python generate_examples.py --category simple --target 5
        uv run python generate_examples.py --category all
        uv run python generate_examples.py --category simple --model gemini-2.0-flash

Free-tier friendly. Resumable: skips IDs already present in the output file.
"""

from __future__ import annotations

import argparse
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
    existing_ids, validate_call, load_tools_idx,
)

ROOT = Path(__file__).resolve().parent
TOOLS_PATH = ROOT / "tools.json"
SEEDS_PATH = ROOT / "data" / "seeds.json"
OUT_DIR = ROOT / "data" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# How many examples per category in the final 500-set.
TARGETS = {
    "simple": 200,
    "multiple": 100,
    "parallel": 50,
    "multi_turn": 100,
    "irrelevance": 50,
}

LANGUAGES = ["english", "hindi", "hinglish", "tamil_transliterated", "bengali_transliterated"]
LANGUAGE_WEIGHTS = [0.35, 0.20, 0.30, 0.075, 0.075]

# Fixed "as_of_date" so date-relative queries are deterministic across re-runs.
AS_OF_DATE = "2026-05-30"

# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are generating evaluation examples for BFCL-India, a function-calling benchmark for Indian-context APIs.

You will be given:
  1. A JSON registry of available tools (with strict JSON Schema parameters).
  2. A few hand-written EXAMPLE EVALUATION cases in the SAME format you must produce.
  3. Generation constraints (category, language, count).

Output STRICTLY a JSON array of N example objects. No prose, no markdown fences. Each object must have exactly these keys:
  id, category, available_tools, language, messages, ground_truth.
Use realistic Indian names, places, phone numbers, VPAs, PINs, etc. Do not reuse the example queries verbatim — vary phrasing, tools, parameters, and intents.

Treat AS_OF_DATE = {as_of_date} as today. Resolve relative dates ('tomorrow', 'next Friday', 'in 3 days') to absolute YYYY-MM-DD against this anchor and put the resolved date in ground_truth args. Never put words like 'tomorrow' inside ground_truth.

For Hinglish examples, use Roman script with code-switching ("paaji ko 500 bhej de"), not pure Devanagari. For Hindi examples, use Devanagari. Preserve the user's casual tone in messages but ensure ground_truth is always strictly-typed.

Every ground_truth tool call MUST validate against the JSON Schema of the called tool. Do not invent argument keys. Do not omit required arguments.

CRITICAL SCHEMA RULES:
- If a field's `type` is `object`, you MUST emit a JSON object containing ONLY the keys listed in `object_fields`. Never emit a string or array where an object is required.
- If a field has `object_required`, those nested keys MUST be present.
- If a field's `type` is `array` with `array_item_fields`, every array element must be an object using ONLY those keys.
- Honour ALL regex patterns. Examples:
    - bank account_number: 9-18 digits (e.g. "50100123456789", NEVER "1234")
    - PAN: 5 letters + 4 digits + 1 letter (e.g. "ABCDE1234F")
    - mobile: 10 digits starting 6-9 (e.g. "9876543210", NEVER "+91...")
    - PIN code: 6 digits, first non-zero (e.g. "110092")
    - VPA: name@bank format (e.g. "rohit@okhdfc")
    - IFSC: 4 letters + "0" + 6 alphanum (e.g. "HDFC0000234")
    - vehicle reg: e.g. "DL3CAB1234", "KA01MJ7777"
- Honour enum constraints exactly — match the casing shown in the schema.
- Never emit fields that are not declared in `fields` or `object_fields`.
"""

CATEGORY_PROMPTS = {
    "simple": """Generate {n} SIMPLE category examples.
- One user turn, one tool call in ground_truth.
- 'available_tools' is a list of 1-2 tool names (the correct tool, optionally one near-distractor).
- About 30% of queries should include small-talk/noise BEFORE the actionable request.
- Pick tools from this subset: {tool_subset}.
- Languages must be sampled from: {languages}.
- IDs must follow the pattern bfcl_india_simple_{start:03d} ... bfcl_india_simple_{end:03d}.""",

    "multiple": """Generate {n} MULTIPLE category examples.
- One user turn, one tool call in ground_truth.
- 'available_tools' is a list of 2-5 tool names. The correct tool is in the list along with NEAR-DUPLICATES from the same family.
- The user's query should be ambiguous enough that a weak model would pick the wrong one.
- Pick tool families from: {tool_subset}.
- Languages: {languages}.
- IDs: bfcl_india_multiple_{start:03d} ... bfcl_india_multiple_{end:03d}.""",

    "parallel": """Generate {n} PARALLEL category examples.
- One user turn, MULTIPLE independent tool calls in ground_truth (2-3 calls per example).
- The user's request must compose multiple actions in a single sentence.
- Pick tools from: {tool_subset}.
- Languages: {languages}.
- IDs: bfcl_india_parallel_{start:03d} ... bfcl_india_parallel_{end:03d}.""",

    "multi_turn": """Generate {n} MULTI-TURN category examples.
- 2-4 turns total. Structure: user -> assistant.tool_calls -> tool (result, JSON-string content) -> user -> ground_truth.predicted_calls.
- The TOOL message MUST contain a realistic JSON-string payload representing the API's response. Include both happy-path AND error/recovery results.
- Ground truth is the FINAL tool call (or empty list if the final correct action is to answer in natural language).
- Pick tools from: {tool_subset}.
- Languages: {languages}.
- IDs: bfcl_india_multiturn_{start:03d} ... bfcl_india_multiturn_{end:03d}.""",

    "irrelevance": """Generate {n} IRRELEVANCE category examples.
- One user turn. ground_truth.predicted_calls MUST be an empty list [].
- The user query should look superficially related to one of the available tools but actually requires capabilities NONE of the tools have.
- 'available_tools' is a list of 2-4 tools that a weak model might force-fit.
- Pick decoy tools from: {tool_subset}.
- Languages: {languages}.
- IDs: bfcl_india_irrelevance_{start:03d} ... bfcl_india_irrelevance_{end:03d}.""",
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def load_tools() -> list[dict[str, Any]]:
    return json.loads(TOOLS_PATH.read_text(encoding="utf-8"))


def load_seeds() -> list[dict[str, Any]]:
    return json.loads(SEEDS_PATH.read_text(encoding="utf-8"))


def schema_index(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {t["name"]: t for t in tools}


def validate_example(ex: dict[str, Any], tools_idx: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    """Returns (ok, reason). Drops anything that can't be scored."""
    required_top = {"id", "category", "available_tools", "language", "messages", "ground_truth"}
    if not required_top.issubset(ex):
        return False, f"missing top-level keys: {required_top - set(ex)}"
    if not isinstance(ex["available_tools"], list) or not ex["available_tools"]:
        return False, "available_tools must be a non-empty list"
    for t in ex["available_tools"]:
        if t not in tools_idx:
            return False, f"unknown tool '{t}'"
    gt = ex.get("ground_truth")
    if not isinstance(gt, dict):
        return False, "ground_truth must be an object"
    calls = gt.get("predicted_calls")
    if calls is None or not isinstance(calls, list):
        return False, "ground_truth.predicted_calls missing or not a list"
    if ex["category"] == "irrelevance" and calls:
        return False, "irrelevance must have empty predicted_calls"
    if ex["category"] != "irrelevance" and not calls:
        return False, "non-irrelevance category requires at least one call"
    for call in calls:
        if not isinstance(call, dict):
            return False, f"call is not an object: {type(call).__name__}"
        if call.get("tool") not in tools_idx:
            return False, f"called unknown tool '{call.get('tool')}'"
        schema = tools_idx[call["tool"]]["parameters"]
        try:
            jsonschema.validate(call.get("args", {}), schema)
        except jsonschema.ValidationError as e:
            return False, f"args fail schema for {call['tool']}: {e.message}"
    return True, "ok"


def pick_seed_examples(seeds: list[dict[str, Any]], category: str, k: int = 1) -> list[dict[str, Any]]:
    pool = [s for s in seeds if s["category"] == category]
    if not pool:
        return []
    random.shuffle(pool)
    return pool[:k]


# ----------------------------------------------------------------------------
# Generation loop
# ----------------------------------------------------------------------------


def generate_batch(
    model: Any,
    tools: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    tools_idx: dict[str, dict[str, Any]],
    category: str,
    n: int,
    start_id: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Ask Gemini for `n` examples in one call. Returns the validated subset."""
    subset = pick_tool_subset(tools, k=4)
    languages = rng.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS, k=n)
    seed_demos = pick_seed_examples(seeds, category, k=1)

    compact_subset = [compact_tool(tools_idx[name]) for name in subset if name in tools_idx]

    prompt = "\n\n".join(
        [
            SYSTEM_PROMPT.format(as_of_date=AS_OF_DATE),
            "TOOL REGISTRY (compact, this batch only):\n"
            + json.dumps(compact_subset, indent=1),
            "EXAMPLE EVALUATION CASE (study format, do not copy):\n"
            + json.dumps(seed_demos, indent=1, ensure_ascii=False),
            CATEGORY_PROMPTS[category].format(
                n=n,
                tool_subset=", ".join(subset),
                languages=", ".join(languages),
                start=start_id,
                end=start_id + n - 1,
            ),
            "Output: a single JSON array of exactly N objects. No markdown.",
        ]
    )

    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.9,
            "max_output_tokens": 4000,
        },
    )
    raw = strip_fences(response.text)
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # If the model returns {"examples": [...]} or similar, unwrap.
    if isinstance(items, dict):
        for v in items.values():
            if isinstance(v, list):
                items = v
                break
        else:
            return []
    if not isinstance(items, list):
        return []

    valid: list[dict[str, Any]] = []
    for ex in items:
        if not isinstance(ex, dict):
            print(f"  [drop] non-dict item: {type(ex).__name__}")
            continue
        try:
            ok, reason = validate_example(ex, tools_idx)
        except Exception as e:
            print(f"  [drop] {ex.get('id', '?')}: malformed ({type(e).__name__}: {e})")
            continue
        if ok:
            valid.append(ex)
        else:
            print(f"  [drop] {ex.get('id', '?')}: {reason}")
    return valid


def init_client(model_name: str) -> Any:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing — populate .env first.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def run(category: str, target: int, batch_size: int, model_name: str, sleep_s: float) -> None:
    load_dotenv()
    model = init_client(model_name)

    tools = load_tools()
    tools_idx = schema_index(tools)
    seeds = load_seeds()

    rng = random.Random(hash(category) & 0xFFFFFFFF)

    out_path = OUT_DIR / f"{category}.jsonl"
    seen = existing_ids(out_path)
    print(f"[{category}] model={model_name}, target={target}, "
          f"already_have={len(seen)}, output={out_path.relative_to(ROOT)}")

    written = len(seen)
    pbar = tqdm(total=target, initial=written, desc=category)
    next_id = written + 1

    with out_path.open("a", encoding="utf-8") as fout:
        while written < target:
            n = min(batch_size, target - written)
            try:
                batch = generate_batch(model, tools, seeds, tools_idx, category, n, next_id, rng)
            except Exception as e:
                print(f"  [error] {type(e).__name__}: {e}")
                time.sleep(15)
                continue
            for ex in batch:
                if ex["id"] in seen:
                    continue
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                seen.add(ex["id"])
                written += 1
                next_id += 1
                pbar.update(1)
                if written >= target:
                    break
            fout.flush()
            time.sleep(sleep_s)
    pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=[*TARGETS.keys(), "all"], default="all")
    parser.add_argument("--target", type=int, default=None,
                        help="Override default count for the chosen category.")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Examples per Gemini request. Lower = more requests, smaller prompts.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite",
                        help="Gemini model id. gemini-3.1-flash-lite (default, 15 RPM free), "
                             "gemini-3.5-flash (better quality, 5 RPM), "
                             "gemini-2.5-flash-lite (backup, 10 RPM).")
    parser.add_argument("--sleep", type=float, default=4.5,
                        help="Seconds between requests. Free tier = 15 RPM, so >=4 is safe.")
    args = parser.parse_args()

    cats = list(TARGETS.keys()) if args.category == "all" else [args.category]
    for cat in cats:
        target = args.target if args.target is not None else TARGETS[cat]
        run(cat, target, batch_size=args.batch_size, model_name=args.model, sleep_s=args.sleep)


if __name__ == "__main__":
    main()
