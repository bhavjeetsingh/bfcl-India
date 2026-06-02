"""
prepare_training_data.py — Build train.jsonl + val.jsonl for fine-tuning.

Reads:  data/glaive.jsonl       (110K Glaive function-calling)
        data/xlam.jsonl         (60K Salesforce xLAM)
        data/apigen_mt.jsonl    (5K APIGen-MT, multi-turn)
        data/seeds.json         (BFCL-India hand-written seeds)
Writes: data/train.jsonl
        data/val.jsonl
        data/data_stats.json    (counts per source, dedup stats)

Output format: Qwen-2.5 chat template with assistant emitting {"calls": [...]}.
This MUST match eval.py's expected output format exactly — otherwise the
fine-tune teaches the wrong shape and eval scores get worse.

Usage:
    uv run python prepare_training_data.py
    uv run python prepare_training_data.py --max-train 50000
    uv run python prepare_training_data.py --skip glaive   (if a source is broken)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Same anchor as generate_examples.py / eval.py.
AS_OF_DATE = "2026-05-30"

# Aligned with eval.py — assistant must learn to emit this exact shape.
SYSTEM_PROMPT_TEMPLATE = """You are a tool-calling assistant. The user's request must be answered ONLY by calling one or more of the tools provided.

Today's date is {date}. When the user says "tomorrow", "next Friday", "in 3 days", resolve to an absolute YYYY-MM-DD date against this anchor and put the resolved date in the tool args. Never put words like "tomorrow" inside args.

Output a single JSON object: {{"calls": [{{"tool": "<name>", "args": {{...}}}}, ...]}}.

Rules:
- If multiple tools are needed (parallel), include all of them in the "calls" array.
- For multi-turn conversations, output ONLY the next call(s) given the conversation so far.
- If NO available tool can satisfy the request, output {{"calls": []}}.
- Do not invent tool names. Do not invent argument keys not in the schema.
- Honour all regex patterns, enums, and required fields.
- Output strict JSON. No markdown, no prose, no explanation.

AVAILABLE TOOLS:
{tools_json}"""


# ----------------------------------------------------------------------------
# Source parsers — each yields {messages, tools, source} records.
# Each parser is defensive: bad records are skipped, not raised.
# ----------------------------------------------------------------------------


def parse_xlam(path: Path) -> Iterator[dict[str, Any]]:
    """xLAM: {query, tools (JSON string), answers (JSON string with tool calls)}"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            answers = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
            if not isinstance(tools, list) or not isinstance(answers, list):
                continue
            calls = []
            for a in answers:
                tool = a.get("name") or a.get("tool")
                args = a.get("arguments") or a.get("args") or {}
                if tool:
                    calls.append({"tool": tool, "args": args})
            if not calls:
                continue
            yield {
                "messages": [{"role": "user", "content": r["query"]}],
                "tools": tools,
                "calls": calls,
                "source": "xlam",
            }
        except Exception:
            continue


def parse_glaive(path: Path) -> Iterator[dict[str, Any]]:
    """Glaive: {system (tools embedded in text), chat (string with USER:/ASSISTANT:)}.

    Glaive's format: assistant turn may contain prose THEN a <functioncall> tag,
    or be a pure refusal with no tag. Refusals are skipped.
    """
    if not path.exists():
        return
    user_re = re.compile(r"USER:\s*(.+?)(?=ASSISTANT:|FUNCTION RESPONSE:|<\|endoftext\|>|$)", re.DOTALL)
    # Match the call object after <functioncall>, regardless of preceding prose.
    call_re = re.compile(r"<functioncall>\s*(\{.+?\})\s*(?:<\|endoftext\|>|$|USER:|ASSISTANT:)", re.DOTALL)
    tools_re = re.compile(r"\{\s*\"name\"\s*:\s*\"[^\"]+\".*?\n\}", re.DOTALL)

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            sys_text = r.get("system", "") or ""
            chat = r.get("chat", "") or ""

            # Extract tool definitions from system text (each is a multi-line JSON object).
            tools = []
            for m in tools_re.finditer(sys_text):
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict) and "name" in obj:
                        tools.append(obj)
                except Exception:
                    continue
            if not tools:
                continue

            user_match = user_re.search(chat)
            call_match = call_re.search(chat)
            if not user_match or not call_match:
                continue  # refusal or malformed — skip

            user_q = user_match.group(1).strip()
            try:
                call_obj = json.loads(call_match.group(1).strip())
            except Exception:
                continue

            tool_name = call_obj.get("name")
            args = call_obj.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    continue
            if not tool_name or not isinstance(args, dict):
                continue

            yield {
                "messages": [{"role": "user", "content": user_q}],
                "tools": tools,
                "calls": [{"tool": tool_name, "args": args}],
                "source": "glaive",
            }
        except Exception:
            continue


def parse_apigen_mt(path: Path) -> Iterator[dict[str, Any]]:
    """APIGen-MT: multi-turn function-calling traces with conversation field."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            tools = r.get("tools") or r.get("functions") or []
            if isinstance(tools, str):
                tools = json.loads(tools)
            if not isinstance(tools, list) or not tools:
                continue

            convo = r.get("conversation") or r.get("messages") or []
            if not isinstance(convo, list) or len(convo) < 2:
                continue

            # Find the last assistant turn that contains tool calls — that's our target.
            history: list[dict[str, str]] = []
            target_calls: list[dict[str, Any]] | None = None
            for turn in convo:
                role = turn.get("role") or turn.get("from")
                content = turn.get("content") or turn.get("value") or ""
                if role in ("user", "human"):
                    history.append({"role": "user", "content": str(content)})
                elif role in ("assistant", "gpt", "model"):
                    tc = turn.get("tool_calls") or turn.get("function_calls")
                    if tc:
                        calls = []
                        for c in tc:
                            tool = c.get("name") or (c.get("function") or {}).get("name")
                            args = c.get("arguments") or (c.get("function") or {}).get("arguments")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {}
                            if tool:
                                calls.append({"tool": tool, "args": args or {}})
                        if calls:
                            target_calls = calls
                            break  # train on the FIRST tool-call turn (simpler)
                    history.append({"role": "assistant", "content": str(content)})
                elif role in ("tool", "function"):
                    history.append({"role": "tool", "content": str(content)})

            if not target_calls or not history:
                continue

            yield {
                "messages": history,
                "tools": tools,
                "calls": target_calls,
                "source": "apigen_mt",
            }
        except Exception:
            continue


def parse_indian_train(path: Path) -> Iterator[dict[str, Any]]:
    """BFCL-India training set produced by generate_indian_training.py +
    generate_topup.py. Records are already in the {messages, tools, calls}
    shape this script's pipeline expects — we normalise the source label so
    the main run + top-up records share a single bucket in target_mix."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if not r.get("messages") or not r.get("tools"):
                continue
            yield {
                "messages": r["messages"],
                "tools": r["tools"],
                "calls": r.get("calls", []),
                "source": "indian_train",  # collapse main + topup labels
            }
        except Exception:
            continue


def parse_bfcl_seeds(path: Path) -> Iterator[dict[str, Any]]:
    """BFCL-India hand-written seeds — high-quality Indian-context examples.

    NOTE: We do NOT include data/generated/ here — those are held out as the
    test set. Training on them would leak ground truth into the model.
    """
    if not path.exists():
        return
    try:
        seeds = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    tools_path = ROOT / "tools.json"
    try:
        all_tools = json.loads(tools_path.read_text(encoding="utf-8"))
    except Exception:
        return
    tools_idx = {t["name"]: t for t in all_tools}

    for ex in seeds:
        try:
            available = ex.get("available_tools") or []
            tools = [tools_idx[n] for n in available if n in tools_idx]
            if not tools:
                continue
            calls = ex.get("ground_truth", {}).get("predicted_calls")
            if calls is None:
                continue
            messages = []
            for m in ex.get("messages", []):
                role = m.get("role", "user")
                if role == "tool":
                    messages.append({
                        "role": "tool",
                        "content": f"[{m.get('tool_name', '?')}] {m.get('content', '')}",
                    })
                elif role == "assistant" and m.get("tool_calls"):
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({"calls": m["tool_calls"]}, ensure_ascii=False),
                    })
                else:
                    messages.append({"role": role, "content": m.get("content", "")})
            yield {
                "messages": messages,
                "tools": tools,
                "calls": calls,
                "source": "bfcl_seeds",
            }
        except Exception:
            continue


# ----------------------------------------------------------------------------
# Conversion to Qwen chat-template format
# ----------------------------------------------------------------------------


def to_chat_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a normalized record to {messages: [{role, content}, ...]}.

    The assistant output is the JSON {"calls": [...]} that eval.py expects.
    """
    tools_json = json.dumps(rec["tools"], ensure_ascii=False, indent=1)
    if len(tools_json) > 12000:
        return None  # too long — would blow context window

    system = SYSTEM_PROMPT_TEMPLATE.format(date=AS_OF_DATE, tools_json=tools_json)
    chat = [{"role": "system", "content": system}]
    chat.extend(rec["messages"])
    chat.append({
        "role": "assistant",
        "content": json.dumps({"calls": rec["calls"]}, ensure_ascii=False),
    })
    return {"messages": chat, "source": rec["source"]}


# ----------------------------------------------------------------------------
# Filter + dedup
# ----------------------------------------------------------------------------


def first_user_text(rec: dict[str, Any]) -> str:
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip().lower()
    return ""


def query_hash(rec: dict[str, Any]) -> str:
    return hashlib.sha1(first_user_text(rec).encode("utf-8")).hexdigest()


def looks_valid(rec: dict[str, Any]) -> bool:
    if not rec.get("tools") or not rec.get("messages"):
        return False
    calls = rec.get("calls")
    # Empty list is OK — refusal training data has calls=[] on purpose.
    # None means the parser didn't extract any structure, which is broken.
    if calls is None:
        return False
    if not isinstance(calls, list):
        return False
    for c in calls:
        if not c.get("tool") or not isinstance(c.get("args"), dict):
            return False
    return True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=30000,
                        help="Hard cap on total training examples (Kaggle T4 budget).")
    parser.add_argument("--strict-mix", action="store_true",
                        help="Honour target_mix proportions EXACTLY by scaling the total "
                             "down to whatever the scarcest source supports. Prevents the "
                             "Indian fraction from being silently diluted by backfill when a "
                             "source (e.g. indian_train) is smaller than its target slice.")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", action="append", default=[],
                        choices=["glaive", "xlam", "apigen", "bfcl", "indian"],
                        help="Skip a source if its file is broken.")
    args = parser.parse_args()

    sources = {
        "glaive": (DATA_DIR / "glaive.jsonl", parse_glaive),
        "xlam": (DATA_DIR / "xlam.jsonl", parse_xlam),
        "apigen": (DATA_DIR / "apigen_mt.jsonl", parse_apigen_mt),
        "bfcl": (DATA_DIR / "seeds.json", parse_bfcl_seeds),
        "indian": (DATA_DIR / "train_indian.jsonl", parse_indian_train),
    }

    rng = random.Random(args.seed)
    raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, (path, parser_fn) in sources.items():
        if name in args.skip:
            print(f"[skip] {name}")
            continue
        if not path.exists():
            print(f"[miss] {path.name} not found — skipping {name}")
            continue
        before = len(raw[name])
        for rec in parser_fn(path):
            if looks_valid(rec):
                raw[name].append(rec)
        print(f"[load] {name}: {len(raw[name]) - before} valid records")

    # Dedup by user-query hash. Within a duplicate set, prefer the
    # higher-quality/more-relevant source. Indian training data is the
    # whole point — it wins all ties.
    priority = {"indian_train": 0, "bfcl_seeds": 1, "apigen_mt": 2, "xlam": 3, "glaive": 4}
    seen: dict[str, dict[str, Any]] = {}
    for source, recs in raw.items():
        for r in recs:
            h = query_hash(r)
            if not h:
                continue
            if h not in seen or priority.get(r["source"], 99) < priority.get(seen[h]["source"], 99):
                seen[h] = r
    deduped = list(seen.values())
    print(f"[dedup] {sum(len(v) for v in raw.values())} -> {len(deduped)} after dedup")

    # Balance: target proportions across sources. The BFCL-India TEST set is
    # ~68% non-English (Hinglish/Hindi/Tamil/Bengali), so the training mix must
    # be Indian-heavy or the model is evaluated on a distribution it never saw.
    # Aggressive Indian weighting is the project's whole differentiation.
    target_mix = {
        "indian_train": 0.40,
        "xlam": 0.40,
        "glaive": 0.13,
        "apigen_mt": 0.04,
        "bfcl_seeds": 0.03,
    }
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in deduped:
        by_source[r["source"]].append(r)
    for s in by_source:
        rng.shuffle(by_source[s])

    # Determine the effective total. In --strict-mix, anchor the total on the
    # INDIAN data so we use as much of it as possible (it's the project's whole
    # point), then fill the other slices to their target proportions. Sources
    # that can't fill their slice contribute whatever they have — they only
    # lower the realised total, never inflate a non-Indian source past target.
    if args.strict_mix:
        anchor_avail = len(by_source.get("indian_train", []))
        anchor_frac = target_mix["indian_train"]
        cap = min(args.max_train, int(anchor_avail / anchor_frac)) if anchor_frac else args.max_train
        backfill = False
        print(f"[mix] strict-mix on — anchored on {anchor_avail} Indian examples "
              f"({anchor_frac:.0%} target) -> effective total {cap}")
    else:
        cap = args.max_train
        backfill = True

    picked: list[dict[str, Any]] = []
    for src, frac in target_mix.items():
        n = int(cap * frac)
        picked.extend(by_source[src][:n])  # slice auto-truncates if source is short
    # Only backfill when NOT in strict-mix — backfill trades proportion fidelity
    # for hitting the exact cap, which is the opposite of what we want here.
    if backfill and len(picked) < cap:
        leftovers = [r for src in by_source for r in by_source[src][int(cap * target_mix.get(src, 0)):]]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: cap - len(picked)])
    rng.shuffle(picked)
    print(f"[pick] {len(picked)} examples after balance + cap")

    # Convert to chat format. Drop any whose tools_json is too long.
    chat_records: list[dict[str, Any]] = []
    for r in picked:
        c = to_chat_record(r)
        if c is not None:
            chat_records.append(c)
    print(f"[format] {len(chat_records)} examples after chat-format conversion")

    # 90/10 train/val split.
    rng.shuffle(chat_records)
    n_val = max(1, int(len(chat_records) * args.val_frac))
    val = chat_records[:n_val]
    train = chat_records[n_val:]

    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    train_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n",
        encoding="utf-8",
    )

    # Stats.
    src_counts = Counter(r["source"] for r in chat_records)
    stats = {
        "raw_per_source": {k: len(v) for k, v in raw.items()},
        "after_dedup": len(deduped),
        "final_per_source": dict(src_counts),
        "train_count": len(train),
        "val_count": len(val),
        "max_train": args.max_train,
        "val_frac": args.val_frac,
        "seed": args.seed,
    }
    (DATA_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n" + json.dumps(stats, indent=2))
    print(f"\nWrote: {train_path.relative_to(ROOT)}")
    print(f"Wrote: {val_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
