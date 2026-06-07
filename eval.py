"""
eval.py â€” Score a model on BFCL-India.

Loads:    data/generated/*.jsonl (the 421-example test set)
          tools.json (50-tool registry)
Calls:    a model via Gemini / Groq / OpenRouter / local HuggingFace transformers
Computes: per-category accuracy + weighted overall (matching SPEC.md Â§4)
Writes:   reports/{model_name}_predictions.jsonl
          reports/{model_name}_report.json (numbers + failure modes)

Usage:
    uv run python eval.py --model gemini-3.1-flash-lite --provider gemini
    uv run python eval.py --model llama-3.3-70b-versatile --provider groq
    uv run python eval.py --model Qwen/Qwen2.5-3B-Instruct --provider hf --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonschema
from dotenv import load_dotenv
from tqdm import tqdm

from utils import strip_fences

ROOT = Path(__file__).resolve().parent
TOOLS_PATH = ROOT / "tools.json"
DATA_DIR = ROOT / "data" / "generated"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CATEGORY_WEIGHTS = {
    "simple": 0.40,
    "multiple": 0.20,
    "parallel": 0.10,
    "multi_turn": 0.20,
    "irrelevance": 0.10,
}

# Same anchor as generate_examples.py â€” gold args use absolute dates resolved
# against this. The model must use the same anchor or all date-relative
# queries fail value-match.
AS_OF_DATE = "2026-05-30"

SYSTEM_PROMPT = f"""You are a tool-calling assistant. The user's request must be answered ONLY by calling one or more of the tools provided.

Today's date is {AS_OF_DATE}. When the user says "tomorrow", "next Friday", "in 3 days", resolve to an absolute YYYY-MM-DD date against this anchor and put the resolved date in the tool args. Never put words like "tomorrow" inside args.

Output a single JSON object: {{"calls": [{{"tool": "<name>", "args": {{...}}}}, ...]}}.

Rules:
- If multiple tools are needed (parallel), include all of them in the "calls" array.
- For multi-turn conversations, output ONLY the next call(s) given the conversation so far.
- If NO available tool can satisfy the request, output {{"calls": []}}.
- Do not invent tool names. Do not invent argument keys not in the schema.
- Honour all regex patterns, enums, and required fields.
- Output strict JSON. No markdown, no prose, no explanation.
"""


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def load_tools() -> dict[str, dict[str, Any]]:
    return {t["name"]: t for t in json.loads(TOOLS_PATH.read_text(encoding="utf-8"))}


def load_examples(shuffle_seed: int = 42, split: str = "all") -> list[dict[str, Any]]:
    """Load BFCL-India examples.

    split=
      "all"  -> the original 421-example pool (data/generated/*.jsonl)
      "dev"  -> data/eval/dev.jsonl     (used freely during HP tuning)
      "test" -> data/eval/test.jsonl    (secret split â€” run ONCE after final HP selection)

    Shuffling matters for partial runs: if quota hits mid-eval, the examples
    seen so far are spread across all 5 categories instead of one. Seeded so
    reruns are reproducible.
    """
    examples: list[dict[str, Any]] = []
    if split == "dev":
        path = ROOT / "data" / "eval" / "dev.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing â€” run `python split_test_set.py` first.")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    elif split == "test":
        path = ROOT / "data" / "eval" / "test.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing â€” run `python split_test_set.py` first.")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    else:
        for path in sorted(DATA_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rng = random.Random(shuffle_seed)
    rng.shuffle(examples)
    return examples


def already_predicted(out_path: Path) -> dict[str, dict[str, Any]]:
    """Read existing predictions, deduplicate (newest wins), and rewrite cleanly.

    Append-mode runs can create duplicates; this normalises the file so each
    example appears exactly once with its latest prediction.
    """
    if not out_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            out[obj["id"]] = obj  # later entries overwrite earlier â€” newest wins
        except Exception:
            continue
    # Rewrite the file deduped, in stable id order, so it stops growing forever.
    if out:
        sorted_records = [out[k] for k in sorted(out.keys())]
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in sorted_records) + "\n",
            encoding="utf-8",
        )
    return out


# ----------------------------------------------------------------------------
# Model backends
# ----------------------------------------------------------------------------


class Backend:
    def predict(self, messages: list[dict[str, Any]], available_tools: list[dict[str, Any]]) -> str:
        raise NotImplementedError


class GeminiBackend(Backend):
    def __init__(self, model_name: str):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY missing.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        # Money / KYC / card-block queries trip Gemini's default safety filters
        # and return empty responses, biasing the baseline downward. Disable.
        self.safety = {
            "HARASSMENT": "BLOCK_NONE",
            "HATE_SPEECH": "BLOCK_NONE",
            "SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "DANGEROUS": "BLOCK_NONE",
        }

    def predict(self, messages, available_tools):
        prompt = build_prompt(messages, available_tools)
        resp = self.model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
                "max_output_tokens": 2000,
            },
            safety_settings=self.safety,
        )
        return resp.text or ""


class GroqBackend(Backend):
    def __init__(self, model_name: str):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("GROQ_API_KEY missing.")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def predict(self, messages, available_tools):
        prompt = build_prompt(messages, available_tools)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class OpenRouterBackend(Backend):
    def __init__(self, model_name: str):
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY missing.")
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model_name = model_name

    def predict(self, messages, available_tools):
        prompt = build_prompt(messages, available_tools)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        return resp.choices[0].message.content or ""


class HFBackend(Backend):
    """Local HuggingFace transformers — for evaluating Qwen/Llama on GPU."""

    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.device = device
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def predict(self, messages, available_tools):
        import torch
        chat = build_chat_messages(messages, available_tools)
        prompt = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                ids,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return text


def build_chat_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Render a multi-turn example as proper role-tagged chat turns.

    The system message contains BFCL-India instructions + the available tools.
    Subsequent turns preserve user / assistant / tool roles so chat-template
    models see the conversation structure they were trained on.
    """
    system = SYSTEM_PROMPT + "\n\nAVAILABLE TOOLS:\n" + json.dumps(tools, indent=1)
    chat: list[dict[str, str]] = [{"role": "system", "content": system}]
    for m in messages:
        role = m.get("role", "user")
        if role == "assistant" and m.get("tool_calls"):
            content = json.dumps({"calls": m["tool_calls"]}, ensure_ascii=False)
            chat.append({"role": "assistant", "content": content})
        elif role == "tool":
            chat.append({
                "role": "tool",
                "content": f"[{m.get('tool_name', '?')}] {m.get('content', '')}",
            })
        else:
            chat.append({"role": role, "content": m.get("content", "")})
    return chat


def build_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Render a single-string prompt for API backends without native chat roles."""
    parts = [SYSTEM_PROMPT, "AVAILABLE TOOLS:\n" + json.dumps(tools, indent=1)]
    convo: list[str] = []
    for m in messages:
        role = m.get("role", "user").upper()
        if role == "ASSISTANT" and m.get("tool_calls"):
            convo.append(f"ASSISTANT (tool calls): {json.dumps(m['tool_calls'])}")
        elif role == "TOOL":
            convo.append(f"TOOL ({m.get('tool_name', '?')}): {m.get('content', '')}")
        else:
            convo.append(f"{role}: {m.get('content', '')}")
    parts.append("CONVERSATION:\n" + "\n".join(convo))
    parts.append("Now respond with the JSON object.")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------------


def parse_predicted_calls(raw: str) -> tuple[list[dict[str, Any]] | None, str]:
    """Returns (calls_or_None, error_reason). None means JSON-invalid."""
    raw = strip_fences(raw)
    if not raw:
        return None, "empty output"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"json decode: {e.msg}"
    # Accept both {"calls": [...]} and a bare list.
    if isinstance(obj, list):
        calls = obj
    elif isinstance(obj, dict):
        calls = obj.get("calls")
        if calls is None:
            # Maybe model emitted the call directly: {"tool": "...", "args": {...}}
            if "tool" in obj and "args" in obj:
                calls = [obj]
            else:
                return None, "no 'calls' key in JSON object"
    else:
        return None, f"unexpected JSON top-level: {type(obj).__name__}"
    if not isinstance(calls, list):
        return None, "'calls' is not a list"
    return calls, ""


# ----------------------------------------------------------------------------
# Scoring (matches SPEC.md Â§4)
# ----------------------------------------------------------------------------


def score_call(predicted: dict[str, Any], gold: dict[str, Any], tools_idx: dict[str, dict[str, Any]], lenient: bool = False) -> dict[str, Any]:
    """Score a single (predicted, gold) tool call pair.

    lenient=True drops the schema_compliant gate from `fully_correct`. Use this
    to estimate production-style accuracy: real agents try the call and fail
    at the API layer, they don't pre-validate against JSON Schema. The
    schema_compliant flag is still reported for reference.
    """
    p_tool = predicted.get("tool")
    g_tool = gold.get("tool")
    tool_name_correct = p_tool == g_tool

    p_args = predicted.get("args", {}) or {}
    g_args = gold.get("args", {}) or {}

    # Check required args present (SPEC Â§4.1 metric)
    required_args_present = True
    if p_tool in tools_idx:
        schema = tools_idx[p_tool]["parameters"]
        required_fields = schema.get("required", [])
        if required_fields:
            p_keys = set(p_args.keys()) if isinstance(p_args, dict) else set()
            required_args_present = all(r in p_keys for r in required_fields)

    schema_compliant = False
    if p_tool in tools_idx:
        schema = tools_idx[p_tool]["parameters"]
        try:
            jsonschema.validate(p_args, schema)
            schema_compliant = True
        except jsonschema.ValidationError:
            schema_compliant = False
        except Exception:
            schema_compliant = False

    p_keys = set(p_args.keys()) if isinstance(p_args, dict) else set()
    g_keys = set(g_args.keys()) if isinstance(g_args, dict) else set()
    if p_keys or g_keys:
        tp = len(p_keys & g_keys)
        precision = tp / len(p_keys) if p_keys else 0
        recall = tp / len(g_keys) if g_keys else 0
        arg_keys_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    else:
        arg_keys_f1 = 1.0  # both empty

    # Per-arg value match for keys in gold (gold is ground truth)
    matching = 0
    total = 0
    for k, gv in g_args.items() if isinstance(g_args, dict) else []:
        total += 1
        pv = p_args.get(k) if isinstance(p_args, dict) else None
        if values_match(pv, gv):
            matching += 1
    arg_values_match = (matching / total) if total else 1.0

    fully_correct = tool_name_correct and arg_values_match == 1.0 and (lenient or schema_compliant)

    return {
        "tool_name_correct": tool_name_correct,
        "required_args_present": required_args_present,
        "schema_compliant": schema_compliant,
        "arg_keys_f1": round(arg_keys_f1, 3),
        "arg_values_match": round(arg_values_match, 3),
        "fully_correct": fully_correct,
    }


def _coerce_number(v: Any) -> float | None:
    """Best-effort parse of a value into a number. Many models emit numerics
    as JSON strings ("0.05", "25,000", "â‚¹1500"). Treating those as type
    mismatches would deflate every model's score â€” including the baselines â€”
    so we coerce before comparing."""
    if isinstance(v, bool):
        return None  # don't treat True/False as 1/0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower().replace(",", "")
        s = re.sub(r"^(rs\.?|inr|â‚¹|\$)\s*", "", s)
        try:
            return float(s)
        except ValueError:
            return None
    return None


def values_match(p: Any, g: Any) -> bool:
    """Compare predicted vs gold argument values. Liberal-equality for primitives.

    Booleans are checked first (bool is a subclass of int in Python). Numbers
    are compared after coercion so string-encoded numerics ("0.05", "1,500")
    match their numeric gold value â€” a real-world tolerance that keeps scoring
    honest rather than punishing JSON-string formatting.
    """
    if isinstance(g, bool) or isinstance(p, bool):
        return p == g
    if isinstance(g, list) and isinstance(p, list):
        if len(p) != len(g):
            return False
        return all(values_match(x, y) for x, y in zip(p, g))
    if isinstance(g, dict) and isinstance(p, dict):
        if set(p.keys()) != set(g.keys()):
            return False
        return all(values_match(p[k], g[k]) for k in g)
    # Numeric comparison with string-coercion tolerance.
    gn, pn = _coerce_number(g), _coerce_number(p)
    if gn is not None and pn is not None:
        return abs(pn - gn) < 1e-6
    if isinstance(g, str) and isinstance(p, str):
        return p.strip().lower() == g.strip().lower()
    return p == g


def score_example(ex: dict[str, Any], predicted_calls: list[dict[str, Any]] | None, tools_idx: dict[str, dict[str, Any]], lenient: bool = False) -> dict[str, Any]:
    """Score a single eval example."""
    category = ex["category"]
    gold_calls = ex["ground_truth"]["predicted_calls"]

    if predicted_calls is None:
        return {
            "category": category,
            "json_valid": False,
            "correct": False,
            "failure": "json_invalid",
        }

    json_valid = True

    if category == "irrelevance":
        # Correct iff model emitted no calls.
        correct = len(predicted_calls) == 0
        return {
            "category": category,
            "json_valid": json_valid,
            "correct": correct,
            "failure": None if correct else "should_have_refused",
        }

    if not predicted_calls:
        return {
            "category": category,
            "json_valid": json_valid,
            "correct": False,
            "failure": "empty_calls_when_required",
        }

    if category == "parallel":
        return _score_call_set(predicted_calls, gold_calls, tools_idx, category, lenient)

    # multi_turn: usually 1 gold call, but some turns require several. Score as
    # an order-insensitive set, same as parallel, so multi-call turns aren't
    # silently truncated to the first call.
    if category == "multi_turn" and len(gold_calls) > 1:
        return _score_call_set(predicted_calls, gold_calls, tools_idx, category, lenient)

    # multiple: exactly ONE tool should be selected from several candidates.
    # Emitting extra calls must be penalised, otherwise a model can spray calls
    # and win as long as the first one happens to match.
    if category == "multiple" and len(predicted_calls) != len(gold_calls):
        return {"category": category, "json_valid": True, "correct": False,
                "failure": "wrong_call_count"}

    # simple, multiple, single-call multi_turn â€” score the first (and only) call.
    if not gold_calls:
        return {"category": category, "json_valid": True, "correct": True, "failure": None}

    g = gold_calls[0]
    p = predicted_calls[0]
    s = score_call(p, g, tools_idx, lenient=lenient)
    failure = None
    if not s["fully_correct"]:
        if not s["tool_name_correct"]:
            failure = "wrong_tool"
        elif not (lenient or s["schema_compliant"]):
            failure = "schema_violation"
        elif s["arg_values_match"] < 1.0:
            failure = "arg_values_off"
        else:
            failure = "other"
    return {
        "category": category,
        "json_valid": True,
        "correct": s["fully_correct"],
        "tool_name_correct": s["tool_name_correct"],
        "schema_compliant": s["schema_compliant"],
        "arg_keys_f1": s["arg_keys_f1"],
        "arg_values_match": s["arg_values_match"],
        "failure": failure,
    }


def _score_call_set(predicted_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]],
                    tools_idx: dict[str, dict[str, Any]], category: str, lenient: bool) -> dict[str, Any]:
    """Order-insensitive set match: every gold call must be matched by exactly
    one distinct predicted call, and counts must be equal (no extra calls)."""
    if len(predicted_calls) != len(gold_calls):
        return {"category": category, "json_valid": True, "correct": False, "failure": "wrong_call_count"}
    used = [False] * len(predicted_calls)
    all_matched = True
    for g in gold_calls:
        matched = False
        for i, p in enumerate(predicted_calls):
            if used[i]:
                continue
            s = score_call(p, g, tools_idx, lenient=lenient)
            if s["fully_correct"]:
                used[i] = True
                matched = True
                break
        if not matched:
            all_matched = False
            break
    return {"category": category, "json_valid": True, "correct": all_matched,
            "failure": None if all_matched else f"{category}_mismatch"}


# ----------------------------------------------------------------------------
# Main eval loop
# ----------------------------------------------------------------------------


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    cat_acc: dict[str, float] = {}
    for cat, items in by_cat.items():
        if not items:
            cat_acc[cat] = 0.0
            continue
        cat_acc[cat] = sum(1 for r in items if r["correct"]) / len(items)

    overall_unweighted = sum(1 for r in rows if r["correct"]) / len(rows) if rows else 0.0
    overall_weighted = sum(CATEGORY_WEIGHTS.get(c, 0) * acc for c, acc in cat_acc.items())

    json_validity = sum(1 for r in rows if r.get("json_valid")) / len(rows) if rows else 0.0

    failure_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("failure"):
            failure_counts[r["failure"]] += 1

    return {
        "overall_weighted": round(overall_weighted, 4),
        "overall_unweighted": round(overall_unweighted, 4),
        "json_validity_pct": round(json_validity, 4),
        "per_category_accuracy": {k: round(v, 4) for k, v in cat_acc.items()},
        "per_category_count": {k: len(v) for k, v in by_cat.items()},
        "failure_modes": dict(sorted(failure_counts.items(), key=lambda x: -x[1])),
        "n_examples": len(rows),
    }


def init_backend(provider: str, model_name: str, device: str) -> Backend:
    if provider == "gemini":
        return GeminiBackend(model_name)
    if provider == "groq":
        return GroqBackend(model_name)
    if provider == "openrouter":
        return OpenRouterBackend(model_name)
    if provider == "hf":
        return HFBackend(model_name, device=device)
    raise SystemExit(f"unknown provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "groq", "openrouter", "hf"], required=True)
    parser.add_argument("--model", required=True, help="Model id (provider-specific)")
    parser.add_argument("--device", default="cuda", help="Device for hf provider (cuda/cpu/mps)")
    parser.add_argument("--limit", type=int, default=None, help="Eval only the first N examples (smoke test)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between API calls (seconds)")
    parser.add_argument("--label", default=None, help="Override report file name")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="dev",
                        help="Which split to evaluate. dev = use freely during HP tuning. "
                             "test = the secret 100-example split, run EXACTLY ONCE after final HP "
                             "selection. all = the full 421-example pool (legacy / sanity).")
    parser.add_argument("--lenient", action="store_true",
                        help="Production-style scoring: drop strict JSON Schema gating, only "
                             "require correct tool name + matching arg values. Lenient scores "
                             "are an upper-ish bound for real agent deployment.")
    args = parser.parse_args()

    load_dotenv()

    base_label = args.label or args.model.replace("/", "_").replace(":", "_")
    # Suffix split so dev runs and the secret test run don't share files.
    label = f"{base_label}_{args.split}"
    pred_path = REPORTS_DIR / f"{label}_predictions.jsonl"
    report_path = REPORTS_DIR / f"{label}_report.json"

    if args.split == "test":
        print("=" * 60)
        print("WARNING: Running on the SECRET TEST split.")
        print("This split should be evaluated EXACTLY ONCE per model,")
        print("AFTER final hyperparameter selection on the dev split.")
        print("=" * 60)

    tools_idx = load_tools()
    examples = load_examples(split=args.split)
    if args.limit:
        examples = examples[: args.limit]

    backend = init_backend(args.provider, args.model, args.device)

    seen = already_predicted(pred_path)
    print(f"[eval] model={args.model} provider={args.provider} examples={len(examples)} "
          f"already_done={len(seen)}")

    rows: list[dict[str, Any]] = []
    consecutive_skips = 0
    pbar = tqdm(examples, desc=label)
    with pred_path.open("a", encoding="utf-8") as fout:
        for ex in pbar:
            ex_id = ex["id"]
            if ex_id in seen:
                cached = seen[ex_id]
                calls = cached.get("predicted_calls")
                row = score_example(ex, calls, tools_idx, lenient=args.lenient)
                row["id"] = ex_id
                rows.append(row)
                continue

            available = [tools_idx[n] for n in ex["available_tools"] if n in tools_idx]
            try:
                raw = backend.predict(ex["messages"], available)
            except Exception as e:
                msg = str(e)
                # Daily / per-day caps look different across providers â€” bail cleanly.
                if any(s in msg.lower() for s in ["tokens per day", "requests per day", "tpd", "rpd", "quota_id\":\"generaterequestsperday"]):
                    print(f"\n  [stop] Daily quota exhausted on {args.model}. "
                          f"Predictions saved. Resume tomorrow or switch model/provider.")
                    break
                # Per-minute / per-second hint â€” wait and retry once.
                m = re.search(r"retry in ([\d.]+)\s*s", msg, re.IGNORECASE) \
                    or re.search(r"retry_delay.*?seconds:\s*(\d+)", msg, re.DOTALL) \
                    or re.search(r"try again in ([\d]+)m([\d.]+)s", msg)
                if m:
                    if m.lastindex == 2:  # m+s format
                        wait = int(m.group(1)) * 60 + float(m.group(2))
                    else:
                        wait = float(m.group(1))
                    if wait > 120:
                        print(f"\n  [stop] Long backoff ({int(wait/60)} min) on {args.model}. "
                              f"Likely daily cap. Predictions saved.")
                        break
                    print(f"\n  [wait] {wait:.0f}s for rate limit, then retrying {ex_id}")
                    time.sleep(wait + 1)
                    try:
                        raw = backend.predict(ex["messages"], available)
                    except Exception as e2:
                        print(f"  [skip] {ex_id}: still failing after wait ({type(e2).__name__})")
                        consecutive_skips += 1
                        if consecutive_skips >= 5:
                            print(f"\n  [stop] {consecutive_skips} consecutive failures â€” "
                                  f"likely daily cap on {args.model}. Predictions saved.")
                            break
                        continue
                else:
                    print(f"\n  [error] {ex_id}: {type(e).__name__}: {msg[:200]}")
                    consecutive_skips += 1
                    if consecutive_skips >= 5:
                        print(f"\n  [stop] {consecutive_skips} consecutive errors. Bailing.")
                        break
                    time.sleep(5)
                    continue
            consecutive_skips = 0  # reset on successful predict

            calls, parse_err = parse_predicted_calls(raw)
            record = {"id": ex_id, "predicted_calls": calls, "raw": raw[:1500]}
            if parse_err:
                record["parse_error"] = parse_err
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            seen[ex_id] = record

            row = score_example(ex, calls, tools_idx, lenient=args.lenient)
            row["id"] = ex_id
            rows.append(row)

            # Per-category running accuracy â€” meaningful signal during partial runs.
            cat_seen: dict[str, list[bool]] = defaultdict(list)
            for r in rows:
                cat_seen[r["category"]].append(bool(r["correct"]))
            postfix = {
                cat[:3]: f"{(sum(v) / len(v)):.0%}({len(v)})"
                for cat, v in sorted(cat_seen.items())
            }
            postfix["all"] = f"{(sum(1 for r in rows if r['correct']) / len(rows)):.0%}"
            pbar.set_postfix(postfix)

            if args.sleep > 0:
                time.sleep(args.sleep)
    pbar.close()

    summary = aggregate(rows)
    summary["model"] = args.model
    summary["provider"] = args.provider
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"REPORT â€” {args.model}")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {report_path.relative_to(ROOT)}")
    print(f"Saved: {pred_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
