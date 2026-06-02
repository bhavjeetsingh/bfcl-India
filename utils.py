"""
utils.py — Shared utilities across BFCL-India generation and eval scripts.

Consolidates duplicated functions from generate_examples.py,
generate_indian_training.py, generate_topup.py, and eval.py.
"""

from __future__ import annotations

import hashlib
import json
import re
import random
from pathlib import Path
from typing import Any

import jsonschema


def strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences from model output."""
    text = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*)\s*```$", text, flags=re.DOTALL)
    return fence.group(1) if fence else text


def compact_field(v: dict[str, Any]) -> dict[str, Any]:
    """Compact a single property, recursing one level into nested object schemas."""
    out: dict[str, Any] = {
        "type": v.get("type"),
        "enum": v.get("enum"),
        "desc": (v.get("description") or "")[:80],
    }
    if v.get("type") == "object" and isinstance(v.get("properties"), dict):
        out["object_fields"] = {
            ik: {"type": iv.get("type"), "enum": iv.get("enum")}
            for ik, iv in v["properties"].items()
        }
        if v.get("required"):
            out["object_required"] = v["required"]
    if v.get("type") == "array" and isinstance(v.get("items"), dict):
        items = v["items"]
        if items.get("type") == "object" and isinstance(items.get("properties"), dict):
            out["array_item_fields"] = {
                ik: {"type": iv.get("type"), "enum": iv.get("enum")}
                for ik, iv in items["properties"].items()
            }
            if items.get("required"):
                out["array_item_required"] = items["required"]
        else:
            out["array_item_type"] = items.get("type")
    return out


def compact_tool(tool: dict[str, Any], desc_limit: int = 200) -> dict[str, Any]:
    """Strip schemas to name + brief description + required keys for prompt economy."""
    params = tool.get("parameters", {})
    props = params.get("properties", {})
    return {
        "name": tool["name"],
        "description": tool["description"][:desc_limit],
        "required": params.get("required", []),
        "fields": {k: compact_field(v) for k, v in props.items()},
    }


def pick_tool_subset(tools: list[dict[str, Any]], k: int, rng: random.Random | None = None) -> list[str]:
    """Pick k tool names, mixing categories so output isn't dominated by one prefix."""
    by_prefix: dict[str, list[str]] = {}
    for t in tools:
        prefix = t["name"].split("_", 1)[0]
        by_prefix.setdefault(prefix, []).append(t["name"])
    picked: list[str] = []
    prefixes = list(by_prefix.keys())
    if rng:
        rng.shuffle(prefixes)
    else:
        random.shuffle(prefixes)
    while len(picked) < k and prefixes:
        for p in prefixes:
            if by_prefix[p]:
                picked.append(by_prefix[p].pop())
                if len(picked) >= k:
                    break
    return picked


def existing_ids(out_path: Path) -> set[str]:
    """Read existing example IDs from a JSONL file for dedup."""
    if not out_path.exists():
        return set()
    ids: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["id"])
        except Exception:
            continue
    return ids


def example_id(query: str, calls: list[dict[str, Any]]) -> str:
    """Deterministic ID from query + calls for dedup."""
    h = hashlib.sha1((query + json.dumps(calls, sort_keys=True)).encode("utf-8")).hexdigest()[:12]
    return f"bfcl_train_{h}"


def validate_call(call: dict[str, Any], tools_idx: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    """Validate a single tool call against its JSON Schema.

    Returns (ok, reason) tuple for detailed error reporting.
    """
    if not isinstance(call, dict):
        return False, "call not a dict"
    tool = call.get("tool")
    args = call.get("args", {})
    if tool not in tools_idx:
        return False, f"unknown tool {tool}"
    if not isinstance(args, dict):
        return False, "args not a dict"
    schema = tools_idx[tool]["parameters"]
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as e:
        return False, f"schema invalid: {e.message}"
    return True, "ok"


def load_tools_idx(tools_path: Path) -> dict[str, dict[str, Any]]:
    """Load tools.json and index by name."""
    return {t["name"]: t for t in json.loads(tools_path.read_text(encoding="utf-8"))}
