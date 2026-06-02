# BFCL-India: Indian-Context Function Calling Benchmark

**Version:** 0.1.0
**Status:** Draft (active development)
**License:** MIT
**Companion model:** ToolCaller-Qwen-3B (link added on release)

---

## 1. Motivation

Function calling is the bridge between language models and real-world action. Modern AI agents — from customer-support chatbots to in-app assistants — depend on a model's ability to (a) select the right tool from a registry, (b) emit a syntactically valid JSON tool call, and (c) populate the call with correctly-typed arguments.

The de-facto standard for measuring this capability is the **Berkeley Function Calling Leaderboard (BFCL)**, which evaluates models across simple, multiple, parallel, and multi-turn tool use. BFCL is excellent — but its tool registry is overwhelmingly **Western-context**: Google Calendar, Stripe, Twilio, NYC restaurant search.

Indian product companies — Razorpay, Swiggy, CRED, Postman, Sarvam, Krutrim, Flipkart, Ola, PhonePe — operate over a fundamentally different tool surface: **UPI VPAs, IRCTC PNRs, Aadhaar OTPs, GSTIN lookups, BBPS bills, IMPS/NEFT, DigiLocker, MNP porting, BBMP property tax**. A model that scores 95% on BFCL standard may score 65% on Indian-context tools because it has never seen the schema shapes, the regex constraints, or the disambiguation patterns these APIs use.

**BFCL-India closes this gap.** It is the first open benchmark for function calling over Indian-context tools, mirroring BFCL's eval methodology so results are directly comparable.

### Why this matters now

1. Open small models (3B-8B) are getting good enough to replace closed frontier models for tool calling — but only when fine-tuned on representative data.
2. Privacy-constrained Indian enterprises (banks, hospitals, government) cannot send data to OpenAI or Anthropic. They need local models that work on local APIs.
3. No public benchmark exists for evaluating models on the Indian tool surface, so there is no way to measure progress.

---

## 2. Tool Registry

50 tools across 10 categories — the complete registry is in `tools.json`. Tools mirror real public-facing Indian APIs (IRCTC, BBPS, UIDAI Auth API, BookMyShow, Swiggy/Zomato).

| Category | Tools | Domain |
|---|---|---|
| **UPI** | upi_send, upi_collect, check_upi_balance, upi_mandate_create, upi_transaction_history | Payments via NPCI's UPI |
| **Travel** | irctc_search_trains, irctc_book_ticket, irctc_pnr_status, redbus_search, cab_book | Train, bus, intra-city cab |
| **Food / Delivery** | swiggy_search_restaurant, zomato_place_order, swiggy_track_delivery, swiggy_apply_coupon, cancel_food_order | Hyperlocal food ordering |
| **Government** | aadhaar_verify, pan_lookup, gst_search, digilocker_fetch_doc, efile_itr_status | Identity, tax, document retrieval |
| **Banking** | account_balance, mini_statement, block_card, request_chequebook, fd_create | Retail banking |
| **E-commerce** | flipkart_search, amazon_track_order, meesho_buy, myntra_return, ajio_apply_coupon | Online retail |
| **Telecom** | recharge_mobile, check_data_balance, port_number, pay_postpaid, dnd_toggle | Mobile recharge / MNP / TRAI |
| **Entertainment** | bms_search_movie, bms_book_seats, hotstar_remind, prime_resume, spotify_play | Movies, OTT, music |
| **Local services** | urban_company_book, dunzo_send, dunzo_pickup, pharm_easy_order, medlife_refill | Home services, P2P courier, e-pharmacy |
| **Civic / utility** | bbmp_property_tax, electricity_bill_pay, water_bill_pay, gas_book_cylinder, pollution_check | Bills, municipal, vehicle |

### Schema design principles

Every schema in `tools.json` follows these rules:

1. **Strict regex / enum constraints on India-specific identifiers** — UPI VPA, PAN, GSTIN, IFSC, PIN code, mobile, vehicle registration, IRCTC station codes, ISO 3166-2 state codes.
2. **Disambiguation prose in `description`** — every tool says "Use when…" and "Do NOT use for… (use X instead)". Critical for the *Multiple* eval category.
3. **`additionalProperties: false`** at every object level — the model cannot invent extra args.
4. **`"const": true` on consent fields** — sensitive tools (Aadhaar, DigiLocker, MNP) require explicit consent at the schema level. The model cannot skip it.
5. **Geographic bounds** on lat/lng (`6 ≤ lat ≤ 37`, `68 ≤ lng ≤ 98`) — restricts hallucinated coordinates to the Indian subcontinent.
6. **Tool families share enums** — all IRCTC tools share `travel_class` and `quota`. All food tools share `platform`. Reuse mirrors real APIs.

---

## 3. Evaluation Categories

421 evaluation examples split across 5 categories, modelled after BFCL v3:

| Category | Count | What it tests |
|---|---|---|
| **Simple** | 152 | Single user query → single tool call. Tests baseline ability to pick a tool and fill required args correctly. |
| **Multiple** | 100 | Query is provided alongside 2-5 candidate tools (some near-duplicates). Tests disambiguation. |
| **Parallel** | 50 | One query → multiple independent tool calls (e.g., "block my debit card AND request a chequebook"). Tests structured array output. |
| **Multi-turn** | 69 | 2-4 turns where the assistant makes a tool call, receives a synthetic tool result, and decides the next action. Tests trajectory completion. |
| **Irrelevance** | 50 | Query has no matching tool in the registry — model must refuse rather than force a wrong call. Tests calibration / refusal. |

### Coverage guarantees

- Every tool appears in at least 4 examples across the 421-set.
- Every category has examples in **English, Hindi (Devanagari), Hinglish (Roman)**, and a small share of **Tamil / Bengali** (transliterated where appropriate).
- 30% of Simple examples include "noisy" extra context (small talk, irrelevant detail) before the actionable request.

### Train / test separation

- Phrasings used in training data generation (Phase 5 of the master plan) are **disjoint** from BFCL-India test queries.
- 100 of the 500 examples are held back as a **secret test split**, run only once after final hyperparameter selection. Prevents overfitting to the public 400.

---

## 4. Scoring Methodology

BFCL-India scores each category independently and reports a **weighted overall accuracy**.

### 4.1 Per-call scoring primitives

| Metric | Definition |
|---|---|
| `json_valid` | Boolean. Does the model's output parse as valid JSON conforming to a tool-call schema? |
| `tool_name_correct` | Boolean. Does the chosen `tool` match ground truth? |
| `required_args_present` | Boolean. Are all `required` fields from the tool schema present? |
| `arg_keys_f1` | F1 between predicted and ground-truth argument keys. |
| `arg_values_match` | Per-argument exact match for primitives. For dates/numbers: parsed-equal. For free-text: cosine ≥ 0.85 on `bge-small-en-v1.5` (LLM-as-judge fallback for code-switched fields). |
| `schema_compliant` | Boolean. Does the call validate against the tool's JSON Schema (`additionalProperties: false`, `pattern`, `enum`, `min/max`, `required`)? |

A call is **fully correct** iff `json_valid AND tool_name_correct AND schema_compliant AND arg_values_match (per-key)`.

### 4.2 Per-category accuracy

| Category | Formula |
|---|---|
| Simple | fraction of examples where the call is fully correct |
| Multiple | fully correct AND chosen tool is the ground-truth one among the candidate set |
| Parallel | exact-match on the **set** of (tool, args) calls (order-insensitive) |
| Multi-turn | trajectory completion: final state matches ground-truth state across all turns |
| Irrelevance | fraction where the model **refuses** (no tool call OR designated `__no_tool__` sentinel) |

### 4.3 Overall score

```
overall = 0.40 × simple
        + 0.20 × multiple
        + 0.10 × parallel
        + 0.20 × multi_turn
        + 0.10 × irrelevance
```

Weights mirror BFCL v3 conventions, biased toward Simple to match real agent traffic.

---

## 5. Submission Format

A submission is a single JSONL file with one line per evaluation example.

### 5.1 Input file (provided to submitters)

`bfcl-india-test.jsonl` — each line:

```json
{
  "id": "bfcl_india_simple_001",
  "category": "simple",
  "available_tools": ["upi_send", "check_upi_balance"],
  "language": "hinglish",
  "messages": [
    {"role": "user", "content": "paaji ko 500 rupees bhej de upi pe, unka id paaji@okaxis"}
  ]
}
```

For Multi-turn examples, `messages` may include multiple turns and synthetic `tool` role entries.

### 5.2 Output file (submitter produces)

`{model_name}_predictions.jsonl` — each line:

```json
{
  "id": "bfcl_india_simple_001",
  "predicted_calls": [
    {
      "tool": "upi_send",
      "args": {
        "recipient_vpa": "paaji@okaxis",
        "amount": 500,
        "currency": "INR"
      }
    }
  ]
}
```

For Irrelevance examples, `"predicted_calls": []` is the correct answer.

### 5.3 Running the scorer

```bash
python eval.py \
  --predictions my_model_predictions.jsonl \
  --gold bfcl-india-test-gold.jsonl \
  --tools tools.json \
  --output report.json
```

The scorer prints per-category accuracy and the weighted overall, and writes a JSON breakdown of per-example failure modes (broken JSON, wrong tool, missing required arg, etc.).

---

## 6. Reproducibility

- All evaluation examples are released under MIT.
- All 50 tool schemas are documented in `tools.json` with strict JSON Schema Draft 2020-12.
- Reference scorer (`eval.py`) is part of this repository.
- Synthetic data was generated using Gemini-2.0-Flash / Gemini-2.5-Flash-Lite (free tier) seeded with hand-written examples; generation prompts are documented in `generate_examples.py`.

### 6.1 Dataset Construction (full disclosure)

The companion training set (`bhavjeetsingh2912/toolcaller-train-mix` on HuggingFace, ~9K examples) is built from four sources. Actual contribution after parsing, dedup, and balancing:

| Source | Raw downloaded | Final in mix | % of training |
|---|---|---|---|
| Salesforce xLAM-FC-60K | 60,000 | ~6,865 | ~76% |
| BFCL-India custom Indian-context (this work) | 1,779 | ~1,773 | ~20% |
| Glaive Function Calling v2 | 112,960 | ~348 | ~4% |
| BFCL-India hand-written seeds (this work) | 10 | 10 | <1% |
| Salesforce APIGen-MT-5K | 5,000 | **0** | 0% |

**Why the gaps:**

- **Glaive contributed only 2% of its 113K records (348 / 112,960).** The dataset uses a custom string-encoded chat format (`USER:`/`ASSISTANT:`/`<functioncall>` tags) where many records contain only refusals or prose without tool calls. The parser conservatively skips ambiguous records. A more aggressive parser would marginally increase contribution but at quality cost; xLAM dominates the mix regardless.
- **APIGen-MT contributed 0 records.** Its tool schemas live in a separate metadata file not joined in v0.1, and role labels (`function_call`/`observation`) require non-trivial mapping. xLAM provides comparable multi-turn signal in the meantime.
- **Net effect:** the training mix is effectively **xLAM (76%) + Indian-context (20%) + minor seeds**. Claims about Glaive/APIGen-MT in early commits do not hold for the final shipped dataset.

### 6.2 Performance Characteristics

Documented for honest deployment expectations:

- **Strict-schema evaluation.** `eval.py` rejects calls that fail JSON Schema validation (regex, enum, required fields). A production agent would attempt the call and fail at the API layer instead. Strict-schema accuracy is therefore a **lower bound** on production usefulness — pass `--lenient` to `eval.py` for the production-style score (skips schema-compliance gating, still requires correct tool name and matching arg values).
- **Tokenizer inefficiency on Indic scripts.** Qwen-2.5's tokenizer encodes Devanagari at roughly 3× the token count of equivalent English text. Hindi and Bengali queries are correspondingly more expensive in inference latency and per-call cost. Hinglish (Roman script with code-switching) does NOT incur this cost.
- **Date anchoring at inference time.** The system prompt MUST be regenerated with the actual current date on every request. Stale anchors silently produce wrong absolute dates for relative queries ("tomorrow", "next Friday"). The companion model card includes a deployment-time helper.
- **Baseline confidence intervals.** Where baseline numbers are reported with `n < 200`, treat them as preliminary. Per-category numbers from samples of n=8-15 have ±20% error bars and should be read directionally, not as point estimates. Full 321-example dev-split baselines are the headline numbers; partial runs are documented as such.

---

## 7. Known Limitations

This is a v0.1 release. Each limitation below is documented so reviewers and reproducers can judge results in context. The v0.2 roadmap addresses the items flagged.

1. **Synthetic generation, not human-collected.** All 421 evaluation queries were generated by Gemini-2.5-Flash from 10 hand-written seeds, with schema-validation gating dropping ~20% of outputs. Synthetic data risks embedding model-specific phrasing biases. This is industry standard for function-calling benchmarks (xLAM, ToolACE, APIGen-MT all use synthetic generation), but is acknowledged. v0.2 plan: 200 human-collected queries from real Indian users to complement.

2. **Train and test were generated by the same model family.** Both BFCL-India test (421) and the companion training set (1,779) come from Gemini. Models fine-tuned on the companion set may inherit a distributional advantage on this eval. For honest comparison, baselines (Llama-3.3-70B, Gemini-2.5-Flash) are evaluated by other model families.

3. **Final dataset size is 421, not 500.** Free-tier API quotas exhausted during generation. Categories complete: irrelevance (50), parallel (50), multiple (100). Partial: simple (152/200), multi_turn (69/100). Per-category counts are documented in the report and model card.

4. **Held-out test split is mandatory.** 100 examples are held out as a secret test split (`data/eval/test.jsonl`). All hyperparameter selection happens on the dev split (`data/eval/dev.jsonl`, ~321 examples). The test split is run EXACTLY ONCE after final HP selection. Audit log: `data/eval/SPLIT.md`.

5. **No live API calls.** All "tool results" in multi-turn examples are synthetic — we do not call IRCTC or Swiggy in real time; we simulate plausible responses.

6. **Language coverage is skewed.** Hindi and Hinglish are strong (~55% combined); Tamil, Bengali, Marathi, Kannada, Telugu, Punjabi, Gujarati are <10% combined. v0.2: per-language balance enforced.

7. **Tools are India-only.** Cross-border edge cases (international UPI, foreign passport DigiLocker) are out of scope.

8. **No adversarial robustness category.** Future work: jailbreak-style queries that try to trick the model into emitting destructive calls (transfer all funds, delete records).

9. **Date anchoring.** Examples assume `today = 2026-05-30`. At inference time the system prompt MUST be updated with the actual current date; otherwise relative-date queries silently produce stale absolute dates. Documented in the model card.

10. **Same tool registry for train and test.** v0.1 measures in-distribution accuracy on a fixed 50-tool registry (matches BFCL's own design). v0.2: held-out tools for zero-shot generalization.

---

## 8. Citation

```bibtex
@misc{bfcl-india-2026,
  title  = {BFCL-India: An Indian-Context Function Calling Benchmark},
  author = {Singh, Bhavjeet},
  year   = {2026},
  url    = {https://github.com/bhavjeetsingh/bfcl-India}
}
```

---

## 9. Acknowledgements

- The Berkeley Function Calling Leaderboard team for setting the gold standard in tool-use evaluation.
- The Salesforce / xLAM team for releasing high-quality function-calling training data.
- The AI4Bharat and Sarvam teams for raising the bar on Indic language modelling.
