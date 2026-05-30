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

500 evaluation examples split across 5 categories, modelled after BFCL v3:

| Category | Count | What it tests |
|---|---|---|
| **Simple** | 200 | Single user query → single tool call. Tests baseline ability to pick a tool and fill required args correctly. |
| **Multiple** | 100 | Query is provided alongside 2-5 candidate tools (some near-duplicates). Tests disambiguation. |
| **Parallel** | 50 | One query → multiple independent tool calls (e.g., "block my debit card AND request a chequebook"). Tests structured array output. |
| **Multi-turn** | 100 | 2-4 turns where the assistant makes a tool call, receives a synthetic tool result, and decides the next action. Tests trajectory completion. |
| **Irrelevance** | 50 | Query has no matching tool in the registry — model must refuse rather than force a wrong call. Tests calibration / refusal. |

### Coverage guarantees

- Every tool appears in at least 4 examples across the 500-set.
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
python score.py \
  --predictions my_model_predictions.jsonl \
  --gold bfcl-india-test-gold.jsonl \
  --tools tools.json \
  --output report.json
```

The scorer prints per-category accuracy and the weighted overall, and writes a JSON breakdown of per-example failure modes (broken JSON, wrong tool, missing required arg, etc.).

---

## 6. Reproducibility

- All 500 evaluation examples are released under MIT.
- All 50 tool schemas are documented in `tools.json` with strict JSON Schema Draft 2020-12.
- Reference scorer (`score.py`) is part of this repository.
- Synthetic data was generated using Gemini-2.0-Flash (free tier) seeded with hand-written examples; generation prompts are documented in `data/PROMPTS.md` for full reproducibility.

---

## 7. Known Limitations

1. **No live API calls.** All "tool results" in multi-turn examples are synthetic — we do not call IRCTC or Swiggy in real time; we simulate plausible responses.
2. **Coverage skew.** Hindi and Hinglish coverage is strong; Tamil, Bengali, Marathi, Kannada, Telugu, Punjabi, and Gujarati are present but under-represented (planned for v0.2).
3. **No adversarial robustness category.** Future work: jailbreak-style queries that try to trick the model into emitting destructive calls.
4. **Single-region focus.** All tools are India-only.

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
