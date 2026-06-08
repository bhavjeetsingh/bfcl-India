# BFCL-India

**The first function-calling benchmark for Indian APIs.**

BFCL-India evaluates language models on their ability to call Indian-context tools — UPI, IRCTC, Aadhaar, Swiggy, and 46 others. It mirrors the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) methodology so results are directly comparable.

## Why this exists

The de-facto standard for measuring function calling is BFCL, but its tool registry is overwhelmingly Western-context (Google Calendar, Stripe, Twilio). Indian product companies — Razorpay, Swiggy, CRED, PhonePe — operate over a fundamentally different tool surface: UPI VPAs, IRCTC PNRs, Aadhaar OTPs, GSTIN lookups, BBPS bills.

A model that scores 95% on BFCL standard may score 65% on Indian-context tools. **BFCL-India closes this gap.**

## Quick start

```bash
# Clone
git clone https://github.com/bhavjeetsingh/bfcl-India.git
cd bfcl-India

# Install
pip install -r requirements.txt

# Evaluate a model
python eval.py --model gemini-2.5-flash --provider gemini
python eval.py --model llama-3.3-70b-versatile --provider groq
python eval.py --model Qwen/Qwen2.5-3B-Instruct --provider hf --device cuda
```

## Benchmark

421 evaluation examples across 5 categories:

| Category | Count | Weight | What it tests |
|---|---|---|---|
| Simple | 152 | 40% | Single tool call — baseline tool selection + arg filling |
| Multiple | 100 | 20% | Disambiguation among near-duplicate tools |
| Parallel | 50 | 10% | Multiple independent tool calls in one response |
| Multi-turn | 69 | 20% | Trajectory completion across 2-4 conversation turns |
| Irrelevance | 50 | 10% | Refusal when no tool matches (calibration) |

## Baselines

| Model | Params | Weighted accuracy | Notes |
|---|---|---|---|
| GPT-4o-mini | unknown | TBD | Complete dev run |
| Llama-3.3-70B | 70B | 77.1% (n=71) | Partial |
| Gemini-2.5-Flash | unknown | 70.0% (n=22) | Partial |

## Tool registry

50 tools across 10 categories — the complete Indian API surface:

| Category | Example tools |
|---|---|
| UPI | `upi_send`, `upi_collect`, `check_upi_balance` |
| Travel | `irctc_search_trains`, `irctc_book_ticket`, `irctc_pnr_status` |
| Food | `swiggy_search_restaurant`, `zomato_place_order` |
| Government | `aadhaar_verify`, `pan_lookup`, `gst_search` |
| Banking | `account_balance`, `mini_statement`, `block_card` |
| E-commerce | `flipkart_search`, `amazon_track_order` |
| Telecom | `recharge_mobile`, `port_number` |
| Entertainment | `bms_search_movie`, `hotstar_remind` |
| Local services | `urban_company_book`, `dunzo_send` |
| Civic/utility | `bbmp_property_tax`, `electricity_bill_pay` |

Schemas use strict JSON Schema Draft 2020-12 with regex constraints, enum validation, `additionalProperties: false`, and `const: true` on consent fields.

## Companion model

**ToolCaller-Qwen-3B** — Qwen2.5-3B fine-tuned with QLoRA on 29K examples (xLAM + Glaive + APIGen-MT + Indian-context training data). Runs on-prem at $0 inference cost.

- Model: [bhavjeetsingh2912/toolcaller-qwen-3b](https://huggingface.co/bhavjeetsingh2912/toolcaller-qwen-3b)
- Training data: [bhavjeetsingh2912/toolcaller-train-mix](https://huggingface.co/datasets/bhavjeetsingh2912/toolcaller-train-mix)

## Project structure

```
bfcl-india/
├── SPEC.md                    # Full benchmark specification
├── tools.json                 # 50-tool registry (JSON Schema Draft 2020-12)
├── eval.py                    # Scorer — 4 backends (Gemini, Groq, OpenRouter, HF)
├── prepare_training_data.py   # Build training mix from multiple sources
├── generate_examples.py       # Generate eval examples via Gemini
├── generate_indian_training.py# Generate Indian-context training data
├── data/
│   ├── generated/             # 421 eval examples (JSONL)
│   ├── eval/                  # Train/test split (321 dev + 100 test)
│   ├── train.jsonl            # Final training set
│   ├── val.jsonl              # Validation set
│   └── train_indian.jsonl     # Indian-context training examples
└── reports/                   # Evaluation reports per model
```

## Scoring

```bash
python eval.py \
  --predictions my_model_predictions.jsonl \
  --gold data/eval/dev.jsonl \
  --tools tools.json \
  --output reports/my_model_report.json
```

Use `--lenient` for production-style scoring (skips schema-compliance gating).

## Gradio Demo Showcase

The repository includes a Gradio interface to interactively test user queries against the Indian-context tools in mock or live model modes.

```bash
# Run the Gradio dashboard
python app.py
```

Open `http://127.0.0.1:7860` in your browser. It includes preset sample queries (Hinglish UPI payments, IRCTC booking, Swiggy returns, GSTIN lookups) to demonstrate model predictions and simulated API execution logs.

## Known limitations

See [SPEC.md §7](SPEC.md) for a full disclosure including:
- Synthetic generation (not human-collected)
- 421 examples (not planned 500) due to API quota
- Train and test generated by same model family
- Language coverage skewed toward Hindi/Hinglish

## Citation

```bibtex
@misc{bfcl-india-2026,
  title  = {BFCL-India: An Indian-Context Function Calling Benchmark},
  author = {Singh, Bhavjeet},
  year   = {2026},
  url    = {https://github.com/bhavjeetsingh/bfcl-India}
}
```

## License

MIT

## Acknowledgements

- Berkeley Function Calling Leaderboard team for setting the gold standard
- Salesforce / xLAM team for high-quality function-calling training data
- AI4Bharat and Sarvam teams for Indic language modelling
