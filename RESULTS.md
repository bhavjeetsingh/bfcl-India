# BFCL-India Evaluation Results — ToolCaller-Qwen-3B-v3

This document summarizes the final evaluation results of **ToolCaller-Qwen-3B-v3** on the **BFCL-India** dev split (321 examples), comparing it against standard baselines (GPT-4o-mini, Llama-3.3-70B-Versatile, and Gemini-2.5-Flash).

---

## 1. Executive Summary

By fine-tuning **Qwen-2.5-3B-Instruct** on an undiluted, Indian-context-heavy SFT mix (65.62% `train_indian.jsonl`), **ToolCaller-Qwen-3B-v3** achieves an overall weighted accuracy of **68.81%**. This is a **+5.3% absolute gain** over the v1 baseline (63.50%) and brings the local 3B model within **0.26%** of **GPT-4o-mini** (69.07%), while outperforming it significantly on tool choice disambiguation and refusal categories.

### Key Highlights
* **100% JSON Validity:** Emits strict JSON schemas with zero syntax or format errors.
* **Refusal Mastery (+13% over GPT-4o-mini):** Calibrated to identify irrelevant queries and refuse to call a tool, scoring **73.91%** on irrelevance.
* **Specialized Disambiguation (+9.2% over GPT-4o-mini):** Correctly identifies and selects the correct tool from highly similar candidate sets, scoring **75.00%** on the Multiple category.
* **Edge-Device Efficiency:** Executes fully locally at 3B parameter scale on commodity hardware (Kaggle T4 GPUs) at zero API costs.

---

## 2. Benchmark Leaderboard (Dev Split — n=321)

| Model | Size | Overall Weighted | Simple | Multiple | Parallel | Multi-turn | Irrelevance | JSON Validity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Llama-3.3-70B-Versatile** | 70B | **74.30%** | 70.73% | 68.88% | 56.25% | 95.65% | 75.00% | 100% |
| **GPT-4o-Mini** (Cloud) | — | **69.07%** | 74.29% | 65.79% | 36.84% | 82.14% | 60.87% | 92.83% |
| **ToolCaller-Qwen-3B-v3 (Ours)**| **3B** | **68.81%** | 69.52% | **75.00%** | 28.95% | 78.57% | **73.91%** | **100%** |
| *ToolCaller-Qwen-3B-v1* | 3B | *63.50%* | — | — | — | — | — | — |

*Note: Llama-3.3-70B and Gemini-2.5-Flash scores are based on partial dev split runs (n=128 and n=175 respectively) due to API rate constraints. GPT-4o-mini and ToolCaller-Qwen-3B-v3 are scored on the full 321-example dev split.*

---

## 3. Category Breakdown and Analysis

### 3.1 Simple Category (Acc: 69.52% | n=105)
Matches user queries to single-action tool calls. 
* **Strengths:** Strong schema adherence (100% pass rate) on India-specific identifiers like GSTINs, PAN numbers, and mobile validations.
* **Failures:** `arg_values_off` (specifically timezone calculations and date offsets).

### 3.2 Multiple Category (Acc: 75.00% | n=76)
Forces selection between similar tool names (e.g. `upi_send` vs. `upi_collect` vs. `upi_mandate_create`).
* **Strengths:** Outperforms GPT-4o-mini by **+9.2%**. The model successfully parses contextual triggers (e.g., distinguishing "send money" from "setup auto-pay" or "request a payment").

### 3.3 Irrelevance Category (Acc: 73.91% | n=46)
Calibrates model's ability to refuse execution when no tool satisfies the user query.
* **Strengths:** Outperforms GPT-4o-mini by **+13.0%**. The model is highly calibrated and outputs `{"calls": []}` when appropriate, instead of hallucinating tool calls.

### 3.4 Parallel Category (Acc: 28.95% | n=38)
Executes multiple independent calls in one user turn.
* **Analysis:** This remains the lowest scoring category due to the strict order-insensitive exact set matching. If the model misses one parameter or call, the entire example is scored as `0.0`.

---

## 4. Path to v4 (+70% Accuracy)

To close the remaining gap to Llama-3.3-70B and achieve 70%+ overall weighted score:
1. **Augment Parallel Examples:** Expand the parallel training portion in the dataset generation step to teach multi-call structures.
2. **Date Math Generalization:** Introduce synthetic relative date prompts to fine-tune calendar parsing (resolving timezone offsets and day arithmetic).
3. **Optional Default Matching:** Refine prompts to enforce default argument fills where expected by the benchmark's gold standard.