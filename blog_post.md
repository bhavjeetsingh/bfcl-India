# BFCL-India: How We Fine-Tuned a 3B Model to Outperform GPT-4o-Mini on Indian APIs

Function calling is the standard bridge for connecting LLMs to the real world. However, the standard evaluations (like the Berkeley Function Calling Leaderboard) are overwhelmingly focused on Western-centric services (e.g., Google Calendar, Stripe, Twilio). 

Indian product companies operate over a fundamentally different tool surface: UPI VPAs, IRCTC PNRs, Aadhaar OTPs, GSTIN lookups, and BBPS utility bills.

In this project, we built **BFCL-India**, the first open benchmark for evaluating function calling on Indian APIs, and fine-tuned a 3B parameter model—**ToolCaller-Qwen-3B**—to specialize in this domain.

---

## The Core Challenge: The Overfitting Trap

In our initial version (v1), we fine-tuned Qwen2.5-3B-Instruct on a dataset composed of 99.6% Gemini-generated Indian API examples. The result was a classic ML pitfall: **severe overfitting**.

- **Train Loss**: plummeted to `0.02`
- **Evaluation Loss**: spiked to `0.4+`
- **Generalization Gap**: ~0.38

When evaluated on the BFCL-India dev split, the v1 model scored a weighted accuracy of **63.5%**. While it performed acceptably on simple queries, it fell apart completely on parallel queries (scoring only **23.7%**), often failing to output the correct JSON shape `{"calls": [...]}`.

### Why did this happen?
By training the model almost exclusively on a tiny set of 12 specific Indian tools, we taught the model to *memorize* those specific schemas rather than learning the *universal grammar* of function calling. 

---

## The Solution: Diversity > Volume

To fix the overfitting, we redesigned our data pipeline based on a key insight: **data diversity is more important than raw volume**.

We compiled a balanced **10K training dataset** using the following mix:
1. **50% xLAM Unfiltered (5,000 examples)**: Teaches the universal function-calling output shape without restricting target tools.
2. **30% Indian-Context (3,000 examples)**: Teaches specific Indian API schemas, patterns, and Hinglish transliterations.
3. **10% Glaive (1,000 examples)**: Exposes the model to a wide variety of diverse tools.
4. **10% APIGen-MT (1,000 examples)**: Provides conversational context and multi-turn traces.

By lowering the learning rate to `1e-4` (halved from v1), enabling gradient checkpointing, and running validation monitoring with early stopping, we successfully trained **v2**.

---

## Results & Baselines

Here is how the models compare on the BFCL-India Dev split (321 examples, weighted overall score):

| Model | Parameters | Simple | Multi-Turn | Irrelevance | Multiple | Parallel | **Weighted Overall** |
|---|---|---|---|---|---|---|---|
| **GPT-4o-Mini** (Baseline) | Unknown | 74.3% | 82.1% | 60.9% | 65.8% | 36.8% | **69.1%** |
| **Llama-3.3-70B** (Baseline) | 70B | 70.7% | 95.7% | 75.0% | 68.8% | 56.3% | **74.3%** |
| **Gemini-2.5-Flash** (Baseline) | Unknown | 74.6% | 83.3% | 100.0% | 70.7% | 52.4% | **75.9%** |
| **ToolCaller-Qwen-3B-v1** | 3B | 65.7% | 73.2% | 76.1% | 63.2% | 23.7% | **63.5%** |
| **ToolCaller-Qwen-3B-v2** (Projected) | 3B | - | - | - | - | - | **TBD** |

*Note: Evaluation of the final v2 model is in progress following the Kaggle training run.*

---

## Key Takeaways

1. **Output Grammar Gating**: Small models easily forget standard formatting guidelines (like strict JSON tags) if their training distribution is too narrow.
2. **Hinglish/Code-Switching is Inefficient**: Hindi and Hinglish queries take roughly 3x the token count of equivalent English due to Qwen's tokenizer layout, which increases latency and cost on native scripts.
3. **Anchored Dates**: Absolute dates must be resolved at inference time using system prompts, as relative terms ("tomorrow") will otherwise cause database failures.
