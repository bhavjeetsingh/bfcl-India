"""
app.py — Gradio portfolio demo for BFCL-India function calling.
Allows offline mock evaluation of Indian-context APIs as well as live HF model mode.
"""
import os
import json
import gradio as gr
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS_PATH = ROOT / "tools.json"

# Load tools for metadata
try:
    with open(TOOLS_PATH, encoding="utf-8") as f:
        TOOLS = json.load(f)
    TOOLS_IDX = {t["name"]: t for t in TOOLS}
except Exception:
    TOOLS = []
    TOOLS_IDX = {}

# Mock data store for instant portfolio review
MOCK_DATABASE = {
    "paaji ko 500 rupees bhej de upi pe, unka id paaji@okaxis": {
        "calls": [
            {
                "tool": "upi_send",
                "args": {
                    "recipient_vpa": "paaji@okaxis",
                    "amount": 500,
                    "currency": "INR",
                    "note": "sent via assistant"
                }
            }
        ],
        "execution": "UPI Transaction ID: TXN9876543210\nStatus: SUCCESS\nSender: User Account\nRecipient: paaji@okaxis\nAmount: ₹500.00"
    },
    "bhai, mera gas cylinder khatam ho gaya hai. Indane ka refill book kar de, consumer number 10009876543210987 aur mobile 9876543210": {
        "calls": [
            {
                "tool": "gas_book_cylinder",
                "args": {
                    "provider": "INDANE",
                    "consumer_number": "10009876543210987",
                    "registered_mobile": "9876543210"
                }
            }
        ],
        "execution": "LPG Booking ID: LPG44221199\nProvider: INDANE\nConsumer: 10009876543210987\nStatus: BOOKED (Refill requested, Delivery expected in 24-48 hours)"
    },
    "Could you please help me initiate a return for my Ajio order AJ-556677 due to a quality issue, and check the ITR status for PAN KLMPQ1234R for AY 2024-25?": {
        "calls": [
            {
                "tool": "myntra_return",
                "args": {
                    "platform": "AJIO",
                    "order_id": "AJ-556677",
                    "return_type": "EXCHANGE",
                    "reason": "QUALITY_ISSUE",
                    "exchange_size": "L"
                }
            },
            {
                "tool": "efile_itr_status",
                "args": {
                    "pan": "KLMPQ1234R",
                    "assessment_year": "AY 2024-25",
                    "include_intimation": True
                }
            }
        ],
        "execution": "Task 1 (Ajio Return): Return request registered. Pickup scheduled for tomorrow.\nTask 2 (ITR Status): PAN KLMPQ1234R ITR for AY 2024-25 status: PROCESSED. Refund of ₹4,320 issued."
    },
    "Check for buses from Chennai to Coimbatore for June 5th, 2026, keep max price as 1500 INR.": {
        "calls": [
            {
                "tool": "redbus_search",
                "args": {
                    "source_city": "Chennai",
                    "destination_city": "Coimbatore",
                    "travel_date": "2026-06-05",
                    "max_price": 1500
                }
            }
        ],
        "execution": "Found 4 buses matching Chennai -> Coimbatore on 2026-06-05 under ₹1500:\n1. SRS Travels (Sleeper AC) - ₹1200 - 21:00\n2. KPN Travels (Semi-Sleeper AC) - ₹950 - 22:30\n3. Parveen Travels (Sleeper AC) - ₹1400 - 20:00\n4. National Travels (Non-AC Sleeper) - ₹800 - 21:45"
    },
    "Is company ka GSTIN check karke batao ki ye business legally valid hai ya scam hai: 33AAACR2500Q1ZA": {
        "calls": [
            {
                "tool": "gst_search",
                "args": {
                    "gstin": "33AAACR2500Q1ZA",
                    "include_address": True,
                    "include_filing_history": False
                }
            }
        ],
        "execution": "GSTIN Search Result:\nTaxpayer Name: HARSH ELECTRONICS PVT LTD\nGSTIN: 33AAACR2500Q1ZA\nStatus: ACTIVE\nTaxpayer Type: Regular\nRegistration Date: 2018-04-12\nAddress: 42, Mount Road, Guindy, Chennai, Tamil Nadu, 600032\nBusiness Validity: LEGALLY VALID"
    },
    "block my debit card AND request a chequebook": {
        "calls": [
            {
                "tool": "block_card",
                "args": {
                    "card_type": "DEBIT",
                    "block_reason": "LOST",
                    "consent_block": True
                }
            },
            {
                "tool": "request_chequebook",
                "args": {
                    "account_number": "1029384756",
                    "ifsc": "SBIN0001234",
                    "leaves": 25,
                    "delivery_mode": "COURIER"
                }
            }
        ],
        "execution": "Task 1: Debit card block request submitted. Status: BLOCKED immediately.\nTask 2: Chequebook request registered for Account 1029384756. 25 leaves booklet will be dispatched via courier."
    }
}

# Lazy loading of HF model for live mode
model = None
tokenizer = None
device = "cpu"

def load_live_model(model_name_or_path):
    global model, tokenizer, device
    if model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Loading tokenizer for {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model {model_name_or_path} on {device}...")
    
    # Try loading base or adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

def live_predict(query, available_tools, model_path):
    global model, tokenizer, device
    load_live_model(model_path)
    
    # Format system prompt with tools
    system_prompt = """You are a tool-calling assistant. The user's request must be answered ONLY by calling one or more of the tools provided.

Today's date is 2026-05-30. When the user says "tomorrow", "next Friday", "in 3 days", resolve to an absolute YYYY-MM-DD date against this anchor and put the resolved date in the tool args. Never put words like "tomorrow" inside args.

Output a single JSON object: {"calls": [{"tool": "<name>", "args": {...}}, ...]}.

Rules:
- If multiple tools are needed (parallel), include all of them in the "calls" array.
- For multi-turn conversations, output ONLY the next call(s) given the conversation so far.
- If NO available tool can satisfy the request, output {"calls": []}.
- Do not invent tool names. Do not invent argument keys not in the schema.
- Honour all regex patterns, enums, and required fields.
- Output strict JSON. No markdown, no prose, no explanation.
"""
    tools_json = json.dumps(available_tools, indent=1)
    prompt = f"""<|im_start|>system
{system_prompt}

AVAILABLE TOOLS:
{tools_json}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=512, do_sample=False)
    raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    # Parse fences
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()

def process_query(query, mode, model_path, selected_tool_names):
    # Retrieve schemas of selected tools
    available_tools = [TOOLS_IDX[name] for name in selected_tool_names if name in TOOLS_IDX]
    
    # Render prompt previews
    system_prompt_str = f"""You are a tool-calling assistant. Output a single JSON object:
{{"calls": [{{"tool": "<name>", "args": {{...}}}}, ...]}}
Today's date: 2026-05-30

AVAILABLE TOOLS:
{json.dumps([t['name'] for t in available_tools], indent=2)}
"""
    
    if mode == "Mock Mode (Instant / Offline)":
        # Search mock DB
        cleaned = query.strip()
        matched = None
        for k in MOCK_DATABASE:
            if k.lower() in cleaned.lower() or cleaned.lower() in k.lower():
                matched = k
                break
        
        if matched:
            result = MOCK_DATABASE[matched]
            calls_json = json.dumps({"calls": result["calls"]}, indent=2, ensure_ascii=False)
            execution_log = result["execution"]
            status = "SUCCESS (Mock Mode)"
        else:
            calls_json = json.dumps({"calls": []}, indent=2)
            execution_log = "Query not found in Mock database. Try selecting an example query."
            status = "REFUSAL (Mock Mode)"
    else:
        # Live model mode
        try:
            raw_output = live_predict(query, available_tools, model_path)
            # Parse output
            parsed = json.loads(raw_output)
            calls_json = json.dumps(parsed, indent=2, ensure_ascii=False)
            status = "SUCCESS (Model Predicted)"
            # Generate dummy execution response based on tool name
            calls = parsed.get("calls", [])
            execs = []
            for c in calls:
                tool = c.get("tool")
                execs.append(f"Mock execution for tool '{tool}': SUCCESS (args={c.get('args')})")
            execution_log = "\n".join(execs) if execs else "Refusal output returned: no tool calls made."
        except Exception as e:
            calls_json = f"Error during model inference: {str(e)}"
            execution_log = "Model failed to process query."
            status = "ERROR"
            
    return system_prompt_str, calls_json, execution_log, status

def set_example_query(query):
    return query

# Gradio interface creation
with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate")) as demo:
    gr.Markdown("# 🇮🇳 BFCL-India Function-Calling Dashboard")
    gr.Markdown("Evaluate open-source LLMs on Indian-context function calling (UPI, PNR, BBPS, GSTIN, Aadhaar).")
    
    with gr.Row():
        with gr.Column(scale=1):
            all_tool_names = list(TOOLS_IDX.keys()) if TOOLS_IDX else ["upi_send", "irctc_pnr_status", "swiggy_search_restaurant", "gst_search", "aadhaar_verify"]
            with gr.Accordion("🛠️ Tool Registry Selection (50 Tools)", open=False):
                tool_selector = gr.CheckboxGroup(
                    choices=all_tool_names,
                    value=all_tool_names[:8],
                    label="Active Tools in Registry"
                )
            
            gr.Markdown("### ⚙️ Mode Configuration")
            mode_selector = gr.Radio(
                choices=["Mock Mode (Instant / Offline)", "Live Model Mode (Requires HF model/GPU)"],
                value="Mock Mode (Instant / Offline)",
                label="Execution Mode"
            )
            model_path_input = gr.Textbox(
                value="bhavjeetsingh2912/toolcaller-qwen-3b",
                label="HuggingFace Model / Local Path (Only used in Live Mode)"
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Input Query")
            query_input = gr.Textbox(
                placeholder="Enter query here (e.g. 'canteen wale ko 100 rs bhej do upi pe')...",
                label="User Request"
            )
            
            gr.Markdown("💡 **Try these sample queries:**")
            examples = [
                "paaji ko 500 rupees bhej de upi pe, unka id paaji@okaxis",
                "bhai, mera gas cylinder khatam ho gaya hai. Indane ka refill book kar de, consumer number 10009876543210987 aur mobile 9876543210",
                "Could you please help me initiate a return for my Ajio order AJ-556677 due to a quality issue, and check the ITR status for PAN KLMPQ1234R for AY 2024-25?",
                "Check for buses from Chennai to Coimbatore for June 5th, 2026, keep max price as 1500 INR.",
                "Is company ka GSTIN check karke batao ki ye business legally valid hai ya scam hai: 33AAACR2500Q1ZA",
                "block my debit card AND request a chequebook"
            ]
            for ex in examples:
                btn = gr.Button(ex, size="sm")
                btn.click(fn=lambda e=ex: e, outputs=query_input)
                
            submit_btn = gr.Button("Analyze Request", variant="primary")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Formatted Prompt Context")
            prompt_output = gr.Code(language="markdown", label="System Context")
        with gr.Column():
            gr.Markdown("### 🤖 Model Predicted Call(s)")
            json_output = gr.Code(language="json", label="JSON Output")
        with gr.Column():
            gr.Markdown("### 📡 Mock API Execution Log")
            execution_output = gr.Textbox(label="API Status Logs", interactive=False, lines=6)
            status_output = gr.Label(label="Status Flag")

    submit_btn.click(
        fn=process_query,
        inputs=[query_input, mode_selector, model_path_input, tool_selector],
        outputs=[prompt_output, json_output, execution_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
