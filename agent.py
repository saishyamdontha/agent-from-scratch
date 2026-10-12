"""A minimal LLM agent built from scratch: a loop, three tools, and a human check
before any action that changes something."""
import ast
import json
import operator
import re
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
MAX_STEPS = 8

# ---------- Tools: plain Python functions the model may ask us to run ----------

SUPPLIERS = {
    "Nordic Parts": {"price_per_unit": 12.5, "lead_time_days": 6, "reliability": 0.97},
    "Baltic Supply": {"price_per_unit": 10.9, "lead_time_days": 14, "reliability": 0.88},
    "Delta Components": {"price_per_unit": 11.8, "lead_time_days": 9, "reliability": 0.93},
}


def list_suppliers():
    return SUPPLIERS


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression):
    """Safe arithmetic: numbers and + - * / only, never eval()."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("only numbers and + - * / are allowed")
    return walk(ast.parse(expression, mode="eval").body)


def place_order(supplier, quantity, reason="", assumptions=""):
    if supplier not in SUPPLIERS:
        return {"error": f"unknown supplier {supplier}"}
    cost = SUPPLIERS[supplier]["price_per_unit"] * quantity
    return {"status": "order placed", "supplier": supplier,
            "quantity": quantity, "total_cost": round(cost, 2)}


# name -> (function, needs human confirmation?, description, parameters)
TOOLS = {
    "list_suppliers": (list_suppliers, False,
                       "List all suppliers with price per unit, lead time and reliability.",
                       {"type": "object", "properties": {}, "required": []}),
    "calculate": (calculate, False,
                  "Evaluate an arithmetic expression such as '12.5 * 200'.",
                  {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}),
    "place_order": (place_order, True,
                    "Place an order with a supplier. This changes the real world.",
                    {"type": "object",
                     "properties": {"supplier": {"type": "string"},
                                    "quantity": {"type": "integer"},
                                    "reason": {"type": "string",
                                               "description": "Why this choice, in one sentence."},
                                    "assumptions": {"type": "string",
                                                    "description": "Anything assumed but not checked, "
                                                    "including trade-offs the user should know about."}},
                     "required": ["supplier", "quantity", "reason", "assumptions"]}),
}

TOOL_SCHEMAS = [{"type": "function",
                 "function": {"name": name, "description": desc, "parameters": params}}
                for name, (_, _, desc, params) in TOOLS.items()]


# ---------- The human check: the part that matters for human-in-the-loop work ----------

def human_approves(name, args):
    print(f"\n  [CONFIRM] The agent wants to run {name}({args.get('supplier')}, {args.get('quantity')})")
    print(f"  Reason:      {args.get('reason', '(none given)')}")
    print(f"  Assumptions: {args.get('assumptions', '(none given)')}")
    answer = input("  Allow it? [y/n]: ").strip().lower()
    return answer == "y"


# ---------- The agent loop ----------

def call_model(messages):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL, "messages": messages,
        "tools": TOOL_SCHEMAS, "stream": False, "think": False,
    }, timeout=300)
    response.raise_for_status()
    return response.json()["message"]


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def run(task):
    messages = [
        {"role": "system", "content": "You are a purchasing assistant. Use the tools to "
         "get facts and do arithmetic; never guess numbers. Explain your choice briefly."},
        {"role": "user", "content": task},
    ]
    for step in range(1, MAX_STEPS + 1):
        reply = call_model(messages)
        messages.append(reply)
        calls = reply.get("tool_calls") or []

        if not calls:                       # no tool requested: the agent is done
            print(f"\n[step {step}] FINAL ANSWER:\n{strip_thinking(reply.get('content'))}")
            return

        for call in calls:                  # run every tool the model asked for
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            print(f"[step {step}] tool call: {name}({json.dumps(args)})")

            if name not in TOOLS:
                result = {"error": f"no tool called {name}"}
            else:
                fn, needs_confirmation, _, _ = TOOLS[name]
                if needs_confirmation and not human_approves(name, args):
                    result = {"error": "The human declined this action. Do not retry it; "
                              "explain what you would have done instead."}
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:  # report errors to the model, don't crash
                        result = {"error": str(exc)}

            print(f"          result: {json.dumps(result)}")
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})

    print(f"\nStopped after {MAX_STEPS} steps without a final answer.")


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or \
        "Which supplier is cheapest for 200 units? Work out the total cost and order from it."
    run(task)
