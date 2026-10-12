# Agent from scratch

A minimal LLM agent written without frameworks, running on a local open-weight model
(qwen3:8b through Ollama).

- A tool-calling loop: the model decides, the code runs a tool, the result goes back, repeat
- Three tools: list suppliers, safe arithmetic, place an order
- A human-approval step before any action that changes state
- Before asking for approval, the agent must state its reason and its assumptions

## First finding

Asked for the cheapest supplier, the agent chose the one with the lowest reliability (0.88
against 0.97) and the longest lead time. When required to state assumptions, it said
reliability and lead time were "not considered", but never gave the numbers. Stating an
assumption is not the same as making it useful for the person approving.

## Run

    ollama pull qwen3:8b
    pip install requests
    python3 agent.py
