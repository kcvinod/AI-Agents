# AI Promot Qaulity Scoring Agent

This project contains a minimal AI support agent that parses incoming customer emails, classifies intent, urgency and complexity, and chooses an action (auto-reply, create ticket, escalate, assign to engineering).

Quick start:

1. Install dependencies :

This script assume the Ollama Python client is available. 

It verifies the Ollama runtime and that the LLM model is available and prints actionable errors if not.

Ensure Ollama is installed and the requested model is available (for example: `ollama pull gemma2:2b`).


2. Usage:
  python3 -W ignore Prompt_QA_Agent.py 
  Provide a prompt when prompted.


