"""
Prompt Quality Scoring Agent using Ollama (on-prem) with model `gemma2:2b`.
"""
# pip install langchain-core langchain-ollama
# IMport the necessary modules
from xml.parsers.expat import model
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
import subprocess
import sys

# Deifne the prompt template for evaluating prompt quality
TEMPLATE = '''
You are a prompt-quality evaluator. Given the following user prompt, evaluate it using the four criteria below.

Criteria (each 0-10):
1. Clarity — Is the prompt easy to understand with a clear goal?
2. Specificity — Are specific details and requirements provided?
3. Output Format and Constraints — Does the prompt specify output format, tone, or length constraints?
4. Persona defined — Does the prompt assign a specific role/persona to the assistant?

For each criterion return an integer score 0-10 and a one-sentence explanation. Then compute an overall score (Final scope) 0-10 (round the average of the four scores to nearest integer) and give a short one-sentence explanation.

Also provide 2-3 concrete suggestions (short) to improve the prompt.

Input variable: {user_prompt}
'''

def _run_cmd(cmd: list, capture_output: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture_output, text=True)

def _ensure_model_available_cli(model: str) -> bool:
    # Check `ollama list` for model name
    try:
        proc = _run_cmd(["ollama", "list"])  # lists installed models
        if proc.returncode != 0:
            return False
        out = proc.stdout or proc.stderr or ""
        return model in out
    except Exception:
        return False

def evaluate_prompt(user_prompt: str) -> str:
    """
    Docstring for evaluate_prompt
    Evaluate the quality of a user prompt using specified criteria.
    """
        # check if model is available
    model = "gemma2:2b"
    if not _ensure_model_available_cli(model):
        print(f"Error: Model '{model}' is not available.")
        sys.exit(1)
    

    # Format the full prompt
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    # Initialize the Ollama LLM with the desired model
    model = ChatOllama(model=model, temperature=0.5)
    parser = StrOutputParser()

    # Create the processing chain
    chain = prompt | model | parser

    # Invoke the LLM with the prompt
    
    response = chain.invoke({"user_prompt": user_prompt})
    return response
    
def main():
    user_prompt = input("Enter Prompt: ")
 
    try:
        result = evaluate_prompt(user_prompt=user_prompt )
        print("\nPrompt Quality Evaluation:\n")
        print(result) 
        with open("Prompt_Evaluation.txt", "a", encoding="utf-8") as f:
            f.write(f"Prompt:\n {user_prompt}\n")
            f.write(f"Evaluation:\n {result}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
