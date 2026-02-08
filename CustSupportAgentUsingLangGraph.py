from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import Dict
from langgraph.graph import StateGraph, START, END
from pathlib import Path
import json
import re
from pydantic import BaseModel

# Initialize the Ollama LLM with the desired model
llm = ChatOllama(model="gemma2:2b", temperature=0.5)


# Define data models for email and knowledge base results
class Email(BaseModel):
    subject: str
    from_email: str
    to_email: str
    body: str
    raw: str

class KBResult(BaseModel):
    article_path: str
    relevance_score: float
    snippet: str

class State(TypedDict):
    intent: str 
    urgency: str 
    complexity: str 
    summary: str
    email : Email
    kb_results: KBResult


# Define functions for each node in the graph
def readcustemail(state:State):
    """
    Reads customer email from the Sample folder path under project and parses them by calling parse email function
    In real implementation, this would read from an email server or API to fetch new customer emails.
    """
    print("Reading customer email...")
        
    # use below when running from python file
    BASE = Path(__file__).parent  
    # use below when running from jupyter notebook
    #BASE = Path().resolve()
    file_path = BASE / "samples" / "emails5.txt"

    state['email'] = parse_email(file_path.read_text(encoding="utf-8"))
    return state


def parse_email(raw: str) -> Dict[str, str]:
    """
    Parse a minimal raw email text into a dict with headers and body.
    """
    print("Parsing email...")
    lines = raw.strip().splitlines()
    headers = {}
    body_lines = []
    in_body = False
    SEPARATOR_HEADERS = ["Subject:", "From:", "To:"]

    for line in lines:
        if not in_body and any(line.startswith(h) for h in SEPARATOR_HEADERS):
            k, v = line.split(":", 1)
            headers[k.lower()] = v.strip()
            continue
        # blank line separates headers and body
        if line.strip() == "":
            in_body = True
            continue
        if in_body:
            body_lines.append(line)

    return {
        "subject": headers.get("subject", "(no subject)"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "body": "\n".join(body_lines).strip(),
        "raw": raw,
    }

    
def classify_email(state:State): 
    """
    Classify email based on topic,urgency, complexity, intent 
    """
    print("Classifying email based on intent, urgency, complexity and topic...")

    prompt = f"""
    You are an AI assistant that helps classify customer support emails.
    Return a JSON object with keys: intent, urgency, complexity, summary.
    Intent must be one of: account, billing, bug, feature_request, technical_issue, general_inquiry.
    Possible value for urgency are Low, Medium and High.
    Possible value for complexity are Low, Medium and High.
    summary should be a concise summary of the email body in 1-2 sentences.

    
    Classify the following email based on intent, urgency, and complexity:
    Email Subject: {state['email']['subject']}
    Email Body: {state['email']['body']}
    
    Provide the classification in the following format:
    intent: <intent>
    urgency: <urgency>
    complexity: <complexity>
    summary: <summary>  

    Do not include any explanation, only return the JSON object with the classification results.
    """
    response = llm.invoke(prompt)

    # Parse LLM response to extract intent, urgency, complexity and summary 
    llm_output = str(response.content)
    print("llm_output: ", llm_output)
    parsed_value = extract_json(llm_output)
    print("parsed_value: ", parsed_value)
    
    # normalize fields
    state['intent'] = parsed_value.get("intent", "general_inquiry")
    state['urgency'] = parsed_value.get("urgency", "low")
    state['complexity'] = parsed_value.get("complexity", "low")
    state['summary'] = parsed_value.get("summary", state['email']['body'][:100] + "...")   

    return state

def extract_json(text: str):
    print("Extracting JSON from text...")
    # Remove markdown fences like ```json ... ``` or ~~~json ... ~~~
    cleaned = re.sub(r"```json|```|~~~json|~~~", "", text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("JSON parsing failed:", e)
        print("Cleaned text was:", cleaned)
        return None


def search_kb(state:State):
    """
    Function searches knowledge base to determine response for customer query 
    On real implementation, this would query a vector database or search index to retrieve relevant articles based on email classification and summary.
     For simplicity, we return hardcoded results here.
    """
    print("Searching knowledge base for relevant articles...")

    # Placeholder function to simulate knowledge base search
    # In real implementation, this would query a vector database or search index
    kb_results = [
        ("kb/article1.txt", 0.95, "This article explains how to reset your password."),
        ("kb/article2.txt", 0.90, "This article provides troubleshooting steps for login issues."),
    ]
    state['kb_results'] = kb_results
    return state

def should_escalate(state:State):
    """
    Determine if the issue should be escalated based on urgency and complexity
    """
    print("Determining if issue should be escalated based on urgency and complexity...")

    if state['urgency'].lower() == "high":
        print("Issue classified as high urgency. Should escalate to human support agent.")
        return "escalate_issue"
    elif state['complexity'].lower() == "high":
        print("Issue classified as high complexity. Should escalate to human support agent.")
        return "escalate_issue"
    else:
        print("Issue classified as low/medium urgency and complexity. Can draft response without escalation.")
        return "draft_response"


def draft_response(state:State):
    """
    Draft response based on email classification and knowledge base search results
    """
    print("Drafting response based on email classification and knowledge base search results...")

    prompt = f"""
    You are an Customer Support Agent that helps draft professional customer support email responses on behalf of Helpdesk team. 
    Draft a response to the following email based on its classification and knowledge base search results:
   
    Email Subject: {state['email']['subject']}
    Email Body: {state['email']['body']}
    
    Intent: {state['intent']}
    Urgency: {state['urgency']}
    Complexity: {state['complexity']}
    Summary: {state['summary']}
    
    Knowledge Base Search Results: {state.get('kb_results', 'No results found')}
    
    Do not include any explanation, only return the drafted email response that can be sent to customer. 
    """

    response = llm.invoke(prompt)
    
    # implement email sending logic here and send email to customer
    print("Response drafted and sent to customer:")
    print(response.content)
    
    return state


def escalate_issue(state:State):
    """
    Escalate urgent and complex issue to human support agent
    """
    escalate_text = {}
    print("Escalating issue to human support agent...")
    # urgent messages should be escalated to on-call/support
    if state['urgency'].lower() == "high":
        escalate_text = {
            "action": "escalate",
            "reason": "high urgency",
            "payload": {"ticket_summary": state['email']['body'], "assignee": "on-call"},
        }

    # high complexity -> assign engineer / create internal ticket
    if state['complexity'].lower() == "high":
        escalate_text = {
            "action": "assign_engineer",
            "reason": "high complexity",
            "payload": {"assignee": "engineering_team", "ticket_summary": state['email']['body']}
        }
        
    print("Issue escalated to human support agent.")
    print(escalate_text)
    return state

def main():

    # Create state graph
    custsupport_graph = StateGraph(State)

    # Define nodes
    custsupport_graph.add_node('readcustemail', readcustemail)
    custsupport_graph.add_node('classify_email', classify_email)
    custsupport_graph.add_node('search_kb', search_kb)
    custsupport_graph.add_node('draft_response', draft_response)
    custsupport_graph.add_node('escalate_issue', escalate_issue)


    # Define edges
    custsupport_graph.add_edge(START, 'readcustemail')
    custsupport_graph.add_edge('readcustemail', 'classify_email')
    #custsupport_graph.add_edge('classify_email', 'search_kb')
    #custsupport_graph.add_conditional_edges('search_kb', should_escalate)
    custsupport_graph.add_conditional_edges('classify_email', should_escalate)
    custsupport_graph.add_edge('search_kb', 'draft_response')   
    custsupport_graph.add_edge('escalate_issue', END)
    custsupport_graph.add_edge('draft_response', END)
    # Compile and run the graph
    customer_support_agent_graph = custsupport_graph.compile()
    state = customer_support_agent_graph.invoke({})

    # Generate and save the graph visualization
    graph_image = customer_support_agent_graph.get_graph(xray=True).draw_mermaid_png()
    with open("customer_support_agent_graph.png", "wb") as f:
        f.write(graph_image)


# Main entry point of the program
if __name__ == "__main__":
    main()