# %%
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import Dict
from langgraph.graph import StateGraph, START, END
from pathlib import Path
import json

# %%
# Initialize the Ollama LLM with the desired model
llm = ChatOllama(model="qwen3:4b", temperature=0.5)

# %%
class State(TypedDict):
    intent: str 
    urgency: str 
    complexity: str 
    summary: str
    email : Dict[str,str]
    kb_results: [tuple()]
    


# %%
def readcustemail(state:State):
    """
    reads customer email from the path and parses them by calling parse email function

    """
    # use below when running from python file
    BASE = Path(__file__).parent 
    # use below when running from jupyter notebook
    #BASE = Path().resolve()
    
    file_path = BASE / "samples" / "emails.txt"

    state['email'] = parse_email(file_path.read_text(encoding="utf-8"))
    return state


def parse_email(raw: str) -> Dict[str, str]:
    """
    Parse a minimal raw email text into a dict with headers and body.
    """
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
    prompt = f"""
    You are an AI assistant that helps classify customer support emails.
    Return a JSON object with keys: intent, urgency, complexity, summary.
    Intent must be one of: account, billing, bug, feature_request, technical_issue, general_inquiry.
    Possible value for urgency are Low, Medium and High.
    Possible value for complexity are Low, Medium and High.

    
    Classify the following email based on intent, urgency, and complexity:
    Email Subject: {state['email']['subject']}
    Email Body: {state['email']['body']}
    
    Provide the classification in the following format:
    Intent: <intent>
    Urgency: <urgency>
    Complexity: <complexity>
    Topic: <topic> 
    """

    response = llm.invoke(prompt)
    # Parse response to extract intent, urgency, complexity and summary 
    llm_output = response.content.strip()
    parsed_value = json.loads(llm_output)
    # normalize fields
    state['intent'] = parsed_value.get("intent", "general_inquiry")
    state['urgency'] = parsed_value.get("urgency", "low")
    state['complexity'] = parsed_value.get("complexity", "low")
    state['summary'] = parsed_value.get("summary", state['email']['body'][:100] + "...")   
    return state

def search_kb(state:State):
    """
    Function searches knowledge base to determine response for customer query 
    Requires scikit-learn. If not installed, returns empty list
    Return list of top_n matches as tuples (path, score, excerpt).
    """
    # Placeholder function to simulate knowledge base search
    # In real implementation, this would query a vector database or search index
    kb_results = [
        ("kb/article1.txt", 0.95, "This article explains how to reset your password."),
        ("kb/article2.txt", 0.90, "This article provides troubleshooting steps for login issues."),
    ]
    state['kb_results'] = kb_results
    return state

def should_escalate(state:State) -> bool:
    """
    Determine if the issue should be escalated based on urgency and complexity
    """
    if state.get('urgency') == "high":
        return "escalate_issue"
    elif state.get('complexity') == "high":
        return "escalate_issue"
    else:
        return "draft_response"

def draft_response(state:State):
    """
    Draft response based on email classification and knowledge base search results
    """
        
    prompt = f"""
    You are an AI assistant that helps draft professional customer support email responses.
    Draft a response to the following email based on its classification and knowledge base search results:
   
    Email Subject: {state['email']['subject']}
    Email Body: {state['email']['body']}
    
    Intent: {state['intent']}
    Urgency: {state['urgency']}
    Complexity: {state['complexity']}
    Summary: {state['summary']}
    
    Knowledge Base Search Results: {state.get('kb_results', 'No results found')}
    
    """
    response = llm.invoke(prompt)
    # implement email sending logic here and send email to customer
    #return response.content
    print("Response drafted and sent to customer:")
    print(response.content)
    return state


def escalate_issue(state:State):
    """
    Escalate urgent and complex issue to human support agent
    """
    # urgent messages should be escalated to on-call/support
    if urgency == "high":
        escalate_text = {
            "action": "escalate",
            "reason": "high urgency",
            "payload": {"ticket_summary": state['email']['body'], "assignee": "on-call"},
        }

    # high complexity -> assign engineer / create internal ticket
    if complexity == "high":
        escalate_text = {
            "action": "assign_engineer",
            "reason": "high complexity",
            "payload": {"assignee": "engineering_team", "ticket_summary": state['email']['body']}
        }
        
    print("Issue escalated to human support agent.")
    print(escalate_text)
    return state

# %%
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
custsupport_graph.add_edge('classify_email', 'search_kb')
custsupport_graph.add_conditional_edges('search_kb', should_escalate)
custsupport_graph.add_edge('escalate_issue', END)
custsupport_graph.add_edge('draft_response', END)

customer_support_agent_graph = custsupport_graph.compile()


# %%
state = customer_support_agent_graph.invoke({})



# %%
graph_image = customer_support_agent_graph.get_graph(xray=True).draw_mermaid_png()

with open("customer_support_agent_graph.png", "wb") as f:
    f.write(graph_image)





