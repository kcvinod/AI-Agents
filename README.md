# AI Support Agent

This project contains a minimal AI support agent that parses incoming customer emails, classifies intent, urgency and complexity, and chooses an action (auto reple or escalate to human engineer or L2 team ) 

Quick start:

1. Install dependencies :

```bash
install ollama framework from 
https://ollama.com/download
ollama pull qwen3:4  # run it from command line
pip install langchain 
pip install langgraph 
```

2. Put sample emails in local path under folder samples and then run the main program

```bash
python CustSupportAgentUsingLangGrapg.py
```


