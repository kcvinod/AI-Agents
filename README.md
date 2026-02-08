# AI Support Agent

This project contains a minimal AI support agent that parses incoming customer emails, classifies intent, urgency and complexity, and chooses an action either auto responds to cusotmer or escalates to  engineering or on call team based on email classification.

Quick start:

1. Install dependencies :

```bash
pip install ollama 
ollama pull gemma2:2b # run it from command line 
pip install langgraph 
pip install langchain-ollama

```

2. Place this file  CustSupportAgentUsingLangGrapg.py into your project folder 

3. Create a folder "samples" inside folder folder and put all testing emails in the folder 

4. Update the email file name in the function "def readcustemail(state:State):" for testing specific email file 

def readcustemail(state:State):
...
...
file_path = BASE / "samples" / "emails5.txt" 


5. Run the python program 

```bash 
python3 CustSupportAgentUsingLangGrapg.py
```

