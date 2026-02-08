
# pip install langchain-core langchain-ollama
# IMport the necessary modules
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import AIMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the Ollama LLM with the desired model
llm = ChatOllama(model="gemma2:2b")

template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("{system_message}"),
    HumanMessagePromptTemplate.from_template("{prompt}"),
])

# Lets define different prompts and invoke them as required using CO-STAR model 
# CO-STAR ( Context, Objective, Style, Tone, Audience, and Response) model for prompt engineering

# Prompt 1: Executive Summary of Quarterly Performance Report
system_message = """
Role : 
- You are an executive assistant .

Rules : 
- You respond based on contextual information provided in the prompt and provide references where applicable.

Response Format: 
- Concise and formal language.
- Provide references or sources when applicable.

"""

prompt1 = """
Context:
Please refer to the following quarterly performance report 

Question: 
Please provide 150 word executive summary ofthe quarterly performance report. 
Pls include tables of key metrics, risks and opportunities.
"""

# Prompt 2: Project Update Email to Client
system_message = """
Role : 
- You are an Project Email Assistant .

Rules : 
- You avoid referening sensitive information in the email. 

Response Format: 
- Professional and courteous tone.

"""

prompt2 = """
Context:
client = {client}
project = {project}
deadlines = {deadlines}


Question: 
Draft a professional email to a {client} summarizing project {project} progress, milestones achieved, requesting feedback, and next steps.
Include bullet list of action items and project {deadlines} in the email. 

"""

# Prompt 3: HR Policy Compliance Audit
system_message = """
Role : 
- You are a HR Policy Compliance Auditor.

Rules : 
- You refer to company policies and regulations in your response. 

Response Format: 
- Professional and concise language, with clear references to relevant policies and regulations.

"""

prompt3= """
Review the following HR policy document and identify any missing compliance clauses or ambiguous language.
Provide a summary of your findings and recommendations for addressing any identified issues.

Provide your response in a json format with the following structure:
{
"Issues": [list of identified issues],
"Severity": [severity level of each issue],
"Recommendations": [recommendations for addressing each issue]
}

Context:
Document = {HR_DOcument}
POlicies = {HR_Policies}

"""

# Prompt 4: Meeting Minutes Summarization
system_message = """
Role : 
- You are a meeting minutes assistant.
Rules :
- Avoid adding information that is not explicitly mentioned in the transcript. Ensure clarity and accuracy in summarizing the meeting content

"""

prompt4 = """
SUmmarize the following meeting transcript in to structured Markdown with the following sections:
## Decisions Made
- List of decisions made during the meeting
## Action Items with Assigned Owners and confidence scores 
- List of action items with assigned owners and deadlines and associated confidence scores

Context:
Meeting Transcript = {Meeting_Transcript_text}

"""

# Prompt 5: Market Research Report Generation
system_message = """
Role : 
- You are a Market Research Analyst.

Rules : 
- You don't add information that is not supported by the data. 
- You provide clear references to the data sources and methodologies used in your analysis

Response Format: 
- You provide a your report in json format. You also provide a concise narrative summary of your findings and insights in less than 200 words.

"""

prompt5 = """
Generate a market research report based on the following articles present in dataset and research question. 
Include SWOT analysis, top 3 trends with citations and references to the data sources used in your analysis.

Provide your response in json format with the following structure:
{"SWOT Analysis": {SWOT Analysis},
"Top Trends": [list of top 3 trends with citations and references],
"citiations": [list of citations and references to data sources used in the analysis]}

Narrative Summary:
Provide a concise narrative summary of your findings and insights in less than 200 words.

Context:
Dataset = {Dataset}
Research Question = {Research_Question}
"""

# Function to invoke the LLM with the formatted prompt
def ask_bot(prompt: str) -> str:
    # Format the prompt with the user input and then get the response from the LLM
    response = llm.invoke(template.format(
        system_message=system_message,
        prompt=prompt
    ))
    return response.content

# Example usage
#prompt = prompt1
#prompt = prompt2
#prompt = prompt3
#prompt = prompt4
prompt = prompt5

print(ask_bot(prompt))



