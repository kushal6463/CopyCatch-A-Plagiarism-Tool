import os
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

def load_llm():
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def load_and_chunk_documents(pdf_path, chunk_size=2000, chunk_overlap=100):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def get_prompt(template_text):
    return PromptTemplate(input_variables=["context"], template=template_text)

summary_template = """
You're an expert in understanding and analysing research papers.
Given the following paper, provide a detailed and accurate overview in bullet points.
Avoid any hallucination—base all information strictly on the content of the paper.

Input paper: {context}

Output format:
Strict Instructions:
- Only include information found in the text.
- Do not hallucinate or add external knowledge.
- Do not use phrases like "Alternatively" or "Based on the second part".

Provide the overview in bullet points

Use clear, concise language

Cover key areas:
- Objective / Problem addressed
- Methods / Approach used
- Datasets / Experiments (if any)
- Results / Findings
- Conclusions / Implications
- Any noted limitations or future work
"""

novelty_template = """
You're an expert in analyzing research papers.
Given the following paper, identify and clearly explain its novelty — what makes this work new, original, or unique compared to prior work in the field.

Input paper: {context}

Output format:

Clearly state the main novelty of the paper

Briefly mention what has been done in previous work (if discussed)

Highlight how this paper is different or improves upon existing methods

Use concise bullet points or a short paragraph for clarity

Do not hallucinate — rely only on the paper's content.
"""

def run_stuff_chain(llm, docs, prompt):
    chain = create_stuff_documents_chain(
        llm=llm, prompt=prompt, document_variable_name="context"
    )
    result = chain.invoke({"context": docs})
    return result.strip()

def analyze_paper(pdf_path):
    llm = load_llm()
    docs = load_and_chunk_documents(pdf_path)
    summary = run_stuff_chain(llm, docs, get_prompt(summary_template))
    novelty = run_stuff_chain(llm, docs, get_prompt(novelty_template))
    return summary, novelty