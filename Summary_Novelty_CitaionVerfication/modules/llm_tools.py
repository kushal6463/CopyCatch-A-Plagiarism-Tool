import os
import json
from langchain_openai import ChatOpenAI
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_tavily import TavilySearch
from langchain.tools import Tool
from modules.utils import (
    clean_arxiv_id,
    process_tavily_result,
    load_pdf_content,
)

def _extract_abstract_from_text(document_text: str, llm):
    if not document_text:
        return "No document content provided for abstract extraction."
    text_to_process = document_text[:6000]
    prompt = f"""
    You are an AI assistant tasked with extracting the abstract from a research paper.
    Carefully read the following text, which is from the beginning of a paper.
    Identify and extract the complete abstract section.
    Provide ONLY the text of the abstract, with no additional commentary or formatting.
    If you cannot find a clear abstract section, state "Abstract not found.".

    Document Text Snippet:
    ---
    {text_to_process}
    ---

    Extracted Abstract:
    """
    try:
        response = llm.invoke(prompt)
        extracted_abstract = response.content.strip()
        if extracted_abstract == "Abstract not found.":
            return "Abstract section not found or could not be extracted by LLM."
        else:
            return extracted_abstract
    except Exception as e:
        return f"ERROR: An error occurred during LLM abstract extraction: {e}"

def _extract_references_section_from_text(document_text: str, llm):
    if not document_text:
        return ""
    last_pages_text = document_text[-8000:]
    prompt = f"""
    You are an AI assistant tasked with extracting the References or Bibliography section from a research paper.
    Carefully read the following text from the end of a paper.
    Identify and extract the complete References or Bibliography section.
    Look for sections that start with words like:
    - References
    - Bibliography
    - Works Cited
    - Citations
    Provide ONLY the text of the references section, with no additional commentary or formatting.
    If you cannot find a clear references section, respond with exactly: "REFERENCES_NOT_FOUND"

    Document Text (from end of paper):
    ---
    {last_pages_text}
    ---
    Extracted References Section:
    """
    try:
        response = llm.invoke(prompt)
        extracted_references = response.content.strip()
        if (
            extracted_references == "REFERENCES_NOT_FOUND"
            or len(extracted_references) < 50
        ):
            full_prompt = f"""
            You are an AI assistant tasked with extracting the References or Bibliography section from a research paper.
            Carefully read the following full document text.
            Identify and extract the complete References or Bibliography section.
            Look for sections that start with words like:
            - References
            - Bibliography
            - Works Cited
            - Citations
            Provide ONLY the text of the references section, with no additional commentary or formatting.
            If you cannot find a clear references section, respond with exactly: "REFERENCES_NOT_FOUND"

            Full Document Text:
            ---
            {document_text}
            ---
            Extracted References Section:
            """
            response = llm.invoke(full_prompt)
            extracted_references = response.content.strip()
        if extracted_references == "REFERENCES_NOT_FOUND":
            return "NO_REFERENCES_SECTION_FOUND"
        return extracted_references
    except Exception as e:
        return f"ERROR: An error occurred during LLM references section extraction: {e}"

def _parse_references_from_text(references_text: str, llm) -> str:
    if not references_text or references_text == "NO_REFERENCES_SECTION_FOUND":
        return json.dumps([])
    prompt = f"""
    You are an AI assistant tasked with parsing a list of academic paper references.
    The following text is the raw content extracted from the References or Bibliography section of a research paper.
    Please carefully read the text and identify each distinct cited paper entry.
    For each entry, extract the following information:
    1. The title of the paper (usually in quotes or italics)
    2. The arXiv ID (if present). Look for patterns like "arXiv:2301.07041", "arxiv.org/abs/2301.07041", or just "2301.07041"
    3. The authors of the paper (usually at the beginning of each reference)
    4. The entire text of the citation entry as it appears

    IMPORTANT: Format your output as a valid JSON array of objects. Each object should have these exact keys:
    - "title": (string, the title of the paper)
    - "arxiv_id": (string or null, the arXiv ID if found, otherwise null)
    - "authors": (string, the authors of the paper)
    - "entire_citation": (string, the full text of the citation entry)

    Your response should contain ONLY the JSON array, with no markdown formatting, no ```json blocks, and no additional text.
    If the formatting is very difficult to parse and you cannot confidently identify individual entries or extract the requested fields,
    return an empty JSON array [].

    Raw References Text:
    ---
    {references_text}
    ---
    """
    try:
        response = llm.invoke(prompt)
        json_string = response.content.strip()
        if json_string.startswith("```json"):
            json_string = json_string[7:]
        if json_string.startswith("```"):
            json_string = json_string[3:]
        if json_string.endswith("```"):
            json_string = json_string[:-3]
        json_string = json_string.strip()
        try:
            parsed_data = json.loads(json_string)
        except json.JSONDecodeError:
            bracket_start = json_string.find("[")
            if bracket_start > 0:
                json_string = json_string[bracket_start:]
            bracket_end = json_string.rfind("]")
            if bracket_end != -1 and bracket_end < len(json_string) - 1:
                json_string = json_string[: bracket_end + 1]
            try:
                parsed_data = json.loads(json_string)
            except json.JSONDecodeError:
                return json.dumps([])
        if isinstance(parsed_data, list):
            valid_entries = []
            for entry in parsed_data:
                if isinstance(entry, dict):
                    cleaned_entry = {
                        "title": str(entry.get("title", "")).strip(),
                        "arxiv_id": entry.get("arxiv_id"),
                        "authors": str(entry.get("authors", "")).strip(),
                        "entire_citation": str(
                            entry.get("entire_citation", "")
                        ).strip(),
                    }
                    if cleaned_entry["arxiv_id"] in ["null", "None", ""]:
                        cleaned_entry["arxiv_id"] = None
                    valid_entries.append(cleaned_entry)
            return json.dumps(valid_entries)
        else:
            return json.dumps([])
    except Exception as e:
        return json.dumps([])

def _get_arxiv_summary_internal(arxiv_id: str, arxiv_tool) -> str:
    cleaned_id = clean_arxiv_id(arxiv_id)
    if not cleaned_id:
        return ""
    try:
        result_docs = arxiv_tool.invoke(cleaned_id)
        result = ""
        if isinstance(result_docs, str):
            result = result_docs
        elif isinstance(result_docs, list) and result_docs:
            result = result_docs[0].page_content
        if not result:
            return ""
        summary_text = ""
        if "Summary:" in result:
            summary_start = result.find("Summary:") + len("Summary:")
            summary_end_candidates = [
                result.find("\n\n", summary_start),
                result.find("\nPublished:", summary_start),
                result.find("\nAuthors:", summary_start),
                result.find("\nTitle:", summary_start),
            ]
            summary_end = len(result)
            for cand_pos in summary_end_candidates:
                if cand_pos != -1:
                    summary_end = min(summary_end, cand_pos)
            summary_text = result[summary_start:summary_end].strip()
        elif "Abstract:" in result:
            summary_start = result.find("Abstract:") + len("Abstract:")
            summary_end_candidates = [
                result.find("\n\n", summary_start),
                result.find("\nPublished:", summary_start),
                result.find("\nAuthors:", summary_start),
                result.find("\nTitle:", summary_start),
            ]
            summary_end = len(result)
            for cand_pos in summary_end_candidates:
                if cand_pos != -1:
                    summary_end = min(summary_end, cand_pos)
            abstract_text = result[summary_start:summary_end].strip()
            summary_text = abstract_text
        else:
            lines = result.split("\n")
            content_lines = [
                line.strip()
                for line in lines
                if line.strip()
                and not any(
                    header in line
                    for header in [
                        "Title:",
                        "Authors:",
                        "Published:",
                        "Entry ID:",
                        "Links:",
                    ]
                )
                and len(line.strip()) > 20
            ]
            summary_text = " ".join(content_lines[:15])
        return " ".join(summary_text.splitlines()).strip()
    except Exception as e:
        return f"ERROR: Could not retrieve arXiv summary for {cleaned_id}: {e}"

def _evaluate_citation_quality_internal(
    main_abstract: str, citation_summary: str, citation_title: str, llm
) -> str:
    default_error_response = {
        "evaluation": "unable_to_evaluate",
        "confidence": 0.0,
        "reasoning": "Error during evaluation",
        "relevance_score": 0.0,
        "relationship_type": "unrelated",
    }
    if not citation_summary or not main_abstract:
        default_error_response["reasoning"] = (
            "Insufficient information: missing citation summary or main abstract for evaluation."
        )
        return json.dumps(default_error_response)
    prompt = f"""
    You are an expert academic reviewer. Evaluate the relevance of a CITED PAPER to a MAIN PAPER based on their abstracts.

    MAIN PAPER ABSTRACT:
    ---
    {main_abstract}
    ---

    CITED PAPER TITLE: {citation_title}
    CITED PAPER SUMMARY/ABSTRACT:
    ---
    {citation_summary}
    ---

    Provide your evaluation in JSON format ONLY. The JSON object must include these keys:
    - "evaluation": (string) "good", "bad", or "marginal".
    - "confidence": (float) Your confidence in this evaluation, from 0.0 to 1.0.
    - "reasoning": (string) A brief explanation for your evaluation (max 2-3 sentences).
    - "relevance_score": (float) A score from 0.0 (irrelevant) to 1.0 (highly relevant).
    - "relationship_type": (string) One of: "foundational", "methodological", "comparative", "supportive", "tangential", "unrelated".

    Criteria:
    - "good": Highly relevant, directly supports or relates to the main paper's core research.
    - "bad": Irrelevant or very loosely related, offers no clear contribution.
    - "marginal": Some relation, perhaps to background or a minor aspect, but not core.

    Consider topical overlap, methodological connections, and if the cited work strengthens the main paper.
    Respond with ONLY the JSON object.
    """
    try:
        response = llm.invoke(prompt)
        evaluation_text = response.content.strip()
        if evaluation_text.startswith("```json"):
            evaluation_text = evaluation_text[7:]
        if evaluation_text.startswith("```"):
            evaluation_text = evaluation_text[3:]
        if evaluation_text.endswith("```"):
            evaluation_text = evaluation_text[:-3]
        evaluation_text = evaluation_text.strip()
        try:
            evaluation_result = json.loads(evaluation_text)
            required_keys = [
                "evaluation",
                "confidence",
                "reasoning",
                "relevance_score",
                "relationship_type",
            ]
            for key in required_keys:
                if key not in evaluation_result:
                    if key == "evaluation":
                        evaluation_result[key] = "marginal"
                    elif key in ["confidence", "relevance_score"]:
                        evaluation_result[key] = 0.5
                    elif key == "reasoning":
                        evaluation_result[key] = "Reasoning not fully provided by LLM."
                    elif key == "relationship_type":
                        evaluation_result[key] = "tangential"
            evaluation_result["confidence"] = max(
                0.0, min(1.0, float(evaluation_result.get("confidence", 0.5)))
            )
            evaluation_result["relevance_score"] = max(
                0.0, min(1.0, float(evaluation_result.get("relevance_score", 0.5)))
            )
            if evaluation_result.get("evaluation") not in ["good", "bad", "marginal"]:
                evaluation_result["evaluation"] = "marginal"
            return json.dumps(evaluation_result)
        except json.JSONDecodeError:
            current_default = default_error_response.copy()
            current_default["reasoning"] = (
                f"Failed to parse LLM evaluation response"
            )
            current_default["evaluation"] = "marginal"
            current_default["confidence"] = 0.3
            return json.dumps(current_default)
    except Exception as e:
        current_default = default_error_response.copy()
        current_default["reasoning"] = f"Error during evaluation LLM call: {str(e)}"
        return json.dumps(current_default)

def get_llm_and_tools():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
    arxiv_tool_instance = None
    try:
        arxiv_wrapper = ArxivAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=4000,
            load_max_docs=1,
            load_all_available_meta=True,
        )
        arxiv_tool_instance = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    except Exception:
        pass
    tavily_search_tool_instance = None
    if TAVILY_API_KEY:
        try:
            tavily_search_tool_instance = TavilySearch(
                max_results=1,
                search_depth="basic",
                include_raw_content=False,
                include_answer=True,
            )
        except Exception:
            pass
    tools = []
    tools.append(
        Tool(
            name="LoadPDFContent",
            func=load_pdf_content,
            description="Loads a PDF document from a given file path and returns its full text content as a string. Input is the file path.",
        )
    )
    tools.append(
        Tool(
            name="ExtractAbstract",
            func=lambda text: _extract_abstract_from_text(text, llm),
            description="Extracts the abstract from a given document text using an LLM. Input is the full document text. Returns the abstract text or 'Abstract not found.'.",
        )
    )
    tools.append(
        Tool(
            name="ExtractReferencesSection",
            func=lambda text: _extract_references_section_from_text(text, llm),
            description="Extracts the raw references/bibliography section from a given document text using an LLM. Input is the full document text. Returns the raw text or 'NO_REFERENCES_SECTION_FOUND'.",
        )
    )
    tools.append(
        Tool(
            name="ParseReferences",
            func=lambda text: _parse_references_from_text(text, llm),
            description="Parses a raw references section text into a JSON string of structured citation objects. Input is the raw references section text. Each object has 'title', 'arxiv_id', 'authors', 'entire_citation'. Returns an empty JSON array string if parsing fails.",
        )
    )
    if arxiv_tool_instance:
        tools.append(
            Tool(
                name="ArxivSearch",
                func=lambda arxiv_id: _get_arxiv_summary_internal(arxiv_id, arxiv_tool_instance),
                description="Searches arXiv for a paper's summary/abstract using its arXiv ID. Input is the arXiv ID (e.g., '2301.07041'). Returns the summary text or an error message.",
            )
        )
    if tavily_search_tool_instance:
        tools.append(
            Tool(
                name="WebSearch",
                func=lambda query: process_tavily_result(
                    tavily_search_tool_instance.invoke(
                        f"Find abstract or summary for research paper: {query}"
                    )
                ),
                description="Performs a web search to find the abstract or summary of a research paper. Input is a search query (e.g., 'paper title authors'). Returns the summary text or an error message.",
            )
        )
    tools.append(
        Tool(
            name="EvaluateCitationQuality",
            func=lambda json_input: _evaluate_citation_quality_internal(
                json.loads(json_input)["main_abstract"],
                json.loads(json_input)["citation_summary"],
                json.loads(json_input)["citation_title"],
                llm,
            ),
            description="Evaluates the relevance and quality of a cited paper to a main paper. Input is a JSON string with keys: 'main_abstract', 'citation_summary', 'citation_title'. Returns a JSON string with evaluation details.",
        )
    )
    return llm, tools