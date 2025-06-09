import os
import json
from modules.llm_tools import get_llm_and_tools

def verify_citations(pdf_path):
    llm, tools = get_llm_and_tools()
    load_pdf_tool = next((t for t in tools if t.name == "LoadPDFContent"), None)
    extract_abstract_tool = next((t for t in tools if t.name == "ExtractAbstract"), None)
    extract_references_section_tool = next((t for t in tools if t.name == "ExtractReferencesSection"), None)
    parse_references_tool = next((t for t in tools if t.name == "ParseReferences"), None)
    arxiv_search_tool = next((t for t in tools if t.name == "ArxivSearch"), None)
    web_search_tool = next((t for t in tools if t.name == "WebSearch"), None)
    evaluate_citation_quality_tool = next((t for t in tools if t.name == "EvaluateCitationQuality"), None)

    pdf_content = load_pdf_tool.func(pdf_path)
    if "ERROR" in pdf_content:
        return {"error": f"Failed to load PDF: {pdf_content}"}

    main_abstract_content = extract_abstract_tool.func(pdf_content)
    if "ERROR" in main_abstract_content or "Abstract not found." in main_abstract_content:
        main_abstract_content = "Abstract could not be extracted or was not found."

    references_content_raw = extract_references_section_tool.func(pdf_content)
    if "ERROR" in references_content_raw or references_content_raw == "NO_REFERENCES_SECTION_FOUND":
        return {
            "main_abstract": main_abstract_content,
            "enhanced_references": [],
        }

    parsed_references_json_str = parse_references_tool.func(references_content_raw)
    references_array = json.loads(parsed_references_json_str)
    if not references_array:
        return {
            "main_abstract": main_abstract_content,
            "enhanced_references": [],
        }

    enhanced_references = []
    for ref in references_array:
        enhanced_ref = {
            "title": ref.get("title", ""),
            "arxiv_id": ref.get("arxiv_id", ""),
            "authors": ref.get("authors", ""),
            "entire_citation": ref.get("entire_citation", ""),
            "summary": "",
            "citation_evaluation": {},
        }
        arxiv_id_val = ref.get("arxiv_id")
        summary_found = False

        if arxiv_id_val and arxiv_search_tool:
            arxiv_summary = arxiv_search_tool.func(arxiv_id_val)
            if arxiv_summary and not arxiv_summary.startswith("ERROR"):
                enhanced_ref["summary"] = arxiv_summary
                summary_found = True

        if not summary_found and web_search_tool:
            query_parts = []
            if ref.get("title"):
                query_parts.append(f'title: "{ref["title"]}"')
            if ref.get("authors"):
                query_parts.append(f'authors: "{ref["authors"]}"')
            query = " ".join(query_parts).strip()
            if not query and ref.get("entire_citation"):
                query = ref.get("entire_citation", "")
            if query:
                web_summary = web_search_tool.func(query)
                if web_summary and not web_summary.startswith("ERROR"):
                    enhanced_ref["summary"] = web_summary
                    summary_found = True

        if enhanced_ref["summary"] and main_abstract_content:
            evaluation_input = json.dumps(
                {
                    "main_abstract": main_abstract_content,
                    "citation_summary": enhanced_ref["summary"],
                    "citation_title": enhanced_ref["title"],
                }
            )
            citation_evaluation_json_str = (
                evaluate_citation_quality_tool.func(evaluation_input)
            )
            citation_evaluation = json.loads(citation_evaluation_json_str)
            enhanced_ref["citation_evaluation"] = citation_evaluation
        else:
            reason = "Missing citation summary for evaluation."
            if not main_abstract_content:
                reason = "Missing main abstract for evaluation."
            if not enhanced_ref["summary"] and not main_abstract_content:
                reason = "Missing main abstract and citation summary."
            enhanced_ref["citation_evaluation"] = {
                "evaluation": "unable_to_evaluate",
                "confidence": 0.0,
                "reasoning": reason,
                "relevance_score": 0.0,
                "relationship_type": "unrelated",
            }
        enhanced_references.append(enhanced_ref)

    return {
        "main_abstract": main_abstract_content,
        "enhanced_references": enhanced_references,
    }