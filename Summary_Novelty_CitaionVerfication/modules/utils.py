import re
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_content(file_path: str) -> str:
    """Loads a PDF document and returns its concatenated page content."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} pages from {file_path}")
        return "".join([doc.page_content for doc in documents])
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return "ERROR: PDF file not found."
    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")
        return f"ERROR: An error occurred while loading the PDF: {e}"

def clean_arxiv_id(arxiv_id: str) -> str:
    """Cleans and normalizes an arXiv ID to the standard format."""
    if not arxiv_id or str(arxiv_id).lower() in ["null", "none", ""]:
        return ""
    arxiv_id_str = str(arxiv_id).strip()
    arxiv_id_str = re.sub(r"^(arXiv:|arxiv:)", "", arxiv_id_str, flags=re.IGNORECASE)
    arxiv_id_str = re.sub(r"^https?://arxiv\.org/abs/", "", arxiv_id_str)
    arxiv_id_str = re.sub(r"^abs/", "", arxiv_id_str)
    match = re.search(
        r"(\d{4}\.\d{4,5}(v\d+)?|[a-z-]+/\d{7}(v\d+)?)", arxiv_id_str
    )  # Added version support
    return match.group(1) if match else ""

def process_tavily_result(tavily_result) -> str:
    """Helper function to extract string summary from TavilySearch results."""
    if isinstance(tavily_result, str):
        return tavily_result
    elif isinstance(tavily_result, list) and tavily_result:
        # Prioritize 'answer' key if available, otherwise 'content'
        if 'answer' in tavily_result[0] and tavily_result[0]['answer']:
            return tavily_result[0]['answer']
        elif 'content' in tavily_result[0] and tavily_result[0]['content']:
            return tavily_result[0]['content']
    elif isinstance(tavily_result, dict) and tavily_result.get("answer"):
        return tavily_result.get("answer")
    return "ERROR: No usable summary found from web search."