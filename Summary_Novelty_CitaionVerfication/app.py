import streamlit as st
import tempfile
import os
from modules.summarizer import analyze_paper
from modules.citation_verifier import verify_citations

st.set_page_config(page_title="Research Paper Analyzer", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Summary", "Novelty", "Citation Verification", "AI Content Detection"])

if "pdf_uploaded" not in st.session_state:
    st.session_state["pdf_uploaded"] = False
if "summary" not in st.session_state:
    st.session_state["summary"] = None
if "novelty" not in st.session_state:
    st.session_state["novelty"] = None
if "citation_results" not in st.session_state:
    st.session_state["citation_results"] = None
if "processing" not in st.session_state:
    st.session_state["processing"] = False
if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    st.session_state["pdf_path"] = tmp_path
    summary, novelty = analyze_paper(tmp_path)
    citation_results = verify_citations(tmp_path)
    return summary, novelty, citation_results

if page == "Home":
    st.title("Copy_Catch - Research Paper Analyzer")
    st.markdown("""
        Upload your research paper (PDF), and we'll extract:
        - A detailed summary
        - The paper's novelty
        - Citation verification
    """)
    uploaded_file = st.file_uploader("Upload a research paper", type=["pdf"])
    if st.button("Analyze Paper") and uploaded_file:
        st.session_state["processing"] = True
        with st.spinner("Analyzing... (this may take a few minutes)"):
            summary, novelty, citation_results = process_pdf(uploaded_file)
            st.session_state["summary"] = summary
            st.session_state["novelty"] = novelty
            st.session_state["citation_results"] = citation_results
            st.session_state["pdf_uploaded"] = True
            st.session_state["processing"] = False
        st.success("Analysis complete! You can now view results from the sidebar.")

    if st.session_state["processing"]:
        st.info("Processing... Please wait.")

elif page == "Summary":
    st.title("Summary of Research Paper")
    if st.session_state.get("pdf_uploaded"):
        st.markdown(st.session_state["summary"])
    else:
        st.info("Please upload and analyze a paper first from the Home page.")

elif page == "Novelty":
    st.title("Novelty in This Research Paper")
    if st.session_state.get("pdf_uploaded"):
        st.markdown(st.session_state["novelty"])
    else:
        st.info("Please upload and analyze a paper first from the Home page.")

elif page == "Citation Verification":
    st.title("Citation Verification")
    if st.session_state.get("pdf_uploaded"):
        citation_results = st.session_state["citation_results"]
        st.subheader("Main Paper Abstract")
        st.markdown(f"> {citation_results.get('main_abstract', 'N/A')}")
        st.subheader("References Evaluation")
        enhanced_references = citation_results.get("enhanced_references", [])
        if enhanced_references:
            import pandas as pd
            table_data = []
            for ref in enhanced_references:
                eval_data = ref.get("citation_evaluation", {})
                relevance_score_val = eval_data.get("relevance_score", "N/A")
                title_display = ref.get("title", "").strip()
                if not title_display:
                    title_display = (
                        (ref.get("entire_citation", "N/A")[:100] + "...")
                        if ref.get("entire_citation")
                        else "N/A"
                    )
                relevance_score_str = (
                    f"{relevance_score_val:.2f}"
                    if isinstance(relevance_score_val, (float, int))
                    else str(relevance_score_val)
                )
                # table_data.append(
                #     {
                #         "Title": title_display,
                #         "Evaluation": eval_data.get("evaluation", "N/A"),
                #         "Relevance Score": relevance_score_str,
                #         "Relationship Type": eval_data.get("relationship_type", "N/A"),
                #     }
                # )
                table_data.append(
                    {
                        "Title": title_display,
                        "Evaluation": eval_data.get("evaluation", "N/A"),
                        "Reasoning": eval_data.get("reasoning", "N/A"),
                        "Relevance Score": relevance_score_str,
                        "Relationship Type": eval_data.get("relationship_type", "N/A"),
                    }
                )
            df = pd.DataFrame(table_data)
            st.dataframe(df, height=min(len(df) * 35 + 38, 600), use_container_width=True)

            st.subheader("📊 Citation Quality Summary")
            good = sum(
                1
                for item in citation_results.get("enhanced_references", [])
                if item["citation_evaluation"].get("evaluation") == "good"
            )
            bad = sum(
                1
                for item in citation_results.get("enhanced_references", [])
                if item["citation_evaluation"].get("evaluation") == "bad"
            )
            marginal = sum(
                1
                for item in citation_results.get("enhanced_references", [])
                if item["citation_evaluation"].get("evaluation") == "marginal"
            )
            unable = sum(
                1
                for item in citation_results.get("enhanced_references", [])
                if item["citation_evaluation"].get("evaluation") == "unable_to_evaluate"
            )
            total = len(citation_results.get("enhanced_references", []))

            if total > 0:
                st.write(f"Total references processed: {total}")
                cols = st.columns(4)
                cols[0].metric(
                    "👍 Good", good, f"{good / total * 100:.1f}%" if total else "0%"
                )
                cols[1].metric(
                    "👎 Bad", bad, f"{bad / total * 100:.1f}%" if total else "0%"
                )
                cols[2].metric(
                    "🤔 Marginal",
                    marginal,
                    f"{marginal / total * 100:.1f}%" if total else "0%",
                )
                cols[3].metric(
                    "❓ Unable to Evaluate",
                    unable,
                    f"{unable / total * 100:.1f}%" if total else "0%",
                )
            else:
                st.info("No references were processed to summarize.")
        else:
            st.info("No references were found or parsed from the document to evaluate.")
    else:
        st.info("Please upload and analyze a paper first from the Home page.")

elif page == "AI Content Detection":
    st.title("AI Content Detection in Research Paper")
    st.markdown("""
    Upload a research paper (PDF) on the Home page, then view AI content detection results here.
    """)
    if st.session_state.get("pdf_uploaded") and st.session_state.get("pdf_path"):
        from modules.ai_detector import analyze_pdf_for_ai_content
        st.markdown("""
        #### 🧭 AI Prediction Scale
        <div style="display: flex; justify-content: space-between; font-size: 14px;">
            <span style="color: red;">🟥 AI Generated<br><small>0–20</small></span>
            <span style="color: orange;">🟧 Likely AI<br><small>20–40</small></span>
            <span style="color: goldenrod;">🟨 Uncertain<br><small>40–60</small></span>
            <span style="color: limegreen;">🟩 Mostly Human<br><small>60–80</small></span>
            <span style="color: green;">🟩 Human Written<br><small>80–100</small></span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Detect AI-generated Content"):
            with st.spinner("Analyzing for AI-generated content..."):
                result = analyze_pdf_for_ai_content(st.session_state["pdf_path"])
            if result:
                score = result.get("score", 0)
                if score <= 20:
                    label = "🟥 AI Generated"
                    color = "red"
                elif score <= 40:
                    label = "🟧 Likely AI"
                    color = "orange"
                elif score <= 60:
                    label = "🟨 Uncertain"
                    color = "goldenrod"
                elif score <= 80:
                    label = "🟩 Mostly Human"
                    color = "limegreen"
                else:
                    label = "🟩 Human Written"
                    color = "green"
                st.markdown(f"### 🧠 Prediction Score: `{score:.2f}`")
                st.markdown(f"**Prediction Category:** <span style='color:{color}; font-size: 18px;'>{label}</span>", unsafe_allow_html=True)
                st.progress(int(score))
                with st.expander("📋 Raw API Response"):
                    st.json(result)
            else:
                st.warning("No result returned from AI detection module.")
    else:
        st.info("Please upload and analyze a paper first from the Home page.")