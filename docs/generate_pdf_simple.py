#!/usr/bin/env python3
"""
Simple PDF Generator using basic Python
Creates a text-based PDF from the paper outline

This is a minimal implementation that doesn't require external PDF libraries.
"""

import os
from datetime import datetime


def create_pdf(output_path):
    """Create a simple PDF file manually"""

    # PDF header
    pdf_content = []

    # PDF objects
    objects = []

    # Object 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # Object 2: Pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R 6 0 R] /Count 4 >>\nendobj\n")

    # Content for each page
    page_contents = []

    # Page 1 content - Title and Abstract
    page1_text = """BT
/F1 18 Tf
50 750 Td
(Neurosymbolic AI for ADRD Classification:) Tj
/F1 14 Tf
0 -25 Td
(Comparing Pure LLM vs. Hybrid Approaches) Tj
/F1 12 Tf
0 -20 Td
(A Comprehensive Paper Outline and Pipeline Code Review) Tj
/F1 10 Tf
0 -30 Td
(Medical University of South Carolina, Biomedical Informatics Center) Tj
0 -15 Td
(Date: """ + datetime.now().strftime("%Y-%m-%d") + """) Tj
0 -40 Td
/F1 14 Tf
(PART I: PAPER OUTLINE) Tj
/F1 12 Tf
0 -30 Td
(Abstract) Tj
/F1 10 Tf
0 -20 Td
(Background: Alzheimer's Disease and Related Dementias \\(ADRD\\) classification from) Tj
0 -12 Td
(clinical notes remains challenging due to complex diagnostic criteria requiring) Tj
0 -12 Td
(integration of cognitive assessments, functional status, and differential diagnosis.) Tj
0 -20 Td
(Objective: To compare the performance of pure LLM-based classification versus a) Tj
0 -12 Td
(neurosymbolic AI approach that integrates LLM reasoning with deterministic symbolic) Tj
0 -12 Td
(computation and knowledge retrieval for ADRD classification using NIA-AA criteria.) Tj
0 -20 Td
(Methods: We developed ClinOrchestra, a neurosymbolic orchestration platform that) Tj
0 -12 Td
(combines: \\(1\\) LLM reasoning for natural language understanding, \\(2\\) symbolic) Tj
0 -12 Td
(functions for deterministic clinical score interpretation \\(MoCA, MMSE, CDR\\),) Tj
0 -12 Td
(\\(3\\) RAG-based retrieval from clinical guidelines, and \\(4\\) domain-specific hints.) Tj
0 -30 Td
/F1 12 Tf
(1. Introduction) Tj
/F1 10 Tf
0 -20 Td
(Research Questions:) Tj
0 -15 Td
(RQ1: How does pure LLM classification compare to neurosymbolic hybrid approaches?) Tj
0 -12 Td
(RQ2: Does integrating symbolic functions improve classification accuracy?) Tj
0 -12 Td
(RQ3: What role does RAG-based guideline retrieval play in performance?) Tj
0 -12 Td
(RQ4: Can neurosymbolic approaches provide more interpretable reasoning chains?) Tj
ET"""
    page_contents.append(page1_text)

    # Page 2 content - Methods
    page2_text = """BT
/F1 14 Tf
50 750 Td
(2. Methods) Tj
/F1 12 Tf
0 -25 Td
(2.1 Study Design) Tj
/F1 10 Tf
0 -20 Td
(Comparison Arms:) Tj
0 -15 Td
(- Arm 1 \\(Control\\): Pure LLM classification with NIA-AA prompt only) Tj
0 -12 Td
(- Arm 2 \\(Treatment\\): Neurosymbolic approach \\(LLM + Functions + RAG + Extras\\)) Tj
0 -25 Td
/F1 12 Tf
(2.2 Neurosymbolic Architecture) Tj
/F1 10 Tf
0 -20 Td
(The ClinOrchestra platform implements a 4-stage pipeline:) Tj
0 -15 Td
(Stage 1: Task Analysis - LLM analyzes clinical text and determines required tools) Tj
0 -12 Td
(Stage 2: Tool Execution - Functions, RAG, and Extras execute in parallel) Tj
0 -12 Td
(Stage 3: Synthesis - LLM combines all tool outputs into structured JSON) Tj
0 -12 Td
(Stage 4: RAG Refinement - Optional evidence-based verification) Tj
0 -25 Td
/F1 12 Tf
(2.3 Symbolic Functions for ADRD) Tj
/F1 10 Tf
0 -20 Td
(| Function              | Purpose                    | Output               |) Tj
0 -12 Td
(| interpret_moca        | MoCA score interpretation  | impairment_level     |) Tj
0 -12 Td
(| interpret_mmse        | MMSE score interpretation  | severity             |) Tj
0 -12 Td
(| calculate_cdr_severity| CDR global score           | category             |) Tj
0 -12 Td
(| count_domains         | NIA-AA domain count        | meets_criteria       |) Tj
0 -12 Td
(| check_dementia_criteria| Full criteria check       | classification       |) Tj
0 -25 Td
/F1 12 Tf
(3. Expected Results) Tj
/F1 10 Tf
0 -20 Td
(Hypotheses:) Tj
0 -15 Td
(H1: Neurosymbolic approach will achieve higher accuracy than pure LLM) Tj
0 -12 Td
(H2: Symbolic function integration will reduce score interpretation errors) Tj
0 -12 Td
(H3: RAG retrieval will improve guideline adherence) Tj
0 -12 Td
(H4: Neurosymbolic reasoning will be more interpretable) Tj
ET"""
    page_contents.append(page2_text)

    # Page 3 content - Pipeline Code Review
    page3_text = """BT
/F1 14 Tf
50 750 Td
(PART II: PIPELINE CODE REVIEW) Tj
/F1 12 Tf
0 -30 Td
(1. ClinOrchestra Architecture Overview) Tj
/F1 10 Tf
0 -20 Td
(ClinOrchestra is a task-agnostic neurosymbolic AI orchestration platform that combines:) Tj
0 -15 Td
(- Neural reasoning \\(LLMs\\) for natural language understanding) Tj
0 -12 Td
(- Symbolic computation \\(deterministic functions\\) for grounded calculations) Tj
0 -12 Td
(- Knowledge retrieval \\(RAG\\) for guideline-based evidence) Tj
0 -12 Td
(- Domain hints \\(Extras\\) for contextual knowledge injection) Tj
0 -25 Td
(Key Files:) Tj
0 -15 Td
(- core/agent_system.py: STRUCTURED 4-stage extraction pipeline) Tj
0 -12 Td
(- core/function_registry.py: Manages deterministic Python functions) Tj
0 -12 Td
(- core/regex_preprocessor.py: Text normalization patterns) Tj
0 -12 Td
(- core/rag_engine.py: FAISS-based retrieval from guidelines) Tj
0 -12 Td
(- core/extras_manager.py: Domain-specific knowledge hints) Tj
0 -30 Td
/F1 12 Tf
(2. Strengths of Current Implementation) Tj
/F1 10 Tf
0 -20 Td
(Clinical Accuracy:) Tj
0 -15 Td
(- Functions correctly implement NIA-AA and DSM-5 guidelines) Tj
0 -12 Td
(- Education-adjusted scoring \\(MoCA, SLUMS\\)) Tj
0 -12 Td
(- CDR 0.5 ambiguity correctly handled \\(critical MCI vs dementia distinction\\)) Tj
0 -20 Td
(Architecture Quality:) Tj
0 -15 Td
(- Clean separation of neural vs. symbolic concerns) Tj
0 -12 Td
(- Task-agnostic design \\(reusable for other clinical tasks\\)) Tj
0 -12 Td
(- Parallel tool execution for performance) Tj
0 -12 Td
(- Adaptive retry with metrics tracking) Tj
ET"""
    page_contents.append(page3_text)

    # Page 4 content - NIA-AA Criteria
    page4_text = """BT
/F1 14 Tf
50 750 Td
(Appendix: NIA-AA Criteria Summary) Tj
/F1 12 Tf
0 -30 Td
(Dementia Criteria \\(Table 1\\)) Tj
/F1 10 Tf
0 -20 Td
(Cognitive or behavioral symptoms that:) Tj
0 -15 Td
(1. Interfere with ability to function at work or usual activities) Tj
0 -12 Td
(2. Represent decline from previous levels of functioning) Tj
0 -12 Td
(3. Not explained by delirium or major psychiatric disorder) Tj
0 -20 Td
(AND cognitive impairment in minimum TWO domains:) Tj
0 -15 Td
(- Memory: repetitive questions, misplacing items, forgetting events, getting lost) Tj
0 -12 Td
(- Executive: poor safety understanding, cannot manage finances, poor decisions) Tj
0 -12 Td
(- Visuospatial: cannot recognize faces/objects, cannot find objects) Tj
0 -12 Td
(- Language: word-finding difficulty, hesitations, speech/writing errors) Tj
0 -12 Td
(- Behavior: mood fluctuations, agitation, apathy, loss of drive, withdrawal) Tj
0 -30 Td
/F1 12 Tf
(MCI Criteria \\(Table 1\\)) Tj
/F1 10 Tf
0 -20 Td
(- Cognitive concern reflecting change, reported by patient/informant/clinician) Tj
0 -12 Td
(- Objective evidence of impairment in >= 1 cognitive domain) Tj
0 -12 Td
(- PRESERVATION OF INDEPENDENCE in functional abilities) Tj
0 -12 Td
(- Not demented) Tj
0 -30 Td
/F1 12 Tf
(KEY DISTINCTION: MCI vs. Dementia = Functional Independence) Tj
/F1 10 Tf
0 -20 Td
(- MCI: Cognitive impairment WITH preserved independence) Tj
0 -12 Td
(- Dementia: Cognitive impairment WITH functional decline) Tj
ET"""
    page_contents.append(page4_text)

    # Build pages
    obj_num = 3
    content_obj_start = 7

    for i, content in enumerate(page_contents):
        # Page object
        objects.append(f"{obj_num} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_obj_start + i} 0 R /Resources << /Font << /F1 {content_obj_start + len(page_contents)} 0 R >> >> >>\nendobj\n")
        obj_num += 1

    # Content streams
    for i, content in enumerate(page_contents):
        stream = f"stream\n{content}\nendstream"
        objects.append(f"{content_obj_start + i} 0 obj\n<< /Length {len(content)} >>\n{stream}\nendobj\n")

    # Font object
    font_obj = content_obj_start + len(page_contents)
    objects.append(f"{font_obj} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    # Build the PDF
    pdf = "%PDF-1.4\n"

    # Calculate object offsets
    offset = len(pdf)
    offsets = []

    for obj in objects:
        offsets.append(offset)
        pdf += obj
        offset = len(pdf)

    # Cross-reference table
    xref_offset = len(pdf)
    pdf += "xref\n"
    pdf += f"0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n"

    # Trailer
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    pdf += f"startxref\n{xref_offset}\n%%EOF"

    # Write to file
    with open(output_path, 'wb') as f:
        f.write(pdf.encode('latin-1'))

    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "neurosymbolic_adrd_paper_outline.pdf")
    create_pdf(output_path)
