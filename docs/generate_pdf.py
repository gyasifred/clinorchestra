#!/usr/bin/env python3
"""
Generate PDF from the Neurosymbolic ADRD Paper Outline

This script converts the markdown document to a well-formatted PDF.
Requires: fpdf2 or reportlab

Usage:
    python generate_pdf.py
"""

import os
import sys
from datetime import datetime

# Try to use fpdf2 first, fallback to reportlab
try:
    from fpdf import FPDF
    USE_FPDF = True
except ImportError:
    USE_FPDF = False
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    except ImportError:
        print("Error: Neither fpdf2 nor reportlab is installed.")
        print("Install with: pip install fpdf2 or pip install reportlab")
        sys.exit(1)


class PDFDocument:
    """PDF generator for the paper outline"""

    def __init__(self, output_path):
        self.output_path = output_path

    def generate(self):
        """Generate the PDF document"""
        if USE_FPDF:
            self._generate_fpdf()
        else:
            self._generate_reportlab()

    def _generate_fpdf(self):
        """Generate PDF using fpdf2"""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font('Helvetica', 'B', 18)
        pdf.multi_cell(0, 10, 'Neurosymbolic AI for ADRD Classification:', align='C')
        pdf.set_font('Helvetica', 'B', 14)
        pdf.multi_cell(0, 8, 'Comparing Pure LLM vs. Hybrid Approaches', align='C')
        pdf.ln(5)

        pdf.set_font('Helvetica', 'I', 12)
        pdf.multi_cell(0, 8, 'A Comprehensive Paper Outline and Pipeline Code Review', align='C')
        pdf.ln(5)

        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, 'Medical University of South Carolina, Biomedical Informatics Center', align='C')
        pdf.multi_cell(0, 6, f'Date: {datetime.now().strftime("%Y-%m-%d")}', align='C')
        pdf.ln(10)

        # Divider
        pdf.set_draw_color(100, 100, 100)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(10)

        # PART I Header
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 10, 'PART I: PAPER OUTLINE', ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Abstract
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, 'Abstract (Proposed)', ln=True)
        pdf.set_font('Helvetica', '', 10)

        abstract_text = """Background: Alzheimer's Disease and Related Dementias (ADRD) classification from clinical notes remains challenging due to complex diagnostic criteria requiring integration of cognitive assessments, functional status, and differential diagnosis. While Large Language Models (LLMs) show promise in clinical NLP tasks, their "black-box" reasoning raises concerns about reliability and interpretability in high-stakes medical decisions.

Objective: To compare the performance of pure LLM-based classification versus a neurosymbolic AI approach that integrates LLM reasoning with deterministic symbolic computation and knowledge retrieval for ADRD classification using NIA-AA criteria.

Methods: We developed ClinOrchestra, a neurosymbolic orchestration platform that combines: (1) LLM reasoning for natural language understanding, (2) symbolic functions for deterministic clinical score interpretation (MoCA, MMSE, CDR), (3) RAG-based retrieval from clinical guidelines, and (4) domain-specific knowledge hints. We evaluated both approaches on a clinical note dataset with ground-truth ADRD labels."""

        pdf.multi_cell(0, 5, abstract_text)
        pdf.ln(5)

        # Section 1: Introduction
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, '1. Introduction', ln=True)
        pdf.ln(3)

        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 6, '1.1 Background and Motivation', ln=True)
        pdf.set_font('Helvetica', '', 10)
        intro_text = """- ADRD Prevalence and Impact: Growing burden of Alzheimer's disease globally
- Clinical Classification Challenges: Complex multi-domain criteria (NIA-AA guidelines), requires integration of cognitive testing, functional assessment, and differential diagnosis
- Need for AI Assistance: Shortage of specialists, time constraints in primary care"""
        pdf.multi_cell(0, 5, intro_text)
        pdf.ln(3)

        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 6, '1.2 Research Questions', ln=True)
        pdf.set_font('Helvetica', '', 10)
        rq_text = """RQ1: How does pure LLM classification compare to neurosymbolic hybrid approaches for ADRD detection?
RQ2: Does integrating symbolic functions for clinical score interpretation improve classification accuracy?
RQ3: What role does RAG-based guideline retrieval play in classification performance?
RQ4: Can neurosymbolic approaches provide more interpretable reasoning chains?"""
        pdf.multi_cell(0, 5, rq_text)
        pdf.ln(5)

        # Section 2: Methods
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, '2. Methods', ln=True)
        pdf.ln(3)

        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 6, '2.1 Study Design', ln=True)
        pdf.set_font('Helvetica', '', 10)
        methods_text = """Comparison Arms:
- Arm 1 (Control): Pure LLM classification with NIA-AA prompt only
- Arm 2 (Treatment): Neurosymbolic approach (LLM + Functions + RAG + Extras)"""
        pdf.multi_cell(0, 5, methods_text)
        pdf.ln(3)

        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 6, '2.2 Neurosymbolic Architecture', ln=True)
        pdf.set_font('Helvetica', '', 10)
        arch_text = """The ClinOrchestra platform implements a 4-stage pipeline:

Stage 1: Task Analysis - LLM analyzes clinical text and determines required tools
Stage 2: Tool Execution - Functions, RAG, and Extras execute in parallel
Stage 3: Synthesis - LLM combines all tool outputs into structured JSON
Stage 4: RAG Refinement - Optional evidence-based verification

Key Components:
1. Symbolic Functions: Deterministic interpretation of MoCA, MMSE, CDR scores
2. RAG Engine: Vector search in NIA-AA 2024 clinical guidelines
3. Extras Manager: Domain-specific hints (differential diagnoses, terminology)"""
        pdf.multi_cell(0, 5, arch_text)
        pdf.ln(5)

        # Symbolic Functions Table
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 6, '2.3 Symbolic Functions for ADRD', ln=True)
        pdf.set_font('Helvetica', '', 9)

        # Table header
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(50, 6, 'Function', 1, 0, 'C', True)
        pdf.cell(70, 6, 'Purpose', 1, 0, 'C', True)
        pdf.cell(60, 6, 'Output', 1, 1, 'C', True)

        # Table rows
        functions = [
            ('interpret_moca', 'MoCA score interpretation', 'impairment_level, interpretation'),
            ('interpret_mmse', 'MMSE score interpretation', 'severity, dementia_suggested'),
            ('calculate_cdr_severity', 'CDR global score', 'category, implications'),
            ('count_domains', 'NIA-AA domain count', 'count, meets_criteria'),
            ('check_dementia_criteria', 'Full criteria check', 'classification (YES/NO)'),
        ]

        for func in functions:
            pdf.cell(50, 5, func[0], 1)
            pdf.cell(70, 5, func[1], 1)
            pdf.cell(60, 5, func[2], 1, 1)
        pdf.ln(5)

        # Section 3: Expected Results
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, '3. Expected Results', ln=True)
        pdf.set_font('Helvetica', '', 10)
        results_text = """Hypotheses:
H1: Neurosymbolic approach will achieve higher accuracy than pure LLM
H2: Symbolic function integration will reduce score interpretation errors
H3: RAG retrieval will improve guideline adherence
H4: Neurosymbolic reasoning will be more interpretable

Ablation Study Design:
- LLM only (baseline)
- LLM + Functions only
- LLM + RAG only
- LLM + Extras only
- LLM + Functions + RAG
- LLM + Functions + RAG + Extras (full neurosymbolic)"""
        pdf.multi_cell(0, 5, results_text)
        pdf.ln(5)

        # PART II Header
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 10, 'PART II: PIPELINE CODE REVIEW', ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Code Review
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, '1. ClinOrchestra Architecture Overview', ln=True)
        pdf.set_font('Helvetica', '', 10)
        code_review_text = """ClinOrchestra is a task-agnostic neurosymbolic AI orchestration platform that combines:
- Neural reasoning (LLMs) for natural language understanding
- Symbolic computation (deterministic functions) for grounded calculations
- Knowledge retrieval (RAG) for guideline-based evidence
- Domain hints (Extras) for contextual knowledge injection

Key Files:
- core/agent_system.py: STRUCTURED 4-stage extraction pipeline
- core/function_registry.py: Manages deterministic Python functions
- core/regex_preprocessor.py: Text normalization patterns
- core/rag_engine.py: FAISS-based retrieval from guidelines
- core/extras_manager.py: Domain-specific knowledge hints"""
        pdf.multi_cell(0, 5, code_review_text)
        pdf.ln(5)

        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, '2. Strengths of Current Implementation', ln=True)
        pdf.set_font('Helvetica', '', 10)
        strengths_text = """Clinical Accuracy:
- Functions correctly implement NIA-AA and DSM-5 guidelines
- Education-adjusted scoring (MoCA, SLUMS)
- CDR 0.5 ambiguity correctly handled (critical MCI vs dementia distinction)

Architecture Quality:
- Clean separation of neural vs. symbolic concerns
- Task-agnostic design (reusable for other clinical tasks)
- Parallel tool execution for performance
- Adaptive retry with metrics tracking

Interpretability:
- Explicit tool call logs visible
- Reasoning chain from LLM captured
- Function outputs provide deterministic justification
- RAG citations provide evidence trail"""
        pdf.multi_cell(0, 5, strengths_text)
        pdf.ln(5)

        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, '3. Recommended Experiments', ln=True)
        pdf.set_font('Helvetica', '', 10)
        experiments_text = """Experiment 1: Pure LLM vs. Neurosymbolic
- A/B comparison with identical test set
- Metrics: Accuracy, F1, PPV, NPV, sensitivity, specificity

Experiment 2: Ablation Study
- Remove one component at a time
- Measure performance delta from full pipeline

Experiment 3: Score Interpretation Accuracy
- Focus on cases with explicit MoCA/MMSE/CDR scores
- Compare interpretation accuracy between approaches

Experiment 4: MCI vs. Dementia Distinction
- Subset analysis on borderline cases (CDR 0.5, MoCA 18-25)
- Compare classification of ambiguous cases"""
        pdf.multi_cell(0, 5, experiments_text)
        pdf.ln(5)

        # NIA-AA Criteria Summary
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, 'Appendix: NIA-AA Criteria Summary', ln=True)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 6, 'Dementia Criteria (Table 1)', ln=True)
        pdf.set_font('Helvetica', '', 10)
        dementia_text = """Cognitive or behavioral symptoms that:
1. Interfere with ability to function at work or usual activities
2. Represent decline from previous levels of functioning
3. Not explained by delirium or major psychiatric disorder

AND cognitive impairment in minimum TWO domains:
- Memory (repetitive questions, misplacing items, forgetting events, getting lost)
- Executive (poor safety understanding, cannot manage finances, poor decisions)
- Visuospatial (cannot recognize faces/objects, cannot find objects)
- Language (word-finding difficulty, hesitations, speech/writing errors)
- Behavior (mood fluctuations, agitation, apathy, loss of drive, withdrawal)"""
        pdf.multi_cell(0, 5, dementia_text)
        pdf.ln(3)

        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 6, 'MCI Criteria (Table 1)', ln=True)
        pdf.set_font('Helvetica', '', 10)
        mci_text = """- Cognitive concern reflecting change, reported by patient/informant/clinician
- Objective evidence of impairment in >= 1 cognitive domain
- PRESERVATION OF INDEPENDENCE in functional abilities
- Not demented

KEY DISTINCTION: MCI vs. Dementia = Functional Independence
- MCI: Cognitive impairment WITH preserved independence
- Dementia: Cognitive impairment WITH functional decline"""
        pdf.multi_cell(0, 5, mci_text)

        # Save
        pdf.output(self.output_path)
        print(f"PDF generated successfully: {self.output_path}")

    def _generate_reportlab(self):
        """Generate PDF using reportlab"""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=12
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=6
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=6,
            textColor=colors.HexColor('#003366')
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )

        story = []

        # Title
        story.append(Paragraph("Neurosymbolic AI for ADRD Classification:", title_style))
        story.append(Paragraph("Comparing Pure LLM vs. Hybrid Approaches", subtitle_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph("A Comprehensive Paper Outline and Pipeline Code Review", styles['Italic']))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Medical University of South Carolina, Biomedical Informatics Center", styles['Normal']))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        story.append(Spacer(1, 24))

        # Part I Header
        story.append(Paragraph("PART I: PAPER OUTLINE", heading_style))
        story.append(Spacer(1, 12))

        # Abstract
        story.append(Paragraph("Abstract (Proposed)", styles['Heading3']))
        abstract = """<b>Background:</b> Alzheimer's Disease and Related Dementias (ADRD) classification from
        clinical notes remains challenging due to complex diagnostic criteria requiring integration of cognitive
        assessments, functional status, and differential diagnosis. While Large Language Models (LLMs) show promise
        in clinical NLP tasks, their "black-box" reasoning raises concerns about reliability and interpretability
        in high-stakes medical decisions.<br/><br/>
        <b>Objective:</b> To compare the performance of pure LLM-based classification versus a neurosymbolic AI
        approach that integrates LLM reasoning with deterministic symbolic computation and knowledge retrieval
        for ADRD classification using NIA-AA criteria.<br/><br/>
        <b>Methods:</b> We developed ClinOrchestra, a neurosymbolic orchestration platform that combines:
        (1) LLM reasoning for natural language understanding, (2) symbolic functions for deterministic clinical
        score interpretation (MoCA, MMSE, CDR), (3) RAG-based retrieval from clinical guidelines, and
        (4) domain-specific knowledge hints."""
        story.append(Paragraph(abstract, body_style))
        story.append(Spacer(1, 12))

        # Section 1
        story.append(Paragraph("1. Introduction", styles['Heading2']))
        story.append(Paragraph("1.1 Research Questions", styles['Heading3']))
        rqs = """<b>RQ1:</b> How does pure LLM classification compare to neurosymbolic hybrid approaches for ADRD detection?<br/>
        <b>RQ2:</b> Does integrating symbolic functions for clinical score interpretation improve classification accuracy?<br/>
        <b>RQ3:</b> What role does RAG-based guideline retrieval play in classification performance?<br/>
        <b>RQ4:</b> Can neurosymbolic approaches provide more interpretable reasoning chains?"""
        story.append(Paragraph(rqs, body_style))
        story.append(Spacer(1, 12))

        # Section 2
        story.append(Paragraph("2. Methods", styles['Heading2']))
        story.append(Paragraph("2.1 Study Design", styles['Heading3']))
        methods = """<b>Comparison Arms:</b><br/>
        - Arm 1 (Control): Pure LLM classification with NIA-AA prompt only<br/>
        - Arm 2 (Treatment): Neurosymbolic approach (LLM + Functions + RAG + Extras)"""
        story.append(Paragraph(methods, body_style))
        story.append(Spacer(1, 6))

        story.append(Paragraph("2.2 Neurosymbolic Architecture", styles['Heading3']))
        arch = """The ClinOrchestra platform implements a 4-stage pipeline:<br/>
        <b>Stage 1:</b> Task Analysis - LLM analyzes clinical text and determines required tools<br/>
        <b>Stage 2:</b> Tool Execution - Functions, RAG, and Extras execute in parallel<br/>
        <b>Stage 3:</b> Synthesis - LLM combines all tool outputs into structured JSON<br/>
        <b>Stage 4:</b> RAG Refinement - Optional evidence-based verification"""
        story.append(Paragraph(arch, body_style))
        story.append(Spacer(1, 12))

        # Functions Table
        story.append(Paragraph("2.3 Symbolic Functions for ADRD", styles['Heading3']))

        table_data = [
            ['Function', 'Purpose', 'Output'],
            ['interpret_moca', 'MoCA score interpretation', 'impairment_level'],
            ['interpret_mmse', 'MMSE score interpretation', 'severity'],
            ['calculate_cdr_severity', 'CDR global score', 'category'],
            ['count_domains', 'NIA-AA domain count', 'meets_criteria'],
            ['check_dementia_criteria', 'Full criteria check', 'classification'],
        ]

        table = Table(table_data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        # Page Break for Part II
        story.append(PageBreak())

        # Part II Header
        story.append(Paragraph("PART II: PIPELINE CODE REVIEW", heading_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("1. ClinOrchestra Architecture Overview", styles['Heading2']))
        overview = """ClinOrchestra is a task-agnostic neurosymbolic AI orchestration platform that combines:<br/>
        - <b>Neural reasoning</b> (LLMs) for natural language understanding<br/>
        - <b>Symbolic computation</b> (deterministic functions) for grounded calculations<br/>
        - <b>Knowledge retrieval</b> (RAG) for guideline-based evidence<br/>
        - <b>Domain hints</b> (Extras) for contextual knowledge injection<br/><br/>
        <b>Key Files:</b><br/>
        - core/agent_system.py: STRUCTURED 4-stage extraction pipeline<br/>
        - core/function_registry.py: Manages deterministic Python functions<br/>
        - core/rag_engine.py: FAISS-based retrieval from guidelines<br/>
        - core/extras_manager.py: Domain-specific knowledge hints"""
        story.append(Paragraph(overview, body_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("2. Strengths of Current Implementation", styles['Heading2']))
        strengths = """<b>Clinical Accuracy:</b><br/>
        - Functions correctly implement NIA-AA and DSM-5 guidelines<br/>
        - Education-adjusted scoring (MoCA, SLUMS)<br/>
        - CDR 0.5 ambiguity correctly handled (critical MCI vs dementia distinction)<br/><br/>
        <b>Architecture Quality:</b><br/>
        - Clean separation of neural vs. symbolic concerns<br/>
        - Task-agnostic design (reusable for other clinical tasks)<br/>
        - Parallel tool execution for performance<br/><br/>
        <b>Interpretability:</b><br/>
        - Explicit tool call logs visible<br/>
        - Function outputs provide deterministic justification<br/>
        - RAG citations provide evidence trail"""
        story.append(Paragraph(strengths, body_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("3. Recommended Experiments", styles['Heading2']))
        experiments = """<b>Experiment 1:</b> Pure LLM vs. Neurosymbolic - A/B comparison with identical test set<br/>
        <b>Experiment 2:</b> Ablation Study - Remove one component at a time to measure contribution<br/>
        <b>Experiment 3:</b> Score Interpretation Accuracy - Focus on cases with explicit scores<br/>
        <b>Experiment 4:</b> MCI vs. Dementia Distinction - Subset analysis on borderline cases"""
        story.append(Paragraph(experiments, body_style))

        # Build PDF
        doc.build(story)
        print(f"PDF generated successfully: {self.output_path}")


def main():
    """Main entry point"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'neurosymbolic_adrd_paper_outline.pdf')

    generator = PDFDocument(output_path)
    generator.generate()

    return output_path


if __name__ == "__main__":
    pdf_path = main()
    print(f"\nPDF saved to: {pdf_path}")
