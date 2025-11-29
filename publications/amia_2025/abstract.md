# ClinOrchestra: A Universal LLM-Powered Platform for Clinical Data Extraction and Classification

**Authors:** Frederick Gyasi¹, [Co-authors TBD]

**Affiliations:**
¹Biomedical Informatics Center, Medical University of South Carolina, Charleston, SC

---

## Abstract

### Objective

To develop and validate a universal, large language model (LLM)-powered platform for extracting structured clinical information from unstructured medical narratives, with applications in Alzheimer's Disease and Related Dementias (ADRD) classification and pediatric malnutrition assessment.

### Background

Clinical data extraction from unstructured text remains a significant bottleneck in healthcare AI development, clinical research, and quality improvement initiatives. Traditional natural language processing (NLP) approaches require task-specific model training, extensive labeled datasets, and domain expertise for each new extraction task. Recent advances in large language models offer potential for flexible, general-purpose clinical data extraction, but lack the reliability, domain knowledge integration, and systematic tool orchestration needed for production deployment.

### Methods

We developed ClinOrchestra, a modular platform implementing dual execution modes: (1) STRUCTURED mode—a deterministic 4-stage pipeline (task analysis, parallel tool execution, structured extraction, and retrieval-augmented generation refinement), and (2) ADAPTIVE mode—an autonomous iterative loop with dynamic tool selection. The system integrates 44 medical calculation functions (e.g., growth percentiles, clinical scores), 177 text normalization patterns, 192 clinical knowledge extras, and retrieval-augmented generation (RAG) for guideline-based validation. We evaluated the platform on two distinct clinical tasks: (a) binary classification of ADRD vs. non-ADRD from geriatric neurology notes (N=[PLACEHOLDER] patients, [PLACEHOLDER] clinical notes), and (b) pediatric malnutrition classification using WHO and ASPEN criteria from growth and nutrition documentation (N=[PLACEHOLDER] patients, [PLACEHOLDER] encounters).

### Results

**ADRD Classification:**
- Accuracy: [PLACEHOLDER]% (95% CI: [PLACEHOLDER])
- Sensitivity: [PLACEHOLDER]% (95% CI: [PLACEHOLDER])
- Specificity: [PLACEHOLDER]% (95% CI: [PLACEHOLDER])
- Positive Predictive Value: [PLACEHOLDER]%
- Negative Predictive Value: [PLACEHOLDER]%
- F1-Score: [PLACEHOLDER]
- Area Under ROC Curve (AUC): [PLACEHOLDER]
- Comparison to baseline NLP: [PLACEHOLDER]% improvement in F1-score

**Pediatric Malnutrition Classification:**
- Accuracy: [PLACEHOLDER]% (95% CI: [PLACEHOLDER])
- Sensitivity: [PLACEHOLDER]% (95% CI: [PLACEHOLDER])
- Specificity: [PLACEHOLDER]% (95% CI: [PLACEHOLDER])
- WHO Criteria Agreement (κ): [PLACEHOLDER] ([PLACEHOLDER] agreement)
- ASPEN Criteria Agreement (κ): [PLACEHOLDER] ([PLACEHOLDER] agreement)
- Multi-indicator validation accuracy: [PLACEHOLDER]%
- Comparison to rule-based approach: [PLACEHOLDER]% improvement

**Cross-Domain Performance:**
- Average processing time per document: [PLACEHOLDER] seconds
- Cache utilization efficiency: [PLACEHOLDER]x speedup
- Zero-shot transfer capability: [PLACEHOLDER]% accuracy on held-out clinical domain

### Conclusion

ClinOrchestra demonstrates that a universal LLM-powered platform with systematic tool orchestration and knowledge integration can achieve high performance across diverse clinical extraction tasks without task-specific model training. The dual execution mode architecture balances reliability (STRUCTURED mode) with flexibility (ADAPTIVE mode), enabling both production deployment and complex case handling. Integration of domain-specific functions, retrieval-augmented generation, and clinical knowledge significantly enhances LLM performance beyond zero-shot prompting. This approach offers a scalable solution for accelerating clinical AI development, supporting quality improvement initiatives, and enabling rapid adaptation to new extraction tasks. Future work will extend validation to additional clinical domains and evaluate real-world deployment in clinical workflows.

**Keywords:** Natural Language Processing, Large Language Models, Clinical Data Extraction, Alzheimer's Disease, Pediatric Malnutrition, Retrieval-Augmented Generation, Tool Orchestration

**Word Count:** [PLACEHOLDER] words
