# ClinOrchestra: A Universal Platform for Autonomous Clinical AI Through Task-Driven LLM Orchestration

**Authors:** Frederick Gyasi¹, [Co-authors TBD]

**Affiliations:**
¹Biomedical Informatics Center, Medical University of South Carolina, Charleston, SC

---

## Abstract

### Objective

To develop and validate ClinOrchestra, a universal platform for autonomous clinical AI that enables task-driven language model orchestration, and evaluate its performance on pediatric malnutrition classification and Alzheimer's Disease and Related Dementias (ADRD) diagnostic classification.

### Background

Clinical AI development faces significant barriers from task-specific model requirements, extensive labeled datasets, and limited adaptability to new clinical domains [1]. While large language models (LLMs) offer potential for flexible clinical task automation, they lack systematic integration of computational tools, domain knowledge, and clinical guidelines needed for reliable autonomous execution [2, 3]. Current approaches require custom development for each clinical task, limiting scalability and generalizability. We developed ClinOrchestra as a universal, autonomous clinical AI platform that accomplishes any clinical task through intelligent orchestration of multiple tool types—functions (Python computations), retrieval-augmented generation (RAG) resources (clinical guidelines and publications), extras (task-specific hints), and patterns (text transformations)—driven entirely by user-defined task descriptions, prompts, and configurations.

### Methods

ClinOrchestra implements dual workflow types: (1) STRUCTURED workflows—a predefined 4-stage pipeline (autonomous task analysis, parallel tool execution, task completion, and optional RAG refinement) for predictable, production-ready tasks, and (2) ADAPTIVE workflows—dynamic autonomous workflows that adjust tool selection and execution strategy based on task requirements and intermediate results. The platform is fully user-configurable: users write custom Python functions, define text patterns, provide task-specific hints (extras), and supply RAG resources via JSON/YAML configuration files. We validated the platform on two distinct in-house clinical datasets: (a) pediatric malnutrition classification using WHO and ASPEN criteria (N=[PLACEHOLDER] patients, [PLACEHOLDER] encounters), and (b) ADRD vs. non-ADRD binary classification from geriatric neurology notes (N=[PLACEHOLDER] patients, [PLACEHOLDER] clinical notes). Both tasks were configured with user-defined schemas, custom functions for clinical calculations, and RAG resources containing relevant clinical guidelines.

### Results

**Pediatric Malnutrition Classification:**
- Overall accuracy: [PLACEHOLDER]% (95% CI: [PLACEHOLDER]–[PLACEHOLDER])
- Sensitivity: [PLACEHOLDER]% (95% CI: [PLACEHOLDER]–[PLACEHOLDER])
- Specificity: [PLACEHOLDER]% (95% CI: [PLACEHOLDER]–[PLACEHOLDER])
- F1-score: [PLACEHOLDER]
- WHO criteria agreement (κ): [PLACEHOLDER] ([PLACEHOLDER] agreement)
- ASPEN criteria agreement (κ): [PLACEHOLDER] ([PLACEHOLDER] agreement)

**ADRD Classification:**
- Overall accuracy: [PLACEHOLDER]% (95% CI: [PLACEHOLDER]–[PLACEHOLDER])
- Sensitivity: [PLACEHOLDER]% (95% CI: [PLACEHOLDER]–[PLACEHOLDER])
- Specificity: [PLACEHOLDER]% (95% CI: [PLACEHOLDER]–[PLACEHOLDER])
- Positive predictive value: [PLACEHOLDER]%
- Negative predictive value: [PLACEHOLDER]%
- F1-score: [PLACEHOLDER]
- AUC-ROC: [PLACEHOLDER]

**Platform Performance:**
- Average processing time per case: [PLACEHOLDER] seconds
- Tool orchestration accuracy: [PLACEHOLDER]% (correct tool selection and execution)
- RAG retrieval relevance: [PLACEHOLDER]% top-3 accuracy
- Cross-task adaptability: [PLACEHOLDER] hours to configure new clinical task

### Conclusion

ClinOrchestra demonstrates that a universal, user-configurable platform for autonomous clinical AI can achieve high performance across diverse clinical tasks through task-driven LLM orchestration. The platform's ability to autonomously analyze task requirements, intelligently select and execute tools (functions, RAG, extras, patterns), and adapt execution strategies enables rapid deployment to new clinical domains without custom model development. Integration of user-defined computational functions, clinical guidelines via RAG, and domain-specific knowledge significantly enhances LLM performance beyond zero-shot prompting. The dual workflow architecture balances predictable execution (STRUCTURED) with adaptive clinical reasoning (ADAPTIVE), supporting both production deployment and complex case handling. This user-driven, autonomous approach offers a scalable solution for clinical AI development, quality improvement initiatives, and research data collection. Future work will extend validation to additional clinical domains, evaluate real-world clinical workflow integration, and assess generalizability across healthcare institutions.

**Keywords:** Clinical Artificial Intelligence, Large Language Models, Autonomous Systems, Task-Driven Orchestration, Clinical Decision Support, Alzheimer's Disease, Pediatric Malnutrition, Retrieval-Augmented Generation

### References

[1] Rajkomar A, Dean J, Kohane I. Machine learning in medicine. N Engl J Med. 2019;380(14):1347-1358.

[2] Thirunavukarasu AJ, Ting DSJ, Elangovan K, Gutierrez L, Tan TF, Ting DSW. Large language models in medicine. Nat Med. 2023;29(8):1930-1940.

[3] Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge. Nature. 2023;620(7972):172-180.

**Word Count:** [PLACEHOLDER] words
