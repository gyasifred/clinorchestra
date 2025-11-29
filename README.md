# ClinOrchestra v1.0.0

**Universal, Autonomous Clinical AI Platform for Task-Driven LLM Orchestration**

ClinOrchestra is a **universal, autonomous clinical AI platform** that enables task-driven language model orchestration to accomplish **any clinical task**. The platform integrates multiple tool types—functions (Python computations), RAG (clinical guidelines and publications), extras (task-specific hints), and patterns (text transformations)—that LLMs leverage autonomously based on task requirements. From malnutrition classification to ADRD diagnosis, from medication extraction to guideline adherence assessment, ClinOrchestra adapts to **any clinical task** through user-defined prompts, schemas, and configurations.

![ClinOrchestra Logo](assets/clinorchestra_logo.svg)

---

## Key Features

- **🌐 Universal Platform**: Accomplishes ANY clinical task through autonomous LLM orchestration
  - No hardcoded task logic - driven by user-defined prompts and schemas
  - Examples (malnutrition, ADRD) demonstrate capability, not limits
  - Platform is fully user-configurable: custom functions, patterns, extras, and RAG resources
- **⚙️ Dual Workflow Types**:
  - **STRUCTURED Workflows**: Predefined 4-stage pipeline for predictable, production-ready tasks
  - **ADAPTIVE Workflows**: Dynamic, autonomous workflows that adjust based on task requirements
- **🔧 Multi-Column Prompt Variables**: Pass multiple dataset columns as prompt placeholders
  - Configure which columns feed into prompts (e.g., patient_id, age, gender, diagnosis)
  - Full backward compatibility with simple text-only workflows
- **🤖 Autonomous Task Execution**: LLM analyzes tasks, determines required tools, and orchestrates execution
  - Adaptive workflow adjusts strategy based on intermediate results
  - Intelligent tool selection and composition for complex clinical reasoning
- **🧮 Tool Integration**: Multiple tool types for comprehensive task support
  - **Functions**: Python functions for medical calculations (20+ built-in, fully customizable)
  - **RAG**: Clinical guidelines and publications from PDFs/URLs
  - **Extras**: Task-specific hints and domain knowledge (49+ clinical extras)
  - **Patterns**: Text preprocessing and postprocessing transformations (33+ patterns)
- **🎯 User-Driven Customization**: Complete control over platform behavior
  - Write custom functions, define patterns, provide extras via JSON/YAML
  - Selective tool enablement for each task
  - Configure workflow type based on task complexity
- **🤖 Multi-LLM Support**: OpenAI, Anthropic, Google, Azure, Unsloth (local models)
- **⚡ High Performance**: Parallel processing, caching, async tool execution (60-75% faster)
- **🚀 Multi-GPU Processing**: Automatic multi-GPU support for local models (2-4x faster on H100 clusters)
- **🖥️ Web Interface**: User-friendly Gradio UI with real-time progress tracking

---

## Installation

```bash
# From PyPI
pip install clinorchestra

# With local LLM support
pip install clinorchestra[local]

# From source
git clone https://github.com/gyasifred/clinorchestra.git
cd clinorchestra
pip install -e .
```

---

## Quick Start

```bash
# Launch application
clinorchestra
```

Web interface opens at `http://localhost:7860`

### Basic Workflow

1. **Model Setup**: Select LLM provider and configure API key
2. **Define Task**: Write extraction prompt and JSON schema
3. **Upload Data**: Load CSV file with clinical text
4. **Configure Tools**: Enable patterns, functions, extras, RAG documents
5. **Test**: Use Playground to test single extraction
6. **Process**: Batch process entire dataset

---

## System Architecture

ClinOrchestra features a **6-layer modular architecture** designed for autonomous clinical task orchestration, scalability, and production-ready performance.

### Architecture Overview

![System Architecture](assets/diagrams/overall_architecture.svg)

**The 6 Layers:**
1. **Web Interface** - Gradio-based UI with configuration tabs
2. **Application State** - Central configuration manager (Observer pattern)
3. **Orchestration Engines** - Dual workflow types (STRUCTURED & ADAPTIVE)
4. **Core Services** - LLM integration, RAG engine, preprocessing
5. **Tool Systems** - Functions, patterns, extras, RAG for autonomous task execution
6. **Optimization** - Caching (400x faster), parallel processing, adaptive retry

**For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)**

---

## Workflow Types

### STRUCTURED Workflows (Default)

Predefined 4-stage pipeline for predictable, production-ready task execution:

![STRUCTURED Workflow](assets/diagrams/structured_mode_workflow.svg)

```
Stage 1: Task Analysis → LLM determines required tools autonomously
Stage 2: Tool Execution → Execute functions, RAG, extras (async/parallel)
Stage 3: Task Completion → Generate structured output based on task requirements
Stage 4: RAG Refinement → Enhance specific fields with evidence (optional)
```

**Best for**: Predictable workflows, production deployments, maximum reliability

**Key Features**: Fixed sequence, deterministic execution, optimized for scale

### ADAPTIVE Workflows

Dynamic, autonomous workflows that adjust based on task requirements and intermediate results:

![ADAPTIVE Workflow](assets/diagrams/adaptive_mode_workflow.svg)

```
Continuous Autonomous Loop:
1. LLM analyzes task and current state
2. Decides which tools to call based on needs
3. Executes tools in parallel (async)
4. Analyzes results and adjusts strategy
5. Iterates until task objective achieved
6. Outputs final structured result
```

**Best for**: Complex clinical reasoning, evolving requirements, maximum flexibility

**Key Features**: Dynamic tool selection, adaptive strategy, self-correcting execution

Enable ADAPTIVE workflows in the Config tab with max iterations and tool call limits.

### Component Interactions

![Component Interactions](assets/diagrams/component_interactions.svg)

See how all components work together seamlessly through the Observer pattern, with AppState managing configuration and agents orchestrating LLM + Tools + RAG.

---

## Example

**JSON Schema:**
```json
{
  "diagnosis": {"type": "string", "required": true},
  "severity": {"type": "string"},
  "medications": {"type": "array"}
}
```

**Clinical Text:**
```
Patient presents with Type 2 DM, HbA1c 8.2%.
BP 145/92. Current meds: Metformin 1000mg BID.
```

**Output:**
```json
{
  "diagnosis": "Type 2 Diabetes Mellitus",
  "severity": "Uncontrolled (HbA1c 8.2%)",
  "medications": ["Metformin 1000mg twice daily"],
  "blood_pressure": "145/92 mmHg - Stage 2 Hypertension"
}
```

---

## Core Components

### Functions (20+ medical calculations)
- BMI, BSA, IBW calculations
- Growth percentiles and z-scores
- Lab value corrections (calcium, anion gap)
- Unit conversions (kg/lbs, cm/inches)
- Clinical scores (MAP, CrCl, HEART, TIMI, CAGE, NIHSS, KDIGO, GOLD)

### Patterns (33+ text normalizations)
- Vital signs: BP, HR, RR, temperature, SpO2
- Lab values: glucose, HbA1c, electrolytes
- Medications: dosing, frequency, routes
- Diagnosis abbreviations: DM→diabetes mellitus

### Extras (49+ clinical hints)
- Growth standards (WHO, CDC)
- Malnutrition criteria (ASPEN)
- Diagnostic criteria (diabetes, hypertension, sepsis)
- Assessment scales (APGAR, Glasgow, NYHA)
- Reference ranges and clinical norms

### RAG (Knowledge retrieval)
- Upload clinical guidelines (PDF/URL)
- Automatic chunking and embedding
- Similarity search during extraction
- Source citation in outputs

---

## Advanced Features

- **PHI Redaction**: Detect and redact protected health information
- **Multi-format Output**: Save redacted/normalized text alongside extractions
- **Error Handling**: Configurable retry strategies with intelligent fallback
- **Progress Tracking**: Real-time batch processing status
- **Configuration Persistence**: Auto-save all settings
- **Performance Monitoring**: Track extraction timing metrics
- **Prompt Variables**: Pass any dataset columns as template variables

---

## Use Cases (Platform is Universal - Not Limited to These)

The platform is **universal** and autonomously accomplishes any clinical task you define. Examples include:

- **Clinical Classification**: ADRD diagnosis, malnutrition assessment, disease staging
- **Data Extraction & Curation**: Create annotated datasets for AI/ML training
- **Chart Review Automation**: Process large medical record sets with structured outputs
- **Quality Improvement**: Measure guideline adherence across patient populations
- **Research Data Collection**: Extract research variables from clinical narratives
- **Clinical Decision Support**: Generate parameters and recommendations for CDS systems
- **Comprehensive Annotation**: Multi-evidence diagnostic reasoning for medical AI
- **Risk Stratification**: Calculate clinical scores and assess patient risk levels
- **Medication Reconciliation**: Extract, normalize, and reconcile medication lists
- **Clinical Trial Screening**: Assess eligibility criteria from patient charts
- **Adverse Event Detection**: Identify, classify, and report adverse events
- **Guideline Compliance**: Assess adherence to clinical practice guidelines

**The only limit is your task description** - ClinOrchestra provides the universal, autonomous infrastructure.

---

## Documentation

### Core Documentation
- **`SDK_GUIDE.md`**: **Comprehensive guide for programmatic usage (Python SDK)**
- `ARCHITECTURE.md`: System design and component overview
- `MULTI_GPU_GUIDE.md`: Multi-GPU processing for local models (H100 clusters)
- `README.md`: This file (quick start and overview)

### SDK vs UI Usage

**🖥️ Web UI** (Default): Launch with `clinorchestra` command - best for interactive exploration and testing

**🐍 Python SDK**: Import and use programmatically - best for:
- Integration into data pipelines
- Batch processing automation
- Custom applications
- Reproducible research workflows

**→ See [`SDK_GUIDE.md`](SDK_GUIDE.md) for complete programmatic usage examples**

### Examples (Demonstrating Universal Capability)
- `examples/malnutrition_classification/`: Malnutrition assessment example
- `examples/`: Additional sample datasets and use cases
- `mimic-iv/`: MIMIC-IV diagnosis annotation example (comprehensive clinical AI training)

### Evaluation
- `evaluation/`: Benchmarking and testing tools

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

MIT License

---

## Contact

- **GitHub**: https://github.com/gyasifred/clinorchestra
- **Email**: gyasi@musc.edu
- **Institution**: Medical University of South Carolina, Biomedical Informatics Center

---

## Acknowledgments

- HeiderLab, ClinicalNLP Lab, MUSC
- CDC (growth chart data)
- Clinical guideline organizations (WHO, ASPEN, ADA, ACC/AHA)

---

**Version**: 1.0.0 (Production Release)
**Platform Type**: Universal, Autonomous Clinical AI Platform for Task-Driven LLM Orchestration
**Author**: Frederick Gyasi
