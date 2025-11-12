# ClinOrchestra

**Universal LLM-Powered Clinical Data Extraction Platform**

ClinOrchestra is an intelligent system for extracting structured information from clinical text using large language models (LLMs), retrieval-augmented generation (RAG), custom functions, and task-specific hints.

![ClinOrchestra Logo](assets/clinorchestra_logo.svg)

---

## Features

- **Universal System**: Define any extraction task via prompts and JSON schemas
- **Dual Execution Modes**:
  - **STRUCTURED**: Reliable 4-stage pipeline for production workloads
  - **ADAPTIVE**: Autonomous iterative loop for complex cases
- **RAG Integration**: Retrieve clinical guidelines from PDFs and URLs
- **Custom Functions**: Medical calculations (BMI, growth percentiles, lab conversions)
- **Clinical Hints**: Pre-loaded domain knowledge (WHO, ASPEN, diagnostic criteria)
- **Pattern Normalization**: Standardize clinical abbreviations and formats
- **Multi-LLM Support**: OpenAI, Anthropic, Google, Azure, Unsloth (local)
- **Batch Processing**: Process large datasets with error handling
- **Web Interface**: User-friendly Gradio UI

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

## Execution Modes

### STRUCTURED Mode (Default)

Systematic 4-stage pipeline for reliable, production-ready extraction:

```
Stage 1: Task Analysis → Determine required tools
Stage 2: Tool Execution → Run functions, RAG, extras
Stage 3: Extraction → Generate structured JSON output
Stage 4: RAG Refinement → Enhance selected fields (optional)
```

### ADAPTIVE Mode

Autonomous iterative loop for complex extractions:

```
Continuous Loop:
1. LLM analyzes clinical text
2. Decides which tools to call
3. Executes tools in parallel
4. Analyzes results
5. Iterates until extraction complete
6. Outputs final JSON
```

Enable ADAPTIVE mode in the Config tab with max iterations and tool call limits.

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
- Clinical scores (MAP, CrCl)

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
- **Multi-format Output**: Save redacted/normalized text
- **Error Handling**: Configurable retry strategies
- **Progress Tracking**: Real-time batch processing status
- **Configuration Persistence**: Auto-save all settings
- **Performance Monitoring**: Track extraction timing metrics

---

## Use Cases

- **Clinical Data Curation**: Create annotated datasets for AI/ML
- **Chart Review**: Extract information from large record sets
- **Quality Improvement**: Measure guideline adherence
- **Research**: Extract variables from clinical narratives
- **Decision Support**: Extract parameters for clinical alerts

---

## Documentation

### Architecture
- `ARCHITECTURE.md`: System design and component overview

### Examples
- `examples/`: Sample datasets and use cases
- `scripts/`: Utility scripts for setup

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

**Version**: 1.0.0
**Author**: Frederick Gyasi
