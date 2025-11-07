# ClinOrchestra

**🎯 Truly Universal Platform for Clinical Data Extraction & Orchestration**

ClinOrchestra is an intelligent, LLM-powered platform for extracting structured information from **ANY clinical task** using agentic orchestration with RAG, custom functions, and task-specific hints.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🌟 UNIVERSAL SYSTEM - Not Task-Specific!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**This system works for ANY clinical extraction task:**

✅ **Nutritional:** Malnutrition, obesity, feeding disorders
✅ **Metabolic:** Diabetes, thyroid, metabolic syndrome
✅ **Cardiovascular:** Hypertension, heart failure, arrhythmias
✅ **Renal:** AKI, CKD, dialysis
✅ **Infectious:** Sepsis, pneumonia, UTI
✅ **Oncology:** Cancer staging, treatment response
✅ **Medications:** Drug lists, adverse events, adherence
✅ **Social:** SDOH, living conditions, support systems
✅ **YOUR CUSTOM TASK:** Define via prompts and JSON schema!

**How it's universal:**
- You define the task via **prompts** and **JSON schema**
- The LLM makes **independent decisions** based on YOUR task
- Built-in templates (malnutrition, diabetes) are **examples** to learn from
- System orchestrates tools to extract **YOUR data**, not predefined data

**No hardcoded tasks!** The framework adapts to whatever you define.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🚀 Quick Start

```bash
# Install
pip install clinorchestra

# Launch
clinorchestra
```

Web interface opens at `http://localhost:7860`

## ✨ Key Features

- **🤖 Dual Execution Modes**:
  - **STRUCTURED Mode** (ExtractionAgent v1.0.0): Reliable 4-stage pipeline
  - **ADAPTIVE Mode** (AgenticAgent v1.0.0): Continuous loop with autonomous tool calling + async/parallel execution (60-75% faster)
- **📚 RAG Integration**: Retrieve clinical guidelines and evidence from PDFs/URLs
- **🧮 Custom Functions**: Medical calculations (BMI, conversions, growth percentiles, etc.)
- **💡 Extras (Hints)**: 49+ pre-loaded clinical hints (WHO, ASPEN, diagnostic criteria)
- **🔄 Pattern Normalization**: 33+ regex patterns for standardizing clinical text
- **🎯 Multi-LLM Support**: OpenAI, Anthropic, Google, Azure, Unsloth (local)
- **📊 Batch Processing**: Process large datasets with error handling
- **🎨 Web UI**: User-friendly Gradio interface

## 📖 Documentation

### Installation

```bash
# From PyPI
pip install clinorchestra

# With local LLM support
pip install clinorchestra[local]

# From source
git clone https://github.com/yourusername/clinorchestra.git
cd clinorchestra
pip install -e .
```

### Execution Modes

ClinOrchestra supports TWO execution modes:

#### STRUCTURED Mode (v1.0.0) - Default
**Systematic 4-stage pipeline** - Best for: Production workloads, predictable behavior

```
STAGE 1: TASK ANALYSIS
├── Analyzes extraction task and clinical text
├── Determines required tools and parameters
└── Generates intelligent queries

STAGE 2: TOOL EXECUTION
├── Executes medical calculation functions
├── Retrieves relevant guidelines (RAG)
└── Matches task-specific hints (Extras)

STAGE 3: EXTRACTION
├── LLM extracts structured JSON
└── Uses preprocessed text + tool results

STAGE 4: RAG REFINEMENT (Optional)
├── Refines selected fields with RAG evidence
└── Cites specific sources
```

#### ADAPTIVE Mode (v1.0.0) - Advanced
**Continuous autonomous loop** - Best for: Complex cases, evolving tasks, maximum accuracy

```
CONTINUOUS LOOP
├── LLM analyzes clinical text
├── Autonomously decides which tools to call
├── PAUSE → Execute tools in PARALLEL (async/await)
├── RESUME → Analyze results
├── Refine queries and call more tools if needed
├── Iterate until extraction complete
└── Output final JSON

Performance: 60-75% faster due to parallel tool execution
```

**Enable ADAPTIVE Mode:**
```python
app_state.set_agentic_config(
    enabled=True,
    max_iterations=20,
    max_tool_calls=50
)
```

See [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) for detailed comparison and [AGENTIC_USER_GUIDE.md](AGENTIC_USER_GUIDE.md) for usage instructions.

### Core Components

**Functions** - Medical calculations (20+ examples):
- `calculate_bmi`, `kg_to_lbs`, `calculate_growth_percentile`
- `calculate_mean_arterial_pressure`, `calculate_bsa`
- `calculate_corrected_calcium`, `calculate_anion_gap`
- Unit conversions, weight change calculations, and more

**Patterns** - Text normalization (33+ examples):
- Vital signs: BP, HR, RR, temperature, SpO2
- Lab values: glucose, HbA1c, electrolytes, CBC
- Medications: dosing, frequency (BID, TID), routes (PO, IV)
- Diagnosis expansions: DM→diabetes mellitus, HTN→hypertension

**Extras** - Clinical hints (49+ examples):
- Growth standards (WHO, CDC)
- Malnutrition criteria (ASPEN)
- Diagnostic criteria (diabetes, hypertension, CKD, sepsis)
- Assessment scales (APGAR, Glasgow Coma, NYHA, qSOFA)
- Laboratory reference ranges, vital sign norms

**RAG** - Knowledge retrieval:
- Upload clinical guidelines (PDFs, URLs)
- Automatic chunking and embedding
- Similarity search during extraction
- Source citation in outputs

### Configuration

1. **Model Setup**: Choose LLM provider, enter API key, set parameters
2. **Define Task**: Write extraction prompt and JSON schema
3. **Prepare Data**: Upload CSV with clinical text
4. **Configure Tools**: Enable patterns, add functions/extras, upload RAG documents
5. **Test**: Use Playground for single extractions
6. **Process**: Batch process entire dataset

### Example Use Case

```python
# JSON Schema
{
  "diagnosis": {"type": "string", "required": true},
  "severity": {"type": "string", "required": false},
  "symptoms": {"type": "array", "required": false}
}

# Clinical Text
"Patient presents with Type 2 DM, HbA1c 8.2%. BP 145/92. Current meds: Metformin 1000mg BID."

# Output
{
  "diagnosis": "Type 2 Diabetes Mellitus",
  "severity": "Uncontrolled (HbA1c 8.2%)",
  "symptoms": ["Hyperglycemia"],
  "blood_pressure": "145/92 mmHg - Stage 2 Hypertension",
  "medications": ["Metformin 1000mg twice daily"]
}
```

## 🔧 Advanced Features

- **PHI Redaction**: Auto-detect and redact protected health information
- **Temporal Tracking**: Capture trends and changes over time
- **Multi-format Output**: Save redacted/normalized text alongside extractions
- **Error Handling**: Configurable retry strategies
- **Progress Tracking**: Real-time progress monitoring
- **Configuration Persistence**: Auto-save all settings

## 📚 Example Medical Functions

Run the included script to create 20+ functions:

```bash
python scripts/create_medical_examples.py
```

Includes: BMI, BSA, IBW, anion gap, corrected calcium, CrCl, eGFR, fluid requirements, calorie/protein requirements, QTc interval, pack-years, and more.

## 🎯 Use Cases

1. **Clinical Data Curation**: Create annotated training datasets for AI
2. **Chart Review**: Extract specific information from large record sets
3. **Quality Improvement**: Measure guideline adherence, identify gaps
4. **Research**: Extract variables from clinical narratives
5. **Clinical Decision Support**: Extract parameters for alerts/recommendations

## 🤝 Contributing

Contributions welcome! Please fork, create feature branch, add tests, and submit PR.

## 📄 License

MIT License

## 📧 Contact

- **Issues**: https://github.com/gyasifred/clinannotate/issues
- **Email**: gyasi@musc.edu
- **Institution**: Medical University of South Carolina, Biomedical Informatics Center

## 🙏 Acknowledgments

- HeiderLab, ClinicalNLP Lab, MUSC
- CDC (growth chart data)
- Clinical guideline organizations (WHO, ASPEN, ADA, ACC/AHA)

---

**Version**: 1.0.0 | **Author**: Frederick Gyasi
