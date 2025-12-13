# ClinOrchestra - Universal Clinical AI Platform

**YAML-Based Clinical Data Extraction with Autonomous LLM Orchestration**

ClinOrchestra is a universal clinical AI platform that enables autonomous language model orchestration for any clinical task. The platform uses **YAML configuration** for all components (functions, patterns, extras) and integrates multiple tool types that LLMs leverage autonomously.

![ClinOrchestra Logo](assets/clinorchestra_logo.svg)

---

## Key Features

- **Universal Platform**: Accomplishes ANY clinical task through autonomous LLM orchestration
  - Task-driven by user-defined prompts and schemas
  - Pre-configured tasks: Malnutrition, ADRD, MIMIC-IV critical care
  - Fully customizable with YAML configurations

- **Dual Workflow Types**:
  - **STRUCTURED**: Predefined 4-stage pipeline for production tasks
  - **ADAPTIVE**: Dynamic workflows that adjust based on task requirements

- **YAML-Only Configuration**: Clean, organized, human-readable
  - Task-specific YAMLs in `yaml_configs/`
  - Example templates in `examples/yaml_templates/`
  - Import via UI → Creates individual `.yaml` files

- **Tool Integration**:
  - **Functions**: Python calculations (z-scores, BMI, clinical scores)
  - **RAG**: Clinical guidelines and publications
  - **Extras**: Task-specific clinical knowledge
  - **Patterns**: Text normalization

- **Multi-LLM Support**: OpenAI, Anthropic, Google, Azure, local models
- **High Performance**: Parallel processing, caching, async execution
- **Web Interface**: Gradio UI with real-time tracking

---

## Installation

```bash
# From PyPI
pip install clinorchestra

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

### Getting Started with Pre-Configured Tasks

**1. Import Task-Specific YAMLs:**

For **Malnutrition Assessment**:
```
1. Functions Tab → Import → yaml_configs/malnutrition_functions.yaml
2. Patterns Tab → Import → yaml_configs/malnutrition_patterns.yaml
3. Extras Tab → Import → yaml_configs/malnutrition_extras.yaml
```

For **ADRD/Cognitive Assessment**:
```
1. Functions Tab → Import → yaml_configs/adrd_functions.yaml
2. Patterns Tab → Import → yaml_configs/adrd_patterns.yaml
3. Extras Tab → Import → yaml_configs/adrd_extras.yaml
```

**2. Use Shared/Common Resources** (optional but recommended):
```
Functions Tab → Import → yaml_configs/shared_functions.yaml
Patterns Tab → Import → yaml_configs/shared_patterns.yaml
Extras Tab → Import → yaml_configs/shared_extras.yaml
```

**3. Process Your Data:**
- Go to Processing Tab
- Upload CSV with clinical notes
- Define extraction task and schema
- Run extraction

---

## System Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    YAML Configuration                        │
│  yaml_configs/                                              │
│  ├── malnutrition_functions.yaml (13 functions)            │
│  ├── malnutrition_patterns.yaml (10 patterns)              │
│  ├── malnutrition_extras.yaml (90 extras)                  │
│  └── ... (ADRD, MIMIC-IV, shared)                          │
└─────────────────────────────────────────────────────────────┘
                          ↓ Import via UI
┌─────────────────────────────────────────────────────────────┐
│                  Individual YAML Files                       │
│  functions/                                                 │
│  ├── calculate_zscore.yaml                                 │
│  ├── calculate_bmi.yaml                                    │
│  └── ...                                                    │
│  patterns/                                                  │
│  ├── normalize_bp.yaml                                     │
│  └── ...                                                    │
│  extras/                                                    │
│  ├── aspen_criteria.yaml                                   │
│  └── ...                                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓ Loaded by
┌─────────────────────────────────────────────────────────────┐
│                      Managers                               │
│  • FunctionRegistry (loads .yaml functions)                │
│  • RegexPreprocessor (loads .yaml patterns)                │
│  • ExtrasManager (loads .yaml extras)                      │
└─────────────────────────────────────────────────────────────┘
                          ↓ Used by
┌─────────────────────────────────────────────────────────────┐
│                 Orchestration Engines                       │
│  • STRUCTURED Pipeline (4 stages)                          │
│  • ADAPTIVE Pipeline (dynamic)                             │
└─────────────────────────────────────────────────────────────┘
                          ↓ Produces
┌─────────────────────────────────────────────────────────────┐
│                  Extraction Results                         │
│  Structured JSON matching your schema                       │
└─────────────────────────────────────────────────────────────┘
```

### Component Flow

1. **User imports task-specific YAML** via UI
2. **System creates individual `.yaml` files** for each function/pattern/extra
3. **Managers load on startup** from individual YAML files
4. **LLM uses tools** during extraction based on task requirements
5. **Results extracted** matching defined schema

---

## Available Tasks

### Malnutrition Assessment (13 functions, 10 patterns, 90 extras)
- ASPEN Pediatric Malnutrition Criteria
- WHO Z-Score Classification
- Growth velocity calculations
- Anthropometric assessments

### ADRD/Cognitive Assessment (3 functions, 12 patterns, 22 extras)
- CDR (Clinical Dementia Rating) scoring
- MMSE interpretation
- NIA-AA diagnostic criteria
- Biomarker assessments

### MIMIC-IV Critical Care (3 functions, 20 extras)
- SOFA score calculation
- KDIGO AKI staging
- Critical care protocols

### Shared Utilities (25 functions, 149 patterns, 53 extras)
- BMI, age, unit conversions
- Vital signs normalization
- Common medical abbreviations
- Lab value standardization

---

## Creating Custom Tasks

See `examples/yaml_templates/README.md` for comprehensive guides on creating:
- Custom functions
- Text normalization patterns
- Clinical knowledge extras

**Example templates provided:**
- `functions_example.yaml` - 5 complete function examples with documentation
- `patterns_example.yaml` - 40+ pattern examples
- `extras_example.yaml` - Clinical knowledge examples

---

## Configuration

### YAML Format

**Functions:**
```yaml
name: calculate_bmi
description: "Calculate Body Mass Index"
enabled: true
code: |
  def calculate_bmi(weight_kg, height_m):
      return round(weight_kg / (height_m ** 2), 2)
parameters:
  weight_kg:
    type: number
    required: true
returns: "BMI value"
```

**Patterns:**
```yaml
name: normalize_bp
pattern: '\b(BP)\s*(\d+)/(\d+)\b'
replacement: 'BP \2/\3'
description: "Standardize blood pressure"
enabled: true
```

**Extras:**
```yaml
id: aspen_criteria
name: "ASPEN Malnutrition Criteria"
type: criteria
content: |
  ASPEN Pediatric Malnutrition requires ≥2 of:
  1. Insufficient energy intake
  2. Weight loss or deceleration
keywords:
  - malnutrition
  - ASPEN
metadata:
  category: malnutrition
  priority: high
```

---

## Documentation

- **Main README**: This file
- **YAML Configs Guide**: `yaml_configs/README.md` - How to import task-specific YAMLs
- **Template Examples**: `examples/yaml_templates/README.md` - How to create custom YAMLs
- **Evaluation Guide**: `evaluation/README.md` - Testing and benchmarking

---

## Project Structure

```
clinorchestra/
├── yaml_configs/              # Task-specific YAMLs to import
│   ├── README.md
│   ├── malnutrition_*.yaml
│   ├── adrd_*.yaml
│   ├── mimic_iv_*.yaml
│   └── shared_*.yaml
├── examples/
│   └── yaml_templates/        # Example templates for creating new tasks
│       ├── README.md
│       ├── functions_example.yaml
│       ├── patterns_example.yaml
│       └── extras_example.yaml
├── core/                      # Core managers and engines
│   ├── function_registry.py   # Loads .yaml functions
│   ├── regex_preprocessor.py  # Loads .yaml patterns
│   ├── extras_manager.py      # Loads .yaml extras
│   ├── agent_system.py        # STRUCTURED pipeline
│   └── agentic_agent.py       # ADAPTIVE pipeline
├── ui/                        # Gradio UI
│   ├── functions_tab.py       # Import/manage functions
│   ├── patterns_tab.py        # Import/manage patterns
│   └── extras_tab.py          # Import/manage extras
├── functions/                 # Individual function YAMLs (created on import)
├── patterns/                  # Individual pattern YAMLs (created on import)
└── extras/                    # Individual extra YAMLs (created on import)
```

---

## Citation

If you use ClinOrchestra in your research, please cite:

```bibtex
@software{clinorchestra2024,
  title={ClinOrchestra: Universal Clinical AI Platform},
  author={Gyasi, Frederick},
  year={2024},
  url={https://github.com/gyasifred/clinorchestra}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Issues**: https://github.com/gyasifred/clinorchestra/issues
- **Documentation**: See `yaml_configs/README.md` and `examples/yaml_templates/README.md`
- **Email**: gyasi@musc.edu

---

**ClinOrchestra** - Universal, YAML-based clinical AI for any task
