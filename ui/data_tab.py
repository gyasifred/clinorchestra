"""
Data Configuration Tab for ClinOrchestra - COMPLETE VERSION
Supports multiple label mappings and proper state management

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: HeiderLab
Version: 1.0.0
"""

import gradio as gr
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def create_data_tab(app_state) -> Dict[str, Any]:
    """
    Create complete data configuration tab.
    
    Args:
        app_state: Application state manager
        
    Returns:
        Dictionary of Gradio components
    """
    
    components = {}
    
    gr.Markdown("### Data Input Configuration")
    
    # FILE INPUT SECTION
    with gr.Accordion("📁 File Input", open=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Upload CSV File")
                
                file_upload = gr.File(
                    label="CSV File",
                    file_types=[".csv"],
                    type="filepath"
                )
                components['file_upload'] = file_upload
                
                upload_status = gr.Textbox(label="Status", interactive=False)
                components['upload_status'] = upload_status
            
            with gr.Column():
                gr.Markdown("#### Or Provide Path")
                
                file_path = gr.Textbox(
                    label="File Path",
                    placeholder="/path/to/data.csv"
                )
                components['file_path'] = file_path
                
                load_path_btn = gr.Button("Load", variant="primary")
                components['load_path_btn'] = load_path_btn
    
    # DATA PREVIEW SECTION
    with gr.Accordion("📊 Data Preview", open=True):
        data_preview = gr.Dataframe(
            label="First 10 Rows",
            row_count=(1, "dynamic")
        )
        components['data_preview'] = data_preview
        
        data_info = gr.Textbox(label="Dataset Info", interactive=False)
        components['data_info'] = data_info
    
    # COLUMN CONFIGURATION SECTION
    with gr.Accordion("🔧 Column Configuration", open=True):
        available_columns = gr.State(value=[])
        components['available_columns'] = available_columns
        
        gr.Markdown("""
        **Column Selection Guide:**
        - **Text Column**: The clinical text to analyze (REQUIRED)
        - **Identifier Columns**: Patient IDs, encounter numbers, etc. (preserved in output)
        - **Additional Columns**: Any other columns to include in final output
        """)
        
        with gr.Row():
            text_column = gr.Dropdown(
                choices=[],
                label="Text Column (REQUIRED)",
                info="Column containing clinical text to analyze"
            )
            components['text_column'] = text_column
        
        with gr.Row():
            deid_columns = gr.CheckboxGroup(
                choices=[],
                label="Identifier Columns",
                info="Columns to preserve in output (IDs, patient numbers, etc.)"
            )
            components['deid_columns'] = deid_columns
        
        with gr.Row():
            additional_columns = gr.CheckboxGroup(
                choices=[],
                label="Additional Columns",
                info="Other columns to include in final output"
            )
            components['additional_columns'] = additional_columns
    
    # LABEL CONFIGURATION SECTION
    with gr.Accordion("🏷️ Label Configuration", open=False):
        gr.Markdown("""
        **Label Mapping:**
        
        If your data has labels (e.g., diagnosis codes, categories), you can provide 
        context for each label to help the LLM understand what they mean.
        
        **Important:** You can add mappings for ALL your label categories. The system
        will store all mappings and apply them during processing.
        """)
        
        has_labels = gr.Checkbox(
            label="Dataset Has Labels",
            value=False
        )
        components['has_labels'] = has_labels
        
        with gr.Column(visible=False) as label_config_panel:
            label_column = gr.Dropdown(
                choices=[],
                label="Label Column",
                info="Column containing label values"
            )
            components['label_column'] = label_column
            
            gr.Markdown("#### Analyze Labels")
            
            analyze_labels_btn = gr.Button("📊 Analyze Label Column", variant="primary")
            components['analyze_labels_btn'] = analyze_labels_btn
            
            label_analysis = gr.TextArea(
                label="Label Analysis",
                lines=10,
                interactive=False
            )
            components['label_analysis'] = label_analysis
            
            unique_labels_state = gr.State(value=[])
            components['unique_labels_state'] = unique_labels_state
            
            gr.Markdown("---")
            gr.Markdown("#### Add Label Mappings")
            gr.Markdown("**Add meaning/context for each label value - you can map ALL labels**")
            
            with gr.Row():
                select_label = gr.Dropdown(
                    choices=[],
                    label="Select Label Value",
                    allow_custom_value=True
                )
                components['select_label'] = select_label
                
                label_meaning = gr.TextArea(
                    label="Label Meaning/Context",
                    placeholder="Describe what this label means...",
                    lines=4,
                    info="This context helps the LLM understand the label"
                )
                components['label_meaning'] = label_meaning
            
            with gr.Row():
                add_mapping_btn = gr.Button("➕ Add/Update Mapping", variant="primary")
                clear_single_mapping_btn = gr.Button("🗑️ Clear This Mapping")
                clear_all_mappings_btn = gr.Button("🗑️ Clear All Mappings", variant="stop")
            
            components['add_mapping_btn'] = add_mapping_btn
            components['clear_single_mapping_btn'] = clear_single_mapping_btn
            components['clear_all_mappings_btn'] = clear_all_mappings_btn
            
            # State to hold ALL mappings (FIXED: was losing mappings)
            current_label_mappings = gr.State(value={})
            components['current_label_mappings'] = current_label_mappings
            
            mapping_status = gr.Textbox(label="Mapping Status", interactive=False)
            components['mapping_status'] = mapping_status
            
            gr.Markdown("#### Current Mappings")
            gr.Markdown("**All defined label mappings will be shown below:**")
            
            mappings_display = gr.Dataframe(
                headers=["Label Value", "Meaning/Context"],
                datatype=["str", "str"],
                label="Defined Mappings (All Categories)",
                interactive=False,
                row_count=(1, "dynamic")
            )
            components['mappings_display'] = mappings_display
            
            with gr.Row():
                save_mappings_btn = gr.Button(
                    "💾 Save All Mappings to Configuration",
                    variant="primary",
                    size="lg"
                )
                export_mappings_btn = gr.Button("📥 Export Mappings to JSON")

            components['save_mappings_btn'] = save_mappings_btn
            components['export_mappings_btn'] = export_mappings_btn

            download_mappings = gr.File(
                label="Download Mappings JSON",
                interactive=False,
                visible=False
            )
            components['download_mappings'] = download_mappings

            gr.Markdown("---")
            gr.Markdown("#### Import Label Mappings")
            gr.Markdown("**Upload a YAML or JSON file with label mappings to quickly populate all mappings**")

            with gr.Row():
                import_mappings_file = gr.File(
                    label="Upload Mappings File (YAML or JSON)",
                    file_types=[".yaml", ".yml", ".json"],
                    type="filepath"
                )
                components['import_mappings_file'] = import_mappings_file

            import_status = gr.Textbox(label="Import Status", interactive=False)
            components['import_status'] = import_status
        
        components['label_config_panel'] = label_config_panel
    
    # PHI REDACTION SECTION
    with gr.Accordion("🔒 PHI/PII Redaction", open=False):
        gr.Markdown("""
        **Protected Health Information (PHI) Redaction:**
        
        Automatically detect and redact sensitive information before LLM processing.
        Uses regex patterns and NLP models for comprehensive PHI removal.
        
        **What Gets Saved:**
        - `{text_column}`: Always the ORIGINAL unredacted text
        - `{text_column}_redacted`: Redacted version (only if you enable saving below)
        - LLM processes the redacted text to protect privacy
        - Final output includes both original and redacted versions
        """)
        
        enable_phi_redaction = gr.Checkbox(
            label="Enable PHI/PII Redaction",
            value=False,
            info="Redact sensitive information before LLM processing"
        )
        components['enable_phi_redaction'] = enable_phi_redaction
        
        with gr.Column(visible=False) as phi_config_panel:
            phi_entity_types = gr.CheckboxGroup(
                choices=[
                    "PERSON", "LOCATION", "DATE", "AGE",
                    "PHONE", "EMAIL", "SSN", "MEDICAL_RECORD",
                    "ID", "HOSPITAL"
                ],
                value=["PERSON", "DATE", "PHONE", "EMAIL", "SSN", "MEDICAL_RECORD"],
                label="Entity Types to Redact"
            )
            components['phi_entity_types'] = phi_entity_types
            
            redaction_method = gr.Radio(
                choices=["Replace with tag", "Replace with placeholder", "Remove"],
                value="Replace with tag",
                label="Redaction Method",
                info="How to handle detected PHI"
            )
            components['redaction_method'] = redaction_method
            
            save_redacted_text = gr.Checkbox(
                label="Save Redacted Text Column in Output",
                value=True,
                info="Include a '_redacted' column with PHI removed text"
            )
            components['save_redacted_text'] = save_redacted_text
            
            gr.Markdown("#### Test Redaction")
            
            phi_test_text = gr.TextArea(
                label="Test Text",
                placeholder="Enter clinical text to test PHI redaction...",
                lines=4
            )
            components['phi_test_text'] = phi_test_text
            
            test_phi_btn = gr.Button("Test Redaction")
            components['test_phi_btn'] = test_phi_btn
            
            with gr.Row():
                phi_test_result = gr.TextArea(
                    label="Redacted Output",
                    lines=4,
                    interactive=False
                )
                phi_test_summary = gr.TextArea(
                    label="Summary",
                    lines=4,
                    interactive=False
                )
            
            components['phi_test_result'] = phi_test_result
            components['phi_test_summary'] = phi_test_summary
        
        components['phi_config_panel'] = phi_config_panel
    
    # PATTERN NORMALIZATION SECTION
    with gr.Accordion("🔤 Regex Pattern Normalization", open=False):
        gr.Markdown("""
        **Text Preprocessing with Regex Patterns:**
        
        Apply regex patterns (from Patterns tab) to normalize text BEFORE LLM processing.
        This fixes common formatting issues that might confuse the LLM.
        
        **What Gets Saved:**
        - `{text_column}`: Always the ORIGINAL text (before pattern application)
        - `{text_column}_normalized`: Pattern-normalized version (only if you enable saving below)
        - LLM processes the normalized text for better extraction
        - You control whether to save the normalized version
        
        **Note:** Normalization happens AFTER PHI redaction if both are enabled.
        """)
        
        enable_pattern_normalization = gr.Checkbox(
            label="Apply Regex Pattern Normalization",
            value=True,
            info="Apply patterns from Patterns tab before LLM processing"
        )
        components['enable_pattern_normalization'] = enable_pattern_normalization
        
        save_normalized_text = gr.Checkbox(
            label="Save Normalized Text Column in Output",
            value=False,
            info="Include a '_normalized' column showing pattern-applied text"
        )
        components['save_normalized_text'] = save_normalized_text
        
        gr.Markdown("""
        **Processing Flow:**
        ```
        Original Text → [PHI Redaction] → [Pattern Normalization] → LLM Processing
                             ↓              ↓                    ↓
                       Always Saved   Optional Save      Optional Save
        ```
        """)
    
    # OUTPUT COLUMN REFERENCE
    gr.Markdown("---")
    gr.Markdown("### Output Column Reference")
    
    with gr.Accordion("📋 What Columns Will Be in My Output?", open=False):
        gr.Markdown("""
        **Your Final CSV Output Will Contain:**

        **1. Identifier Columns** (from "Identifier Columns" selection above)
        - All columns you marked as identifiers
        - Examples: patient_id, encounter_id, mrn

        **2. Original Text Column** (from "Text Column" selection)
        - Your original clinical text (ALWAYS saved)
        - Column name: Whatever you selected as Text Column

        **3. Optional Processed Text Columns** (based on your settings)
        - `{text_column}_redacted`: If PHI redaction enabled AND you chose to save it
        - `{text_column}_normalized`: If pattern normalization enabled AND you chose to save it

        **4. Label Information** (if labels enabled)
        - `input_label_value`: The original label value from your dataset
        - `label_context_used`: The meaning/context text you defined for that label

        **5. Extraction Results** (from LLM)
        - `stage1_{field_name}`: For each field in your JSON schema
        - Example: If you defined "diagnosis" field, you'll get "stage1_diagnosis"

        **6. Additional Columns** (from "Additional Columns" selection)
        - Any other columns you selected to include

        **7. Processing Metadata**
        - `processing_timestamp`: When this row was processed
        - `llm_provider`: Which LLM was used (e.g., "openai")
        - `llm_model`: Which model was used (e.g., "gpt-4o")
        - `extras_used`: Number of extras/hints used
        - `rag_used`: Number of RAG documents used

        **Example Output Structure:**
        ```
        patient_id, clinical_note, clinical_note_redacted, input_label_value,
        label_context_used, stage1_diagnosis, stage1_medications, stage1_vitals,
        processing_timestamp, llm_provider, llm_model
        ```
        """)
    
    # VALIDATION SECTION
    gr.Markdown("---")
    gr.Markdown("### Validate Configuration")
    
    validate_btn = gr.Button("✅ Validate Configuration", variant="primary", size="lg")
    components['validate_btn'] = validate_btn
    
    validation_result = gr.TextArea(
        label="Validation Result",
        lines=15,
        interactive=False
    )
    components['validation_result'] = validation_result
    
    # EVENT HANDLERS
    
    def load_file(file_path_str):
        """Load CSV file"""
        if not file_path_str:
            return (
                "No file", gr.update(value=None), "No data",
                [], gr.update(choices=[]), gr.update(choices=[]),
                gr.update(choices=[]), gr.update(choices=[])
            )
        
        try:
            df = pd.read_csv(file_path_str)
            n_rows, n_cols = df.shape
            columns = df.columns.tolist()
            
            info = f"Loaded: {n_rows:,} rows x {n_cols} columns"
            preview = df.head(10)
            
            return (
                f"✅ Loaded {n_rows:,} rows",
                gr.update(value=preview),
                info,
                columns,
                gr.update(choices=columns, value=None),
                gr.update(choices=columns, value=[]),
                gr.update(choices=columns, value=[]),
                gr.update(choices=columns, value=None)
            )
        except Exception as e:
            return (
                f"❌ Error: {str(e)}", gr.update(value=None), "Failed",
                [], gr.update(choices=[]), gr.update(choices=[]),
                gr.update(choices=[]), gr.update(choices=[])
            )
    
    def analyze_labels(file_upload_path, file_path_str, label_col):
        """Analyze unique labels in the selected column"""
        if not label_col:
            return "❌ Select a label column first", [], gr.update(choices=[])
        
        input_file = file_upload_path if file_upload_path else file_path_str
        
        if not input_file:
            return "❌ Load a data file first", [], gr.update(choices=[])
        
        try:
            df = pd.read_csv(input_file)
            
            if label_col not in df.columns:
                return f"❌ Column '{label_col}' not found in dataset", [], gr.update(choices=[])
            
            unique_vals = df[label_col].dropna().unique().tolist()
            
            # Sort unique values
            try:
                unique_vals = sorted(unique_vals)
            except:
                unique_vals = sorted([str(v) for v in unique_vals])
            
            analysis = f"✅ Found {len(unique_vals)} unique label values:\n\n"
            
            # Show distribution
            for val in unique_vals[:50]:
                count = len(df[df[label_col] == val])
                percentage = (count / len(df)) * 100
                analysis += f"  '{val}': {count} occurrences ({percentage:.1f}%)\n"
            
            if len(unique_vals) > 50:
                analysis += f"\n  ... and {len(unique_vals) - 50} more labels\n"
            
            analysis += f"\n💡 You can now map ALL {len(unique_vals)} labels below"
            
            return analysis, unique_vals, gr.update(choices=unique_vals, value=unique_vals[0] if unique_vals else None)
        
        except Exception as e:
            logger.error(f"Label analysis failed: {e}")
            return f"❌ Error: {str(e)}", [], gr.update(choices=[])
    
    def add_mapping(label_val, label_mean, current_mappings):
        """Add or update a single label mapping"""
        if not label_val or not label_mean:
            return "❌ Both label value and meaning are required", current_mappings, gr.update()
        
        if not label_mean.strip():
            return "❌ Label meaning cannot be empty", current_mappings, gr.update()
        
        # Create a copy to preserve all existing mappings
        new_mappings = dict(current_mappings) if current_mappings else {}
        
        # Add or update this mapping
        label_key = str(label_val)
        new_mappings[label_key] = label_mean.strip()
        
        # Update display
        display_data = [
            [key, value[:200] if len(value) > 200 else value] 
            for key, value in sorted(new_mappings.items())
        ]
        
        action = "Updated" if label_key in current_mappings else "Added"
        status = f"✅ {action} mapping for '{label_val}'\n\n"
        status += f"Total mappings defined: {len(new_mappings)}"
        
        logger.info(f"Label mapping {action.lower()}: {label_key} -> {label_mean[:50]}...")
        
        return status, new_mappings, gr.update(value=display_data)
    
    def clear_single_mapping(label_val, current_mappings):
        """Clear a specific label mapping"""
        if not label_val:
            return "❌ Select a label value to clear", current_mappings, gr.update()
        
        label_key = str(label_val)
        
        if label_key in current_mappings:
            new_mappings = dict(current_mappings)
            del new_mappings[label_key]
            
            display_data = [
                [key, value[:200] if len(value) > 200 else value] 
                for key, value in sorted(new_mappings.items())
            ]
            
            status = f"✅ Cleared mapping for '{label_val}'\n\n"
            status += f"Remaining mappings: {len(new_mappings)}"
            
            logger.info(f"Label mapping cleared: {label_key}")
            
            return status, new_mappings, gr.update(value=display_data)
        else:
            return f"⚠️ No mapping found for '{label_val}'", current_mappings, gr.update()
    
    def clear_all_mappings():
        """Clear all label mappings"""
        empty_mappings = {}
        logger.info("All label mappings cleared")
        
        return (
            "✅ All mappings cleared", 
            empty_mappings, 
            gr.update(value=[])
        )
    
    def load_existing_meaning(selected_label, current_mappings):
        """Load existing meaning for selected label"""
        if selected_label and current_mappings and str(selected_label) in current_mappings:
            return current_mappings[str(selected_label)]
        else:
            return ""
    
    def save_mappings(current_mappings):
        """Save all mappings to app state configuration"""
        if not current_mappings or len(current_mappings) == 0:
            return "❌ No mappings defined. Add at least one mapping first."
        
        try:
            # Save to app state
            success = app_state.set_label_mappings(current_mappings)
            
            if success:
                status = f"✅ Saved {len(current_mappings)} label mappings to configuration!\n\n"
                status += "Mapped labels:\n"
                for key in sorted(list(current_mappings.keys())[:10]):
                    status += f"  - {key}\n"
                if len(current_mappings) > 10:
                    status += f"  ... and {len(current_mappings) - 10} more\n"
                
                logger.info(f"Saved {len(current_mappings)} label mappings to app state")
                return status
            else:
                return "❌ Failed to save mappings to configuration"
                
        except Exception as e:
            logger.error(f"Failed to save label mappings: {e}")
            return f"❌ Error saving mappings: {str(e)}"
    
    def export_mappings(current_mappings):
        """Export mappings to JSON file"""
        if not current_mappings:
            return "❌ No mappings to export", gr.update(visible=False)

        try:
            import json
            from datetime import datetime
            from pathlib import Path

            output_dir = Path("./configs")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"label_mappings_{timestamp}.json"

            with open(filepath, 'w') as f:
                json.dump(current_mappings, f, indent=2)

            logger.info(f"Label mappings exported to {filepath}")

            return f"✅ Exported {len(current_mappings)} mappings to:\n{filepath}", gr.update(value=str(filepath), visible=True)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return f"❌ Export failed: {str(e)}", gr.update(visible=False)

    def import_mappings(filepath, current_mappings):
        """Import mappings from YAML or JSON file"""
        if not filepath:
            return "❌ No file uploaded", current_mappings, gr.update()

        try:
            import json
            import yaml
            from pathlib import Path

            filepath = Path(filepath)

            # Determine file type and load
            if filepath.suffix in ['.yaml', '.yml']:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
            elif filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
            else:
                return f"❌ Unsupported file type: {filepath.suffix}", current_mappings, gr.update()

            # Extract mappings from different possible formats
            imported_mappings = {}

            # Format 1: Direct dictionary (simple key-value)
            if isinstance(data, dict) and not any(k in data for k in ['diagnosis_mapping', 'mappings', 'labels']):
                imported_mappings = data

            # Format 2: MIMIC-IV diagnosis_mapping.yaml format
            elif 'diagnosis_mapping' in data:
                for diagnosis in data['diagnosis_mapping']:
                    # Map diagnosis ID to diagnosis info
                    diag_id = str(diagnosis.get('id', ''))
                    diag_name = diagnosis.get('name', '')
                    diag_desc = diagnosis.get('description', '')
                    diag_category = diagnosis.get('category', '')

                    # Create comprehensive description
                    mapping_text = f"{diag_name}"
                    if diag_category:
                        mapping_text += f" (Category: {diag_category})"
                    if diag_desc:
                        mapping_text += f"\\n{diag_desc}"

                    # Add ICD code information
                    if 'icd_codes' in diagnosis:
                        icd_info = []
                        for icd_version in ['icd9', 'icd10']:
                            if icd_version in diagnosis['icd_codes']:
                                for code_data in diagnosis['icd_codes'][icd_version]:
                                    code = code_data.get('code', '')
                                    desc = code_data.get('description', '')
                                    if code:
                                        icd_info.append(f"{code}: {desc}")
                        if icd_info:
                            mapping_text += f"\\n\\nICD Codes:\\n" + "\\n".join(icd_info)

                    # Map both by ID and by name
                    if diag_id:
                        imported_mappings[diag_id] = mapping_text
                    if diag_name:
                        imported_mappings[diag_name] = mapping_text

            # Format 3: Generic mappings/labels key
            elif 'mappings' in data:
                imported_mappings = data['mappings']
            elif 'labels' in data:
                imported_mappings = data['labels']
            else:
                return "❌ Unrecognized mapping format", current_mappings, gr.update()

            # Merge with existing mappings (imported takes precedence)
            merged_mappings = {**current_mappings, **imported_mappings}

            # Create display dataframe
            display_data = [[k, v] for k, v in merged_mappings.items()]

            logger.info(f"Imported {len(imported_mappings)} mappings from {filepath.name}")

            return (
                f"✅ Successfully imported {len(imported_mappings)} mappings from {filepath.name}\\n"
                f"Total mappings: {len(merged_mappings)}",
                merged_mappings,
                gr.update(value=display_data)
            )

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            return f"❌ YAML parsing error: {str(e)}", current_mappings, gr.update()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return f"❌ JSON parsing error: {str(e)}", current_mappings, gr.update()
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return f"❌ Import failed: {str(e)}", current_mappings, gr.update()
    
    def test_phi_redaction(test_text, entity_types, method):
        """Test PHI redaction"""
        if not test_text or not test_text.strip():
            return "No text provided", ""
        
        if not entity_types:
            return "Select at least one entity type", ""
        
        try:
            from core.pii_redactor import create_redactor
            
            redactor = create_redactor(entity_types=list(entity_types), method=method)
            redacted_text, redactions = redactor.redact(test_text)
            
            summary = f"Redacted {len(redactions)} entities:\n"
            for entity_type in entity_types:
                count = len([r for r in redactions if r['type'] == entity_type])
                if count > 0:
                    summary += f"  {entity_type}: {count}\n"
            
            return redacted_text, summary
            
        except Exception as e:
            logger.error(f"PHI redaction test failed: {e}")
            return f"Error: {str(e)}", ""
    
    def validate_configuration(file_upload_val, file_path_input, text_col,
                         has_labels_val, label_col, current_mappings,
                         deid_cols, additional_cols, available_cols,
                         enable_phi, phi_entities, phi_method, save_redacted,
                         enable_norm, save_norm):
        """Validate configuration"""
        errors = []
        
        if not available_cols:
            errors.append("No data loaded")
        
        if not text_col:
            errors.append("Text column not selected")
        
        if has_labels_val:
            if not label_col:
                errors.append("Label column not selected")
            if not current_mappings or len(current_mappings) == 0:
                errors.append("No label mappings defined")
        
        if enable_phi and not phi_entities:
            errors.append("PHI redaction enabled but no entity types selected")
        
        if errors:
            return "❌ Validation Failed:\n\n" + "\n".join(f"  - {e}" for e in errors)
        
        input_file = file_upload_val if file_upload_val else file_path_input
        
        try:
            df = pd.read_csv(input_file)
            
            # Determine output columns
            output_cols = []
            
            if deid_cols:
                output_cols.extend(deid_cols)
            
            output_cols.append(text_col)
            
            if enable_phi and save_redacted:
                output_cols.append(f"{text_col}_redacted")
            if enable_norm and save_norm:
                output_cols.append(f"{text_col}_normalized")
            
            if has_labels_val:
                output_cols.extend(['input_label_value', 'label_context_used'])
            
            output_cols.append("stage1_*")
            
            if additional_cols:
                output_cols.extend(additional_cols)
            
            output_cols.extend([
                'processing_timestamp',
                'llm_provider',
                'llm_model',
                'extras_used',
                'rag_used'
            ])
            
            success = app_state.set_data_config(
                input_file=input_file,
                text_column=text_col,
                has_labels=has_labels_val,
                label_column=label_col if has_labels_val else None,
                label_mapping=current_mappings if has_labels_val else {},
                deid_columns=list(deid_cols) if deid_cols else [],
                additional_columns=list(additional_cols) if additional_cols else [],
                enable_phi_redaction=enable_phi,
                phi_entity_types=list(phi_entities) if phi_entities else [],
                redaction_method=phi_method,
                save_redacted_text=save_redacted,
                enable_pattern_normalization=enable_norm,
                save_normalized_text=save_norm
            )
            
            if success:
                result = f"""✅ Configuration Valid

**Input Data:**
File: {input_file}
Rows: {len(df):,}
Text Column: {text_col}

**Labels:** {'Enabled' if has_labels_val else 'Disabled'}"""
                
                if has_labels_val:
                    result += f"\nLabel Column: {label_col}"
                    result += f"\nMappings Defined: {len(current_mappings)}"
                
                result += f"\n\n**Text Processing:**"
                result += f"\nPHI Redaction: {'Enabled' if enable_phi else 'Disabled'}"
                if enable_phi:
                    result += f"\n  - Save Redacted Column: {'Yes' if save_redacted else 'No'}"
                    result += f"\n  - Entity Types: {len(phi_entities)}"
                
                result += f"\nPattern Normalization: {'Enabled' if enable_norm else 'Disabled'}"
                if enable_norm:
                    result += f"\n  - Save Normalized Column: {'Yes' if save_norm else 'No'}"
                
                result += f"\n\n**Output Columns (estimated {len(output_cols)} base columns):**\n"
                result += "\n".join(f"  - {col}" for col in output_cols[:15])
                if len(output_cols) > 15:
                    result += f"\n  ... and {len(output_cols) - 15} more"
                
                result += "\n\n✅ Ready for processing!"
                return result
            else:
                return "❌ Failed to save configuration"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    # CONNECT EVENT HANDLERS
    
    file_upload.change(
        fn=load_file,
        inputs=[file_upload],
        outputs=[
            upload_status, data_preview, data_info, available_columns,
            text_column, deid_columns, additional_columns, label_column
        ]
    )
    
    load_path_btn.click(
        fn=load_file,
        inputs=[file_path],
        outputs=[
            upload_status, data_preview, data_info, available_columns,
            text_column, deid_columns, additional_columns, label_column
        ]
    )
    
    has_labels.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[has_labels],
        outputs=[label_config_panel]
    )
    
    enable_phi_redaction.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[enable_phi_redaction],
        outputs=[phi_config_panel]
    )
    
    analyze_labels_btn.click(
        fn=analyze_labels,
        inputs=[file_upload, file_path, label_column],
        outputs=[label_analysis, unique_labels_state, select_label]
    )
    
    add_mapping_btn.click(
        fn=add_mapping,
        inputs=[select_label, label_meaning, current_label_mappings],
        outputs=[mapping_status, current_label_mappings, mappings_display]
    )
    
    clear_single_mapping_btn.click(
        fn=clear_single_mapping,
        inputs=[select_label, current_label_mappings],
        outputs=[mapping_status, current_label_mappings, mappings_display]
    )
    
    clear_all_mappings_btn.click(
        fn=clear_all_mappings,
        outputs=[mapping_status, current_label_mappings, mappings_display]
    )
    
    select_label.change(
        fn=load_existing_meaning,
        inputs=[select_label, current_label_mappings],
        outputs=[label_meaning]
    )
    
    save_mappings_btn.click(
        fn=save_mappings,
        inputs=[current_label_mappings],
        outputs=[mapping_status]
    )
    
    export_mappings_btn.click(
        fn=export_mappings,
        inputs=[current_label_mappings],
        outputs=[mapping_status, download_mappings]
    )

    import_mappings_file.change(
        fn=import_mappings,
        inputs=[import_mappings_file, current_label_mappings],
        outputs=[import_status, current_label_mappings, mappings_display]
    )
    
    test_phi_btn.click(
        fn=test_phi_redaction,
        inputs=[phi_test_text, phi_entity_types, redaction_method],
        outputs=[phi_test_result, phi_test_summary]
    )
    
    validate_btn.click(
        fn=validate_configuration,
        inputs=[
            file_upload, file_path, text_column,
            has_labels, label_column, current_label_mappings,
            deid_columns, additional_columns, available_columns,
            enable_phi_redaction, phi_entity_types, redaction_method, save_redacted_text,
            enable_pattern_normalization, save_normalized_text
        ],
        outputs=[validation_result]
    )
    
    return components