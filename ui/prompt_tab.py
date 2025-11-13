#!/usr/bin/env python3
"""
Prompt Configuration Tab for ClinOrchestra - Dual Prompt Support with RAG Refinement
Supports main and minimal prompts, JSON schema definition, template loading, and RAG refinement
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import gradio as gr
import json
import yaml
from typing import Dict, Any, List
from pathlib import Path
import logging
from core.prompt_templates import get_template, list_templates

logger = logging.getLogger(__name__)

def create_prompt_tab(app_state) -> Dict[str, Any]:
    """
    Create prompt configuration tab with main/minimal/rag prompts and template loading.
    
    Args:
        app_state: Application state manager
        
    Returns:
        Dictionary of Gradio components
    """
    
    components = {}
    
    gr.Markdown("### Prompt Configuration")
    gr.Markdown("""
    Define the extraction task prompts and JSON output structure for the LLM.
    
    **Three Prompt System:**
    - **Main Prompt**: Primary, detailed extraction instructions
    - **Minimal Prompt**: Shorter fallback prompt (used after 3 failed attempts)
    - **RAG Refinement Prompt**: Optional prompt for refining extractions using retrieved evidence
    - All prompts use the same JSON schema
    
    **Important Notes:**
    - Do NOT include `{json_enforcement_instructions}` - this is added automatically
    - Do NOT include `{json_schema_instructions}` - this is added automatically
    - Only use these placeholders: `{clinical_text}`, `{label_context}`, `{rag_outputs}`, `{function_outputs}`, `{extras_outputs}`
    - For RAG refinement: `{stage3_json_output}`, `{retrieved_evidence_chunks}`
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("#### Main Prompt (Required)")
            
            main_prompt = gr.TextArea(
                label="Main Task Prompt",
                placeholder="Describe the extraction task in detail...",
                lines=15,
                info="Primary prompt with full instructions"
            )
            components['main_prompt'] = main_prompt
            
            with gr.Row():
                main_char_count = gr.Textbox(
                    label="Characters",
                    value="0",
                    interactive=False,
                    scale=1
                )
                main_token_estimate = gr.Textbox(
                    label="Est. Tokens",
                    value="0",
                    interactive=False,
                    scale=1
                )
                components['main_char_count'] = main_char_count
                components['main_token_estimate'] = main_token_estimate
        
        with gr.Column(scale=1):
            gr.Markdown("#### Options")
            
            use_minimal = gr.Checkbox(
                label="Enable Minimal Prompt Fallback",
                value=False,
                info="Create a shorter version for fallback scenarios"
            )
            components['use_minimal'] = use_minimal
            
            enable_rag_prompt = gr.Checkbox(
                label="Enable RAG Refinement Prompt",
                value=False,
                info="Create a prompt for refining extractions using RAG evidence"
            )
            components['enable_rag_prompt'] = enable_rag_prompt
            
            gr.Markdown("#### Load Template")
            
            template_list = list_templates()
            example_dropdown = gr.Dropdown(
                choices=list(template_list.values()),
                value="⭐ Universal template - Customize for ANY clinical task",  # v1.0.0: Match actual "blank" template description
                label="Template"
            )
            components['example_dropdown'] = example_dropdown
            
            load_example_btn = gr.Button("Load Template")
            components['load_example_btn'] = load_example_btn
    
    with gr.Column(visible=False) as minimal_prompt_panel:
        gr.Markdown("---")
        gr.Markdown("#### Minimal Prompt (Fallback)")
        gr.Markdown("""
        **When is this used?**
        - After 3 failed attempts with the main prompt (e.g., context window issues)
        - When JSON parsing fails repeatedly
        - Uses the same JSON schema as the main prompt
        """)
        
        minimal_prompt = gr.TextArea(
            label="Minimal Task Prompt",
            placeholder="Shorter version of extraction instructions...",
            lines=10,
            info="Concise version for fallback scenarios"
        )
        components['minimal_prompt'] = minimal_prompt
        
        with gr.Row():
            minimal_char_count = gr.Textbox(
                label="Characters",
                value="0",
                interactive=False,
                scale=1
            )
            minimal_token_estimate = gr.Textbox(
                label="Est. Tokens",
                value="0",
                interactive=False,
                scale=1
            )
            components['minimal_char_count'] = minimal_char_count
            components['minimal_token_estimate'] = minimal_token_estimate
    
    components['minimal_prompt_panel'] = minimal_prompt_panel
    
    with gr.Column(visible=False) as rag_prompt_panel:
        gr.Markdown("---")
        gr.Markdown("#### RAG Refinement Prompt (Optional)")
        gr.Markdown("""
        **When is this used?**
        - After initial extraction completes (Stage 3)
        - Only when RAG successfully retrieves evidence
        - LLM uses evidence to refine/correct the initial extraction
        
        **Available Placeholders:**
        - `{clinical_text}` - Original clinical text
        - `{stage3_json_output}` - JSON from Stage 3 extraction
        - `{retrieved_evidence_chunks}` - Retrieved document chunks from RAG
        - `{label_context}` - Label meaning (if configured)
        - `{json_schema_instructions}` - JSON schema (added automatically)
        
        **Important Notes:**
        - Do NOT include `{rag_json_enforcement_instructions}` - added automatically
        - Do NOT include `{json_schema_instructions}` - added automatically
        """)
        
        default_rag_prompt = """[SYSTEM INSTRUCTION]
You are refining clinical extraction using evidence from authoritative guidelines. The ICD diagnosis is provided. Your refinement must support this diagnosis. Return response EXCLUSIVELY in JSON format.

[ANONYMIZATION]
NEVER use patient/family names. ALWAYS use: "the patient", "the child", "the 7-year-old", "the family"

[REFINEMENT OBJECTIVES]

1. VALIDATE:
- Z-score interpretations (ASPEN: mild -1 to -1.9, moderate -2 to -2.9, severe ≤-3)
- Etiology classification (2013 consensus: illness-related vs non-illness-related)
- NFPE grading accuracy
- Differential diagnosis appropriateness

2. CORRECT:
- Fix severity misclassifications with guideline citations
- Update inappropriate differential diagnoses
- Adjust recommendations to evidence-based practice
- Replace any patient names with "the patient", "the child"

3. ENHANCE:
- Add guideline interpretations: "Per ASPEN criteria, this meets..."
- Include prognostic information when guidelines provide it
- Note diagnostic criteria met

4. FILL GAPS:
- Add severity if z-scores present but classification unstated
- Include guideline-indicated recommendations if missing
- Complete evidence-supported care plan elements

5. ENSURE CONSISTENCY:
- Verify anthropometrics align with severity classification
- Confirm symptoms support diagnoses
- Check recommendations match severity and etiology

[PRINCIPLES]
- Preserve fidelity: NEVER remove correct data or fabricate information
- Quote guidelines when correcting: "Per WHO standards..."
- Flag discrepancies: "Note: anthropometric parameters suggest severe malnutrition, but initial extraction stated moderate"
- Add value ONLY when guidelines clearly apply
- Maintain conversational expert tone
- Write clinical narratives, NOT bullet points
- ANONYMIZE: Remove all names, use "the patient", "the child"

[END TASK]

ORIGINAL CLINICAL TEXT:
{clinical_text}

ICD DIAGNOSIS:
{label_context}

INITIAL EXTRACTION (Stage 3):
{stage3_json_output}

RETRIEVED CLINICAL EVIDENCE:
{retrieved_evidence_chunks}

{json_schema_instructions}

Return ONLY JSON: Start with {{ and end with }}. No markdown or extra text. Refine the extraction using guideline evidence."""
        
        rag_prompt = gr.TextArea(
            label="RAG Refinement Prompt Template",
            placeholder=default_rag_prompt,
            lines=20,
            value=default_rag_prompt
        )
        components['rag_prompt'] = rag_prompt
        
        with gr.Row():
            rag_char_count = gr.Textbox(
                label="Characters",
                value=str(len(default_rag_prompt)),
                interactive=False
            )
            rag_token_estimate = gr.Textbox(
                label="Est. Tokens",
                value=str(len(default_rag_prompt) // 4),
                interactive=False
            )
            components['rag_char_count'] = rag_char_count
            components['rag_token_estimate'] = rag_token_estimate
        
        gr.Markdown("---")
        gr.Markdown("#### RAG Query Configuration")
        gr.Markdown("""
        Select which JSON fields from your extraction schema should be used to build RAG queries.
        The system will combine values from these fields to search for relevant evidence.
        """)
        
        rag_query_fields = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Fields to Use for RAG Queries",
            info="Select fields that will be used to query the knowledge base"
        )
        components['rag_query_fields'] = rag_query_fields
        
        refresh_fields_btn = gr.Button("Refresh from Schema")
        components['refresh_fields_btn'] = refresh_fields_btn
    
    components['rag_prompt_panel'] = rag_prompt_panel
    
    gr.Markdown("---")
    gr.Markdown("### Define JSON Output Structure")
    gr.Markdown("""
    Define the fields for the LLM to extract. Supports nested objects and arrays.
    
    **Field Types:**
    - `string`: Text values
    - `number`: Numeric values
    - `boolean`: True/False values
    - `object`: Nested structure (define properties in description)
    - `array`: List of items
    """)
    
    schema_fields = gr.State(value=[])
    components['schema_fields'] = schema_fields
    
    with gr.Row():
        with gr.Column(scale=2):
            field_name = gr.Textbox(
                label="Field Name",
                placeholder="diagnosis"
            )
            components['field_name'] = field_name
            
            field_description = gr.TextArea(
                label="Field Description",
                placeholder="Primary diagnosis mentioned in the note",
                lines=3
            )
            components['field_description'] = field_description
        
        with gr.Column(scale=1):
            field_type = gr.Dropdown(
                choices=["string", "number", "boolean", "object", "array"],
                value="string",
                label="Field Type"
            )
            components['field_type'] = field_type
            
            field_required = gr.Checkbox(
                label="Required Field",
                value=True
            )
            components['field_required'] = field_required
            
            add_field_btn = gr.Button("➕ Add Field", variant="primary")
            components['add_field_btn'] = add_field_btn
    
    with gr.Row():
        remove_field_idx = gr.Number(
            label="Field Index to Remove",
            value=0,
            precision=0
        )
        remove_field_btn = gr.Button("🗑️ Remove Field")
        clear_fields_btn = gr.Button("🗑️ Clear All Fields", variant="stop")
    
    components['remove_field_idx'] = remove_field_idx
    components['remove_field_btn'] = remove_field_btn
    components['clear_fields_btn'] = clear_fields_btn
    
    schema_display = gr.Dataframe(
        headers=["Index", "Name", "Type", "Required", "Description"],
        datatype=["str", "str", "str", "str", "str"],
        label="Current Schema Fields",
        interactive=False
    )
    components['schema_display'] = schema_display
    
    gr.Markdown("#### Load Schema from File")
    
    with gr.Row():
        schema_file_upload = gr.File(
            label="Upload Schema File (YAML/JSON)",
            file_types=[".yaml", ".yml", ".json"]
        )
        components['schema_file_upload'] = schema_file_upload
        
        load_schema_file_btn = gr.Button("Load Schema from File", variant="primary")
        components['load_schema_file_btn'] = load_schema_file_btn
    
    load_schema_status = gr.Textbox(label="Load Status", interactive=False)
    components['load_schema_status'] = load_schema_status
    
    gr.Markdown("---")
    gr.Markdown("### Preview Prompts")
    gr.Markdown("""
    **Note:** JSON enforcement and schema instructions are added automatically by the system.
    These previews show only your prompt content.
    """)
    
    with gr.Tabs():
        with gr.Tab("Main Prompt"):
            main_prompt_preview = gr.TextArea(
                label="Main Prompt Preview",
                lines=20,
                interactive=False
            )
            components['main_prompt_preview'] = main_prompt_preview
        
        with gr.Tab("Minimal Prompt"):
            minimal_prompt_preview = gr.TextArea(
                label="Minimal Prompt Preview",
                lines=15,
                interactive=False
            )
            components['minimal_prompt_preview'] = minimal_prompt_preview
        
        with gr.Tab("RAG Refinement Prompt"):
            rag_prompt_preview = gr.TextArea(
                label="RAG Refinement Prompt Preview",
                lines=20,
                interactive=False
            )
            components['rag_prompt_preview'] = rag_prompt_preview
        
        with gr.Tab("JSON Schema"):
            json_schema_preview = gr.Code(
                label="JSON Schema Preview",
                language="json",
                interactive=False
            )
            components['json_schema_preview'] = json_schema_preview
    
    update_preview_btn = gr.Button("🔄 Update Preview", variant="secondary")
    components['update_preview_btn'] = update_preview_btn
    
    gr.Markdown("---")
    gr.Markdown("### Save Configuration")
    
    save_prompt_btn = gr.Button("💾 Save Prompt Configuration", variant="primary", size="lg")
    components['save_prompt_btn'] = save_prompt_btn
    
    prompt_status = gr.TextArea(
        label="Status",
        lines=8,
        interactive=False
    )
    components['prompt_status'] = prompt_status
    
    def update_char_count(text):
        """Update character and token count"""
        chars = len(text) if text else 0
        tokens = chars // 4
        return str(chars), str(tokens)
    
    def toggle_minimal_panel(use_min):
        """Toggle minimal prompt panel visibility"""
        return gr.update(visible=use_min)
    
    def toggle_rag_prompt_panel(enable_rag):
        """Toggle RAG prompt panel visibility"""
        return gr.update(visible=enable_rag)
    
    def add_field_to_schema(fields, name, ftype, desc, required):
        """Add field to schema"""
        if not name or not name.strip():
            return fields, gr.update(), "❌ Field name is required"
        
        field = {
            'name': name.strip(),
            'type': ftype,
            'description': desc.strip(),
            'required': required
        }
        
        new_fields = fields + [field]
        
        display_data = [
            [str(i), f['name'], f['type'], "Yes" if f['required'] else "No", f['description'][:50]]
            for i, f in enumerate(new_fields)
        ]
        
        return new_fields, gr.update(value=display_data), f"✅ Added field: {name}"
    
    def remove_field_from_schema(fields, idx):
        """Remove field from schema"""
        try:
            idx = int(idx)
            if 0 <= idx < len(fields):
                removed_field = fields[idx]
                new_fields = fields[:idx] + fields[idx+1:]
                
                display_data = [
                    [str(i), f['name'], f['type'], "Yes" if f['required'] else "No", f['description'][:50]]
                    for i, f in enumerate(new_fields)
                ]
                
                return new_fields, gr.update(value=display_data), f"✅ Removed field: {removed_field['name']}"
            else:
                return fields, gr.update(), "❌ Invalid index"
        except ValueError:
            return fields, gr.update(), "❌ Invalid index"
    
    def clear_all_fields():
        """Clear all schema fields"""
        return [], gr.update(value=[]), "✅ Cleared all fields"
    
    def unflatten_schema(fields: List[Dict]) -> Dict[str, Any]:
        """Convert flat field list to nested schema dictionary"""
        schema = {}
        
        for field in fields:
            name = field['name']
            ftype = field['type']
            desc = field.get('description', '')
            required = field.get('required', True)
            
            schema[name] = {
                'type': ftype,
                'description': desc,
                'required': required
            }
            
            if ftype == 'object':
                schema[name]['properties'] = {}
            elif ftype == 'array':
                schema[name]['items'] = {'type': 'string'}
        
        return schema
    
    def refresh_query_fields(fields):
        """Refresh field choices from current schema"""
        if not fields:
            return gr.update(choices=[], value=[])
        
        field_names = [f['name'] for f in fields]
        current_values = app_state.rag_config.rag_query_fields
        valid_values = [v for v in current_values if v in field_names]
        
        return gr.update(choices=field_names, value=valid_values)
    
    def update_prompt_preview(main_p, minimal_p, rag_p, fields, use_min, enable_rag):
        """Update preview of prompts"""
        if not main_p or not main_p.strip():
            return "", "", "", "{}", gr.update(visible=use_min), gr.update(visible=enable_rag)
        
        if not fields:
            return main_p, minimal_p, rag_p, "{}", gr.update(visible=use_min), gr.update(visible=enable_rag)
        
        schema_dict = unflatten_schema(fields)
        schema_json = json.dumps(schema_dict, indent=2)
        
        main_preview = main_p
        minimal_preview = minimal_p if (use_min and minimal_p) else ""
        rag_preview = rag_p if (enable_rag and rag_p) else ""
        
        return (
            main_preview,
            minimal_preview,
            rag_preview,
            schema_json,
            gr.update(visible=use_min),
            gr.update(visible=enable_rag)
        )
    
    def save_prompt_configuration(main_p, minimal_p, rag_p, fields, use_min, enable_rag, query_fields):
        """
        Save prompt configuration to app state
        FIXED: Properly route rag_query_fields to set_rag_config
        """
        if not main_p or not main_p.strip():
            return "❌ Main prompt is required"
        
        if not fields:
            return "❌ No JSON fields defined"
        
        if use_min and (not minimal_p or not minimal_p.strip()):
            return "❌ Minimal prompt is enabled but empty"
        
        if enable_rag and (not rag_p or not rag_p.strip()):
            return "❌ RAG refinement prompt is enabled but empty"
        
        if enable_rag and not query_fields:
            return "❌ RAG refinement is enabled but no query fields selected"
        
        schema_dict = unflatten_schema(fields)
        
        # FIXED: Save prompt configuration WITHOUT rag_query_fields parameter
        prompt_success = app_state.set_prompt_config(
            main_prompt=main_p,
            minimal_prompt=minimal_p if use_min else None,
            use_minimal=use_min,
            json_schema=schema_dict,
            rag_prompt=rag_p if enable_rag else None
        )
        
        if not prompt_success:
            return "❌ Failed to save prompt configuration"
        
        # FIXED: Save RAG query fields separately via set_rag_config
        if enable_rag and query_fields:
            rag_success = app_state.set_rag_config(
                enabled=app_state.rag_config.enabled,
                rag_query_fields=list(query_fields)
            )
            
            if not rag_success:
                return "⚠️ Prompt configuration saved, but RAG query fields failed to save"
        
        status = f"✅ Prompt configuration saved!\n\n"
        status += f"Main prompt: {len(main_p)} characters\n"
        if use_min:
            status += f"Minimal prompt: {len(minimal_p)} characters\n"
        if enable_rag:
            status += f"RAG refinement prompt: {len(rag_p)} characters\n"
            status += f"RAG query fields: {len(query_fields)}\n"
        status += f"JSON fields: {len(fields)}\n"
        status += f"Fallback enabled: {'Yes' if use_min else 'No'}\n"
        status += f"RAG refinement enabled: {'Yes' if enable_rag else 'No'}\n\n"
        status += "Note: JSON enforcement and schema instructions will be added automatically during processing."
        
        return status
    
    def load_example(selected):
        """Load selected template and auto-populate schema"""
        template_map = {v: k for k, v in list_templates().items()}
        template_key = template_map.get(selected, "blank")
        
        template = get_template(template_key)
        
        main_p = template.get("main", "")
        min_p = template.get("minimal", "")
        use_min = bool(min_p)
        
        rag_p = template.get("rag_prompt", "")
        enable_rag = bool(rag_p)
        
        fields = []
        if "schema" in template:
            for name, props in template["schema"].items():
                fields.append({
                    'name': name,
                    'type': props.get('type', 'string'),
                    'description': props.get('description', ''),
                    'required': props.get('required', True)
                })
        
        display_data = [
            [str(i), f['name'], f['type'], "Yes" if f['required'] else "No", f['description'][:50]]
            for i, f in enumerate(fields)
        ] if fields else []
        
        field_names = [f['name'] for f in fields]
        query_fields_update = gr.update(choices=field_names, value=[])
        
        main_chars, main_tokens = update_char_count(main_p)
        min_chars, min_tokens = update_char_count(min_p)
        rag_chars, rag_tokens = update_char_count(rag_p)
        
        status = f"✅ Loaded template: {selected}\n\n"
        status += "Note: JSON enforcement instructions are NOT in the template.\n"
        status += "They will be added automatically by the agent during processing."
        
        return (
            main_p, min_p, rag_p, use_min, enable_rag, fields, 
            gr.update(value=display_data), status, 
            gr.update(visible=use_min), gr.update(visible=enable_rag),
            main_chars, main_tokens, min_chars, min_tokens, rag_chars, rag_tokens,
            query_fields_update
        )
    
    def load_schema_from_file(file):
        """Load schema from uploaded YAML or JSON file (supports both simplified and JSON Schema formats)"""
        if not file:
            return [], gr.update(), "❌ No file uploaded", gr.update()

        path = Path(file.name)
        try:
            if path.suffix in ('.yaml', '.yml'):
                with open(path) as f:
                    data = yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path) as f:
                    data = json.load(f)
            else:
                return [], gr.update(), "❌ Unsupported file type", gr.update()

            # Detect if this is JSON Schema format (has $schema, properties, type fields)
            is_json_schema = ('$schema' in data or
                            (data.get('type') == 'object' and 'properties' in data))

            fields = []

            if is_json_schema:
                # JSON Schema format - extract from properties
                properties = data.get('properties', {})
                required_fields = data.get('required', [])

                for name, props in properties.items():
                    if isinstance(props, dict):
                        fields.append({
                            'name': name,
                            'type': props.get('type', 'string'),
                            'description': props.get('description', ''),
                            'required': name in required_fields
                        })
            else:
                # Simplified format - flat dictionary of fields
                for name, props in data.items():
                    if isinstance(props, dict):
                        fields.append({
                            'name': name,
                            'type': props.get('type', 'string'),
                            'description': props.get('description', ''),
                            'required': props.get('required', True) if isinstance(props.get('required'), bool) else True
                        })

            display_data = [
                [str(i), f['name'], f['type'], "Yes" if f['required'] else "No", f['description'][:50]]
                for i, f in enumerate(fields)
            ]

            field_names = [f['name'] for f in fields]
            query_fields_update = gr.update(choices=field_names)

            return fields, gr.update(value=display_data), "✅ Schema loaded successfully", query_fields_update
        except Exception as e:
            logger.error(f"Schema loading error: {str(e)}", exc_info=True)
            return [], gr.update(), f"❌ Error loading schema: {str(e)}", gr.update()
    
    # Connect all event handlers
    main_prompt.change(
        fn=update_char_count,
        inputs=[main_prompt],
        outputs=[main_char_count, main_token_estimate]
    )
    
    minimal_prompt.change(
        fn=update_char_count,
        inputs=[minimal_prompt],
        outputs=[minimal_char_count, minimal_token_estimate]
    )
    
    rag_prompt.change(
        fn=update_char_count,
        inputs=[rag_prompt],
        outputs=[rag_char_count, rag_token_estimate]
    )
    
    use_minimal.change(
        fn=toggle_minimal_panel,
        inputs=[use_minimal],
        outputs=[minimal_prompt_panel]
    )
    
    enable_rag_prompt.change(
        fn=toggle_rag_prompt_panel,
        inputs=[enable_rag_prompt],
        outputs=[rag_prompt_panel]
    )
    
    add_field_btn.click(
        fn=add_field_to_schema,
        inputs=[schema_fields, field_name, field_type, field_description, field_required],
        outputs=[schema_fields, schema_display, prompt_status]
    )
    
    remove_field_btn.click(
        fn=remove_field_from_schema,
        inputs=[schema_fields, remove_field_idx],
        outputs=[schema_fields, schema_display, prompt_status]
    )
    
    clear_fields_btn.click(
        fn=clear_all_fields,
        outputs=[schema_fields, schema_display, prompt_status]
    )
    
    refresh_fields_btn.click(
        fn=refresh_query_fields,
        inputs=[schema_fields],
        outputs=[rag_query_fields]
    )
    
    update_preview_btn.click(
        fn=update_prompt_preview,
        inputs=[main_prompt, minimal_prompt, rag_prompt, schema_fields, use_minimal, enable_rag_prompt],
        outputs=[
            main_prompt_preview,
            minimal_prompt_preview,
            rag_prompt_preview,
            json_schema_preview,
            minimal_prompt_preview,
            rag_prompt_preview
        ]
    )
    
    save_prompt_btn.click(
        fn=save_prompt_configuration,
        inputs=[main_prompt, minimal_prompt, rag_prompt, schema_fields, use_minimal, enable_rag_prompt, rag_query_fields],
        outputs=[prompt_status]
    )
    
    load_example_btn.click(
        fn=load_example,
        inputs=[example_dropdown],
        outputs=[
            main_prompt,
            minimal_prompt,
            rag_prompt,
            use_minimal,
            enable_rag_prompt,
            schema_fields,
            schema_display,
            prompt_status,
            minimal_prompt_panel,
            rag_prompt_panel,
            main_char_count,
            main_token_estimate,
            minimal_char_count,
            minimal_token_estimate,
            rag_char_count,
            rag_token_estimate,
            rag_query_fields
        ]
    )
    
    load_schema_file_btn.click(
        fn=load_schema_from_file,
        inputs=[schema_file_upload],
        outputs=[schema_fields, schema_display, load_schema_status, rag_query_fields]
    )
    
    return components