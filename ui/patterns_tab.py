#!/usr/bin/env python3
"""
Patterns Tab - Regex preprocessing UI with file loading
"""

import gradio as gr
import json
import yaml
from typing import Dict, Any
from pathlib import Path
import logging
from core.regex_preprocessor import RegexPreprocessor

logger = logging.getLogger(__name__)

def create_patterns_tab(app_state) -> Dict[str, Any]:
    """Create regex patterns tab"""
    
    # Ensure RegexPreprocessor is initialized
    if app_state.get_regex_preprocessor() is None:
        regex_preprocessor = RegexPreprocessor()
        app_state.set_regex_preprocessor(regex_preprocessor)
        logger.info("Initialized and registered RegexPreprocessor with AppState")
    
    components = {}
    
    gr.Markdown("### Regex Patterns (Text Preprocessing)")
    gr.Markdown("""
    Define regex patterns to preprocess text BEFORE it goes to the LLM.
    Use this to fix patterns that might confuse the LLM.
    
    **Examples:**
    - Standardize formats: "20 mg" → "20mg"
    - Fix inconsistent spacing: "BP 120 / 80" → "BP: 120/80"
    - Clarify negations: "negative for fever" → "No fever"
    
    **Pattern File Schema (YAML or JSON):**
    ```yaml
    patterns:
      - name: standardize_dosage
        pattern: '(\d+)\s+(mg|mcg|g)'
        replacement: '\1\2'
        description: 'Remove space between dose and unit'
        enabled: true
      - name: normalize_bp
        pattern: 'BP[\s:]*(\d+)\s*/\s*(\d+)'
        replacement: 'BP: \1/\2'
        description: 'Standardize blood pressure format'
        enabled: true
    ```
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### Add/Edit Pattern")
            
            pattern_id = gr.Textbox(
                label="Pattern ID",
                placeholder="Will be generated for new patterns",
                interactive=False,
                visible=False
            )
            components['pattern_id'] = pattern_id
            
            pattern_name = gr.Textbox(
                label="Pattern Name",
                placeholder="standardize_dosage"
            )
            components['pattern_name'] = pattern_name
            
            pattern_regex = gr.Textbox(
                label="Regex Pattern",
                placeholder=r'(\d+)\s+(mg|mcg|g)',
                info="Regular expression to match"
            )
            components['pattern_regex'] = pattern_regex
            
            pattern_replacement = gr.Textbox(
                label="Replacement",
                placeholder=r'\1\2',
                info="Replacement text (use \\1, \\2 for groups)"
            )
            components['pattern_replacement'] = pattern_replacement
            
            pattern_description = gr.TextArea(
                label="Description",
                placeholder="Remove space between dose and unit",
                lines=2
            )
            components['pattern_description'] = pattern_description
            
            pattern_enabled = gr.Checkbox(
                label="Enabled",
                value=True,
                info="Apply this pattern during preprocessing"
            )
            components['pattern_enabled'] = pattern_enabled
            
            with gr.Row():
                add_pattern_btn = gr.Button("Add Pattern", variant="primary")
                save_pattern_btn = gr.Button("Save Pattern", variant="primary")
                clear_pattern_btn = gr.Button("Clear")
            
            components['add_pattern_btn'] = add_pattern_btn
            components['save_pattern_btn'] = save_pattern_btn
            components['clear_pattern_btn'] = clear_pattern_btn
            
            pattern_status = gr.Textbox(label="Status", interactive=False)
            components['pattern_status'] = pattern_status
        
        with gr.Column(scale=1):
            gr.Markdown("#### Registered Patterns")
            
            patterns_list = gr.Dataframe(
                headers=["Name", "Enabled", "Description"],
                datatype=["str", "str", "str"],
                label="Available Patterns",
                interactive=False
            )
            components['patterns_list'] = patterns_list
            
            refresh_patterns_btn = gr.Button("Refresh List")
            components['refresh_patterns_btn'] = refresh_patterns_btn
            
            gr.Markdown("#### Manage Patterns")
            
            selected_pattern_name = gr.Textbox(
                label="Pattern Name to View/Edit/Remove/Toggle",
                placeholder="standardize_dosage"
            )
            components['selected_pattern_name'] = selected_pattern_name
            
            with gr.Row():
                view_pattern_btn = gr.Button("View/Edit")
                toggle_pattern_btn = gr.Button("Toggle")
                remove_pattern_btn = gr.Button("Remove", variant="stop")
            
            components['view_pattern_btn'] = view_pattern_btn
            components['toggle_pattern_btn'] = toggle_pattern_btn
            components['remove_pattern_btn'] = remove_pattern_btn
    
    gr.Markdown("---")
    gr.Markdown("### Test Patterns")
    
    with gr.Row():
        with gr.Column():
            test_input_text = gr.TextArea(
                label="Test Input",
                placeholder="Enter clinical text to test patterns...",
                lines=5
            )
            components['test_input_text'] = test_input_text
        
        with gr.Column():
            test_output_text = gr.TextArea(
                label="Test Output",
                lines=5,
                interactive=False
            )
            components['test_output_text'] = test_output_text
    
    with gr.Row():
        test_pattern_name = gr.Textbox(
            label="Pattern to Test (leave empty for all)",
            placeholder="standardize_dosage"
        )
        components['test_pattern_name'] = test_pattern_name
        
        with gr.Column():
            test_pattern_btn = gr.Button("Test Single Pattern")
            test_all_btn = gr.Button("Test All Enabled", variant="primary")
    
    components['test_pattern_btn'] = test_pattern_btn
    components['test_all_btn'] = test_all_btn
    
    gr.Markdown("---")
    gr.Markdown("### Built-in Pattern Examples")
    
    with gr.Accordion("View Examples", open=False):
        gr.Markdown("""
        **standardize_dosage**
        - Pattern: `(\d+)\s+(mg|mcg|g|ml|units)`
        - Replacement: `\1\2`
        - Example: "20 mg" → "20mg"
        
        **standardize_bp**
        - Pattern: `BP[\s:]*(\d+)\s*/\s*(\d+)`
        - Replacement: `BP: \1/\2`
        - Example: "BP 120 / 80" → "BP: 120/80"
        
        **standardize_lab_format**
        - Pattern: `([A-Za-z0-9]+)\s*[:=]\s*(\d+\.?\d*)\s*([a-zA-Z/]+)`
        - Replacement: `\1: \2 \3`
        - Example: "Glucose=120mg/dL" → "Glucose: 120 mg/dL"
        
        **remove_extra_whitespace**
        - Pattern: `\s+`
        - Replacement: ` `
        - Example: "word  with   spaces" → "word with spaces"
        
        **standardize_negation**
        - Pattern: `\b(no|not|negative for|denies)\s+([a-zA-Z\s]+)`
        - Replacement: `No \2`
        - Example: "negative for fever" → "No fever"
        """)
    
    gr.Markdown("---")
    gr.Markdown("### Load from File")
    
    gr.Markdown("""
    Upload a YAML or JSON file containing pattern definitions.
    
    **YAML Format:**
    ```yaml
    patterns:
      - name: pattern_name
        pattern: 'regex_pattern'
        replacement: 'replacement_text'
        description: 'What this pattern does'
        enabled: true
    ```
    
    **JSON Format:**
    ```json
    {
      "patterns": [
        {
          "name": "pattern_name",
          "pattern": "regex_pattern",
          "replacement": "replacement_text",
          "description": "What this pattern does",
          "enabled": true
        }
      ]
    }
    ```
    """)
    
    with gr.Row():
        patterns_file_upload = gr.File(
            label="Upload Pattern File (YAML/JSON)",
            file_types=[".yaml", ".yml", ".json"]
        )
        components['patterns_file_upload'] = patterns_file_upload
        
        load_patterns_file_btn = gr.Button("Load from File", variant="primary")
        components['load_patterns_file_btn'] = load_patterns_file_btn
    
    load_patterns_status = gr.Textbox(label="Load Status", interactive=False)
    components['load_patterns_status'] = load_patterns_status
    
    gr.Markdown("---")
    gr.Markdown("### Import/Export")
    
    with gr.Row():
        export_patterns_btn = gr.Button("Export Patterns")
        import_patterns_btn = gr.Button("Import Patterns")
    
    components['export_patterns_btn'] = export_patterns_btn
    components['import_patterns_btn'] = import_patterns_btn
    
    patterns_json = gr.TextArea(
        label="Patterns JSON",
        lines=10
    )
    components['patterns_json'] = patterns_json
    
    # Event handlers
    
    def add_pattern(name, regex, replacement, description, enabled):
        """Add pattern"""
        if not name or not name.strip():
            return "", "", "", "", "", "No name provided", gr.update()
        
        if not regex or not regex.strip():
            return "", "", "", "", "", "No regex pattern provided", gr.update()
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        success, message = regex_preprocessor.add_pattern(
            name.strip(),
            regex.strip(),
            replacement.strip() if replacement else "",
            description.strip() if description else "",
            enabled
        )
        
        if success:
            patterns_data = [
                [p['name'], "Yes" if p['enabled'] else "No", p['description'][:50]]
                for p in regex_preprocessor.list_patterns()
            ]
            return "", "", "", "", "", message, gr.update(value=patterns_data)
        else:
            return "", "", "", "", "", message, gr.update()
    
    def save_pattern(id, name, regex, replacement, description, enabled):
        """Save edited pattern"""
        if not id:
            return "", "", "", "", "", "No ID provided", gr.update()
        if not name or not name.strip():
            return id, name, regex, replacement, description, enabled, "No name provided", gr.update()
        if not regex or not regex.strip():
            return id, name, regex, replacement, description, enabled, "No regex pattern provided", gr.update()
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        success, message = regex_preprocessor.update_pattern(
            id,
            name.strip(),
            regex.strip(),
            replacement.strip() if replacement else "",
            description.strip() if description else "",
            enabled
        )
        
        if success:
            patterns_data = [
                [p['name'], "Yes" if p['enabled'] else "No", p['description'][:50]]
                for p in regex_preprocessor.list_patterns()
            ]
            return "", "", "", "", "", message, gr.update(value=patterns_data)
        else:
            return id, name, regex, replacement, description, enabled, message, gr.update()
    
    def refresh_patterns():
        """Refresh patterns list"""
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        patterns_data = [
            [p['name'], "Yes" if p['enabled'] else "No", p['description'][:50]]
            for p in regex_preprocessor.list_patterns()
        ]
        return gr.update(value=patterns_data)
    
    def view_pattern(name):
        """View pattern details"""
        if not name or not name.strip():
            return "", "", "", "", "", "No pattern name provided", gr.update()
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        pattern = regex_preprocessor.get_pattern(name.strip())
        if not pattern:
            return "", "", "", "", "", f"Pattern '{name}' not found", gr.update()
        
        return (
            pattern['id'],
            pattern['name'],
            pattern['pattern'],
            pattern['replacement'],
            pattern['description'],
            pattern['enabled'],
            f"Loaded: {name}",
            gr.update()
        )
    
    def toggle_pattern(name):
        """Toggle pattern enabled state"""
        if not name or not name.strip():
            return "No pattern name provided", gr.update()
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        pattern = regex_preprocessor.get_pattern(name.strip())
        if not pattern:
            return f"Pattern '{name}' not found", gr.update()
        
        new_state = not pattern['enabled']
        regex_preprocessor.enable_pattern(name.strip(), new_state)
        
        patterns_data = [
            [p['name'], "Yes" if p['enabled'] else "No", p['description'][:50]]
            for p in regex_preprocessor.list_patterns()
        ]
        
        state_text = "enabled" if new_state else "disabled"
        return f"Pattern '{name}' {state_text}", gr.update(value=patterns_data)
    
    def remove_pattern(name):
        """Remove pattern"""
        if not name or not name.strip():
            return "", "", "", "", "", "No pattern name provided", gr.update()
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        success, message = regex_preprocessor.remove_pattern(name.strip())
        
        if success:
            patterns_data = [
                [p['name'], "Yes" if p['enabled'] else "No", p['description'][:50]]
                for p in regex_preprocessor.list_patterns()
            ]
            return "", "", "", "", "", message, gr.update(value=patterns_data)
        else:
            return "", "", "", "", "", message, gr.update()
    
    def test_pattern(input_text, pattern_name):
        """Test single pattern"""
        if not input_text or not input_text.strip():
            return "No input text provided"
        
        if not pattern_name or not pattern_name.strip():
            return "No pattern name provided"
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        success, result, message = regex_preprocessor.test_pattern(
            pattern_name.strip(),
            input_text
        )
        
        if success:
            return result
        else:
            return f"Error: {message}"
    
    def test_all_patterns(input_text):
        """Test all enabled patterns"""
        if not input_text or not input_text.strip():
            return "No input text provided"
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        result = regex_preprocessor.preprocess(input_text)
        return result
    
    def clear_fields():
        """Clear input fields"""
        return "", "", "", "", "", "Cleared", gr.update()
    
    def load_patterns_from_file(file_path):
        """Load patterns from YAML or JSON file"""
        if not file_path:
            return "No file selected", gr.update()
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        try:
            file_path_obj = Path(file_path)
            
            with open(file_path_obj, 'r') as f:
                if file_path_obj.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_path_obj.suffix == '.json':
                    data = json.load(f)
                else:
                    return "Unsupported file format. Use YAML or JSON.", gr.update()
            
            if 'patterns' not in data:
                return "Invalid file format. Must contain 'patterns' key.", gr.update()
            
            patterns_list = data['patterns']
            
            added_count = 0
            errors = []
            
            for pattern_data in patterns_list:
                try:
                    name = pattern_data.get('name')
                    pattern = pattern_data.get('pattern')
                    replacement = pattern_data.get('replacement', '')
                    description = pattern_data.get('description', '')
                    enabled = pattern_data.get('enabled', True)
                    
                    if not name or not pattern:
                        errors.append(f"Skipping invalid pattern (missing name or pattern)")
                        continue
                    
                    success, message = regex_preprocessor.add_pattern(
                        name, pattern, replacement, description, enabled
                    )
                    
                    if success:
                        added_count += 1
                    else:
                        errors.append(f"{name}: {message}")
                        
                except Exception as e:
                    errors.append(f"Error processing pattern: {str(e)}")
            
            patterns_data = [
                [p['name'], "Yes" if p['enabled'] else "No", p['description'][:50]]
                for p in regex_preprocessor.list_patterns()
            ]
            
            status_msg = f"Loaded {added_count} patterns from file"
            if errors:
                status_msg += f"\n\nWarnings:\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    status_msg += f"\n... and {len(errors) - 5} more"
            
            return status_msg, gr.update(value=patterns_data)
            
        except Exception as e:
            logger.error(f"Failed to load patterns file: {e}")
            return f"Error loading file: {str(e)}", gr.update()
    
    def export_patterns():
        """Export patterns"""
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        return regex_preprocessor.export_patterns()
    
    def import_patterns(json_str):
        """Import patterns"""
        if not json_str or not json_str.strip():
            return "No JSON provided", gr.update()
        
        regex_preprocessor = app_state.get_regex_preprocessor()
        if not regex_preprocessor:
            regex_preprocessor = RegexPreprocessor()
            app_state.set_regex_preprocessor(regex_preprocessor)
            logger.warning("RegexPreprocessor was not set; initialized new instance")
        
        success, count, message = regex_preprocessor.import_patterns(json_str)
        
        if success:
            patterns_data = [
                [p['name'], "Yes" if p['enabled'] else "No", p['description'][:50]]
                for p in regex_preprocessor.list_patterns()
            ]
            return message, gr.update(value=patterns_data)
        else:
            return message, gr.update()
    
    # Connect handlers
    add_pattern_btn.click(
        fn=add_pattern,
        inputs=[pattern_name, pattern_regex, pattern_replacement, pattern_description, pattern_enabled],
        outputs=[pattern_id, pattern_name, pattern_regex, pattern_replacement, pattern_description, pattern_status, patterns_list]
    )
    
    save_pattern_btn.click(
        fn=save_pattern,
        inputs=[pattern_id, pattern_name, pattern_regex, pattern_replacement, pattern_description, pattern_enabled],
        outputs=[pattern_id, pattern_name, pattern_regex, pattern_replacement, pattern_description, pattern_status, patterns_list]
    )
    
    refresh_patterns_btn.click(
        fn=refresh_patterns,
        outputs=[patterns_list]
    )
    
    view_pattern_btn.click(
        fn=view_pattern,
        inputs=[selected_pattern_name],
        outputs=[pattern_id, pattern_name, pattern_regex, pattern_replacement, pattern_description, pattern_enabled, pattern_status, patterns_list]
    )
    
    toggle_pattern_btn.click(
        fn=toggle_pattern,
        inputs=[selected_pattern_name],
        outputs=[pattern_status, patterns_list]
    )
    
    remove_pattern_btn.click(
        fn=remove_pattern,
        inputs=[selected_pattern_name],
        outputs=[pattern_id, pattern_name, pattern_regex, pattern_replacement, pattern_description, pattern_status, patterns_list]
    )
    
    clear_pattern_btn.click(
        fn=clear_fields,
        outputs=[pattern_id, pattern_name, pattern_regex, pattern_replacement, pattern_description, pattern_status, patterns_list]
    )
    
    test_pattern_btn.click(
        fn=test_pattern,
        inputs=[test_input_text, test_pattern_name],
        outputs=[test_output_text]
    )
    
    test_all_btn.click(
        fn=test_all_patterns,
        inputs=[test_input_text],
        outputs=[test_output_text]
    )
    
    load_patterns_file_btn.click(
        fn=load_patterns_from_file,
        inputs=[patterns_file_upload],
        outputs=[load_patterns_status, patterns_list]
    )
    
    export_patterns_btn.click(
        fn=export_patterns,
        outputs=[patterns_json]
    )
    
    import_patterns_btn.click(
        fn=import_patterns,
        inputs=[patterns_json],
        outputs=[pattern_status, patterns_list]
    )
    
    return components