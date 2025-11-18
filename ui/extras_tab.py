#!/usr/bin/env python3
"""
Extras Tab - Task hints for agents with file loading
"""

import gradio as gr
import json
import yaml
from typing import Dict, Any
from pathlib import Path
import logging
from core.extras_manager import ExtrasManager
import uuid

logger = logging.getLogger(__name__)

def create_extras_tab(app_state) -> Dict[str, Any]:
    """Create extras configuration tab"""
    
    # Ensure ExtrasManager is initialized
    if app_state.get_extras_manager() is None:
        extras_manager = ExtrasManager()
        app_state.set_extras_manager(extras_manager)
        logger.info("Initialized and registered ExtrasManager with AppState")
    
    components = {}
    
    gr.Markdown("### Extras (Task Hints)")
    gr.Markdown("""
    Provide task-specific hints that agents can use when struggling.
    Extras include patterns, definitions, guidelines, or any context that helps extraction.
    
    **Extras File Schema (YAML or JSON):**
    ```yaml
    extras:
      - type: pattern
        content: "Lab values format - [test name]: [value] [unit]"
        metadata:
          category: medical
          priority: high
      - type: definition
        content: "BMI categories - Underweight: <18.5, Normal: 18.5-24.9"
        metadata:
          category: clinical
    ```
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### Add/Edit Extra")
            
            extra_id = gr.Textbox(
                label="Extra ID",
                placeholder="Will be generated for new extras",
                interactive=False,
                visible=False
            )
            components['extra_id'] = extra_id

            extra_name = gr.Textbox(
                label="Name (Optional - will be auto-generated if empty)",
                placeholder="e.g., 'WHO Malnutrition Criteria' or 'Z-score Interpretation Guide'",
                lines=1
            )
            components['extra_name'] = extra_name

            extra_type = gr.Dropdown(
                choices=["pattern", "definition", "guideline", "example", "reference", "criteria", "tip"],
                value="pattern",
                label="Extra Type"
            )
            components['extra_type'] = extra_type
            
            extra_content = gr.TextArea(
                label="Content",
                placeholder="Enter the hint content...",
                lines=8
            )
            components['extra_content'] = extra_content
            
            extra_metadata = gr.TextArea(
                label="Metadata (JSON, optional)",
                placeholder='{"category": "medical", "priority": "high"}',
                lines=2
            )
            components['extra_metadata'] = extra_metadata
            
            with gr.Row():
                add_extra_btn = gr.Button("Add Extra", variant="primary")
                save_extra_btn = gr.Button("Save Extra", variant="primary")
                clear_btn = gr.Button("Clear")
            
            components['add_extra_btn'] = add_extra_btn
            components['save_extra_btn'] = save_extra_btn
            components['clear_btn'] = clear_btn
            
            extra_status = gr.Textbox(label="Status", interactive=False)
            components['extra_status'] = extra_status
        
        with gr.Column(scale=1):
            gr.Markdown("#### Registered Extras")

            extras_list = gr.Dataframe(
                headers=["Name", "Type", "Content Preview"],
                datatype=["str", "str", "str"],
                label="Available Extras",
                interactive=False
            )
            components['extras_list'] = extras_list

            refresh_extras_btn = gr.Button("Refresh List")
            components['refresh_extras_btn'] = refresh_extras_btn

            gr.Markdown("#### Enable/Disable Extras")
            gr.Markdown("*Check to enable, uncheck to disable*")

            extras_checkboxes = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Enabled Extras (Click to toggle)",
                interactive=True
            )
            components['extras_checkboxes'] = extras_checkboxes

            gr.Markdown("#### Manage")

            selected_extra_id = gr.Dropdown(
                choices=[],
                label="Select Extra to View/Edit/Remove",
                allow_custom_value=True,
                info="Choose from registered extras (name or ID)"
            )
            refresh_extra_selector_btn = gr.Button("🔄 Refresh Dropdown", size="sm")

            components['selected_extra_id'] = selected_extra_id
            components['refresh_extra_selector_btn'] = refresh_extra_selector_btn

            with gr.Row():
                view_extra_btn = gr.Button("View/Edit")
                remove_extra_btn = gr.Button("Remove", variant="stop")

            components['view_extra_btn'] = view_extra_btn
            components['remove_extra_btn'] = remove_extra_btn
    
    gr.Markdown("---")
    gr.Markdown("### Example Extras")
    
    with gr.Accordion("Example Patterns", open=False):
        gr.Markdown("""
        **Lab Value Pattern:**
        ```
        Type: pattern
        Content: Lab values format - [test name]: [value] [unit]
        Example: Glucose: 120 mg/dL, Hemoglobin: 14.5 g/dL
        ```
        
        **Medication Dosage Pattern:**
        ```
        Type: pattern
        Content: Medication format - [drug name] [dose] [unit] [frequency]
        Example: Metformin 500 mg twice daily
        ```
        
        **BMI Classification:**
        ```
        Type: definition
        Content: BMI categories - Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: >=30
        ```
        """)
    
    gr.Markdown("---")
    gr.Markdown("### Load from File")
    
    gr.Markdown("""
    Upload a YAML or JSON file containing extras definitions.
    
    **YAML Format:**
    ```yaml
    extras:
      - type: pattern
        content: "Your hint content here"
        metadata:
          category: medical
          priority: high
    ```
    
    **JSON Format:**
    ```json
    {
      "extras": [
        {
          "type": "pattern",
          "content": "Your hint content here",
          "metadata": {
            "category": "medical",
            "priority": "high"
          }
        }
      ]
    }
    ```
    """)
    
    with gr.Row():
        extras_file_upload = gr.File(
            label="Upload Extras File (YAML/JSON)",
            file_types=[".yaml", ".yml", ".json"]
        )
        components['extras_file_upload'] = extras_file_upload
        
        load_extras_file_btn = gr.Button("Load from File", variant="primary")
        components['load_extras_file_btn'] = load_extras_file_btn
    
    load_extras_status = gr.Textbox(label="Load Status", interactive=False)
    components['load_extras_status'] = load_extras_status
    
    gr.Markdown("---")
    gr.Markdown("### Import/Export")
    
    with gr.Row():
        export_extras_btn = gr.Button("Export All Extras")
        import_extras_btn = gr.Button("Import Extras")
    
    components['export_extras_btn'] = export_extras_btn
    components['import_extras_btn'] = import_extras_btn
    
    extras_json = gr.TextArea(
        label="Extras JSON",
        lines=10
    )
    components['extras_json'] = extras_json
    
    # Event handlers
    
    def add_extra(extra_name_val, extra_type_val, content, metadata_str):
        """Add extra to manager"""
        if not content or not content.strip():
            return "", "", "", "", "No content provided", gr.update()

        metadata = {}
        if metadata_str and metadata_str.strip():
            try:
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")
                pass

        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")

        success = extras_manager.add_extra(extra_type_val, content.strip(), metadata, name=extra_name_val)

        if success:
            all_extras = extras_manager.list_extras()
            extras_list_data = [
                [e.get('name', e.get('id', 'unknown')), e.get('type', 'unknown'), (e.get('content', '')[:50] + '...') if e.get('content') else '']
                for e in all_extras
            ]
            # Update checkboxes - show all extras, select enabled ones
            checkbox_choices = [e.get('name', e.get('id', 'unknown')) for e in all_extras]
            checkbox_values = [e.get('name', e.get('id', 'unknown')) for e in all_extras if e.get('enabled', True)]
            return "", "", "", "", "Extra added successfully", gr.update(value=extras_list_data), gr.update(choices=checkbox_choices, value=checkbox_values)
        else:
            return "", "", "", "", "Failed to add extra", gr.update(), gr.update()
    
    def save_extra(extra_id_val, extra_name_val, extra_type_val, content, metadata_str):
        """Save edited extra"""
        if not extra_id_val:
            return "", "", "", "", "", "No ID provided", gr.update()
        if not content or not content.strip():
            return extra_id_val, extra_name_val, extra_type_val, "", metadata_str, "No content provided", gr.update()

        metadata = {}
        if metadata_str and metadata_str.strip():
            try:
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")
                return extra_id_val, extra_name_val, extra_type_val, content, metadata_str, "Invalid metadata JSON", gr.update()

        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")

        success = extras_manager.update_extra(extra_id_val, extra_type_val, content.strip(), metadata, name=extra_name_val)

        if success:
            all_extras = extras_manager.list_extras()
            extras_list_data = [
                [e.get('name', e.get('id', 'unknown')), e.get('type', 'unknown'), (e.get('content', '')[:50] + '...') if e.get('content') else '']
                for e in all_extras
            ]
            checkbox_choices = [e.get('name', e.get('id', 'unknown')) for e in all_extras]
            checkbox_values = [e.get('name', e.get('id', 'unknown')) for e in all_extras if e.get('enabled', True)]
            return "", "", "", "", "", f"Extra updated successfully", gr.update(value=extras_list_data), gr.update(choices=checkbox_choices, value=checkbox_values)
        else:
            return extra_id_val, extra_name_val, extra_type_val, content, metadata_str, "Failed to update extra", gr.update(), gr.update()
    
    def refresh_extras_list():
        """Refresh extras list and checkboxes"""
        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")

        all_extras = extras_manager.list_extras()
        extras_list_data = [
            [e.get('name', e.get('id', 'unknown')), e.get('type', 'unknown'), (e.get('content', '')[:50] + '...') if e.get('content') else '']
            for e in all_extras
        ]
        checkbox_choices = [e.get('name', e.get('id', 'unknown')) for e in all_extras]
        checkbox_values = [e.get('name', e.get('id', 'unknown')) for e in all_extras if e.get('enabled', True)]
        return gr.update(value=extras_list_data), gr.update(choices=checkbox_choices, value=checkbox_values)

    def refresh_extra_selector():
        """Refresh extra selector dropdown"""
        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")

        # Create list with both names and IDs for selection
        extras_choices = []
        for e in extras_manager.list_extras():
            name = e.get('name', '')
            if name:
                extras_choices.append(name)  # Prefer name if available
            else:
                extras_choices.append(e['id'])  # Fall back to ID

        return gr.update(choices=extras_choices, value=extras_choices[0] if extras_choices else None)

    def view_extra(extra_name_or_id):
        """View extra for editing - search by name or ID"""
        if not extra_name_or_id or not extra_name_or_id.strip():
            return "", "", "", "", "", "No name or ID provided", gr.update()

        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")

        # Try to find by ID first
        extra = extras_manager.get_extra(extra_name_or_id.strip())

        # If not found by ID, try to find by name
        if not extra:
            for e in extras_manager.list_extras():
                if e.get('name', '').lower() == extra_name_or_id.strip().lower():
                    extra = e
                    break

        if not extra:
            return "", "", "", "", "", f"Extra '{extra_name_or_id}' not found", gr.update()

        metadata_str = json.dumps(extra.get('metadata', {}), indent=2) if extra.get('metadata') else ""

        return (
            extra['id'],
            extra.get('name', ''),
            extra.get('type', 'pattern'),
            extra.get('content', ''),
            metadata_str,
            f"Loaded: {extra.get('name', extra['id'])}",
            gr.update()
        )
    
    def remove_extra(extra_name_or_id):
        """Remove extra by name or ID"""
        if not extra_name_or_id or not extra_name_or_id.strip():
            return "", "", "", "", "", "No name or ID provided", gr.update()

        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")

        # Try to find by ID first
        extra = extras_manager.get_extra(extra_name_or_id.strip())
        extra_id_to_remove = extra_name_or_id.strip() if extra else None

        # If not found by ID, try to find by name
        if not extra:
            for e in extras_manager.list_extras():
                if e.get('name', '').lower() == extra_name_or_id.strip().lower():
                    extra_id_to_remove = e['id']
                    extra = e
                    break

        if not extra_id_to_remove:
            return "", "", "", "", "", f"Extra '{extra_name_or_id}' not found", gr.update()

        success = extras_manager.remove_extra(extra_id_to_remove)

        if success:
            all_extras = extras_manager.list_extras()
            extras_list_data = [
                [e.get('name', e.get('id', 'unknown')), e.get('type', 'unknown'), (e.get('content', '')[:50] + '...') if e.get('content') else '']
                for e in all_extras
            ]
            checkbox_choices = [e.get('name', e.get('id', 'unknown')) for e in all_extras]
            checkbox_values = [e.get('name', e.get('id', 'unknown')) for e in all_extras if e.get('enabled', True)]
            return "", "", "", "", "", f"Removed '{extra.get('name', extra_id_to_remove)}'", gr.update(value=extras_list_data), gr.update(choices=checkbox_choices, value=checkbox_values)
        else:
            return "", "", "", "", "", "Failed to remove extra", gr.update(), gr.update()
    
    def clear_fields():
        """Clear input fields"""
        return "", "", "", "", "", "Cleared", gr.update()
    
    def load_extras_from_file(file_path):
        """Load extras from YAML or JSON file"""
        if not file_path:
            return "No file selected", gr.update()
        
        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")
        
        try:
            file_path_obj = Path(file_path)
            
            with open(file_path_obj, 'r') as f:
                if file_path_obj.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_path_obj.suffix == '.json':
                    data = json.load(f)
                else:
                    return "Unsupported file format. Use YAML or JSON.", gr.update()
            
            if 'extras' not in data:
                return "Invalid file format. Must contain 'extras' key.", gr.update()
            
            extras_list = data['extras']
            
            added_count = 0
            errors = []
            
            for extra_data in extras_list:
                try:
                    extra_type = extra_data.get('type', 'pattern')
                    content = extra_data.get('content')
                    metadata = extra_data.get('metadata', {})
                    
                    if not content:
                        errors.append("Skipping extra with no content")
                        continue
                    
                    success = extras_manager.add_extra(extra_type, content, metadata)
                    
                    if success:
                        added_count += 1
                    else:
                        errors.append(f"Failed to add extra: {content[:30]}...")
                        
                except Exception as e:
                    errors.append(f"Error processing extra: {str(e)}")

            all_extras = extras_manager.list_extras()
            extras_list_data = [
                [e.get('name', e.get('id', 'unknown')), e.get('type', 'unknown'), (e.get('content', '')[:50] + '...') if e.get('content') else '']
                for e in all_extras
            ]
            checkbox_choices = [e.get('name', e.get('id', 'unknown')) for e in all_extras]
            checkbox_values = [e.get('name', e.get('id', 'unknown')) for e in all_extras if e.get('enabled', True)]

            status_msg = f"Loaded {added_count} extras from file"
            if errors:
                status_msg += f"\n\nWarnings:\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    status_msg += f"\n... and {len(errors) - 5} more"

            return status_msg, gr.update(value=extras_list_data), gr.update(choices=checkbox_choices, value=checkbox_values)
            
        except Exception as e:
            logger.error(f"Failed to load extras file: {e}")
            return f"Error loading file: {str(e)}", gr.update(), gr.update()
    
    def export_extras():
        """Export all extras"""
        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")
        
        return extras_manager.export_extras()
    
    def import_extras(json_str):
        """Import extras"""
        if not json_str or not json_str.strip():
            return "No JSON provided", gr.update()

        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")

        success = extras_manager.import_extras(json_str)

        if success:
            all_extras = extras_manager.list_extras()
            extras_list_data = [
                [e.get('name', e.get('id', 'unknown')), e.get('type', 'unknown'), (e.get('content', '')[:50] + '...') if e.get('content') else '']
                for e in all_extras
            ]
            checkbox_choices = [e.get('name', e.get('id', 'unknown')) for e in all_extras]
            checkbox_values = [e.get('name', e.get('id', 'unknown')) for e in all_extras if e.get('enabled', True)]
            return "Extras imported", gr.update(value=extras_list_data), gr.update(choices=checkbox_choices, value=checkbox_values)
        else:
            return "Import failed", gr.update(), gr.update()

    def update_extras_enabled_state(enabled_extras_names):
        """Update enabled state based on checkbox selections"""
        extras_manager = app_state.get_extras_manager()
        if not extras_manager:
            extras_manager = ExtrasManager()
            app_state.set_extras_manager(extras_manager)
            logger.warning("ExtrasManager was not set; initialized new instance")

        enabled_names_set = set(enabled_extras_names) if enabled_extras_names else set()
        updated_count = 0

        # Update all extras based on checkbox state
        for extra in extras_manager.list_extras():
            extra_name = extra.get('name', extra.get('id', 'unknown'))
            should_be_enabled = extra_name in enabled_names_set
            current_state = extra.get('enabled', True)

            # Only update if state changed
            if current_state != should_be_enabled:
                extras_manager.enable_extra(extra['id'], should_be_enabled)
                updated_count += 1

        status_msg = f"Updated {updated_count} extras" if updated_count > 0 else "No changes"
        return status_msg

    # Connect handlers
    add_extra_btn.click(
        fn=add_extra,
        inputs=[extra_name, extra_type, extra_content, extra_metadata],
        outputs=[extra_id, extra_name, extra_type, extra_content, extra_metadata, extra_status, extras_list, extras_checkboxes]
    )

    save_extra_btn.click(
        fn=save_extra,
        inputs=[extra_id, extra_name, extra_type, extra_content, extra_metadata],
        outputs=[extra_id, extra_name, extra_type, extra_content, extra_metadata, extra_status, extras_list, extras_checkboxes]
    )

    refresh_extras_btn.click(
        fn=refresh_extras_list,
        outputs=[extras_list, extras_checkboxes]
    )

    refresh_extra_selector_btn.click(
        fn=refresh_extra_selector,
        outputs=[selected_extra_id]
    )

    view_extra_btn.click(
        fn=view_extra,
        inputs=[selected_extra_id],
        outputs=[extra_id, extra_name, extra_type, extra_content, extra_metadata, extra_status, extras_list]
    )

    remove_extra_btn.click(
        fn=remove_extra,
        inputs=[selected_extra_id],
        outputs=[extra_id, extra_name, extra_type, extra_content, extra_metadata, extra_status, extras_list, extras_checkboxes]
    )

    clear_btn.click(
        fn=clear_fields,
        outputs=[extra_id, extra_name, extra_type, extra_content, extra_metadata, extra_status, extras_list]
    )

    load_extras_file_btn.click(
        fn=load_extras_from_file,
        inputs=[extras_file_upload],
        outputs=[load_extras_status, extras_list, extras_checkboxes]
    )

    export_extras_btn.click(
        fn=export_extras,
        outputs=[extras_json]
    )

    import_extras_btn.click(
        fn=import_extras,
        inputs=[extras_json],
        outputs=[extra_status, extras_list, extras_checkboxes]
    )

    # Update enabled state when checkboxes change
    extras_checkboxes.change(
        fn=update_extras_enabled_state,
        inputs=[extras_checkboxes],
        outputs=[extra_status]
    )
    
    return components