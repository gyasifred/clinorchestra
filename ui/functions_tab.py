#!/usr/bin/env python3
"""
Functions Tab - Custom Function Management with file loading
"""

import gradio as gr
import json
import yaml
from typing import Dict, Any
from pathlib import Path
import logging
from core.function_registry import FunctionRegistry

logger = logging.getLogger(__name__)

def create_functions_tab(app_state) -> Dict[str, Any]:
    """Create custom functions configuration tab"""
    
    # Ensure FunctionRegistry is initialized
    if app_state.get_function_registry() is None:
        function_registry = FunctionRegistry()
        app_state.set_function_registry(function_registry)
        logger.info("Initialized and registered FunctionRegistry with AppState")
    
    components = {}
    
    gr.Markdown("### Custom Functions")
    gr.Markdown("""
    Define Python functions to calculate derived values that LLMs struggle with.
    
    **Functions File Schema (YAML or JSON):**
    ```yaml
    functions:
      - name: calculate_bmi
        description: "Calculate Body Mass Index"
        code: |
          def calculate_bmi(weight_kg, height_m):
              if height_m <= 0:
                  return None
              return round(weight_kg / (height_m ** 2), 2)
        parameters:
          weight_kg:
            type: number
            description: "Weight in kilograms"
          height_m:
            type: number
            description: "Height in meters"
        returns: "BMI value (number)"
    ```
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### Add/Edit Function")
            
            function_id = gr.Textbox(
                label="Function ID",
                placeholder="Will be generated for new functions",
                interactive=False,
                visible=False
            )
            components['function_id'] = function_id
            
            function_name = gr.Textbox(
                label="Function Name",
                placeholder="calculate_bmi"
            )
            components['function_name'] = function_name
            
            function_description = gr.Textbox(
                label="Description",
                placeholder="Calculate Body Mass Index from weight and height"
            )
            components['function_description'] = function_description
            
            function_code = gr.Code(
                label="Function Code",
                language="python",
                value="""def calculate_bmi(weight_kg, height_m):
    '''Calculate Body Mass Index'''
    if height_m <= 0:
        return None
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)""",
                lines=15
            )
            components['function_code'] = function_code
            
            function_returns = gr.Textbox(
                label="Returns",
                placeholder="BMI value (number)"
            )
            components['function_returns'] = function_returns
        
        with gr.Column(scale=1):
            gr.Markdown("#### Parameters")
            
            param_name = gr.Textbox(
                label="Parameter Name",
                placeholder="weight_kg"
            )
            param_type = gr.Dropdown(
                choices=["string", "number", "boolean"],
                value="number",
                label="Type"
            )
            param_desc = gr.Textbox(
                label="Description",
                placeholder="Weight in kilograms"
            )
            
            components['param_name'] = param_name
            components['param_type'] = param_type
            components['param_desc'] = param_desc
            
            add_param_btn = gr.Button("Add Parameter")
            components['add_param_btn'] = add_param_btn
            
            function_params = gr.JSON(
                label="Function Parameters",
                value={}
            )
            components['function_params'] = function_params
    
    with gr.Row():
        register_btn = gr.Button("Register Function", variant="primary")
        save_function_btn = gr.Button("Save Function", variant="primary")
        clear_btn = gr.Button("Clear")
    
    components['register_btn'] = register_btn
    components['save_function_btn'] = save_function_btn
    components['clear_btn'] = clear_btn
    
    function_status = gr.Textbox(
        label="Status",
        interactive=False
    )
    components['function_status'] = function_status
    
    gr.Markdown("---")
    gr.Markdown("### Test Function")
    
    with gr.Row():
        test_func_name = gr.Dropdown(
            choices=[],
            label="Select Function to Test"
        )
        components['test_func_name'] = test_func_name
        
        refresh_test_dropdown_btn = gr.Button("🔄 Refresh", size="sm")
        components['refresh_test_dropdown_btn'] = refresh_test_dropdown_btn
    
    test_func_args = gr.TextArea(
        label="Test Arguments (JSON format)",
        placeholder='{"weight_kg": 70, "height_m": 1.75}',
        lines=3
    )
    components['test_func_args'] = test_func_args
    
    test_func_btn = gr.Button("Run Test")
    components['test_func_btn'] = test_func_btn
    
    test_func_result = gr.Textbox(
        label="Test Result",
        interactive=False
    )
    components['test_func_result'] = test_func_result
    
    gr.Markdown("---")
    gr.Markdown("### Registered Functions")
    
    functions_list = gr.Dataframe(
        headers=["Name", "Description", "Enabled", "Parameters"],
        datatype=["str", "str", "str", "str"],
        label="Available Functions",
        interactive=False
    )
    components['functions_list'] = functions_list
    
    refresh_functions_btn = gr.Button("Refresh List")
    components['refresh_functions_btn'] = refresh_functions_btn
    
    gr.Markdown("#### Manage")

    with gr.Row():
        selected_func_name = gr.Dropdown(
            choices=[],
            label="Select Function to View/Edit/Remove",
            allow_custom_value=True,
            info="Choose from registered functions"
        )
        refresh_function_selector_btn = gr.Button("🔄 Refresh", size="sm")

    components['selected_func_name'] = selected_func_name
    components['refresh_function_selector_btn'] = refresh_function_selector_btn

    with gr.Row():
        view_func_btn = gr.Button("View/Edit")
        toggle_func_btn = gr.Button("Toggle")
        remove_func_btn = gr.Button("Remove", variant="stop")

    components['view_func_btn'] = view_func_btn
    components['toggle_func_btn'] = toggle_func_btn
    components['remove_func_btn'] = remove_func_btn
    
    gr.Markdown("---")
    gr.Markdown("### Load from File")
    
    gr.Markdown("""
    Upload a YAML or JSON file containing function definitions.
    
    **YAML Format:**
    ```yaml
    functions:
      - name: function_name
        description: "What the function does"
        code: |
          def function_name(param1, param2):
              # function body
              return result
        parameters:
          param1:
            type: number
            description: "Parameter description"
        returns: "Return value description"
    ```
    
    **JSON Format:**
    ```json
    {
      "functions": [
        {
          "name": "function_name",
          "description": "What the function does",
          "code": "def function_name(param1, param2):\\n    return result",
          "parameters": {
            "param1": {
              "type": "number",
              "description": "Parameter description"
            }
          },
          "returns": "Return value description"
        }
      ]
    }
    ```
    """)
    
    with gr.Row():
        functions_file_upload = gr.File(
            label="Upload Functions File (YAML/JSON)",
            file_types=[".yaml", ".yml", ".json"]
        )
        components['functions_file_upload'] = functions_file_upload
        
        load_functions_file_btn = gr.Button("Load from File", variant="primary")
        components['load_functions_file_btn'] = load_functions_file_btn
    
    load_functions_status = gr.Textbox(label="Load Status", interactive=False)
    components['load_functions_status'] = load_functions_status
    
    gr.Markdown("---")
    gr.Markdown("### Import/Export")
    
    export_functions_btn = gr.Button("Export All Functions")
    components['export_functions_btn'] = export_functions_btn
    
    functions_json = gr.TextArea(
        label="Functions JSON",
        lines=10
    )
    components['functions_json'] = functions_json
    
    import_functions_btn = gr.Button("Import Functions")
    components['import_functions_btn'] = import_functions_btn
    
    # Event handlers
    
    def add_parameter(param_n, param_t, param_d, current_params):
        """Add parameter to function"""
        if not param_n:
            return "Parameter name required", current_params
        
        new_params = current_params.copy()
        new_params[param_n] = {
            'type': param_t,
            'description': param_d
        }
        
        return f"Added parameter: {param_n}", new_params
    
    def register_function(func_name, func_desc, func_code, func_returns, func_params):
        """Register function"""
        if not func_name or not func_code:
            return "", "", "", "", {}, "Function name and code required", gr.update()
        
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        success, message = function_registry.register_function(
            func_name,
            func_code,
            func_desc,
            func_params,
            func_returns
        )
        
        if success:
            data = []
            for f_name in function_registry.list_functions():
                func_info = function_registry.get_function_info(f_name)
                if func_info:
                    params_str = ', '.join(func_info.get('parameters', {}).keys())
                    data.append([f_name, func_info.get('description', ''), "Yes" if func_info.get('enabled', True) else "No", params_str])
            return "", "", "", "", {}, f"Success: {message}", gr.update(value=data)
        else:
            return func_name, func_desc, func_code, func_returns, func_params, f"Error: {message}", gr.update()
    
    def save_function(func_id, func_name, func_desc, func_code, func_returns, func_params):
        """Save edited function"""
        if not func_id:
            return "", "", "", "", {}, "No ID provided", gr.update()
        if not func_name or not func_code:
            return func_id, func_name, func_desc, func_code, func_returns, func_params, "Function name and code required", gr.update()
        
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        success, message = function_registry.update_function(
            func_id,
            func_name,
            func_code,
            func_desc,
            func_params,
            func_returns
        )
        
        if success:
            data = []
            for f_name in function_registry.list_functions():
                func_info = function_registry.get_function_info(f_name)
                if func_info:
                    params_str = ', '.join(func_info.get('parameters', {}).keys())
                    data.append([f_name, func_info.get('description', ''), "Yes" if func_info.get('enabled', True) else "No", params_str])
            return "", "", "", "", {}, f"Success: {message}", gr.update(value=data)
        else:
            return func_id, func_name, func_desc, func_code, func_returns, func_params, f"Error: {message}", gr.update()
    
    def clear_fields():
        """Clear input fields"""
        return "", "", "", """def my_function():
    pass""", "", {}, "Cleared", gr.update()
    
    def refresh_functions_list():
        """Refresh functions list"""
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        data = []
        for func_name in function_registry.list_functions():
            func_info = function_registry.get_function_info(func_name)
            if func_info:
                params_str = ', '.join(func_info.get('parameters', {}).keys())
                data.append([
                    func_name,
                    func_info.get('description', ''),
                    "Yes" if func_info.get('enabled', True) else "No",
                    params_str
                ])

        return gr.update(value=data)
    
    def refresh_test_dropdown():
        """Refresh test function dropdown with registered functions"""
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")

        functions = function_registry.list_functions()

        if not functions:
            return gr.update(choices=[], value=None)

        return gr.update(choices=functions, value=functions[0] if functions else None)

    def refresh_function_selector():
        """Refresh function selector dropdown"""
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")

        functions = function_registry.list_functions()
        return gr.update(choices=functions, value=functions[0] if functions else None)

    def view_function(func_name):
        """View function details"""
        if not func_name or not func_name.strip():
            return "", "", "", "", "", {}, "No name provided", gr.update()
        
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        func_info = function_registry.get_function_info(func_name.strip())
        if not func_info:
            return "", "", "", "", "", {}, f"Function '{func_name}' not found", gr.update()
        
        return (
            func_info.get('name', ''),  # Use name as ID (functions are keyed by name)
            func_info.get('name', ''),
            func_info.get('description', ''),
            func_info.get('code', ''),
            func_info.get('returns', ''),
            func_info.get('parameters', {}),
            f"Loaded: {func_name}",
            gr.update()
        )
    
    def remove_function(func_name):
        """Remove function"""
        if not func_name or not func_name.strip():
            return "", "", "", "", "", {}, "No name provided", gr.update()
        
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        success, message = function_registry.remove_function(func_name.strip())
        
        if success:
            data = []
            for f_name in function_registry.list_functions():
                func_info = function_registry.get_function_info(f_name)
                if func_info:
                    params_str = ', '.join(func_info.get('parameters', {}).keys())
                    data.append([f_name, func_info.get('description', ''), "Yes" if func_info.get('enabled', True) else "No", params_str])
            return "", "", "", "", "", {}, message, gr.update(value=data)
        else:
            return "", "", "", "", "", {}, message, gr.update()
    
    def test_function(func_name, func_args_json):
        """Test function"""
        if not func_name:
            return "Select a function to test"
        
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        try:
            args = json.loads(func_args_json)
            success, result, message = function_registry.execute_function(func_name, **args)
            
            if success:
                return f"Success: {result}"
            else:
                return f"Error: {message}"
                
        except json.JSONDecodeError:
            return "Invalid JSON arguments"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def load_functions_from_file(file_path):
        """Load functions from YAML or JSON file"""
        if not file_path:
            return "No file selected", gr.update()
        
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        try:
            file_path_obj = Path(file_path)
            
            with open(file_path_obj, 'r') as f:
                if file_path_obj.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_path_obj.suffix == '.json':
                    data = json.load(f)
                else:
                    return "Unsupported file format. Use YAML or JSON.", gr.update()
            
            if 'functions' not in data:
                return "Invalid file format. Must contain 'functions' key.", gr.update()
            
            functions_list = data['functions']
            
            added_count = 0
            errors = []
            
            for func_data in functions_list:
                try:
                    name = func_data.get('name')
                    code = func_data.get('code')
                    description = func_data.get('description', '')
                    parameters = func_data.get('parameters', {})
                    returns = func_data.get('returns', '')
                    
                    if not name or not code:
                        errors.append("Skipping function with missing name or code")
                        continue
                    
                    success, message = function_registry.register_function(
                        name, code, description, parameters, returns
                    )
                    
                    if success:
                        added_count += 1
                    else:
                        errors.append(f"{name}: {message}")
                        
                except Exception as e:
                    errors.append(f"Error processing function: {str(e)}")
            
            func_list_data = []
            for func_name in function_registry.list_functions():
                func_info = function_registry.get_function_info(func_name)
                if func_info:
                    params_str = ', '.join(func_info.get('parameters', {}).keys())
                    func_list_data.append([
                        func_name,
                        func_info.get('description', ''),
                        "Yes" if func_info.get('enabled', True) else "No",
                        params_str
                    ])

            status_msg = f"Loaded {added_count} functions from file"
            if errors:
                status_msg += f"\n\nWarnings:\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    status_msg += f"\n... and {len(errors) - 5} more"
            
            return status_msg, gr.update(value=func_list_data)
            
        except Exception as e:
            logger.error(f"Failed to load functions file: {e}")
            return f"Error loading file: {str(e)}", gr.update()
    
    def export_functions():
        """Export functions"""
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        return function_registry.export_functions()
    
    def import_functions(json_str):
        """Import functions"""
        if not json_str or not json_str.strip():
            return "No JSON provided", gr.update()
        
        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")
        
        success, count, message = function_registry.import_functions(json_str)
        
        if success:
            data = []
            for f_name in function_registry.list_functions():
                func_info = function_registry.get_function_info(f_name)
                if func_info:
                    params_str = ', '.join(func_info.get('parameters', {}).keys())
                    data.append([f_name, func_info.get('description', ''), "Yes" if func_info.get('enabled', True) else "No", params_str])
            return f"Success: {message}", gr.update(value=data)
        else:
            return f"Error: {message}", gr.update()
    
    def toggle_function(func_name):
        """Toggle function enabled state"""
        if not func_name or not func_name.strip():
            return "No function name provided", gr.update()

        function_registry = app_state.get_function_registry()
        if not function_registry:
            function_registry = FunctionRegistry()
            app_state.set_function_registry(function_registry)
            logger.warning("FunctionRegistry was not set; initialized new instance")

        success, message = function_registry.enable_function(
            func_name.strip(),
            not function_registry.get_function_info(func_name.strip()).get('enabled', True)
        )

        if success:
            data = []
            for f_name in function_registry.list_functions():
                func_info = function_registry.get_function_info(f_name)
                if func_info:
                    params_str = ', '.join(func_info.get('parameters', {}).keys())
                    data.append([f_name, func_info.get('description', ''), "Yes" if func_info.get('enabled', True) else "No", params_str])
            return message, gr.update(value=data)
        else:
            return message, gr.update()

    # Connect handlers
    add_param_btn.click(
        fn=add_parameter,
        inputs=[param_name, param_type, param_desc, function_params],
        outputs=[function_status, function_params]
    )
    
    register_btn.click(
        fn=register_function,
        inputs=[function_name, function_description, function_code, function_returns, function_params],
        outputs=[function_id, function_name, function_description, function_code, function_returns, function_params, function_status, functions_list]
    )
    
    save_function_btn.click(
        fn=save_function,
        inputs=[function_id, function_name, function_description, function_code, function_returns, function_params],
        outputs=[function_id, function_name, function_description, function_code, function_returns, function_params, function_status, functions_list]
    )
    
    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[function_id, function_name, function_description, function_code, function_returns, function_params, function_status, functions_list]
    )
    
    refresh_functions_btn.click(
        fn=refresh_functions_list,
        outputs=[functions_list]
    )

    refresh_function_selector_btn.click(
        fn=refresh_function_selector,
        outputs=[selected_func_name]
    )

    refresh_test_dropdown_btn.click(
        fn=refresh_test_dropdown,
        outputs=[test_func_name]
    )
    
    view_func_btn.click(
        fn=view_function,
        inputs=[selected_func_name],
        outputs=[function_id, function_name, function_description, function_code, function_returns, function_params, function_status, functions_list]
    )

    toggle_func_btn.click(
        fn=toggle_function,
        inputs=[selected_func_name],
        outputs=[function_status, functions_list]
    )

    remove_func_btn.click(
        fn=remove_function,
        inputs=[selected_func_name],
        outputs=[function_id, function_name, function_description, function_code, function_returns, function_params, function_status, functions_list]
    )
    
    test_func_btn.click(
        fn=test_function,
        inputs=[test_func_name, test_func_args],
        outputs=[test_func_result]
    )
    
    load_functions_file_btn.click(
        fn=load_functions_from_file,
        inputs=[functions_file_upload],
        outputs=[load_functions_status, functions_list]
    )
    
    export_functions_btn.click(
        fn=export_functions,
        outputs=[functions_json]
    )
    
    import_functions_btn.click(
        fn=import_functions,
        inputs=[functions_json],
        outputs=[function_status, functions_list]
    )
    
    return components