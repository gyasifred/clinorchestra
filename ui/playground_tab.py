#!/usr/bin/env python3
"""
Complete Fixed Playground Tab
Shows detailed logging of extras, RAG, and function usage
"""
import logging
import gradio as gr
import json
import time
import datetime
from typing import Dict, Any
from core.process_persistence import get_process_state

logger = logging.getLogger(__name__)

def create_playground_tab(app_state) -> Dict[str, Any]:
    """Create complete playground tab"""
    
    components = {}
    
    gr.Markdown("### Playground - Experiment With Your Prompt")
    gr.Markdown("""
    Test all aspects with detailed logging:
    - **Full Pipeline**: See extras scanning, RAG retrieval, function calling
    - **Function Testing**: Test custom functions independently
    - **Extras Testing**: Test hints system
    """)
    
    with gr.Tabs():
        with gr.Tab("Full Pipeline Test"):
            gr.Markdown("### Test Complete Extraction with Detailed Logging")
            
            with gr.Row():
                with gr.Column(scale=2):
                    test_text = gr.TextArea(
                        label="Clinical Text",
                        placeholder="Paste clinical text...",
                        lines=15
                    )
                    components['test_text'] = test_text
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Options")
                    
                    test_with_label = gr.Checkbox(
                        label="Test with Label",
                        value=False
                    )
                    components['test_with_label'] = test_with_label
                    
                    with gr.Column(visible=False) as label_panel:
                        label_dropdown = gr.Dropdown(
                            choices=[],
                            label="Label Value"
                        )
                        components['label_dropdown'] = label_dropdown
                        
                        refresh_labels = gr.Button("Refresh Labels")
                        components['refresh_labels'] = refresh_labels
                    
                    components['label_panel'] = label_panel
                    
                    test_btn = gr.Button("Run Test", variant="primary", size="lg")
                    components['test_btn'] = test_btn
                    
                    clear_btn = gr.Button("Clear")
                    components['clear_btn'] = clear_btn
            
            test_with_label.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[test_with_label],
                outputs=[label_panel]
            )
            
            gr.Markdown("---")
            gr.Markdown("### Results")
            
            test_status = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False
            )
            components['test_status'] = test_status
            
            execution_log = gr.TextArea(
                label="Execution Log (Detailed)",
                lines=20,
                interactive=False
            )
            components['execution_log'] = execution_log
            
            with gr.Tabs():
                with gr.Tab("JSON Output"):
                    json_output = gr.JSON(label="Extracted Data", value={})
                    components['json_output'] = json_output
                
                with gr.Tab("Extras Used"):
                    extras_used = gr.JSON(label="Extras Applied", value=[])
                    components['extras_used'] = extras_used
                
                with gr.Tab("RAG Retrieved"):
                    rag_retrieved = gr.JSON(label="RAG Evidence", value=[])
                    components['rag_retrieved'] = rag_retrieved
                
                with gr.Tab("Functions Called"):
                    functions_called = gr.JSON(label="Function Calls", value=[])
                    components['functions_called'] = functions_called
        
        with gr.Tab("Function Testing"):
            gr.Markdown("### Test Custom Functions")
            
            with gr.Row():
                refresh_funcs = gr.Button("Refresh Functions")
                components['refresh_funcs'] = refresh_funcs
            
            func_dropdown = gr.Dropdown(
                choices=[],
                label="Select Function"
            )
            components['func_dropdown'] = func_dropdown
            
            func_info = gr.JSON(label="Function Info", value={})
            components['func_info'] = func_info
            
            func_params = gr.TextArea(
                label="Parameters (JSON)",
                placeholder='{"weight_kg": 70, "height_m": 1.75}',
                lines=5
            )
            components['func_params'] = func_params
            
            test_func_btn = gr.Button("Run Function", variant="primary")
            components['test_func_btn'] = test_func_btn
            
            func_result = gr.TextArea(label="Result", lines=10, interactive=False)
            components['func_result'] = func_result
        
        with gr.Tab("Extras Testing"):
            gr.Markdown("### Test Extras/Hints")
            
            with gr.Row():
                refresh_extras = gr.Button("Refresh Extras")
                components['refresh_extras'] = refresh_extras
            
            extras_dropdown = gr.CheckboxGroup(
                choices=[],
                label="Available Extras"
            )
            components['extras_dropdown'] = extras_dropdown
            
            test_text_extras = gr.TextArea(
                label="Test Text",
                lines=10
            )
            components['test_text_extras'] = test_text_extras
            
            test_extras_btn = gr.Button("Test Extras", variant="primary")
            components['test_extras_btn'] = test_extras_btn
            
            extras_result = gr.TextArea(label="Matched Extras", lines=15, interactive=False)
            components['extras_result'] = extras_result
    
    # ==========================================
    # FULL PIPELINE TEST
    # ==========================================
    def test_full_pipeline(text, use_label, label_val):
        """Test with detailed logging - FIXED to show extras content"""
        
        if not text or not text.strip():
            return "No text", "", {}, [], [], []
        
        if not app_state.config_valid or not app_state.prompt_valid:
            return "Configuration incomplete", "", {}, [], [], []
        
        process_state = get_process_state()
        process_id = f"playground_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        process_state.create_process(process_id, {'total_rows': 1})
        
        log = []
        log.append("=" * 60)
        log.append(f"EXECUTION LOG (Process ID: {process_id})")
        log.append("=" * 60)
        log.append("")
        process_state.add_log(process_id, "Starting full pipeline test")
        
        try:
            start_time = time.time()
            
            from core.agent_factory import create_agent, get_agent_info
            
            llm_manager = app_state.get_llm_manager()
            log.append(f"Using LLM Manager: {llm_manager.provider}/{llm_manager.model_name}")
            
            regex_preprocessor = app_state.get_regex_preprocessor()
            log.append("Using Regex Preprocessor")
            
            extras_manager = app_state.get_extras_manager()
            all_extras = extras_manager.list_extras()
            log.append(f"Extras Manager: {len(all_extras)} extras available")
            
            function_registry = app_state.get_function_registry()
            all_funcs = function_registry.list_functions()
            log.append(f"Function Registry: {len(all_funcs)} functions available")
            
            rag_engine = app_state.get_rag_engine()
            if app_state.rag_config.enabled and rag_engine and rag_engine.initialized:
                log.append(f"RAG Engine: {len(rag_engine.documents_loaded)} documents loaded")
            else:
                log.append("RAG is DISABLED")
            
            log.append("")
            log.append("=" * 60)
            log.append("STARTING AGENT EXECUTION")
            log.append("=" * 60)
            log.append("")

            # Get agent info
            agent_info = get_agent_info(app_state)
            log.append(f"Agent Version: {agent_info['version']}")
            log.append(f"Agent Type: {agent_info['name']}")
            log.append(f"Execution Mode: {agent_info['mode']}")
            if app_state.agentic_config.enabled:
                log.append(f"Max Iterations: {agent_info['config']['max_iterations']}")
                log.append(f"Max Tool Calls: {agent_info['config']['max_tool_calls']}")
            log.append("")

            # Create agent using factory
            agent = create_agent(
                llm_manager=llm_manager,
                rag_engine=rag_engine if app_state.rag_config.enabled else None,
                extras_manager=extras_manager,
                function_registry=function_registry,
                regex_preprocessor=regex_preprocessor,
                app_state=app_state
            )
            log.append(f"{agent_info['name']} initialized")
            
            label_value = label_val if use_label else None
            log.append(f"Processing text ({len(text)} chars) with label: {label_value}")
            
            log.append("")
            log.append("Agent.extract() called...")
            log.append("")
            
            result = agent.extract(text, label_value)
            elapsed = time.time() - start_time
            
            # FIXED: Use stage4 if RAG refinement applied
            if result.get('rag_refinement_applied', False):
                final_output = result.get('stage4_final_output', {})
                log.append("RAG refinement was applied - using Stage 4 output")
            else:
                final_output = result.get('stage3_output', {})
                log.append("No RAG refinement - using Stage 3 output")
            
            extras_used = result.get('extras_used', 0)
            rag_used = result.get('rag_used', 0)
            functions_called = result.get('functions_called', 0)
            
            # FIXED: Get detailed extras content directly from extras_details
            extras_details = result.get('processing_metadata', {}).get('extras_details', [])
            # Use extras_details directly - it already has all the information we need
            extras_full_content = extras_details if extras_details else []
            
            rag_details = result.get('processing_metadata', {}).get('rag_details', [])
            function_details = result.get('processing_metadata', {}).get('function_calls_details', [])
            
            log.append("=" * 60)
            log.append("AGENT EXECUTION SUMMARY")
            log.append("=" * 60)
            log.append("")
            log.append(f"Processing Time: {elapsed:.2f}s")
            log.append(f"Retry Count: {result.get('retry_count', 0)}")
            log.append(f"Prompt Type: {'Minimal' if result.get('used_minimal_prompt', False) else 'Main'}")
            log.append(f"RAG Refinement: {'Applied' if result.get('rag_refinement_applied', False) else 'Not applied'}")
            log.append("")
            
            log.append("COMPONENTS USED:")
            log.append(f"  • Extras: {extras_used}")
            if extras_used > 0 and extras_full_content:
                log.append("    Extras Content:")
                for i, extra in enumerate(extras_full_content, 1):
                    log.append(f"      [{i}] {extra.get('type', 'unknown').upper()}")
                    log.append(f"          ID: {extra.get('id', 'N/A')}")
                    log.append(f"          Relevance: {extra.get('relevance_score', 0):.3f}")
                    log.append(f"          Matched Keywords: {', '.join(extra.get('matched_keywords', []))}")
                    log.append(f"          Content: {extra.get('content', '')[:200]}...")
                    log.append("")
            
            log.append(f"  • RAG Chunks: {rag_used}")
            if rag_used > 0:
                log.append("    RAG Details:")
                for i, rag_item in enumerate(rag_details[:2], 1):
                    log.append(f"      [{i}] Score: {rag_item.get('score', 0):.4f}, Source: {rag_item.get('source', 'Unknown')}")
                    log.append(f"          Query: {rag_item.get('query', 'N/A')}")
                    log.append(f"          Content: {rag_item.get('content', '')[:120]}...")
            
            log.append(f"  • Functions: {functions_called}")
            if functions_called > 0:
                log.append("    Function Details:")
                for func in function_details:
                    log.append(f"      • {func.get('name', 'unknown')}() = {func.get('result', 'N/A')}")
            
            log.append("")
            log.append("=" * 60)
            log.append("EXECUTION COMPLETE")
            log.append("=" * 60)
            
            log_text = "\n".join(log)
            
            status = f"Test completed in {elapsed:.2f}s"
            if use_label:
                status += f"\nLabel: {label_val}"
            status += f"\nExtras: {extras_used} | RAG: {rag_used} | Functions: {functions_called}"
            status += f"\nRAG Refinement: {'Applied' if result.get('rag_refinement_applied', False) else 'Not applied'}"
            
            return (
                status,
                log_text,
                final_output,
                extras_full_content,  # FIXED: Return full extras content
                rag_details,
                function_details
            )
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}", exc_info=True)
            log.append("")
            log.append("=" * 60)
            log.append("ERROR")
            log.append("=" * 60)
            log.append(f"{str(e)}")
            
            return (
                f"Error: {str(e)}",
                "\n".join(log),
                {},
                [],
                [],
                []
            )
    
        
    def clear_test_results():
        """Clear test results"""
        return (
            "Ready",
            "",
            {},
            [],
            [],
            []
        )
    
    def refresh_label_choices():
        """Refresh label choices"""
        if not app_state.data_config.has_labels:
            return gr.update(choices=[], value=None)
        
        label_mapping = app_state.data_config.label_mapping
        choices = sorted(list(label_mapping.keys()))
        
        return gr.update(choices=choices, value=choices[0] if choices else None)
    
    def refresh_functions():
        """Refresh function list"""
        try:
            function_registry = app_state.get_function_registry()
            func_names = function_registry.list_functions()
            
            logger.info(f"Refreshed functions: {func_names}")
            
            if not func_names:
                return gr.update(choices=[], value=None)
            
            return gr.update(choices=func_names, value=func_names[0] if func_names else None)
            
        except Exception as e:
            logger.error(f"Failed to refresh functions: {e}", exc_info=True)
            return gr.update(choices=[], value=None)
    
    def show_function_info(func_name):
        """Show function info"""
        if not func_name:
            return {}
        
        try:
            function_registry = app_state.get_function_registry()
            func_info = function_registry.get_function_info(func_name)
            
            if func_info:
                return {
                    'name': func_info.get('name'),
                    'description': func_info.get('description'),
                    'parameters': func_info.get('parameters', {}),
                    'returns': func_info.get('returns', 'Unknown')
                }
            else:
                return {'error': 'Function not found'}
                
        except Exception as e:
            logger.error(f"Failed to get function info: {e}", exc_info=True)
            return {'error': str(e)}
    
    def test_function(func_name, params_json):
        """Test function call"""
        if not func_name:
            return "Select a function"
        
        try:
            params = json.loads(params_json) if params_json.strip() else {}
            function_registry = app_state.get_function_registry()
            success, result, message = function_registry.execute_function(func_name, **params)
            
            if success:
                output = f"Function executed successfully!\n\n"
                output += f"Function: {func_name}\n"
                output += f"Parameters: {json.dumps(params, indent=2)}\n\n"
                output += f"Result: {json.dumps(result, indent=2)}"
                return output
            else:
                return f"Function failed: {message}"
                
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {str(e)}"
        except Exception as e:
            logger.error(f"Function test failed: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    def refresh_extras_list():
        """Refresh extras list"""
        try:
            extras_manager = app_state.get_extras_manager()
            extras = extras_manager.list_extras()
            
            choices = [
                f"{e['id']} - {e['type']}: {e['content'][:50]}..."
                for e in extras
            ]
            
            return gr.update(choices=choices, value=[])
            
        except Exception as e:
            logger.error(f"Failed to refresh extras: {e}", exc_info=True)
            return gr.update(choices=[], value=[])
    
    def test_extras_matching(text, selected_extras):
        """Test extras matching"""
        if not text or not text.strip():
            return "No text provided"
        
        try:
            extras_manager = app_state.get_extras_manager()
            all_extras = extras_manager.list_extras()
            
            text_lower = text.lower()
            matched = []
            
            for extra in all_extras:
                extra_id = extra.get('id', '')
                content = extra.get('content', '')
                
                if selected_extras:
                    extra_display = f"{extra_id} - {extra.get('type', '')}: {content[:50]}..."
                    if extra_display in selected_extras:
                        matched.append(extra)
                else:
                    keywords = content.lower().split()[:10]
                    if any(kw in text_lower for kw in keywords if len(kw) > 3):
                        matched.append(extra)
            
            if not matched:
                return "No extras matched"
            
            result = f"Matched {len(matched)} extras:\n\n"
            
            for i, extra in enumerate(matched, 1):
                result += f"[{i}] {extra.get('type', 'unknown').upper()}\n"
                result += f"    ID: {extra.get('id', 'N/A')}\n"
                result += f"    Content: {extra.get('content', '')}\n"
                result += f"    Metadata: {extra.get('metadata', {})}\n"
                result += "\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Extras test failed: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    # ==========================================
    # EVENT BINDINGS
    # ==========================================
    test_btn.click(
        fn=test_full_pipeline,
        inputs=[test_text, test_with_label, label_dropdown],
        outputs=[
            test_status,
            execution_log,
            json_output,
            extras_used,
            rag_retrieved,
            functions_called
        ]
    )
    
    clear_btn.click(
        fn=clear_test_results,
        outputs=[
            test_status,
            execution_log,
            json_output,
            extras_used,
            rag_retrieved,
            functions_called
        ]
    )
    
    refresh_labels.click(
        fn=refresh_label_choices,
        outputs=[label_dropdown]
    )
    
    refresh_funcs.click(
        fn=refresh_functions,
        outputs=[func_dropdown]
    )
    
    func_dropdown.change(
        fn=show_function_info,
        inputs=[func_dropdown],
        outputs=[func_info]
    )
    
    test_func_btn.click(
        fn=test_function,
        inputs=[func_dropdown, func_params],
        outputs=[func_result]
    )
    
    refresh_extras.click(
        fn=refresh_extras_list,
        outputs=[extras_dropdown]
    )
    
    test_extras_btn.click(
        fn=test_extras_matching,
        inputs=[test_text_extras, extras_dropdown],
        outputs=[extras_result]
    )
    
    return components