#!/usr/bin/env python3
"""
Config Tab - Model Configuration with Connection Testing
Version: 1.0.0
"""

import gradio as gr
from typing import Dict, Any
import logging
import time

logger = logging.getLogger(__name__)

def create_config_tab(app_state) -> Dict[str, Any]:
    """Create model configuration tab with testing"""
    
    from core.model_config import ModelConfig, ModelConfigManager
    from core.config_persistence import get_persistence_manager
    
    config_manager = ModelConfigManager()
    persistence_manager = get_persistence_manager()
    components = {}
    
    # FIXED: Load saved configuration but DON'T initialize LLM automatically
    saved_config = persistence_manager.load_model_config()
    
    gr.Markdown("### Model Configuration")
    gr.Markdown("Configure your LLM provider and model settings.")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### Provider Settings")
            
            provider_value = saved_config.get('provider', 'openai') if saved_config else 'openai'
            provider = gr.Dropdown(
                choices=["openai", "anthropic", "google", "azure", "local"],
                value=provider_value,
                label="Provider"
            )
            components['provider'] = provider
            
            model_name_value = saved_config.get('model_name', 'gpt-4o') if saved_config else 'gpt-4o'
            model_name = gr.Dropdown(
                choices=config_manager.get_supported_models(provider_value),
                value=model_name_value,
                label="Model"
            )
            components['model_name'] = model_name
            
            context_info = gr.Textbox(
                label="Context Window",
                value=f"Context Window: {config_manager.get_model_context_limit(model_name_value):,} tokens",
                interactive=False
            )
            components['context_info'] = context_info
            
            # FIXED: Don't pre-fill API key even if saved (security)
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Enter API key or load from environment",
                value=""  # Always start empty
            )
            components['api_key'] = api_key
            
            with gr.Row():
                load_from_env_btn = gr.Button("Load from Environment")
                components['load_from_env_btn'] = load_from_env_btn
        
        with gr.Column(scale=1):
            gr.Markdown("#### Model Parameters")
            
            temperature_value = saved_config.get('temperature', 0.0) if saved_config else 0.0
            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=temperature_value,
                step=0.1,
                label="Temperature",
                info="0 = deterministic, higher = more creative"
            )
            components['temperature'] = temperature
            
            max_tokens_value = saved_config.get('max_tokens', 4000) if saved_config else 4000
            max_tokens = gr.Number(
                value=max_tokens_value,
                label="Max Output Tokens",
                precision=0,
                minimum=100
            )
            components['max_tokens'] = max_tokens
            
            model_type_value = saved_config.get('model_type', 'chat') if saved_config else 'chat'
            model_type = gr.Dropdown(
                choices=["chat", "completion"],
                value=model_type_value,
                label="Model Type"
            )
            components['model_type'] = model_type

    gr.Markdown("---")
    gr.Markdown("### Execution Mode Settings")
    gr.Markdown("""
    **Choose your execution mode:**

    🎯 **Use STRUCTURED Mode for predictable workflows; use ADAPTIVE Mode for evolving tasks.**

    - **STRUCTURED Mode** (Default): Systematic 4-stage pipeline - best for production/predictable workflows
    - **ADAPTIVE Mode** (Advanced): Continuous iteration with dynamic adaptation - best for complex/evolving tasks (60-75% faster)

    Both modes are autonomous (agentic) and adapt to ANY clinical task!
    """)

    with gr.Row():
        with gr.Column():
            adaptive_mode_enabled_value = app_state.agentic_config.enabled if hasattr(app_state, 'agentic_config') else False
            adaptive_mode_enabled = gr.Checkbox(
                label="Enable ADAPTIVE Mode (v1.0.0)",
                value=adaptive_mode_enabled_value,
                info="For evolving tasks requiring iterative refinement and dynamic adaptation"
            )
            components['agentic_enabled'] = adaptive_mode_enabled  # Keep internal name for compatibility

        with gr.Column():
            max_iterations_value = app_state.agentic_config.max_iterations if hasattr(app_state, 'agentic_config') else 3
            agentic_max_iterations = gr.Number(
                value=max_iterations_value,
                label="Max Iterations",
                precision=0,
                minimum=3,
                maximum=100,
                info="Maximum conversation iterations in agentic loop (minimum: 3)"
            )
            components['agentic_max_iterations'] = agentic_max_iterations

            max_tool_calls_value = app_state.agentic_config.max_tool_calls if hasattr(app_state, 'agentic_config') else 50
            agentic_max_tool_calls = gr.Number(
                value=max_tool_calls_value,
                label="Max Tool Calls",
                precision=0,
                minimum=10,
                maximum=200,
                info="Maximum total tool calls per extraction"
            )
            components['agentic_max_tool_calls'] = agentic_max_tool_calls

    mode_info = gr.Markdown("""
    **ℹ️ Execution Mode Info:**
    - **STRUCTURED Mode** (v1.0.0): For predictable workflows - systematic 4-stage pipeline
    - **ADAPTIVE Mode** (v1.0.0): For evolving tasks - continuous iteration with dynamic adaptation

    🎯 **Both are autonomous (agentic)** - adapt to ANY clinical task via prompts/schema!

    Use STRUCTURED for predictable workflows. Use ADAPTIVE for evolving tasks.

    See [EXECUTION_MODES_GUIDE.md] for detailed comparison
    """)
    components['agentic_info'] = mode_info  # Keep internal name for compatibility

    gr.Markdown("---")
    gr.Markdown("### Provider-Specific Settings")
    
    with gr.Column(visible=(provider_value == "azure")) as azure_config:
        gr.Markdown("#### Azure OpenAI Settings")
        azure_endpoint_value = saved_config.get('azure_endpoint', '') if saved_config else ''
        azure_endpoint = gr.Textbox(
            label="Azure Endpoint",
            placeholder="https://your-resource.openai.azure.com/",
            value=azure_endpoint_value
        )
        azure_deployment_value = saved_config.get('azure_deployment', '') if saved_config else ''
        azure_deployment = gr.Textbox(
            label="Deployment Name",
            placeholder="your-deployment-name",
            value=azure_deployment_value
        )
        components['azure_endpoint'] = azure_endpoint
        components['azure_deployment'] = azure_deployment
    
    components['azure_config'] = azure_config
    
    with gr.Column(visible=(provider_value == "google")) as google_config:
        gr.Markdown("#### Google AI Settings")
        google_project_id_value = saved_config.get('google_project_id', '') if saved_config else ''
        google_project_id = gr.Textbox(
            label="Project ID",
            placeholder="your-project-id",
            value=google_project_id_value
        )
        components['google_project_id'] = google_project_id
    
    components['google_config'] = google_config
    
    with gr.Column(visible=(provider_value == "local")) as local_config:
        gr.Markdown("#### Local Model Settings")
        local_model_path_value = saved_config.get('local_model_path', '') if saved_config else ''
        local_model_path = gr.Textbox(
            label="Model Path",
            placeholder="/path/to/model or huggingface-model-id",
            value=local_model_path_value
        )
        max_seq_length_value = saved_config.get('max_seq_length', 16384) if saved_config else 16384
        max_seq_length = gr.Number(
            value=max_seq_length_value,
            label="Max Sequence Length (Context Window)",
            precision=0,
            minimum=512,
            info="Model's full context window"
        )
        quantization_value = saved_config.get('quantization', '4bit') if saved_config else '4bit'
        quantization = gr.Dropdown(
            choices=["none", "4bit"],
            value=quantization_value,
            label="Quantization",
            info="4bit recommended for memory efficiency"
        )
        gpu_layers_value = saved_config.get('gpu_layers', -1) if saved_config else -1
        gpu_layers = gr.Slider(
            minimum=-1,
            maximum=100,
            value=gpu_layers_value,
            step=1,
            label="GPU Layers (-1 = all)"
        )
        components['local_model_path'] = local_model_path
        components['max_seq_length'] = max_seq_length
        components['quantization'] = quantization
        components['gpu_layers'] = gpu_layers
    
    components['local_config'] = local_config
    
    gr.Markdown("---")
    gr.Markdown("### Test Connection")
    gr.Markdown("""
    Test if your model configuration is working correctly before processing.
    This will:
    - Verify API credentials
    - Test endpoint connectivity
    - Confirm model availability
    - Validate response format
    """)
    
    test_prompt = gr.Textbox(
        label="Test Prompt",
        value="Say 'Hello, I am ready!' in JSON format: {\"status\": \"...\", \"message\": \"...\"}",
        lines=3,
        info="Simple prompt to test model connectivity"
    )
    components['test_prompt'] = test_prompt
    
    with gr.Row():
        test_connection_btn = gr.Button("Test Connection", variant="secondary", size="lg")
        components['test_connection_btn'] = test_connection_btn
    
    test_result = gr.TextArea(
        label="Test Result",
        lines=10,
        interactive=False
    )
    components['test_result'] = test_result
    
    gr.Markdown("---")
    gr.Markdown("### Save Configuration")
    
    with gr.Row():
        validate_btn = gr.Button("Validate Configuration", variant="secondary")
        save_config_btn = gr.Button("Save Configuration", variant="primary", size="lg")
    
    components['validate_btn'] = validate_btn
    components['save_config_btn'] = save_config_btn
    
    validation_result = gr.Textbox(
        label="Status",
        interactive=False,
        value="Configuration loaded from disk. Please verify API key and test connection." if saved_config else ""
    )
    components['validation_result'] = validation_result
    
    def update_model_list(provider_value):
        """Update model dropdown based on provider"""
        models = config_manager.get_supported_models(provider_value)
        return gr.update(choices=models, value=models[0] if models else None)
    
    def update_context_info(model_value, provider_value):
        """Update context window information and max_seq_length"""
        limit = config_manager.get_model_context_limit(model_value)
        return f"Context Window: {limit:,} tokens", limit
    
    def toggle_provider_configs(provider_value):
        """Show/hide provider-specific configurations"""
        return (
            gr.update(visible=(provider_value == "azure")),
            gr.update(visible=(provider_value == "google")),
            gr.update(visible=(provider_value == "local"))
        )
    
    def load_api_key_from_env(provider_value):
        """Load API key from environment variable"""
        if provider_value == "local":
            return "", "Local model (no API key needed)"
        
        key = config_manager.get_api_key_from_env(provider_value)
        if key:
            return key, "Loaded from environment"
        else:
            return "", "Not found in environment"
    
    def test_connection(*args):
        """Test model connection and response"""
        (prov, model, api_k, temp, max_tok, m_type,
         az_endpoint, az_deploy, g_proj_id, l_path, max_seq, quant, gpu, test_p) = args
        
        if not test_p or not test_p.strip():
            return "Please enter a test prompt"
        
        try:
            config = ModelConfig(
                provider=prov,
                model_name=model,
                api_key=api_k if api_k else None,
                temperature=temp,
                max_tokens=int(max_tok),
                max_seq_length=int(max_seq) if prov == "local" else 16384,
                model_type=m_type,
                azure_endpoint=az_endpoint if prov == "azure" else None,
                azure_deployment=az_deploy if prov == "azure" else None,
                google_project_id=g_proj_id if prov == "google" else None,
                local_model_path=l_path if prov == "local" else None,
                quantization=quant if prov == "local" else None,
                gpu_layers=int(gpu) if prov == "local" else -1
            )
            
            is_valid, message = config.validate()
            if not is_valid:
                return f"Configuration Error:\n{message}"
            
            from core.llm_manager import LLMManager
            
            config_dict = {
                'provider': config.provider,
                'model_name': config.model_name,
                'temperature': config.temperature,
                'max_tokens': config.max_tokens,
            }
            
            if config.api_key:
                config_dict['api_key'] = config.api_key
            
            if prov == "azure" and config.azure_endpoint and config.azure_deployment:
                config_dict['azure_endpoint'] = config.azure_endpoint
                config_dict['azure_deployment'] = config.azure_deployment
            
            if prov == "google" and config.google_project_id:
                config_dict['google_project_id'] = config.google_project_id
            
            if prov == "local":
                if config.local_model_path:
                    config_dict['local_model_path'] = config.local_model_path
                if config.max_seq_length:
                    config_dict['max_seq_length'] = config.max_seq_length
                if config.quantization:
                    config_dict['quantization'] = config.quantization
                if config.gpu_layers is not None:
                    config_dict['gpu_layers'] = config.gpu_layers
            
            llm_manager = LLMManager(config_dict)
            
            start_time = time.time()
            
            response = llm_manager.generate(test_p.strip())
            
            elapsed = time.time() - start_time
            
            # Cleanup after test
            llm_manager.cleanup()
            
            if response:
                result = f"""Connection Test Successful!

Provider: {prov}
Model: {model}
Response Time: {elapsed:.2f}s

Response:
{response}

Status: Model is ready for processing."""
                return result
            else:
                return "Connection test failed: Empty response from model"
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return f"Connection Test Failed:\n\n{str(e)}\n\nPlease check:\n- API key is correct\n- Endpoint is accessible\n- Model name is valid\n- Network connection is available"
    
    def validate_configuration(*args):
        """Validate configuration without saving"""
        (prov, model, api_k, temp, max_tok, m_type,
         az_endpoint, az_deploy, g_proj_id, l_path, max_seq, quant, gpu) = args
        
        try:
            config = ModelConfig(
                provider=prov,
                model_name=model,
                api_key=api_k if api_k else None,
                temperature=temp,
                max_tokens=int(max_tok),
                max_seq_length=int(max_seq) if prov == "local" else 16384,
                model_type=m_type,
                azure_endpoint=az_endpoint if prov == "azure" else None,
                azure_deployment=az_deploy if prov == "azure" else None,
                google_project_id=g_proj_id if prov == "google" else None,
                local_model_path=l_path if prov == "local" else None,
                quantization=quant if prov == "local" else None,
                gpu_layers=int(gpu) if prov == "local" else -1
            )
            
            is_valid, message = config.validate()
            
            if is_valid:
                return f"✓ Configuration is valid!\n\n{message}\n\nYou can now save the configuration."
            else:
                return f"✗ Configuration is invalid:\n\n{message}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def save_configuration(*args):
        """Save configuration to app state and persistence manager - DON'T auto-initialize LLM"""
        (prov, model, api_k, temp, max_tok, m_type,
         az_endpoint, az_deploy, g_proj_id, l_path, max_seq, quant, gpu,
         agen_enabled, agen_max_iter, agen_max_tools) = args

        try:
            config = ModelConfig(
                provider=prov,
                model_name=model,
                api_key=api_k if api_k else None,
                temperature=temp,
                max_tokens=int(max_tok),
                max_seq_length=int(max_seq) if prov == "local" else 16384,
                model_type=m_type,
                azure_endpoint=az_endpoint if prov == "azure" else None,
                azure_deployment=az_deploy if prov == "azure" else None,
                google_project_id=g_proj_id if prov == "google" else None,
                local_model_path=l_path if prov == "local" else None,
                quantization=quant if prov == "local" else None,
                gpu_layers=int(gpu) if prov == "local" else -1
            )

            # Save model config to app state
            success = app_state.set_model_config(config)
            if success:
                # Save to persistence manager
                persistence_manager.save_model_config(config)

                # Save agentic configuration to app_state
                agentic_success = app_state.set_agentic_config(
                    enabled=agen_enabled,
                    max_iterations=int(agen_max_iter),
                    max_tool_calls=int(agen_max_tools)
                )

                # Save agentic configuration to disk
                if agentic_success:
                    persistence_manager.save_agentic_config(app_state.agentic_config)

                # FIXED: DON'T initialize LLM automatically - only on demand
                logger.info(f"Model configuration saved (LLM will be initialized on first use)")
                logger.info(f"Agentic mode: {'ENABLED' if agen_enabled else 'DISABLED'} (max_iterations={agen_max_iter}, max_tool_calls={agen_max_tools})")

                mode = "ADAPTIVE Mode (v1.0.0)" if agen_enabled else "STRUCTURED Mode (v1.0.0)"
                mode_desc = "For evolving tasks" if agen_enabled else "For predictable workflows"
                return f"✓ Model configuration saved successfully!\n\n**Execution Mode**: {mode} - {mode_desc}\n\n🎯 Both modes are autonomous and adapt to ANY clinical task!\n\nLLM will be initialized when you start processing.\n\n✓ Configuration persisted to disk."
            else:
                return "✗ Failed to save configuration"

        except Exception as e:
            logger.error(f"Save configuration error: {e}")
            return f"Error: {str(e)}"
    
    # Connect event handlers
    provider.change(
        fn=update_model_list,
        inputs=[provider],
        outputs=[model_name]
    )
    
    provider.change(
        fn=toggle_provider_configs,
        inputs=[provider],
        outputs=[azure_config, google_config, local_config]
    )
    
    model_name.change(
        fn=update_context_info,
        inputs=[model_name, provider],
        outputs=[context_info, max_seq_length]
    )
    
    load_from_env_btn.click(
        fn=load_api_key_from_env,
        inputs=[provider],
        outputs=[api_key, validation_result]
    )
    
    test_connection_btn.click(
        fn=test_connection,
        inputs=[
            provider, model_name, api_key, temperature, max_tokens, model_type,
            azure_endpoint, azure_deployment, google_project_id,
            local_model_path, max_seq_length, quantization, gpu_layers, test_prompt
        ],
        outputs=[test_result]
    )
    
    validate_btn.click(
        fn=validate_configuration,
        inputs=[
            provider, model_name, api_key, temperature, max_tokens, model_type,
            azure_endpoint, azure_deployment, google_project_id,
            local_model_path, max_seq_length, quantization, gpu_layers
        ],
        outputs=[validation_result]
    )
    
    save_config_btn.click(
        fn=save_configuration,
        inputs=[
            provider, model_name, api_key, temperature, max_tokens, model_type,
            azure_endpoint, azure_deployment, google_project_id,
            local_model_path, max_seq_length, quantization, gpu_layers,
            adaptive_mode_enabled, agentic_max_iterations, agentic_max_tool_calls
        ],
        outputs=[validation_result]
    )
    
    return components