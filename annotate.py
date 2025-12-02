#!/usr/bin/env python3
"""
ClinOrchestra - Intelligent Clinical Data Extraction & Orchestration
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: HeiderLab
Version: 1.0.0

ClinOrchestra orchestrates LLM-powered agents with RAG, functions, and knowledge
to extract structured data from clinical text. Works for ANY clinical task.

What's New in v1.0.0:
- Performance monitoring infrastructure
- LLM response caching system
- Enhanced RAG with batch embeddings
- Comprehensive performance documentation
"""

# GPU ISOLATION: Set CUDA_VISIBLE_DEVICES before any CUDA imports
# This MUST be done before importing torch, transformers, sentence-transformers, etc.
# to prevent GPU context leakage to unintended devices
import os
if 'CLINORCHESTRA_GPU_DEVICE' in os.environ:
    # User explicitly specified which GPU to use
    gpu_device = os.environ['CLINORCHESTRA_GPU_DEVICE']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    print(f"[GPU ISOLATION] Restricting CUDA to device(s): {gpu_device}")
elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # Default: Use only GPU 0 to prevent leakage
    # Users can override by setting CUDA_VISIBLE_DEVICES before running
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("[GPU ISOLATION] Defaulting to GPU 0 only (set CUDA_VISIBLE_DEVICES to override)")

import gradio as gr
import argparse
from pathlib import Path
import unsloth
from ui.config_tab import create_config_tab
from ui.prompt_tab import create_prompt_tab
from ui.data_tab import create_data_tab
from ui.patterns_tab import create_patterns_tab
from ui.extras_tab import create_extras_tab
from ui.functions_tab import create_functions_tab
from ui.rag_tab import create_rag_tab
from ui.playground_tab import create_playground_tab
from ui.processing_tab import create_processing_tab
from ui.retry_metrics_tab import create_retry_metrics_tab
from ui.footer import create_footer
from core.app_state import AppState, StateEvent
from core.config_persistence import get_persistence_manager
from core.logging_config import setup_logging, get_logger
from core.session_manager import get_session_manager
from core.app_state_proxy import AppStateProxy

# Initialize enhanced logging system
setup_logging(
    log_dir="logs",
    log_level="INFO",
    console_level="INFO",
    file_level="DEBUG",
    enable_file_logging=True,
    enable_colors=True
)
logger = get_logger(__name__)

def create_main_interface() -> gr.Blocks:
    """Create main Gradio interface with multi-instance session management (v1.0.0)"""

    # v1.0.0: Multi-instance architecture - each session isolated
    session_manager = get_session_manager()
    session_id = session_manager.create_session()
    persistence_manager = get_persistence_manager()

    logger.info(f"Session created: {session_id}")
    logger.info("Multi-instance SessionManager initialized - tasks will be isolated")

    # Default task for new sessions
    default_task = "ADRD Classification"

    # Get or create AppState for default task
    initial_app_state = session_manager.get_task_context(session_id, default_task)

    # v1.0.0: Create AppState proxy for dynamic task switching
    # This allows tabs to reference the "current" AppState which changes when tasks switch
    # All tabs will work with this proxy, which forwards calls to the active task's AppState
    app_state_proxy = AppStateProxy(initial_app_state)

    # Load saved configurations on startup (task-specific)
    try:
        restore_success = persistence_manager.load_all_configs(initial_app_state)
        if restore_success:
            logger.info(f"Configuration restored for task: {default_task}")
        else:
            logger.info(f"Starting with fresh configuration for task: {default_task}")
    except Exception as e:
        logger.warning(f"Failed to restore configuration: {e}")
    
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .header-section {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, #73000A 0%, #B3A369 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(115, 0, 10, 0.3);
    }

    .header-section h1 {
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .header-section h2 {
        font-weight: 400;
        opacity: 0.95;
    }

    .config-status {
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid #73000A;
    }

    /* MUSC-themed buttons */
    .primary-button {
        background-color: #73000A !important;
        color: white !important;
    }

    .primary-button:hover {
        background-color: #8B0010 !important;
    }

    /* Tab styling with MUSC colors */
    .tab-nav button.selected {
        border-bottom: 3px solid #73000A !important;
        color: #73000A !important;
    }

    /* Horizontal checkbox layout for extras and patterns */
    .horizontal-checkboxes label {
        display: inline-flex !important;
        margin-right: 15px !important;
        margin-bottom: 8px !important;
        align-items: center !important;
    }

    .horizontal-checkboxes .wrap {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
    }

    .horizontal-checkboxes input[type="checkbox"] {
        margin-right: 5px !important;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="stone",
            font=["Inter", "Segoe UI", "sans-serif"]
        ).set(
            button_primary_background_fill="#73000A",
            button_primary_background_fill_hover="#8B0010",
            button_primary_text_color="white",
            slider_color="#73000A",
            checkbox_label_text_color="#000000"
        ),
        css=custom_css,
        title="ClinOrchestra V1.0.0"
    ) as app:
        
        gr.HTML("""
            <div class="header-section">
                <h1>ClinOrchestra</h1>
                <h2>Intelligent Clinical Data Extraction & Orchestration</h2>
                <p>Version 1.0.0 - Multi-Instance Task Isolation</p>
            </div>
        """)

        # v1.0.0: Session and task management state
        session_state = gr.State(value=session_id)
        current_task_state = gr.State(value=default_task)

        # v1.0.0: Task selector for multi-instance isolation
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Task Selection")
                gr.Markdown("Each task maintains isolated configuration and resources")
                task_selector = gr.Dropdown(
                    choices=["ADRD Classification", "Malnutrition Classification", "Custom"],
                    value=default_task,
                    label="Current Task",
                    info="Switch between isolated task contexts",
                    interactive=True
                )
                task_status = gr.Textbox(
                    value=f"Active Task: {default_task} | Session: {session_id[:8]}...",
                    label="Session Info",
                    interactive=False
                )

        # Configuration status panel
        with gr.Row():
            with gr.Column(scale=1):
                def get_status_for_task(task_name: str, sess_id: str):
                    """Get status display for specific task"""
                    # Get task-specific AppState
                    task_app_state = session_manager.get_task_context(sess_id, task_name)
                    if task_app_state is None:
                        return "<div class='config-status'><strong>Error: Task context not found</strong></div>"

                    model_status = "Configured" if task_app_state.config_valid else "Not Configured"
                    prompt_status = "Configured" if task_app_state.prompt_valid else "Not Configured"
                    data_status = "Configured" if task_app_state.data_valid else "Not Configured"
                    processing_status = "Configured" if task_app_state.processing_config.output_path else "Not Configured"

                    # Get last saved information
                    last_saved = persistence_manager.get_last_saved_info()

                    html = f"""
                    <div class="config-status">
                        <strong>Configuration Status ({task_name}):</strong><br/>
                        Model: {model_status}<br/>
                        Prompt: {prompt_status}<br/>
                        Data: {data_status}<br/>
                        Processing: {processing_status}<br/>
                        <hr style="margin: 10px 0;">
                        <small>Last Saved:</small><br/>
                        <small>Model: {last_saved.get('model', 'Never')}</small><br/>
                        <small>Prompt: {last_saved.get('prompt', 'Never')}</small><br/>
                        <small>Data: {last_saved.get('data', 'Never')}</small>
                    </div>
                    """
                    logger.info(f"Status for task '{task_name}': Model={model_status}, Prompt={prompt_status}, Data={data_status}, Processing={processing_status}")
                    return html

                def get_initial_status():
                    """Get initial status display"""
                    return get_status_for_task(default_task, session_id)

                global_status = gr.HTML(value=get_initial_status())
        
        with gr.Tabs():
            with gr.Tab("Model Configuration"):
                config_components = create_config_tab(app_state_proxy)

            with gr.Tab("Prompt Configuration"):
                prompt_components = create_prompt_tab(app_state_proxy)

            with gr.Tab("Data Configuration"):
                data_components = create_data_tab(app_state_proxy)

            with gr.Tab("Regex Patterns"):
                patterns_components = create_patterns_tab(app_state_proxy)

            with gr.Tab("Extras (Hints)"):
                extras_components = create_extras_tab(app_state_proxy)

            with gr.Tab("Custom Functions"):
                functions_components = create_functions_tab(app_state_proxy)

            with gr.Tab("RAG"):
                rag_components = create_rag_tab(app_state_proxy)

            with gr.Tab("Playground"):
                playground_components = create_playground_tab(app_state_proxy)

            with gr.Tab("Processing"):
                processing_components = create_processing_tab(app_state_proxy)

            # v1.0.0: Retry metrics tab for adaptive retry system monitoring
            retry_metrics_components = create_retry_metrics_tab(app_state_proxy)

        # v1.0.0: Task switching callback for multi-instance isolation
        def switch_task(task_name: str, sess_id: str, prev_task: str):
            """Switch to a different task context"""
            logger.info(f"Task switching requested: {prev_task} -> {task_name}")

            # Get or create AppState for new task
            new_app_state = session_manager.switch_task(sess_id, task_name)

            if new_app_state is None:
                logger.error(f"Failed to switch to task: {task_name}")
                return (
                    prev_task,  # Keep previous task
                    f"Error switching to task: {task_name}",
                    get_status_for_task(prev_task, sess_id)
                )

            # v1.0.0: Update the proxy to point to new AppState
            # All tabs will now automatically use the new task's AppState
            app_state_proxy._set_current_app_state(new_app_state)
            logger.info(f"AppState proxy updated to task: {task_name}")

            # Load configuration for new task
            try:
                persistence_manager.load_all_configs(new_app_state)
                logger.info(f"Configuration loaded for task: {task_name}")
            except Exception as e:
                logger.warning(f"Failed to load configuration for task {task_name}: {e}")

            # Update status displays
            new_status = f"Active Task: {task_name} | Session: {sess_id[:8]}..."
            config_status = get_status_for_task(task_name, sess_id)

            logger.info(f"Successfully switched to task: {task_name}")
            logger.info("All tabs now reference the new task's isolated configuration")

            return (
                task_name,  # Update current_task_state
                new_status,  # Update task_status
                config_status  # Update global_status
            )

        # Wire up task selector
        task_selector.change(
            fn=switch_task,
            inputs=[task_selector, session_state, current_task_state],
            outputs=[current_task_state, task_status, global_status]
        )

        create_footer()

        all_components = {
            'config': config_components,
            'prompt': prompt_components,
            'data': data_components,
            'patterns': patterns_components,
            'extras': extras_components,
            'functions': functions_components,
            'rag': rag_components,
            'playground': playground_components,
            'processing': processing_components,
            'global_status': global_status,
            'persistence_manager': persistence_manager,
            'session_id': session_id,
            'session_manager': session_manager
        }

        # v1.0.0: Pass proxy to event handlers so they work with current active task
        setup_event_handlers(app_state_proxy, all_components)
        
        # Auto-save configurations periodically
        def auto_save_configs():
            """Automatically save configurations (silent mode) for current active task"""
            try:
                # v1.0.0: Save current task's config via proxy
                # Proxy forwards all calls to currently active AppState
                persistence_manager.save_all_configs(app_state_proxy, silent=True)
                logger.debug("Auto-saved configurations for current task")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

        # Refresh global status and restore UI state on interface load
        def load_ui_state():
            """Load UI state from persisted config (default task on startup)"""
            status = get_initial_status()

            # Restore model config from current task (via proxy)
            model_config = app_state_proxy.model_config
            model_provider = model_config.provider if model_config else "openai"
            model_name = model_config.model_name if model_config else "gpt-4"
            temperature = model_config.temperature if model_config else 0.0
            max_tokens = model_config.max_tokens if model_config else 4000

            # Restore prompt config from current task (via proxy)
            prompt_config = app_state_proxy.prompt_config
            main_prompt = prompt_config.base_prompt if prompt_config else ""
            minimal_prompt = prompt_config.minimal_prompt if prompt_config else ""

            return (
                status,  # global_status
                model_provider,  # provider dropdown
                model_name,  # model dropdown
                temperature,  # temperature slider
                max_tokens,  # max_tokens slider
                main_prompt,  # main prompt textbox
                minimal_prompt  # minimal prompt textbox
            )

        app.load(
            fn=load_ui_state,
            inputs=None,
            outputs=[
                global_status,
                config_components.get('provider'),
                config_components.get('model_name'),
                config_components.get('temperature'),
                config_components.get('max_tokens'),
                prompt_components.get('main_prompt'),
                prompt_components.get('minimal_prompt')
            ]
        )

        # Set up periodic auto-save (every 5 minutes)
        # Changed from 30 seconds to reduce log spam and disk I/O
        import threading
        def periodic_save():
            import time
            while True:
                time.sleep(300)  # 5 minutes (was 30 seconds)
                auto_save_configs()

        save_thread = threading.Thread(target=periodic_save, daemon=True)
        save_thread.start()
    
    return app

def setup_event_handlers(app_state, components):
    """Setup event handlers with persistence integration"""
    
    persistence_manager = components['persistence_manager']
    
    def update_global_status():
        """Update global status display"""
        model_status = "Configured" if app_state.config_valid else "Not Configured"
        prompt_status = "Configured" if app_state.prompt_valid else "Not Configured"
        data_status = "Configured" if app_state.data_valid else "Not Configured"
        processing_status = "Configured" if app_state.processing_config.output_path else "Not Configured"
        
        # Get last saved information
        last_saved = persistence_manager.get_last_saved_info()
        
        html = f"""
        <div class="config-status">
            <strong>Configuration Status:</strong><br/>
            Model: {model_status}<br/>
            Prompt: {prompt_status}<br/>
            Data: {data_status}<br/>
            Processing: {processing_status}<br/>
            <hr style="margin: 10px 0;">
            <small>Last Saved:</small><br/>
            <small>Model: {last_saved.get('model', 'Never')}</small><br/>
            <small>Prompt: {last_saved.get('prompt', 'Never')}</small><br/>
            <small>Data: {last_saved.get('data', 'Never')}</small>
        </div>
        """
        logger.info(f"Updating global status: Model={model_status}, Prompt={prompt_status}, Data={data_status}, Processing={processing_status}")
        return html
    
    # Observer callbacks with persistence
    def on_model_changed(config):
        """Handle model configuration changes"""
        logger.info("MODEL_CONFIG_CHANGED triggered")
        try:
            persistence_manager.save_model_config(config)
            logger.debug("Model configuration persisted")
        except Exception as e:
            logger.error(f"Failed to persist model config: {e}")
        return update_global_status()
    
    def on_prompt_changed(config):
        """Handle prompt configuration changes"""
        logger.info("PROMPT_CONFIG_CHANGED triggered")
        try:
            persistence_manager.save_prompt_config(config)
            logger.debug("Prompt configuration persisted")
        except Exception as e:
            logger.error(f"Failed to persist prompt config: {e}")
        
        if 'global_status' in components:
            global_status_update = update_global_status()
        else:
            global_status_update = None
        
        if 'rag' in components and 'rag_query_fields' in components['rag']:
            field_names = list(config.json_schema.keys())
            logger.info(f"Updating RAG query fields: {field_names}")
            return global_status_update, gr.update(choices=field_names)
        return global_status_update, None
    
    def on_data_changed(config):
        """Handle data configuration changes"""
        logger.info("DATA_CONFIG_CHANGED triggered")
        try:
            persistence_manager.save_data_config(config)
            logger.debug("Data configuration persisted")
        except Exception as e:
            logger.error(f"Failed to persist data config: {e}")
        return update_global_status()
    
    def on_processing_changed(config):
        """Handle processing configuration changes"""
        logger.info("PROCESSING_CONFIG_CHANGED triggered")
        try:
            persistence_manager.save_processing_config(config)
            logger.debug("Processing configuration persisted")
        except Exception as e:
            logger.error(f"Failed to persist processing config: {e}")
        return update_global_status()
    
    def on_rag_changed(config):
        """Handle RAG configuration changes"""
        logger.info("RAG_CONFIG_CHANGED triggered")
        try:
            persistence_manager.save_rag_config(config)
            logger.debug("RAG configuration persisted")
        except Exception as e:
            logger.error(f"Failed to persist RAG config: {e}")
    
    # Subscribe to events
    app_state.observer.subscribe(StateEvent.MODEL_CONFIG_CHANGED, on_model_changed)
    app_state.observer.subscribe(StateEvent.PROMPT_CONFIG_CHANGED, on_prompt_changed)
    app_state.observer.subscribe(StateEvent.DATA_CONFIG_CHANGED, on_data_changed)
    app_state.observer.subscribe(StateEvent.PROCESSING_CONFIG_CHANGED, on_processing_changed)
    app_state.observer.subscribe(StateEvent.RAG_CONFIG_CHANGED, on_rag_changed)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ClinOrchestra V1.0.0")
    
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ClinOrchestra V1.0.0 - Intelligent Clinical Data Extraction & Orchestration")
    logger.info("="*80)
    logger.info(f"Port: {args.port}")
    logger.info(f"Share: {args.share}")
    logger.info("="*80)
    
    app = create_main_interface()
    
    app.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
        show_error=True
    )

if __name__ == "__main__":
    main()