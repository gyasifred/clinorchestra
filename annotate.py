#!/usr/bin/env python3
"""
ClinAnnotate - Agent-based extraction system
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: HeiderLab
Version: 1.0.0
"""
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
from ui.footer import create_footer
from core.app_state import AppState, StateEvent
from core.config_persistence import get_persistence_manager
from core.logging_config import setup_logging, get_logger

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
    """Create main Gradio interface with persistence integration"""
    
    app_state = AppState()
    persistence_manager = get_persistence_manager()
    logger.info("AppState initialized")
    
    # Load saved configurations on startup
    try:
        restore_success = persistence_manager.load_all_configs(app_state)
        if restore_success:
            logger.info("Configuration restored from previous session")
        else:
            logger.info("Starting with fresh configuration")
    except Exception as e:
        logger.warning(f"Failed to restore configuration: {e}")
    
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    .header-section {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 30px;
    }
    
    .config-status {
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple"
        ),
        css=custom_css,
        title="ClinAnnotate V1.0.0"
    ) as app:
        
        gr.HTML("""
            <div class="header-section">
                <h1>ClinAnnotate</h1>
                <h2>Agent-Based Clinical Data Extraction</h2>
                <p>Version 1.0.0</p>
            </div>
        """)
        
        # Configuration status panel
        with gr.Row():
            with gr.Column(scale=1):
                def get_initial_status():
                    """Get initial status display"""
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
                    logger.info(f"Initial global status: Model={model_status}, Prompt={prompt_status}, Data={data_status}, Processing={processing_status}")
                    return html
                
                global_status = gr.HTML(value=get_initial_status())
        
        with gr.Tabs():
            with gr.Tab("Model Configuration"):
                config_components = create_config_tab(app_state)
            
            with gr.Tab("Prompt Configuration"):
                prompt_components = create_prompt_tab(app_state)
            
            with gr.Tab("Data Configuration"):
                data_components = create_data_tab(app_state)
            
            with gr.Tab("Regex Patterns"):
                patterns_components = create_patterns_tab(app_state)
            
            with gr.Tab("Extras (Hints)"):
                extras_components = create_extras_tab(app_state)
            
            with gr.Tab("Custom Functions"):
                functions_components = create_functions_tab(app_state)
            
            with gr.Tab("RAG"):
                rag_components = create_rag_tab(app_state)
            
            with gr.Tab("Playground"):
                playground_components = create_playground_tab(app_state)
            
            with gr.Tab("Processing"):
                processing_components = create_processing_tab(app_state)
        
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
            'persistence_manager': persistence_manager
        }
        
        setup_event_handlers(app_state, all_components)
        
        # Auto-save configurations periodically
        def auto_save_configs():
            """Automatically save configurations"""
            try:
                persistence_manager.save_all_configs(app_state)
                logger.debug("Auto-saved configurations")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
        
        # Refresh global status and restore UI state on interface load
        def load_ui_state():
            """Load UI state from persisted config"""
            status = get_initial_status()

            # Restore model config
            model_config = app_state.model_config
            model_provider = model_config.provider if model_config else "openai"
            model_name = model_config.model_name if model_config else "gpt-4"
            temperature = model_config.temperature if model_config else 0.0
            max_tokens = model_config.max_tokens if model_config else 4000

            # Restore prompt config
            prompt_config = app_state.prompt_config
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
        
        # Set up periodic auto-save (every 30 seconds)
        import threading
        def periodic_save():
            import time
            while True:
                time.sleep(30)
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
    parser = argparse.ArgumentParser(description="ClinAnnotate V1.0.0")
    
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ClinAnnotate V1.0.0 - Agent-Based Extraction System")
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