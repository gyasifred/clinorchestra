#!/usr/bin/env python3
"""
Processing Tab - Complete with proper text processing integration
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.2
"""
import gradio as gr
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any
from core.output_handler import OutputHandler
from core.process_persistence import get_process_state
from core.agent_system import ExtractionAgent
from core.app_state import StateEvent
from core.logging_config import get_logger

logger = get_logger(__name__)

def create_processing_tab(app_state) -> Dict[str, Any]:
    """Create processing tab with enhanced logging and complete functionality"""
    
    components = {}
    
    gr.Markdown("### Review Configuration & Execute")
    
    with gr.Accordion("Configuration Summary", open=True):
        config_summary = gr.TextArea(
            label="Summary",
            value="Click Refresh",
            lines=20,
            interactive=False
        )
        components['config_summary'] = config_summary
        
        refresh_config_btn = gr.Button("Refresh")
        components['refresh_config_btn'] = refresh_config_btn
    
    gr.Markdown("---")
    gr.Markdown("### Settings")
    
    with gr.Row():
        with gr.Column():
            batch_size = gr.Number(
                value=app_state.processing_config.batch_size,
                label="Batch Size",
                precision=0,
                minimum=1
            )
            components['batch_size'] = batch_size
            
            error_strategy = gr.Radio(
                choices=["skip", "retry", "halt"],
                value=app_state.processing_config.error_strategy,
                label="Error Strategy"
            )
            components['error_strategy'] = error_strategy
        
        with gr.Column():
            output_dir = gr.Textbox(
                value=app_state.processing_config.output_path or "./output",
                label="Output Directory"
            )
            components['output_dir'] = output_dir
            
            dry_run = gr.Checkbox(
                label="Dry Run (5 rows only)",
                value=app_state.processing_config.dry_run
            )
            components['dry_run'] = dry_run
    
    gr.Markdown("---")
    gr.Markdown("### Execute")
    
    with gr.Row():
        start_btn = gr.Button("▶️ Start Processing", variant="primary", size="lg")
        stop_btn = gr.Button("⏹️ Stop", variant="stop")
    
    components['start_btn'] = start_btn
    components['stop_btn'] = stop_btn
    
    processing_status = gr.Textbox(
        label="Status",
        value="Ready",
        interactive=False
    )
    components['processing_status'] = processing_status
    
    gr.Markdown("### Progress")
    
    with gr.Row():
        progress_bar = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            label="Progress (%)",
            interactive=False
        )
        components['progress_bar'] = progress_bar
    
    with gr.Row():
        processed_count = gr.Number(
            value=0,
            label="Processed",
            precision=0,
            interactive=False
        )
        components['processed_count'] = processed_count
        
        failed_count = gr.Number(
            value=0,
            label="Failed",
            precision=0,
            interactive=False
        )
        components['failed_count'] = failed_count
    
    with gr.Accordion("Processing Log (Detailed)", open=True):
        log_display = gr.TextArea(
            label="Log",
            lines=20,
            interactive=False
        )
        components['log_display'] = log_display
    
    gr.Markdown("---")
    gr.Markdown("### Results")
    
    results_summary = gr.TextArea(
        label="Summary",
        value="No processing completed",
        lines=12,
        interactive=False
    )
    components['results_summary'] = results_summary
    
    with gr.Row():
        download_results = gr.File(label="Download Results")
        download_stats = gr.File(label="Download Statistics")
    
    components['download_results'] = download_results
    components['download_stats'] = download_stats
    
    def refresh_configuration():
        """Display configuration"""
        return app_state.get_configuration_summary()
    
    def start_processing(batch_sz, error_strat, out_dir, is_dry_run):
        """Start processing with enhanced logging and process persistence"""
        
        can_start, message = app_state.can_start_processing()
        if not can_start:
            return (
                f"❌ Cannot start: {message}",
                0, 0, 0, "",
                "Cannot start processing",
                None, None
            )
        
        app_state.set_processing_config(
            batch_size=int(batch_sz) if batch_sz is not None else 1,
            error_strategy=error_strat,
            output_path=out_dir,
            dry_run=is_dry_run
        )
        
        app_state.start_processing()
        
        process_state = get_process_state()
        process_id = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        df = pd.read_csv(app_state.data_config.input_file)
        process_state.create_process(process_id, {'total_rows': len(df)})
        
        log_lines = []
        log_lines.append("=" * 80)
        log_lines.append(f"PROCESSING STARTED (Process ID: {process_id})")
        log_lines.append("=" * 80)
        log_lines.append(f"Timestamp: {datetime.now().isoformat()}")
        log_lines.append(f"Dry Run: {is_dry_run}")
        log_lines.append("")
        process_state.add_log(process_id, "Processing started")
        
        try:
            llm_manager = app_state.get_llm_manager()
            log_lines.append(f"✅ Using LLM Manager: {llm_manager.provider}/{llm_manager.model_name}")
            process_state.add_log(process_id, f"Using LLM Manager: {llm_manager.provider}/{llm_manager.model_name}")
            
            regex_preprocessor = app_state.get_regex_preprocessor()
            log_lines.append("✅ Using Regex Preprocessor")
            process_state.add_log(process_id, "Using Regex Preprocessor")
            
            extras_manager = app_state.get_extras_manager()
            all_extras = extras_manager.list_extras()
            log_lines.append(f"✅ Extras Manager: {len(all_extras)} extras available")
            process_state.add_log(process_id, f"Extras Manager: {len(all_extras)} extras available")
            
            function_registry = app_state.get_function_registry()
            all_funcs = function_registry.list_functions()
            log_lines.append(f"✅ Function Registry: {len(all_funcs)} functions available")
            process_state.add_log(process_id, f"Function Registry: {len(all_funcs)} functions available")
            
            rag_engine = app_state.get_rag_engine()
            if app_state.rag_config.enabled and rag_engine and rag_engine.initialized:
                log_lines.append(f"✅ RAG Engine: {len(rag_engine.documents_loaded)} documents loaded")
                process_state.add_log(process_id, f"RAG Engine: {len(rag_engine.documents_loaded)} documents loaded")
            else:
                log_lines.append("ℹ️ RAG Engine: Disabled")
                process_state.add_log(process_id, "RAG Engine: Disabled")
            
            log_lines.append("")
            log_lines.append("=" * 80)
            log_lines.append("TEXT PROCESSING CONFIGURATION")
            log_lines.append("=" * 80)
            log_lines.append(f"PHI Redaction: {'Enabled' if app_state.data_config.enable_phi_redaction else 'Disabled'}")
            if app_state.data_config.enable_phi_redaction:
                log_lines.append(f"  - Save Redacted Column: {'Yes' if app_state.data_config.save_redacted_text else 'No'}")
                log_lines.append(f"  - Entity Types: {', '.join(app_state.data_config.phi_entity_types)}")
            log_lines.append(f"Pattern Normalization: {'Enabled' if app_state.data_config.enable_pattern_normalization else 'Disabled'}")
            if app_state.data_config.enable_pattern_normalization:
                log_lines.append(f"  - Save Normalized Column: {'Yes' if app_state.data_config.save_normalized_text else 'No'}")
            log_lines.append("")
            
            log_lines.append("=" * 80)
            log_lines.append("INITIALIZING AGENT")
            log_lines.append("=" * 80)
            log_lines.append("")
            process_state.add_log(process_id, "Initializing agent")
            
            agent = ExtractionAgent(
                llm_manager=llm_manager,
                rag_engine=rag_engine,
                extras_manager=extras_manager,
                function_registry=function_registry,
                regex_preprocessor=regex_preprocessor,
                app_state=app_state
            )
            log_lines.append("✅ Agent initialized")
            process_state.add_log(process_id, "Agent initialized")
            
            output_handler = OutputHandler(
                data_config=app_state.data_config,
                prompt_config=app_state.prompt_config
            )
            log_lines.append("✅ Output Handler initialized")
            process_state.add_log(process_id, "Output Handler initialized")
            
            if is_dry_run:
                df = df.head(5)
                log_lines.append("🔬 DRY RUN: Processing only 5 rows")
                process_state.add_log(process_id, "Dry run: Processing only 5 rows")
            
            total_rows = len(df)
            text_column = app_state.data_config.text_column
            label_column = app_state.data_config.label_column
            has_labels = app_state.data_config.has_labels
            
            log_lines.append(f"📊 Dataset: {total_rows} rows")
            log_lines.append(f"📝 Text Column: {text_column}")
            if has_labels:
                log_lines.append(f"🏷️ Label Column: {label_column}")
            log_lines.append("")
            log_lines.append("=" * 80)
            log_lines.append("PROCESSING ROWS")
            log_lines.append("=" * 80)
            log_lines.append("")
            process_state.add_log(process_id, "Starting row processing")
            
            processed = 0
            failed = 0
            total_extras_used = 0
            total_rag_used = 0
            total_functions_called = 0
            
            for idx, row in df.iterrows():
                if not app_state.is_processing:
                    log_lines.append("⏹️ Processing stopped by user")
                    process_state.add_log(process_id, "Processing stopped by user")
                    break
                
                clinical_text = str(row[text_column])
                label_value = row[label_column] if has_labels and label_column in row.index else None
                # FIXED: Check for None explicitly, not truthiness (allows 0, False, "" as valid labels)
                label_context = app_state.data_config.label_mapping.get(str(label_value), None) if label_value is not None else None
                
                log_lines.append(f"[Row {idx+1}/{total_rows}] Processing...")
                process_state.add_log(process_id, f"Processing row {idx+1}/{total_rows}")
                
                try:
                    result = agent.extract(clinical_text, label_value)
                    
                    extras_used = result.get('extras_used', 0)
                    rag_used = result.get('rag_used', 0)
                    functions_called = result.get('functions_called', 0)
                    metadata = result.get('processing_metadata', {})
                    
                    total_extras_used += extras_used
                    total_rag_used += rag_used
                    total_functions_called += functions_called
                    
                    if result.get('rag_refinement_applied', False):
                        final_output = result.get('stage4_final_output', {})
                    else:
                        final_output = result.get('stage3_output', {})
                    
                    log_lines.append(f"  ✅ Success")
                    log_lines.append(f"     Extras: {extras_used} | RAG: {rag_used} | Functions: {functions_called}")
                    process_state.add_log(process_id, f"Row {idx+1} processed: Extras={extras_used}, RAG={rag_used}, Functions={functions_called}")
                    
                    if extras_used > 0:
                        extras_details = metadata.get('extras_details', [])
                        log_lines.append(f"     Extras Details:")
                        for extra in extras_details[:3]:
                            log_lines.append(f"       • {extra.get('type', 'unknown')}: {extra.get('content', '')[:60]}...")
                    
                    if rag_used > 0:
                        rag_details = metadata.get('rag_details', [])
                        log_lines.append(f"     RAG Details:")
                        for rag_item in rag_details[:2]:
                            log_lines.append(f"       • Score: {rag_item.get('score', 0):.2f}, Source: {rag_item.get('source', 'Unknown')}")
                    
                    if functions_called > 0:
                        func_details = metadata.get('function_calls_details', [])
                        log_lines.append(f"     Function Calls:")
                        for func in func_details:
                            log_lines.append(f"       • {func.get('function', 'unknown')}() = {func.get('result', 'N/A')}")
                    
                    output_handler.add_record(
                        row=row,
                        llm_output=final_output,
                        redacted_text=result.get('redacted_text'),
                        normalized_text=result.get('normalized_text'),
                        label_context=label_context,
                        metadata={
                            'processing_timestamp': datetime.now().isoformat(),
                            'llm_provider': app_state.model_config.provider,
                            'llm_model': app_state.model_config.model_name,
                            'llm_temperature': app_state.model_config.temperature,
                            'prompt_type': 'minimal' if result.get('used_minimal_prompt', False) else 'main',
                            'retry_count': result.get('retry_count', 0),
                            'extras_used': extras_used,
                            'rag_used': rag_used,
                            'functions_called': functions_called,
                            'rag_refinement_applied': result.get('rag_refinement_applied', False),
                            'parsing_method': result.get('parsing_method_used', 'unknown')
                        }
                    )
                    
                    processed += 1
                    progress = ((idx + 1) / total_rows) * 100
                    process_state.update_progress(process_id, processed, failed, progress)
                    app_state.update_progress(processed, failed, progress)
                    log_lines.append("")
                    
                except Exception as e:
                    logger.error(f"Row {idx} failed: {e}")
                    log_lines.append(f"  ❌ ERROR: {str(e)}")
                    process_state.add_log(process_id, f"Row {idx+1} failed: {str(e)}")
                    log_lines.append("")
                    failed += 1
                    
                    if error_strat == "halt":
                        log_lines.append("⏹️ HALTING due to error strategy")
                        process_state.add_log(process_id, "Halting due to error strategy")
                        break
                    elif error_strat == "retry":
                        log_lines.append("🔄 RETRYING...")
                        process_state.add_log(process_id, f"Retrying row {idx+1}")
                        
                        retry_success = False
                        for attempt in range(1, app_state.processing_config.max_retries + 1):
                            try:
                                result = agent.extract(clinical_text, label_value)
                                
                                extras_used = result.get('extras_used', 0)
                                rag_used = result.get('rag_used', 0)
                                functions_called = result.get('functions_called', 0)
                                metadata = result.get('processing_metadata', {})
                                
                                total_extras_used += extras_used
                                total_rag_used += rag_used
                                total_functions_called += functions_called
                                
                                if result.get('rag_refinement_applied', False):
                                    final_output = result.get('stage4_final_output', {})
                                else:
                                    final_output = result.get('stage3_output', {})
                                
                                log_lines.append(f"  ✅ Success on retry (Attempt {attempt})")
                                log_lines.append(f"     Extras: {extras_used} | RAG: {rag_used} | Functions: {functions_called}")
                                process_state.add_log(process_id, f"Row {idx+1} retry success: Extras={extras_used}, RAG={rag_used}, Functions={functions_called}")
                                
                                if extras_used > 0:
                                    extras_details = metadata.get('extras_details', [])
                                    log_lines.append(f"     Extras Details:")
                                    for extra in extras_details[:3]:
                                        log_lines.append(f"       • {extra.get('type', 'unknown')}: {extra.get('content', '')[:60]}...")
                                
                                if rag_used > 0:
                                    rag_details = metadata.get('rag_details', [])
                                    log_lines.append(f"     RAG Details:")
                                    for rag_item in rag_details[:2]:
                                        log_lines.append(f"       • Score: {rag_item.get('score', 0):.2f}, Source: {rag_item.get('source', 'Unknown')}")
                                
                                if functions_called > 0:
                                    func_details = metadata.get('function_calls_details', [])
                                    log_lines.append(f"     Function Calls:")
                                    for func in func_details:
                                        log_lines.append(f"       • {func.get('function', 'unknown')}() = {func.get('result', 'N/A')}")
                                
                                output_handler.add_record(
                                    row=row,
                                    llm_output=final_output,
                                    redacted_text=result.get('redacted_text'),
                                    normalized_text=result.get('normalized_text'),
                                    label_context=label_context,
                                    metadata={
                                        'processing_timestamp': datetime.now().isoformat(),
                                        'llm_provider': app_state.model_config.provider,
                                        'llm_model': app_state.model_config.model_name,
                                        'llm_temperature': app_state.model_config.temperature,
                                        'prompt_type': 'minimal' if result.get('used_minimal_prompt', False) else 'main',
                                        'retry_count': result.get('retry_count', attempt),
                                        'extras_used': extras_used,
                                        'rag_used': rag_used,
                                        'functions_called': functions_called,
                                        'rag_refinement_applied': result.get('rag_refinement_applied', False),
                                        'parsing_method': result.get('parsing_method_used', 'unknown')
                                    }
                                )
                                
                                processed += 1
                                failed -= 1
                                progress = ((idx + 1) / total_rows) * 100
                                process_state.update_progress(process_id, processed, failed, progress)
                                app_state.update_progress(processed, failed, progress)
                                log_lines.append("")
                                retry_success = True
                                break
                                
                            except Exception as retry_e:
                                logger.error(f"Row {idx} failed on retry {attempt}: {retry_e}")
                                log_lines.append(f"  ❌ RETRY {attempt} FAILED: {str(retry_e)}")
                                process_state.add_log(process_id, f"Row {idx+1} retry {attempt} failed: {str(retry_e)}")
                                
                                if attempt == app_state.processing_config.max_retries:
                                    log_lines.append(f"  ❌ All {app_state.processing_config.max_retries} retries exhausted")
                                    log_lines.append("")
                                continue
                    else:
                        log_lines.append("  ⏭️ Skipping row")
                        continue
            
            log_lines.append("")
            log_lines.append("=" * 80)
            log_lines.append("SAVING RESULTS")
            log_lines.append("=" * 80)
            log_lines.append("")
            process_state.add_log(process_id, "Saving results")
            
            output_dir_path = Path(out_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"results_{timestamp}.csv"
            
            success, results_file = output_handler.save_to_csv(
                output_path=str(output_dir_path),
                filename=results_filename
            )
            if not success:
                raise Exception(f"Failed to save results: {results_file}")
            log_lines.append(f"✅ Results saved: {results_file}")
            process_state.add_log(process_id, f"Results saved: {results_file}")
            
            stats = {
                'total_rows': total_rows,
                'processed': processed,
                'failed': failed,
                'success_rate': (processed / total_rows * 100) if total_rows > 0 else 0,
                'output_file': results_file,
                'timestamp': timestamp,
                'configuration': {
                    'model_provider': app_state.model_config.provider,
                    'model_name': app_state.model_config.model_name,
                    'temperature': app_state.model_config.temperature,
                    'batch_size': batch_sz,
                    'dry_run': is_dry_run,
                    'error_strategy': error_strat,
                    'phi_redaction_enabled': app_state.data_config.enable_phi_redaction,
                    'pattern_normalization_enabled': app_state.data_config.enable_pattern_normalization
                },
                'component_usage': {
                    'total_extras_used': total_extras_used,
                    'total_rag_chunks_retrieved': total_rag_used,
                    'total_functions_called': total_functions_called,
                    'avg_extras_per_row': total_extras_used / processed if processed > 0 else 0,
                    'avg_rag_per_row': total_rag_used / processed if processed > 0 else 0,
                    'avg_functions_per_row': total_functions_called / processed if processed > 0 else 0
                }
            }
            stats['output_statistics'] = output_handler.get_statistics()
            
            stats_filename = f"stats_{timestamp}.json"
            stats_success, stats_file = output_handler.save_statistics(
                output_path=str(output_dir_path),
                filename=stats_filename
            )
            log_lines.append(f"✅ Statistics saved: {stats_file if stats_success else 'Failed'}")
            process_state.add_log(process_id, f"Statistics saved: {stats_file if stats_success else 'Failed'}")
            
            log_lines.append("")
            log_lines.append("=" * 80)
            log_lines.append("PROCESSING COMPLETE")
            log_lines.append("=" * 80)
            log_lines.append("")
            log_lines.append(f"Total: {total_rows}")
            log_lines.append(f"Processed: {processed}")
            log_lines.append(f"Failed: {failed}")
            log_lines.append(f"Success Rate: {stats['success_rate']:.1f}%")
            log_lines.append("")
            log_lines.append("COMPONENT USAGE SUMMARY:")
            log_lines.append(f"  Total Extras Used: {total_extras_used}")
            log_lines.append(f"  Total RAG Chunks: {total_rag_used}")
            log_lines.append(f"  Total Function Calls: {total_functions_called}")
            log_lines.append(f"  Avg Extras/Row: {stats['component_usage']['avg_extras_per_row']:.2f}")
            log_lines.append(f"  Avg RAG/Row: {stats['component_usage']['avg_rag_per_row']:.2f}")
            log_lines.append(f"  Avg Functions/Row: {stats['component_usage']['avg_functions_per_row']:.2f}")
            log_lines.append("")
            
            summary = f"""✅ Processing Complete

Total: {total_rows}
Processed: {processed}
Failed: {failed}
Success Rate: {stats['success_rate']:.1f}%

COMPONENT USAGE:
• Extras Used: {total_extras_used} (avg {stats['component_usage']['avg_extras_per_row']:.2f}/row)
• RAG Chunks: {total_rag_used} (avg {stats['component_usage']['avg_rag_per_row']:.2f}/row)
• Functions Called: {total_functions_called} (avg {stats['component_usage']['avg_functions_per_row']:.2f}/row)

Output File: {Path(results_file).name}
Output Columns: {stats['output_statistics']['total_columns']}
Output Size: {stats['output_statistics']['memory_usage_mb']:.2f} MB

Statistics: {Path(stats_file).name if stats_success else 'Not saved'}
"""
            
            app_state.stop_processing()
            process_state.complete_process(process_id, success=True)
            return (
                "✅ Processing complete",
                100,
                processed,
                failed,
                "\n".join(log_lines),
                summary,
                results_file,
                stats_file if stats_success else None
            )
            
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            log_lines.append("")
            log_lines.append("=" * 80)
            log_lines.append("❌ PROCESSING FAILED")
            log_lines.append("=" * 80)
            log_lines.append(f"Error: {str(e)}")
            process_state.add_log(process_id, f"Processing failed: {str(e)}")
            process_state.complete_process(process_id, success=False)
            app_state.stop_processing()
            
            summary = f"""❌ Processing Failed

Error: {str(e)}
Processed: {processed}
Failed: {failed}
Total: {total_rows}

Check logs for details."""
            
            return (
                f"❌ Error: {str(e)}",
                app_state.current_progress,
                processed,
                failed,
                "\n".join(log_lines),
                summary,
                None,
                None
            )
    
    def stop_processing():
        """Stop processing"""
        app_state.stop_processing()
        return (
            "⏹️ Processing stopped",
            app_state.current_progress,
            app_state.processed_rows,
            app_state.failed_rows,
            log_display.value,
            results_summary.value,
            None,
            None
        )
    
    def update_progress(data: Dict[str, Any]):
        """Update progress from app state"""
        if not app_state.is_processing:
            return (
                processing_status.value,
                progress_bar.value,
                processed_count.value,
                failed_count.value,
                log_display.value,
                results_summary.value,
                download_results.value,
                download_stats.value
            )
        
        processed = data.get('processed', 0)
        failed = data.get('failed', 0)
        progress = data.get('progress', 0.0)
        
        return (
            f"Processing... {processed}/{app_state.data_config.total_rows}",
            progress,
            processed,
            failed,
            log_display.value,
            results_summary.value,
            download_results.value,
            download_stats.value
        )
    
    refresh_config_btn.click(
        fn=refresh_configuration,
        outputs=[config_summary]
    )
    
    start_btn.click(
        fn=start_processing,
        inputs=[batch_size, error_strategy, output_dir, dry_run],
        outputs=[
            processing_status,
            progress_bar,
            processed_count,
            failed_count,
            log_display,
            results_summary,
            download_results,
            download_stats
        ]
    )
    
    stop_btn.click(
        fn=stop_processing,
        outputs=[
            processing_status,
            progress_bar,
            processed_count,
            failed_count,
            log_display,
            results_summary,
            download_results,
            download_stats
        ]
    )
    
    app_state.observer.subscribe(StateEvent.PROCESSING_PROGRESS, update_progress)

    return components