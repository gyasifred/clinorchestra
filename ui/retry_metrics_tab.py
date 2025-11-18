#!/usr/bin/env python3
"""
Retry Metrics Tab - Display adaptive retry system statistics

Shows retry metrics including success rates, context reductions,
and provider-specific performance.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import gradio as gr
import pandas as pd
from typing import Tuple, Dict, Any
from core.app_state import AppState
from core.retry_metrics import get_retry_metrics_tracker
from core.logging_config import get_logger

logger = get_logger(__name__)


def create_retry_metrics_tab(app_state: AppState) -> Tuple:
    """
    Create retry metrics tab for Gradio interface

    Args:
        app_state: Application state

    Returns:
        Tuple of Gradio components
    """

    with gr.Tab("📊 Retry Metrics"):
        gr.Markdown("""
        # Adaptive Retry System Metrics

        View performance statistics for the adaptive retry system, including success rates,
        context reduction effectiveness, and provider-specific metrics.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")

                gr.Markdown("### Filters")
                provider_filter = gr.Dropdown(
                    choices=["All Providers", "openai", "anthropic", "google", "azure", "local"],
                    value="All Providers",
                    label="Filter by Provider"
                )
                days_filter = gr.Slider(
                    minimum=1,
                    maximum=90,
                    value=30,
                    step=1,
                    label="Last N Days"
                )

        # Summary Statistics
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Summary Statistics")
                summary_stats = gr.DataFrame(
                    headers=["Metric", "Value"],
                    label="Overall Performance",
                    interactive=False
                )

        # Success Rate by Attempt
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Success Rate by Attempt Number")
                success_by_attempt = gr.DataFrame(
                    headers=["Attempt", "Success Rate %"],
                    label="Retry Attempt Success Rates",
                    interactive=False
                )

        # Provider-Specific Metrics
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Provider-Specific Performance")
                provider_metrics = gr.DataFrame(
                    headers=["Provider", "Total", "Successful", "Success Rate %", "Avg Attempts"],
                    label="Performance by LLM Provider",
                    interactive=False
                )

        # Top Error Types
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Most Common Error Types")
                error_types = gr.DataFrame(
                    headers=["Error Type", "Count"],
                    label="Errors Leading to Retries",
                    interactive=False
                )

        # Recent Failures
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Recent Failures (Debug)")
                recent_failures = gr.DataFrame(
                    headers=["Extraction ID", "Provider", "Model", "Attempts", "Last Error", "Timestamp"],
                    label="Recent Failed Extractions",
                    interactive=False
                )

        # Configuration Display
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Current Retry Configuration")
                config_display = gr.Textbox(
                    label="Active Configuration",
                    lines=15,
                    interactive=False
                )

        def load_metrics(provider_filter_val: str, days: int) -> Tuple:
            """
            Load and format retry metrics

            Args:
                provider_filter_val: Provider to filter by
                days: Number of days to look back

            Returns:
                Tuple of DataFrames for display
            """
            try:
                tracker = get_retry_metrics_tracker()

                # Get provider for filter (None = all)
                provider = None if provider_filter_val == "All Providers" else provider_filter_val

                # Get summary
                summary = tracker.get_summary(provider=provider, limit_days=days)

                # Format summary statistics
                summary_data = [
                    ["Total Extractions", f"{summary.total_extractions:,}"],
                    ["Successful Extractions", f"{summary.successful_extractions:,}"],
                    ["Failed Extractions", f"{summary.failed_extractions:,}"],
                    ["Success Rate", f"{(summary.successful_extractions / summary.total_extractions * 100) if summary.total_extractions > 0 else 0:.2f}%"],
                    ["Total Retry Attempts", f"{summary.total_retry_attempts:,}"],
                    ["Avg Attempts per Extraction", f"{summary.avg_attempts_per_extraction:.2f}"],
                    ["Avg Context Reduction", f"{summary.avg_context_reduction:.2f}%"],
                ]

                # Format success by attempt
                success_by_attempt_data = [
                    [attempt, f"{rate:.2f}%"]
                    for attempt, rate in sorted(summary.success_rate_by_attempt.items())
                ]

                if not success_by_attempt_data:
                    success_by_attempt_data = [["No data", "N/A"]]

                # Format provider metrics
                provider_metrics_data = [
                    [
                        prov,
                        metrics['total'],
                        metrics['successful'],
                        f"{metrics['success_rate']:.2f}%",
                        f"{metrics['avg_attempts']:.2f}"
                    ]
                    for prov, metrics in summary.metrics_by_provider.items()
                ]

                if not provider_metrics_data:
                    provider_metrics_data = [["No data", "N/A", "N/A", "N/A", "N/A"]]

                # Format error types
                error_types_data = [
                    [error_type, count]
                    for error_type, count in list(summary.error_type_counts.items())[:10]
                ]

                if not error_types_data:
                    error_types_data = [["No errors", 0]]

                # Get recent failures
                failures = tracker.get_recent_failures(limit=10)
                recent_failures_data = [
                    [
                        f["extraction_id"][:8] + "...",  # Truncate ID
                        f["provider"],
                        f["model_name"],
                        f["total_attempts"],
                        f["last_error_type"] or "Unknown",
                        f["timestamp"]
                    ]
                    for f in failures
                ]

                if not recent_failures_data:
                    recent_failures_data = [["No recent failures", "N/A", "N/A", "N/A", "N/A", "N/A"]]

                # Format configuration
                config = app_state.adaptive_retry_config
                config_text = f"""Enabled: {config.enabled}
Max Retry Attempts: {config.max_retry_attempts}

Progressive Reduction Ratios:
  Attempt 2: {config.clinical_text_reduction_ratios[0]*100:.0f}%
  Attempt 3: {config.clinical_text_reduction_ratios[1]*100:.0f}%
  Attempt 4: {config.clinical_text_reduction_ratios[2]*100:.0f}%
  Attempt 5: {config.clinical_text_reduction_ratios[3]*100:.0f}%

History Reduction Levels: {config.history_reduction_levels}
Tool Context Reduction Levels: {config.tool_context_reduction_levels}

Switch to Minimal Prompt: Attempt {config.switch_to_minimal_at_attempt}

Context Preservation:
  Beginning: {config.preserve_context_beginning_ratio*100:.0f}%
  Ending: {config.preserve_context_ending_ratio*100:.0f}%

Smart Context Preservation: {config.use_smart_context_preservation}
Smart Model: {config.smart_preservation_model if config.use_smart_context_preservation else 'N/A'}

Exponential Backoff: {config.enable_exponential_backoff}
  Base: {config.backoff_base_seconds}s
  Max: {config.backoff_max_seconds}s

Metrics Tracking: {config.track_retry_metrics}
"""

                return (
                    summary_data,
                    success_by_attempt_data,
                    provider_metrics_data,
                    error_types_data,
                    recent_failures_data,
                    config_text
                )

            except Exception as e:
                logger.error(f"Failed to load retry metrics: {e}")
                error_data = [[f"Error: {str(e)}", ""]]
                return (error_data, error_data, error_data, error_data, error_data, f"Error loading config: {e}")

        # Wire up refresh button
        refresh_btn.click(
            fn=load_metrics,
            inputs=[provider_filter, days_filter],
            outputs=[
                summary_stats,
                success_by_attempt,
                provider_metrics,
                error_types,
                recent_failures,
                config_display
            ]
        )

        # Also refresh when filters change
        provider_filter.change(
            fn=load_metrics,
            inputs=[provider_filter, days_filter],
            outputs=[
                summary_stats,
                success_by_attempt,
                provider_metrics,
                error_types,
                recent_failures,
                config_display
            ]
        )

        days_filter.change(
            fn=load_metrics,
            inputs=[provider_filter, days_filter],
            outputs=[
                summary_stats,
                success_by_attempt,
                provider_metrics,
                error_types,
                recent_failures,
                config_display
            ]
        )

        # Load initial data
        def load_initial():
            return load_metrics("All Providers", 30)

        # Return components that need initialization
        return (
            refresh_btn,
            provider_filter,
            days_filter,
            summary_stats,
            success_by_attempt,
            provider_metrics,
            error_types,
            recent_failures,
            config_display
        )
