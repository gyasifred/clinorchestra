#!/usr/bin/env python3
"""
Tool Duplication Prevention System

Proactive system to prevent LLMs from requesting duplicate tool calls.
Works by injecting warnings and tracking into prompts before generation.

Features:
- Pre-generation warnings about duplicate calls
- Post-generation duplicate detection and filtering
- Budget tracking and warnings
- Historical call tracking

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import json
from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCallBudget:
    """Track tool call budget and usage"""
    max_calls: int = 100
    calls_used: int = 0
    calls_remaining: int = 100
    unique_calls: int = 0
    duplicate_calls_prevented: int = 0

    def update(self, new_calls: int, unique_new_calls: int):
        """Update budget after processing new calls"""
        self.calls_used += unique_new_calls
        self.calls_remaining = max(0, self.max_calls - self.calls_used)
        self.unique_calls += unique_new_calls
        self.duplicate_calls_prevented += (new_calls - unique_new_calls)

    def get_usage_percentage(self) -> float:
        """Get budget usage as percentage"""
        if self.max_calls == 0:
            return 100.0
        return (self.calls_used / self.max_calls) * 100

    def is_budget_exceeded(self) -> bool:
        """Check if budget is exceeded"""
        return self.calls_used >= self.max_calls

    def is_budget_warning_threshold(self) -> bool:
        """Check if budget is at warning threshold (>80%)"""
        return self.get_usage_percentage() >= 80


class ToolDedupPreventer:
    """
    Proactive tool duplication prevention system

    Prevents LLM from requesting duplicate tool calls by:
    1. Tracking all previous tool calls
    2. Injecting warnings into prompts
    3. Post-processing to remove duplicates
    4. Budget monitoring and warnings
    """

    def __init__(self, max_tool_calls: int = 100):
        """
        Initialize tool deduplication preventer

        Args:
            max_tool_calls: Maximum allowed tool calls
        """
        self.budget = ToolCallBudget(max_calls=max_tool_calls)
        self.call_history: Set[str] = set()  # Set of call signatures
        self.call_details: List[Dict[str, Any]] = []  # Full call details
        logger.info(f"ToolDedupPreventer initialized with budget={max_tool_calls}")

    def create_call_signature(self, tool_call: Dict[str, Any]) -> str:
        """
        Create unique signature for a tool call

        Args:
            tool_call: Tool call dict with type, name, parameters

        Returns:
            Unique signature string
        """
        tool_type = tool_call.get('type', '')
        tool_name = tool_call.get('name', '')
        params = tool_call.get('parameters', {})

        # Sort parameters for consistent comparison
        params_json = json.dumps(params, sort_keys=True)

        signature = f"{tool_type}||{tool_name}||{params_json}"
        return signature

    def is_duplicate(self, tool_call: Dict[str, Any]) -> bool:
        """
        Check if tool call is a duplicate

        Args:
            tool_call: Tool call to check

        Returns:
            True if duplicate, False otherwise
        """
        signature = self.create_call_signature(tool_call)
        return signature in self.call_history

    def record_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """
        Record a tool call in history

        Args:
            tool_call: Tool call to record

        Returns:
            True if recorded (new call), False if duplicate
        """
        signature = self.create_call_signature(tool_call)

        if signature in self.call_history:
            # Duplicate
            logger.debug(f"Duplicate tool call detected: {signature}")
            return False

        # New call - record it
        self.call_history.add(signature)
        self.call_details.append(tool_call)
        logger.debug(f"Recorded new tool call: {signature}")
        return True

    def filter_duplicates(self, tool_calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Filter duplicate tool calls from a list

        Args:
            tool_calls: List of tool calls to filter

        Returns:
            Tuple of (unique_calls, num_duplicates_removed)
        """
        unique_calls = []
        duplicates_removed = 0

        for tool_call in tool_calls:
            if not self.is_duplicate(tool_call):
                unique_calls.append(tool_call)
                self.record_tool_call(tool_call)
            else:
                duplicates_removed += 1
                logger.warning(
                    f"Blocked duplicate: {tool_call.get('type')}.{tool_call.get('name')} "
                    f"with params {tool_call.get('parameters')}"
                )

        # Update budget
        self.budget.update(len(tool_calls), len(unique_calls))

        if duplicates_removed > 0:
            logger.warning(
                f"Removed {duplicates_removed} duplicate tool calls "
                f"({len(unique_calls)} unique from {len(tool_calls)} total)"
            )

        return unique_calls, duplicates_removed

    def generate_prevention_prompt(self) -> str:
        """
        Generate prompt content to prevent duplicate tool calls

        Returns:
            String to inject into prompts
        """
        if not self.call_history:
            # No previous calls yet
            return ""

        # Build warning message
        lines = []
        lines.append("\n⚠️ DUPLICATE TOOL CALL PREVENTION WARNING ⚠️")
        lines.append("=" * 70)
        lines.append("")
        lines.append("The following tool calls have ALREADY been executed:")
        lines.append("")

        # Group by tool type for readability
        func_calls = []
        rag_calls = []
        extras_calls = []

        for call in self.call_details[-20:]:  # Last 20 calls
            tool_type = call.get('type', '')
            tool_name = call.get('name', '')
            params = call.get('parameters', {})

            if tool_type == 'function':
                func_calls.append(f"  ✓ {tool_name}({self._format_params(params)})")
            elif tool_type == 'rag':
                rag_calls.append(f"  ✓ query_rag(query=\"{params.get('query', '')}\")")
            elif tool_type == 'extras':
                extras_calls.append(f"  ✓ query_extras(keywords={params.get('keywords', [])})")

        if func_calls:
            lines.append("FUNCTIONS ALREADY CALLED:")
            lines.extend(func_calls)
            lines.append("")

        if rag_calls:
            lines.append("RAG QUERIES ALREADY RUN:")
            lines.extend(rag_calls)
            lines.append("")

        if extras_calls:
            lines.append("EXTRAS ALREADY QUERIED:")
            lines.extend(extras_calls)
            lines.append("")

        lines.append("🚫 DO NOT call these tools again with the same parameters!")
        lines.append("🚫 Results are already available in tool results above.")
        lines.append("")

        # Budget warning
        usage_pct = self.budget.get_usage_percentage()
        lines.append(f"📊 TOOL CALL BUDGET: {self.budget.calls_used}/{self.budget.max_calls} used ({usage_pct:.0f}%)")

        if self.budget.is_budget_warning_threshold():
            lines.append(f"⚠️ WARNING: Budget at {usage_pct:.0f}% - conserve remaining {self.budget.calls_remaining} calls!")

        if self.budget.is_budget_exceeded():
            lines.append(f"🛑 BUDGET EXCEEDED: No more tool calls allowed!")

        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)

    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters for display"""
        if not params:
            return ""

        # Keep it concise
        items = []
        for k, v in list(params.items())[:3]:  # Max 3 params
            if isinstance(v, str) and len(v) > 30:
                v = v[:30] + "..."
            items.append(f"{k}={repr(v)}")

        result = ", ".join(items)

        if len(params) > 3:
            result += f", ... +{len(params) - 3} more"

        return result

    def get_budget_status(self) -> str:
        """Get budget status summary"""
        usage_pct = self.budget.get_usage_percentage()
        return (
            f"Tool Budget: {self.budget.calls_used}/{self.budget.max_calls} used ({usage_pct:.0f}%), "
            f"{self.budget.duplicate_calls_prevented} duplicates prevented"
        )

    def reset(self):
        """Reset tracker (for new extraction)"""
        self.call_history.clear()
        self.call_details.clear()
        self.budget = ToolCallBudget(max_calls=self.budget.max_calls)
        logger.debug("ToolDedupPreventer reset")


def create_tool_dedup_preventer(max_tool_calls: int = 100) -> ToolDedupPreventer:
    """
    Create a tool deduplication preventer instance

    Args:
        max_tool_calls: Maximum allowed tool calls

    Returns:
        ToolDedupPreventer instance
    """
    return ToolDedupPreventer(max_tool_calls=max_tool_calls)
