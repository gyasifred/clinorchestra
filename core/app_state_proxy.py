#!/usr/bin/env python3
"""
AppState Proxy for Multi-Instance Task Switching

Provides a proxy that allows UI components to reference the "current" AppState
while the actual AppState instance changes when switching between tasks.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

from typing import Any


class AppStateProxy:
    """
    Proxy that forwards all attribute access to the current AppState instance.

    This allows UI components to reference a single "proxy" object while
    the underlying AppState instance changes when switching tasks.
    """

    def __init__(self, initial_app_state):
        """Initialize proxy with initial AppState"""
        # Use object.__setattr__ to bypass our custom __setattr__
        object.__setattr__(self, '_current_app_state', initial_app_state)

    def _set_current_app_state(self, app_state):
        """Update the current AppState being proxied"""
        object.__setattr__(self, '_current_app_state', app_state)

    def _get_current_app_state(self):
        """Get the current AppState being proxied"""
        return object.__getattribute__(self, '_current_app_state')

    def __getattribute__(self, name):
        """Forward attribute access to current AppState"""
        # Internal proxy methods/attributes
        if name in ('_current_app_state', '_set_current_app_state', '_get_current_app_state'):
            return object.__getattribute__(self, name)

        # Forward everything else to current AppState
        current = object.__getattribute__(self, '_current_app_state')
        return getattr(current, name)

    def __setattr__(self, name, value):
        """Forward attribute setting to current AppState"""
        # Internal proxy attribute
        if name == '_current_app_state':
            object.__setattr__(self, name, value)
        else:
            # Forward to current AppState
            current = object.__getattribute__(self, '_current_app_state')
            setattr(current, name, value)

    def __delattr__(self, name):
        """Forward attribute deletion to current AppState"""
        current = object.__getattribute__(self, '_current_app_state')
        delattr(current, name)

    def __repr__(self):
        """String representation"""
        current = object.__getattribute__(self, '_current_app_state')
        return f"<AppStateProxy proxying {current}>"
