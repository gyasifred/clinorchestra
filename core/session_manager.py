#!/usr/bin/env python3
"""
Session Manager for Multi-Instance Task Isolation

Manages multiple isolated sessions with task contexts to prevent resource bleeding
between different tasks (ADRD, malnutrition, custom) and between different users/tabs.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import logging
import uuid
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from core.app_state import AppState

logger = logging.getLogger(__name__)


@dataclass
class TaskContext:
    """Context for a specific task within a session"""
    task_name: str
    app_state: AppState
    created_at: datetime
    last_accessed: datetime


class SessionState:
    """State for a single user session (browser tab)"""

    def __init__(self, session_id: str):
        """Initialize session state"""
        self.session_id = session_id
        self.task_contexts: Dict[str, TaskContext] = {}
        self.active_task: Optional[str] = None
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()

        logger.info(f"Session created: {session_id}")

    def get_task_context(self, task_name: str) -> AppState:
        """Get or create task context within this session"""
        self.last_accessed = datetime.now()

        if task_name not in self.task_contexts:
            logger.info(f"Creating new task context: {task_name} for session {self.session_id}")
            app_state = AppState()
            self.task_contexts[task_name] = TaskContext(
                task_name=task_name,
                app_state=app_state,
                created_at=datetime.now(),
                last_accessed=datetime.now()
            )
        else:
            self.task_contexts[task_name].last_accessed = datetime.now()

        return self.task_contexts[task_name].app_state

    def switch_task(self, task_name: str) -> AppState:
        """Switch active task and return its AppState"""
        logger.info(f"Session {self.session_id}: Switching task to {task_name}")
        self.active_task = task_name
        return self.get_task_context(task_name)

    def get_active_task_context(self) -> Optional[AppState]:
        """Get the currently active task's AppState"""
        if self.active_task is None:
            logger.warning(f"Session {self.session_id}: No active task set")
            return None
        return self.get_task_context(self.active_task)

    def list_tasks(self) -> list:
        """List all task contexts in this session"""
        return list(self.task_contexts.keys())

    def cleanup(self):
        """Clean up all task contexts in this session"""
        logger.info(f"Cleaning up session: {self.session_id}")
        for task_name, context in self.task_contexts.items():
            try:
                context.app_state.cleanup()
                logger.debug(f"Cleaned up task context: {task_name}")
            except Exception as e:
                logger.error(f"Error cleaning up task {task_name}: {e}")

        self.task_contexts.clear()
        self.active_task = None


class SessionManager:
    """
    Global session manager for multi-instance isolation

    Manages multiple user sessions (browser tabs), each with isolated task contexts.
    Ensures complete resource separation between:
    - Different users/tabs (sessions)
    - Different tasks within a session (ADRD, malnutrition, custom)
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern - one manager for entire application"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize session manager (only once due to singleton)"""
        if self._initialized:
            return

        self.sessions: Dict[str, SessionState] = {}
        self._initialized = True
        logger.info("SessionManager initialized (singleton)")

    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SessionState(session_id)
        logger.info(f"New session created: {session_id} (total sessions: {len(self.sessions)})")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get existing session by ID"""
        if session_id not in self.sessions:
            logger.warning(f"Session not found: {session_id}")
            return None
        return self.sessions[session_id]

    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple[str, SessionState]:
        """
        Get existing session or create new one

        Returns:
            Tuple of (session_id, SessionState)
        """
        if session_id is None or session_id not in self.sessions:
            session_id = self.create_session()

        return session_id, self.sessions[session_id]

    def cleanup_session(self, session_id: str):
        """Clean up a specific session"""
        if session_id in self.sessions:
            self.sessions[session_id].cleanup()
            del self.sessions[session_id]
            logger.info(f"Session cleaned up: {session_id} (remaining sessions: {len(self.sessions)})")
        else:
            logger.warning(f"Attempted to cleanup non-existent session: {session_id}")

    def cleanup_inactive_sessions(self, timeout_minutes: int = 60):
        """Clean up sessions inactive for longer than timeout"""
        from datetime import timedelta

        now = datetime.now()
        inactive_sessions = []

        for session_id, session_state in self.sessions.items():
            inactive_duration = now - session_state.last_accessed
            if inactive_duration > timedelta(minutes=timeout_minutes):
                inactive_sessions.append(session_id)

        for session_id in inactive_sessions:
            logger.info(f"Cleaning up inactive session: {session_id}")
            self.cleanup_session(session_id)

        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")

    def get_task_context(self, session_id: str, task_name: str) -> Optional[AppState]:
        """Get task context for a specific session"""
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.get_task_context(task_name)

    def switch_task(self, session_id: str, task_name: str) -> Optional[AppState]:
        """Switch task for a specific session"""
        session = self.get_session(session_id)
        if session is None:
            logger.error(f"Cannot switch task - session not found: {session_id}")
            return None
        return session.switch_task(task_name)

    def get_stats(self) -> dict:
        """Get session manager statistics"""
        total_tasks = sum(len(session.task_contexts) for session in self.sessions.values())

        return {
            'total_sessions': len(self.sessions),
            'total_task_contexts': total_tasks,
            'session_ids': list(self.sessions.keys()),
            'sessions': {
                session_id: {
                    'active_task': session.active_task,
                    'tasks': session.list_tasks(),
                    'created_at': session.created_at.isoformat(),
                    'last_accessed': session.last_accessed.isoformat()
                }
                for session_id, session in self.sessions.items()
            }
        }


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# Convenience functions for common operations
def create_session() -> str:
    """Create new session - returns session_id"""
    return get_session_manager().create_session()


def get_task_context(session_id: str, task_name: str) -> Optional[AppState]:
    """Get AppState for a specific task in a session"""
    return get_session_manager().get_task_context(session_id, task_name)


def switch_task(session_id: str, task_name: str) -> Optional[AppState]:
    """Switch to a different task in a session"""
    return get_session_manager().switch_task(session_id, task_name)


def cleanup_session(session_id: str):
    """Clean up a session"""
    get_session_manager().cleanup_session(session_id)
