#!/usr/bin/env python3
"""
Process Persistence Helper for Long-Running Tasks
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import time

logger = logging.getLogger(__name__)


class ProcessState:
    """Manages persistent state for running processes"""
    
    def __init__(self, state_dir: str = "./process_states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.active_processes = {}
        self.lock = threading.Lock()
        
        logger.info(f"ProcessState initialized: {self.state_dir}")
    
    def create_process(self, process_id: str, metadata: Dict[str, Any]) -> bool:
        """Create a new process state"""
        try:
            with self.lock:
                state = {
                    'process_id': process_id,
                    'status': 'running',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'metadata': metadata,
                    'progress': {
                        'processed': 0,
                        'failed': 0,
                        'total': metadata.get('total_rows', 0),
                        'percentage': 0.0
                    },
                    'logs': []
                }
                
                state_file = self.state_dir / f"{process_id}.json"
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                self.active_processes[process_id] = state
                
                logger.info(f"Created process state: {process_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create process state: {e}")
            return False
    
    def update_progress(self, process_id: str, processed: int, failed: int, percentage: float) -> bool:
        """Update process progress"""
        try:
            with self.lock:
                if process_id not in self.active_processes:
                    state = self.load_process(process_id)
                    if not state:
                        return False
                else:
                    state = self.active_processes[process_id]
                
                state['progress']['processed'] = processed
                state['progress']['failed'] = failed
                state['progress']['percentage'] = percentage
                state['updated_at'] = datetime.now().isoformat()
                
                state_file = self.state_dir / f"{process_id}.json"
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
            return False
    
    def add_log(self, process_id: str, log_entry: str) -> bool:
        """Add a log entry to process"""
        try:
            with self.lock:
                if process_id not in self.active_processes:
                    state = self.load_process(process_id)
                    if not state:
                        return False
                else:
                    state = self.active_processes[process_id]
                
                state['logs'].append({
                    'timestamp': datetime.now().isoformat(),
                    'message': log_entry
                })
                
                # Keep only last 1000 logs
                if len(state['logs']) > 1000:
                    state['logs'] = state['logs'][-1000:]
                
                state['updated_at'] = datetime.now().isoformat()
                
                state_file = self.state_dir / f"{process_id}.json"
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to add log: {e}")
            return False
    
    def complete_process(self, process_id: str, success: bool = True) -> bool:
        """Mark process as completed"""
        try:
            with self.lock:
                if process_id not in self.active_processes:
                    state = self.load_process(process_id)
                    if not state:
                        return False
                else:
                    state = self.active_processes[process_id]
                
                state['status'] = 'completed' if success else 'failed'
                state['completed_at'] = datetime.now().isoformat()
                state['updated_at'] = datetime.now().isoformat()
                
                state_file = self.state_dir / f"{process_id}.json"
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                logger.info(f"Process {process_id} marked as {state['status']}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete process: {e}")
            return False
    
    def load_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Load process state from disk"""
        try:
            state_file = self.state_dir / f"{process_id}.json"
            if not state_file.exists():
                return None
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            with self.lock:
                self.active_processes[process_id] = state
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to load process: {e}")
            return None
    
    def get_active_processes(self) -> Dict[str, Dict[str, Any]]:
        """Get all active processes"""
        active = {}
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                if state['status'] == 'running':
                    process_id = state['process_id']
                    active[process_id] = state
                    
            except Exception as e:
                logger.error(f"Failed to read state file {state_file}: {e}")
        
        return active
    
    def get_logs(self, process_id: str, last_n: int = 100) -> list:
        """Get recent logs for a process"""
        try:
            state = self.load_process(process_id)
            if not state:
                return []
            
            logs = state.get('logs', [])
            return logs[-last_n:]
            
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return []
    
    def cleanup_old_processes(self, days: int = 7):
        """Clean up process states older than specified days"""
        try:
            cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            for state_file in self.state_dir.glob("*.json"):
                try:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    
                    updated_at = datetime.fromisoformat(state['updated_at']).timestamp()
                    
                    if updated_at < cutoff and state['status'] != 'running':
                        state_file.unlink()
                        logger.info(f"Cleaned up old process: {state_file.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to cleanup {state_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Global instance
_process_state = None


def get_process_state() -> ProcessState:
    """Get global ProcessState instance"""
    global _process_state
    if _process_state is None:
        _process_state = ProcessState()
    return _process_state