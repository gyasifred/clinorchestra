# Multi-Instance Task Isolation Architecture (v1.0.0)

## Overview

ClinOrchestra v1.0.0 implements a **multi-instance architecture** that provides complete isolation between different classification tasks (ADRD, Malnutrition, Custom). This prevents resource bleeding and configuration conflicts when working with multiple tasks.

## Architecture Components

### 1. SessionManager (`core/session_manager.py`)

The `SessionManager` is a singleton that manages multiple user sessions (browser tabs/connections).

**Key Features:**
- **Session Isolation**: Each browser tab gets its own session ID
- **Task Contexts**: Each session can have multiple task contexts (ADRD, Malnutrition, Custom)
- **Resource Separation**: Each task context has its own isolated `AppState` instance
- **Automatic Cleanup**: Inactive sessions are automatically cleaned up after 60 minutes

**Example Usage:**
```python
from core.session_manager import get_session_manager

# Get session manager instance
session_manager = get_session_manager()

# Create new session
session_id = session_manager.create_session()

# Get task-specific AppState
app_state = session_manager.get_task_context(session_id, "ADRD Classification")

# Switch to different task
new_app_state = session_manager.switch_task(session_id, "Malnutrition Classification")
```

### 2. AppStateProxy (`core/app_state_proxy.py`)

The `AppStateProxy` is a transparent proxy that forwards all attribute access to the currently active `AppState`.

**Why is this needed?**
- Gradio tab components are created once and capture references to AppState
- When switching tasks, we need tabs to work with a different AppState instance
- The proxy allows dynamic switching without recreating all UI components

**How it works:**
```python
from core.app_state_proxy import AppStateProxy

# Create proxy with initial AppState
proxy = AppStateProxy(initial_app_state)

# Pass proxy to all UI components
create_config_tab(proxy)
create_prompt_tab(proxy)

# When switching tasks, update the proxy
proxy._set_current_app_state(new_app_state)

# All UI components now automatically use the new AppState
```

### 3. Updated annotate.py

The main Gradio interface has been updated to support multi-instance task switching:

**Changes:**
1. Creates a `SessionManager` instance and session ID on startup
2. Creates an `AppStateProxy` for dynamic task switching
3. Adds a **Task Selector** dropdown in the UI
4. Passes the proxy (not raw AppState) to all tab creation functions
5. Implements `switch_task()` callback that updates the proxy when user switches tasks

## Task Isolation Guarantees

### What is Isolated Between Tasks?

Each task maintains its own separate:

1. **Model Configuration**
   - Provider (OpenAI, Anthropic, Google, Azure, Local)
   - Model name and settings
   - Temperature, max tokens, etc.

2. **Prompt Configuration**
   - Main task prompt
   - Minimal prompt
   - JSON schema for output structure

3. **Data Configuration**
   - Input file path
   - Clinical text column
   - ID column, row limits, etc.

4. **RAG Configuration**
   - Document paths
   - Chunk size, overlap
   - Top-k retrieval settings
   - Embeddings

5. **Function Definitions**
   - Custom Python functions
   - Function schemas
   - Tool calling configurations

6. **Patterns and Extras**
   - Regex patterns
   - Extraction hints
   - Context cues

7. **Processing Configuration**
   - Output paths
   - Batch settings
   - Agentic system configuration

8. **LLM Instances**
   - Separate LLM manager instances
   - Independent connection pools
   - Isolated caching

### What is Shared?

- **Logs**: All tasks log to the same logging system
- **Performance Monitoring**: Metrics are collected globally
- **Configuration Persistence**: Saved configs use the same persistence manager (but task-specific keys could be added in future)

## Usage Example

### Starting ClinOrchestra

```bash
python annotate.py --share --port 7860
```

### Working with Multiple Tasks

1. **Select Task**: Use the dropdown at the top to select "ADRD Classification", "Malnutrition Classification", or "Custom"

2. **Configure Task**:
   - Navigate to "Model Configuration" tab and configure for this task
   - Navigate to "Prompt Configuration" and load ADRD or Malnutrition prompts
   - Configure data sources, RAG, functions, etc.

3. **Switch Tasks**:
   - Select a different task from the dropdown
   - The UI automatically loads that task's configuration
   - All tabs now show the new task's settings

4. **Process Data**:
   - Each task processes data independently
   - Processing results are isolated per task

### Task Switching Behavior

When you switch tasks:

1. The `SessionManager` retrieves (or creates) the new task's `AppState`
2. The `AppStateProxy` is updated to point to the new `AppState`
3. Configuration is loaded from persistence (if available)
4. The status panel updates to show the new task's configuration state
5. All tabs automatically work with the new task's isolated resources

**Note**: Currently, UI components don't automatically refresh their displayed values when switching tasks. Navigate to each tab to see the task-specific configuration.

## Technical Details

### Session Lifecycle

```
Browser Opens
    ↓
Session Created (UUID)
    ↓
Default Task: "ADRD Classification"
    ↓
AppState Created for ADRD
    ↓
User Switches to "Malnutrition"
    ↓
AppState Created for Malnutrition (ADRD AppState preserved in SessionManager)
    ↓
Proxy Updated → All tabs now use Malnutrition AppState
    ↓
User Switches back to "ADRD"
    ↓
Proxy Updated → All tabs now use ADRD AppState (no recreation, existing instance reused)
    ↓
Session Inactive > 60 minutes
    ↓
Automatic Cleanup: All task contexts released
```

### Memory Management

- **Lazy Initialization**: AppState components (LLM, RAG, Functions) are only initialized when first used
- **Session Cleanup**: Inactive sessions are cleaned up after 60 minutes timeout
- **Task Context Reuse**: Switching back to a previously-used task reuses its existing AppState

### Thread Safety

- `SessionManager` uses a singleton pattern (one global instance)
- Each session has its own `SessionState` object
- Task contexts within a session are isolated
- **Current Limitation**: The implementation assumes single-threaded Gradio usage

## Future Enhancements

Potential improvements for future versions:

1. **UI Auto-Refresh**: Automatically update all UI components when switching tasks
2. **Task-Specific Persistence**: Save/load configurations with task-specific keys
3. **Concurrent Processing**: Allow multiple tasks to process data simultaneously
4. **Session Sharing**: Enable sharing sessions across browser tabs (same user)
5. **Task Templates**: Pre-configured task templates for common workflows
6. **Task Comparison**: Side-by-side comparison of results from different tasks

## Troubleshooting

### Task Configuration Not Appearing After Switch

**Problem**: Switched tasks but still seeing old task's configuration

**Solution**: Navigate to the specific configuration tab (Model, Prompt, Data, etc.) to trigger UI update

### Memory Usage Growing

**Problem**: Memory usage increases with multiple task switches

**Solution**:
- Sessions clean up automatically after 60 minutes of inactivity
- Restart the application to clear all sessions
- Future version will add manual cleanup button

### Task Contexts Not Isolated

**Problem**: Changes in one task appear to affect another task

**Solution**:
- Verify you're using v1.0.0 with SessionManager
- Check logs for task switching confirmation
- Ensure AppStateProxy is being used (check annotate.py imports)

## API Reference

### SessionManager

```python
class SessionManager:
    def create_session() -> str
        """Create new session, returns session_id"""

    def get_session(session_id: str) -> Optional[SessionState]
        """Get existing session"""

    def get_task_context(session_id: str, task_name: str) -> Optional[AppState]
        """Get AppState for specific task in session"""

    def switch_task(session_id: str, task_name: str) -> Optional[AppState]
        """Switch active task and return its AppState"""

    def cleanup_session(session_id: str)
        """Clean up specific session"""

    def cleanup_inactive_sessions(timeout_minutes: int = 60)
        """Clean up sessions inactive longer than timeout"""

    def get_stats() -> dict
        """Get session manager statistics"""
```

### AppStateProxy

```python
class AppStateProxy:
    def __init__(self, initial_app_state: AppState)
        """Initialize with initial AppState"""

    def _set_current_app_state(self, app_state: AppState)
        """Update current AppState being proxied"""

    def _get_current_app_state(self) -> AppState
        """Get current AppState being proxied"""

    # All other attributes/methods are forwarded to current AppState
```

## Version History

### v1.0.0 (2025-11-30)
- Initial implementation of multi-instance architecture
- SessionManager for session and task isolation
- AppStateProxy for dynamic task switching
- UI task selector dropdown
- Automatic session cleanup (60 min timeout)

---

**Author**: Frederick Gyasi (gyasi@musc.edu)
**Institution**: Medical University of South Carolina, Biomedical Informatics Center
**Version**: 1.0.0
