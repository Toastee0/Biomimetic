# Task Delegation Documents - Model Ducking Implementation

**Purpose**: Copy-paste these task descriptions to other LLMs (Claude, GPT-4, etc.) to parallelize work.

**Context**: We're implementing a "Model Ducking" system where Discord bot switches between specialized LLMs (conversation, coding, reasoning) based on task type.

---

## Task 1: Implement LLMModelManager Class

**For**: Coding-specialized LLM (DeepSeek, GPT-4, Claude)

### Context
You're implementing the core model manager for a Discord bot that switches between different LLM models (Mistral-Small-22B for conversation, DeepSeek-Coder-33B for coding, etc.). Models are loaded via llama-server which requires full restart to switch models (45-120 seconds).

### Requirements
- Read `/home/toastee/BioMimeticAi/config/models.json` for configuration
- Manage llama-server subprocess (start/stop/health check)
- Handle graceful model switching (SIGTERM → SIGKILL if timeout)
- Send status messages to user during switching
- Poll /health endpoint until server ready
- Auto-duck back to conversation mode after 15 min inactivity

### Files to Read First
- `/home/toastee/BioMimeticAi/config/models.json` - Model configurations
- `/home/toastee/BioMimeticAi/docs/MODEL_DUCKING_ARCHITECTURE.md` - Full architecture spec
- `/home/toastee/BioMimeticAi/docs/TEAM_DELEGATION_SUMMARY.md` - Research findings (llama-server control section)

### Deliverable
Create `/home/toastee/BioMimeticAi/src/core/llm_model_manager.py` with:

```python
class LLMModelManager:
    """
    Manages dynamic loading/unloading of LLM models via llama-server

    Key methods:
    - ensure_model_loaded(mode, notify_callback) -> bool
    - _unload_model() -> None
    - _load_model(mode, config) -> bool
    - _wait_for_server_ready() -> bool
    - check_auto_duck() -> None  # Background task
    - get_current_mode() -> str
    - shutdown() -> None
    """
```

### Key Implementation Details
1. Use `asyncio.create_subprocess_exec()` for llama-server
2. Graceful shutdown: `process.terminate()` then `process.kill()` after 10s timeout
3. Health check: Poll `GET http://localhost:53307/health` every 1s for 60s max
4. Notify user via callback: `await notify_callback("🔧 Switching to coding mode...")`
5. Wait 2 seconds after unload for VRAM to release

### Success Criteria
- Can start llama-server with specified model
- Can gracefully switch between models
- Returns True/False for success/failure
- Handles timeouts and errors gracefully
- Logs all operations

---

## Task 2: Implement TaskClassifier

**For**: Any LLM

### Context
Analyze user messages to determine which model should handle the request. This feeds into the ModelManager to trigger automatic model switching.

### Requirements
- Classify messages as: `conversation`, `coding`, `reasoning`, or `vision`
- Use keyword matching from `models.json` configuration
- Support length-based classification (long messages = reasoning)
- Fast execution (< 5ms, runs on every message)

### Files to Read First
- `/home/toastee/BioMimeticAi/config/models.json` - See `task_classification` section
- `/home/toastee/BioMimeticAi/docs/MODEL_DUCKING_ARCHITECTURE.md` - TaskClassifier section

### Deliverable
Create `/home/toastee/BioMimeticAi/src/core/task_classifier.py`:

```python
class TaskClassifier:
    """
    Classifies user messages to determine required model type

    Methods:
    - classify(message: str) -> str  # Returns mode name
    - _check_keywords(message: str, keyword_list: list) -> bool
    """
```

### Classification Logic
1. Check coding keywords: "implement", "code", "function", "debug", etc. → `coding`
2. Check vision keywords: "what do you see", "describe scene", etc. → `vision`
3. Check message length: > 100 words → `reasoning`
4. Default → `conversation`

### Success Criteria
- Correctly classifies coding requests
- Correctly classifies vision requests
- Defaults to conversation for simple messages
- Runs quickly (no LLM calls, just keyword matching)

---

## Task 3: Implement UniversalContextBuilder

**For**: Any LLM

### Context
Build LLM-agnostic context windows that work with any backend. Aggregates episodic memory, contact profiles, and task-specific context into a formatted markdown prompt.

### Requirements
- Pull from EpisodicMemory (recent conversations)
- Pull from ContactMemory (user profile)
- Add task-specific context (docs for coding, camera feeds for vision)
- Format as markdown for any LLM backend
- Keep under reasonable token limits

### Files to Read First
- `/home/toastee/BioMimeticAi/src/memory/episodic.py` - How to get episodes
- `/home/toastee/BioMimeticAi/src/memory/contact_memory.py` - How to get contact info
- `/home/toastee/BioMimeticAi/src/core/dynamic_prompts.py` - Existing prompt building patterns
- `/home/toastee/BioMimeticAi/docs/MODEL_DUCKING_ARCHITECTURE.md` - Universal Context section

### Deliverable
Create `/home/toastee/BioMimeticAi/src/core/universal_context.py`:

```python
class UniversalContextBuilder:
    """
    Builds LLM-agnostic context windows

    Methods:
    - build_context(user_message, contact_id, mode, episodes) -> str
    - _system_identity() -> str
    - _format_user_context(profile) -> str
    - _format_episodes(episodes) -> str
    - _coding_context(message) -> str  # Task-specific
    - _vision_context(message) -> str  # Task-specific
    """
```

### Context Structure
```markdown
# SYSTEM IDENTITY
[Core principles, values]

# CURRENT MODE
{conversation | coding | vision | reasoning}

# USER CONTEXT
Contact: toastee0
Trust: 0.6
Style: Direct, technical

# MEMORY CONTEXT
Recent interactions:
1. [timestamp] Discussed X
2. [timestamp] Worked on Y

# TASK-SPECIFIC CONTEXT
[Docs, code files, camera feeds depending on mode]

# USER REQUEST
[actual message]
```

### Success Criteria
- Correctly formats all context sections
- Integrates with existing memory systems
- Produces markdown output
- Handles missing data gracefully (new users)

---

## Task 4: Discord Bot Integration

**For**: Python/Discord.py specialist

### Context
Integrate the three components (LLMModelManager, TaskClassifier, UniversalContextBuilder) into the existing Discord bot. The bot currently uses a simple TextGenClient and needs to be upgraded for model ducking.

### Requirements
- Minimal changes to existing bot code
- Insert at specific line numbers identified in research
- Maintain existing error handling patterns
- Preserve all existing commands and features

### Files to Read First
- `/home/toastee/BioMimeticAi/src/discord/bot_axiom_review.py` - Main bot file
- `/home/toastee/BioMimeticAi/docs/TEAM_DELEGATION_SUMMARY.md` - Integration points section (Agent 3 findings)
- `/home/toastee/BioMimeticAi/docs/MODEL_DUCKING_ARCHITECTURE.md` - Integration section

### Integration Points (Exact Line Numbers)

**Line 46**: Add imports
```python
from src.core.llm_model_manager import LLMModelManager
from src.core.task_classifier import TaskClassifier
from src.core.universal_context import UniversalContextBuilder
```

**Line 55**: Initialize managers
```python
model_manager = LLMModelManager()
task_classifier = TaskClassifier()
context_builder = UniversalContextBuilder()
```

**Line 508**: Add task classification
```python
# Classify task
task_type = task_classifier.classify(message.content)
logger.info(f"Task classified as: {task_type}")
```

**Line 509**: Ensure model loaded
```python
# Ensure correct model loaded
await model_manager.ensure_model_loaded(
    task_type,
    notify_callback=lambda msg: message.channel.send(msg)
)
```

**Line 526-529**: Replace manual context building
```python
# Build universal context (replaces lines 526-529)
context = context_builder.build_context(
    user_message=message.content,
    contact_id=str(message.author.id),
    mode=task_type,
    episodes=recent_episodes
)
```

**Line 537**: Replace textgen call
```python
# Generate using ModelManager (not TextGenClient directly)
response = await model_manager.generate(
    prompt=message.content,
    context=context,
    max_tokens=500,
    temperature=0.8
)
```

### Success Criteria
- Bot starts without errors
- Correctly classifies tasks
- Switches models when needed
- Sends status updates to user
- Existing commands still work
- Memory integration preserved

---

## Task 5: Testing & Documentation

**For**: Any LLM

### Context
Create comprehensive tests and update documentation after implementation is complete.

### Requirements

#### Test Cases Needed
1. **Model Switching Test**
   - Start bot in conversation mode
   - Send coding request
   - Verify model switches
   - Verify status messages sent
   - Verify response quality

2. **Task Classification Test**
   - Test each keyword category
   - Test edge cases (mixed keywords)
   - Test length-based classification

3. **Context Building Test**
   - Test with new user (no history)
   - Test with established user
   - Test each mode (conversation, coding, etc.)
   - Verify all context sections present

4. **Error Handling Test**
   - Kill llama-server mid-request
   - Invalid model path
   - Timeout during model load
   - Verify graceful degradation

#### Documentation Updates Needed
1. Update `/home/toastee/BioMimeticAi/README.MD` with model ducking feature
2. Update `/home/toastee/BioMimeticAi/CLAUDE.md` with usage examples
3. Create `/home/toastee/BioMimeticAi/docs/MODEL_DUCKING_USAGE.md` user guide

### Deliverable
Create `/home/toastee/BioMimeticAi/tests/test_model_ducking.py` with pytest tests

### Success Criteria
- All tests pass
- Documentation updated
- Usage examples provided
- Troubleshooting guide included

---

## Task 6: Configuration & Deployment

**For**: DevOps/System administration specialist

### Context
Set up the system for production use with proper model files, systemd services, and configuration.

### Requirements

#### Model Files Setup
1. Download/locate model files:
   - Mistral-Small-22B (Q4 GGUF)
   - DeepSeek-Coder-33B (Q4 GGUF)
   - Qwen2.5-Coder-32B (Q4 GGUF)
   - Mistral-Large (Q4 GGUF)

2. Update paths in `config/models.json`

#### Systemd Service
Create `/etc/systemd/system/biomimetic-discord-bot.service`:
```ini
[Unit]
Description=BioMimeticAI Discord Bot with Model Ducking
After=network.target

[Service]
Type=simple
User=toastee
WorkingDirectory=/home/toastee/BioMimeticAi
Environment="DISCORD_TOKEN=..."
ExecStart=/home/toastee/BioMimeticAi/venv/bin/python src/discord/bot_axiom_review.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Auto-Duck Background Task
Add to bot initialization:
```python
@bot.event
async def on_ready():
    # Start auto-duck loop
    bot.loop.create_task(auto_duck_loop())

async def auto_duck_loop():
    while True:
        await asyncio.sleep(60)
        await model_manager.check_auto_duck()
```

### Success Criteria
- Models downloaded and accessible
- Systemd service starts on boot
- Auto-duck works (switches back after timeout)
- Logs to proper location
- Handles crashes and restarts

---

## Coordination Notes

### Execution Order
1. Task 1 (ModelManager) - Core infrastructure, others depend on this
2. Task 2 (TaskClassifier) - Simple, can run parallel with #1
3. Task 3 (ContextBuilder) - Can run parallel with #1 and #2
4. Task 4 (Integration) - Requires #1, #2, #3 complete
5. Task 5 (Testing) - After #4
6. Task 6 (Deployment) - After #5

### Dependencies
- Tasks 1, 2, 3 are independent (can parallelize)
- Task 4 depends on 1, 2, 3
- Task 5 depends on 4
- Task 6 depends on 5

### Communication Between LLMs
If using multiple LLMs for different tasks:
- Share the `/home/toastee/BioMimeticAi/config/models.json` file
- Share research findings in `/home/toastee/BioMimeticAi/docs/TEAM_DELEGATION_SUMMARY.md`
- Each task should write to its designated file path
- No overlapping file writes

### Estimated Time Per Task
- Task 1: 4-5 hours (most complex)
- Task 2: 1 hour (straightforward)
- Task 3: 2-3 hours (memory integration)
- Task 4: 2-3 hours (careful integration)
- Task 5: 2-3 hours (comprehensive tests)
- Task 6: 1-2 hours (configuration)

**Total**: ~15-20 hours if done sequentially, ~6-8 hours if parallelized

---

## Usage Instructions

### For YOU (toastee)
1. Copy a task section (e.g., "Task 1: Implement LLMModelManager Class")
2. Paste into a new chat with another LLM (Claude, GPT-4, etc.)
3. Add: "Please implement this. All file paths are on the server at the specified locations."
4. Review the output before committing

### For the OTHER LLM
- You have full context in the task description
- Read the "Files to Read First" before coding
- Follow the specified file structure exactly
- Test your code if possible
- Document any assumptions or deviations

---

## Quality Checklist

Each task should:
- ✅ Read specified context files first
- ✅ Follow existing code patterns (see TEAM_DELEGATION_SUMMARY.md)
- ✅ Handle errors gracefully
- ✅ Include docstrings
- ✅ Use type hints
- ✅ Match the deliverable specification
- ✅ Be production-ready (not a prototype)

---

## Example Delegation

**What you say to the other LLM:**

> I need you to implement Task 1 from this spec. This is part of a larger system where a Discord bot switches between different LLM models based on task type.
>
> [Paste Task 1 section here]
>
> All files mentioned exist on the server. Please create the complete implementation of `llm_model_manager.py` with proper error handling, logging, and the exact method signatures specified.

**What you get back:**

Complete, tested implementation ready to save to the file.

---

That's it! This document is your delegation toolkit for distributing the Model Ducking implementation across multiple LLMs working in parallel.
