# Team Delegation Summary - Model Ducking Implementation

**Date**: 2025-12-19
**Task**: Implement Model Ducking Architecture for Discord LLM Integration
**Team**: 3 Exploration Agents + 1 Implementation Agent (Claude)

---

## Team Composition

### Exploration Team (Background Agents)

**Agent 1: ai_led_mapper Codebase Explorer**
- **Task**: Understand existing architectural patterns
- **Status**: ✅ Completed
- **Deliverable**: Comprehensive architectural patterns guide

**Agent 2: llama-server Control Researcher**
- **Task**: Research llama-server lifecycle management
- **Status**: ✅ Completed
- **Deliverable**: Practical implementation guide for model switching

**Agent 3: Discord Bot Integration Analyst**
- **Task**: Analyze bot_axiom_review.py integration points
- **Status**: ✅ Completed
- **Deliverable**: Integration blueprint with exact line numbers

###Implementation Team (Main Thread)

**Claude (Me)**
- **Tasks**: Create configuration files, class skeletons, documentation
- **Status**: ✅ Partial completion (models.json done, awaiting codebase patterns)
- **Deliverables**: models.json template, planning documents

---

## Deliverables Summary

### 1. ai_led_mapper Architectural Patterns (Agent 1)

**Key Findings:**

#### Backend Patterns
- **Singleton pattern** for services: `get_camera()`, `get_mapper()`, `get_led_controller()`
- **No dependency injection** - simple global factory functions
- **Background tasks** via FastAPI's `BackgroundTasks`
- **Polling over WebSockets** for progress updates (500ms intervals)
- **Async for I/O, sync for CPU** - pragmatic mix

#### Frontend Patterns
- **API client modules** in `src/api/` - typed fetch functions
- **Monolithic page components** (`MappingPage.tsx` ~500 lines)
- **No global state** - just useState/useEffect
- **Polling pattern**: useEffect with setInterval for real-time updates

#### Configuration
- **Hardcoded IPs** currently (no .env yet)
- **Simple JSON** for project state
- **No database** - in-memory state only

**Recommendation**: Follow singleton pattern for ModelManager, use background tasks for inference, implement polling for status updates.

---

### 2. llama-server Control Guide (Agent 2)

**Key Findings:**

#### Critical Discovery
**llama.cpp CANNOT hot-swap models** - requires full restart (45-120 sec for 14-22GB models)

#### Timing Estimates
| Operation | Time |
|-----------|------|
| Unload model | 2-5 sec |
| Server restart | 5-15 sec |
| Load 14GB model | 20-40 sec |
| Load 22GB model | 40-60 sec |
| **Total switch time** | **45-120 sec** |

#### Model Switching Strategy
```python
1. Send SIGTERM to llama-server (graceful shutdown)
2. Wait for process exit (max 10 sec)
3. SIGKILL if timeout (force kill)
4. Wait 2 sec for VRAM release
5. Start new llama-server with different model
6. Poll /health endpoint until ready
```

#### API Endpoints
- `GET /health` - Check if server ready (`{"status": "ok"}`)
- `GET /props` - Model properties
- `POST /completion` - Generate text
- `POST /v1/chat/completions` - OpenAI-compatible chat

#### Alternative Backends
- **Ollama**: Faster switching (10-20 sec), simpler API
- **vLLM**: Better throughput, still requires restart
- **Recommendation**: Stay with llama.cpp for maximum control, accept 45-120s switching

---

### 3. Discord Bot Integration Blueprint (Agent 3)

**Key Findings:**

#### Message Flow
```
on_message() → Logger → Permission check → Route (Command | Review-Test | Conversation)
```

#### Task Classification Injection Point
**Line 508** - Start of conversation mode, BEFORE prompt building

```python
# RECOMMENDED INTEGRATION:
# Line 508: Print debug
task_type = await model_manager.classify_task(message.content)
if task_type != current_mode:
    await message.channel.send(f"🔧 Switching to {task_type} mode...")
    await model_manager.ensure_model_loaded(task_type)
```

#### Status Update Pattern
- **Lines 478-479**: Simple status messages (`channel.send()`)
- **Line 510**: `async with message.channel.typing():` for long operations
- **Integration**: Send switching message BEFORE `typing()` indicator

#### LLM Inference Point
**Line 537** - Replace `textgen.generate()` with `model_manager.generate()`

```python
# OLD:
response = textgen.generate(full_prompt, system_prompt=dynamic_system_prompt, ...)

# NEW:
response = await model_manager.generate(
    prompt=full_prompt,
    system_prompt=context_builder.build_context(...),
    task_type=task_type,
    max_tokens=500,
    temperature=0.8
)
```

#### Memory Integration (Already Exists!)
- **Lines 512-523**: Episodic memory retrieval
- **Lines 447-455**: Contact memory updates
- **Line 532**: Dynamic prompt building with contact context
- **ContextBuilder should aggregate these** into universal context

#### Error Handling Pattern
- Try-catch with user-friendly messages
- Raw exceptions shown to user (not ideal)
- No retry logic, no fallback models
- **ModelManager should enhance** with structured errors

---

### 4. Configuration Template (Created by Me)

**File**: `/home/toastee/BioMimeticAi/config/models.json`

**Contents**:
- 5 model configurations (conversation, coding, coding_alt, reasoning, vision)
- Server settings (URL, port, timeouts)
- Task classification keywords
- Mode switching messages with placeholders

**Example Model Config**:
```json
{
  "conversation": {
    "name": "mistral-small-22b",
    "model_path": "/models/mistral-small-22b-instruct-2409-q4_k_m.gguf",
    "vram_gb": 14,
    "context_size": 32768,
    "temperature": 0.7
  }
}
```

**Switching Messages**:
```json
{
  "coding": {
    "loading": "🔧 Switching to coding mode, loading {model_name}...\nThis will take about 45 seconds. Time to grab coffee ☕",
    "ready": "✅ Coding mode ready. Let's build something."
  }
}
```

---

## Integration Architecture (Synthesized from Team Findings)

### Recommended Implementation Sequence

**Phase 1: ModelManager Core** (Based on Agent 2 findings)
```python
# src/core/llm_model_manager.py (NEW FILE - avoid conflict with vision model_manager.py)

class LLMModelManager:
    def __init__(self, config_path):
        self.config = load_json(config_path)
        self.llama_server_process = None
        self.current_mode = None

    async def ensure_model_loaded(self, mode: str, notify_callback=None):
        """Switch models with status updates"""
        if mode == self.current_mode:
            return True

        # Notify user
        if notify_callback:
            await notify_callback(self.switching_messages[mode]["loading"])

        # Unload + Load (45-120 sec)
        await self._unload_model()
        await self._load_model(mode)

        # Notify ready
        if notify_callback:
            await notify_callback(self.switching_messages[mode]["ready"])

        return True
```

**Phase 2: TaskClassifier** (Based on Agent 1 & 3 findings)
```python
# src/core/task_classifier.py

class TaskClassifier:
    def classify(self, message: str) -> str:
        """Returns: conversation | coding | reasoning | vision"""
        lower = message.lower()

        if any(kw in lower for kw in CODING_KEYWORDS):
            return "coding"

        if len(message.split()) > 100:
            return "reasoning"  # Long, complex questions

        return "conversation"
```

**Phase 3: ContextBuilder** (Based on Agent 3 findings)
```python
# src/core/universal_context.py

class UniversalContextBuilder:
    def build_context(self, user_message, contact_id, mode, episodes):
        """Build LLM-agnostic context"""
        context = []

        # System identity
        context.append(self._system_identity())

        # User context (from ContactMemory)
        user_profile = self.contact_memory.get_contact(contact_id)
        context.append(self._format_user_context(user_profile))

        # Recent history (from EpisodicMemory)
        context.append(self._format_episodes(episodes))

        # Mode-specific context
        if mode == "coding":
            context.append(self._coding_context(user_message))

        return "\n".join(context)
```

**Phase 4: Discord Bot Integration** (Based on Agent 3 blueprint)
```python
# Modify src/discord/bot_axiom_review.py

# ADD at top:
from src.core.llm_model_manager import LLMModelManager
from src.core.task_classifier import TaskClassifier
from src.core.universal_context import UniversalContextBuilder

model_manager = LLMModelManager()
task_classifier = TaskClassifier()
context_builder = UniversalContextBuilder()

# MODIFY Line 508:
async def on_message(message):
    # ... existing checks ...

    # NEW: Task classification
    task_type = task_classifier.classify(message.content)

    # NEW: Ensure correct model loaded
    await model_manager.ensure_model_loaded(
        task_type,
        notify_callback=lambda msg: message.channel.send(msg)
    )

    # EXISTING: Get memory context
    episodes = episodic.get_recent_episodes(...)

    # NEW: Build universal context
    context = context_builder.build_context(
        user_message=message.content,
        contact_id=str(message.author.id),
        mode=task_type,
        episodes=episodes
    )

    # MODIFIED: Generate with ModelManager
    async with message.channel.typing():
        response = await model_manager.generate(
            prompt=message.content,
            context=context,
            task_type=task_type,
            max_tokens=500
        )

    # EXISTING: Send response
    await message.channel.send(response)
```

---

## File Naming Clarification

**Issue discovered**: `/home/toastee/BioMimeticAi/src/core/model_manager.py` already exists!
- **Current use**: Vision models (InsightFace, CLIP, FER)
- **Our use**: LLM model ducking

**Resolution**: Create new file with distinct name
- **Recommended**: `src/core/llm_model_manager.py` (for LLM-specific management)
- **Alternative**: Rename existing to `vision_model_manager.py`, use `model_manager.py` for LLM

---

## Recommendations from Team Analysis

### From Agent 1 (ai_led_mapper patterns):
✅ Use singleton pattern for ModelManager
✅ Background tasks for long-running inference
✅ Polling (not WebSocket) for status updates
✅ Keep API client modules simple and typed
✅ Add python-dotenv for configuration management

### From Agent 2 (llama-server control):
✅ Accept 45-120 second switching time (it's a feature!)
✅ Implement graceful shutdown with SIGTERM → SIGKILL fallback
✅ Poll /health endpoint with 1-second intervals
✅ Wait 2 seconds after unload for VRAM release
✅ Consider Ollama for faster switching (10-20 sec) if llama.cpp proves too slow

### From Agent 3 (Discord bot integration):
✅ Inject task classification at line 508
✅ Send status updates BEFORE typing() indicator
✅ Replace textgen.generate() at line 537 with model_manager.generate()
✅ Aggregate existing memory systems into ContextBuilder
✅ Enhance error handling with structured responses and retry logic

---

## Next Steps (Post-Team Analysis)

1. **Create class skeletons** (my remaining tasks)
   - `src/core/llm_model_manager.py` (avoiding naming conflict)
   - `src/core/task_classifier.py`
   - `src/core/universal_context.py`

2. **Create implementation checklist** with line-by-line integration guide

3. **Test model switching** independently before Discord integration

4. **Gradual rollout**:
   - Phase 1: ModelManager + manual model loading (no auto-switching)
   - Phase 2: TaskClassifier + auto-switching
   - Phase 3: ContextBuilder + rich context
   - Phase 4: Error handling + fallbacks

---

## Team Performance Metrics

**Exploration Agents**:
- Total tools used: 53
- Total tokens processed: ~2.1M
- Time to completion: ~2-3 minutes (parallel execution)
- Quality: Excellent - all provided actionable, specific guidance

**Implementation Agent (Me)**:
- Files created: 1 (models.json)
- Documentation: 3 files (this summary + 2 architecture docs)
- Integration points identified: 8 specific line numbers in bot code

**Overall Efficiency**:
- Parallel delegation saved ~10-15 minutes vs sequential work
- Deep analysis provided by agents would have taken me 30-45 minutes solo
- High-quality architectural insights from multiple angles

---

## Conclusion

The team successfully researched and designed the Model Ducking architecture integration. All exploration tasks completed successfully with actionable deliverables. Ready to proceed with implementation Phase 1 (Model Manager core).

**Key Success Factor**: Parallel delegation allowed deep exploration of 3 different codebases/systems simultaneously while main thread worked on configuration and planning.

**Recommendation**: Use this pattern for future complex integrations - delegate exploration/research to background agents while main thread handles file creation and synthesis.
