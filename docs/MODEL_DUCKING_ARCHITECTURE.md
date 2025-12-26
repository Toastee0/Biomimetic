# Model Ducking Architecture - Dynamic LLM Backend Switching

**Core Concept**: The BioMimeticAI system generates a continuously modified **universal context window** that can be used with any LLM backend. Models are loaded/unloaded on demand based on task type (conversation vs coding vs vision).

**Analogy**: Like audio ducking (lowering music when speech starts), we "duck" models - unloading inactive models to free VRAM for the active task.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIVERSAL CONTEXT BUILDER                    │
│                                                                 │
│  Generates LLM-agnostic context from:                          │
│  - Episodic memory (recent conversations)                      │
│  - Contact memory (user profile)                               │
│  - Axiom system (relevant principles)                          │
│  - Documentation (when coding)                                 │
│  - Codebase files (when coding)                                │
│  - Vision events (when analyzing scenes)                       │
│                                                                 │
│  Output: Markdown-formatted context + user message             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TASK CLASSIFIER                            │
│                                                                 │
│  Analyzes request → Determines required model:                 │
│  - "conversation" → Mistral-Small-22B                          │
│  - "coding"       → DeepSeek-Coder / CodeLlama                 │
│  - "vision"       → LLaVA / Qwen-VL (future)                   │
│  - "reasoning"    → Mistral-Large (if available)               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL MANAGER                              │
│                                                                 │
│  Current State: {                                               │
│    loaded_model: "mistral-small-22b",                          │
│    vram_used: 14GB,                                            │
│    vram_available: 10GB,                                       │
│    last_switch: timestamp                                      │
│  }                                                              │
│                                                                 │
│  Operations:                                                    │
│  - load_model(model_id) → Unload current, load requested       │
│  - is_loaded(model_id) → Check if already loaded              │
│  - get_capabilities(model_id) → What can this model do?       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM BACKEND ABSTRACTION                      │
│                                                                 │
│  Unified interface for multiple backends:                      │
│  - llama-server (local, supports multiple models)             │
│  - vllm (faster inference, same models)                        │
│  - Claude API (external, no VRAM)                              │
│  - OpenAI API (external, no VRAM)                              │
│                                                                 │
│  Interface:                                                     │
│    generate(context: str, max_tokens: int) → str              │
│    stream(context: str) → AsyncIterator[str]                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                   LLM Response
```

---

## Model Ducking Flow

### Scenario: User switches from conversation to coding

```
1. User: "Let's implement the ring buffer class"

2. Task Classifier detects: coding request

3. Model Manager checks:
   - Current: mistral-small-22b (14GB, conversation)
   - Needed: deepseek-coder-33b (22GB, coding)
   - Action: SWITCH REQUIRED

4. Discord Bot sends:
   "🔧 Switching to coding mode, loading DeepSeek-Coder-33B...
    This will take about 45 seconds. Time to grab coffee ☕"

5. Model Ducking Sequence:
   ┌─────────────────────────────────────────────┐
   │ [14GB] Mistral-Small-22B (conversation)     │ ← Currently loaded
   └─────────────────────────────────────────────┘
                    ⬇ Unload (~30 sec)
   ┌─────────────────────────────────────────────┐
   │ [0GB] No model loaded                       │
   └─────────────────────────────────────────────┘
                    ⬇ Load (~45 sec)
   ┌─────────────────────────────────────────────┐
   │ [22GB] DeepSeek-Coder-33B (coding)          │ ← Now active (LARGER, more capable!)
   └─────────────────────────────────────────────┘

6. Discord Bot sends: "✅ Coding mode ready. Let's build something."

7. Context Builder generates coding context:
   - System prompt: "Expert Python/TypeScript architect"
   - Documentation: CAMERA_MANAGER.md (full text)
   - Existing code: ring buffer related files
   - Recent commits: Last 5 changes
   - User request: "implement ring buffer class"

8. DeepSeek-Coder-33B generates response (higher quality than smaller model)

9. Response sent to Discord

10. Model remains loaded for follow-up coding questions

11. After 15 minutes of inactivity → Auto-duck back to Mistral
    Discord: "💭 Switching back to conversation mode..."
```

**User Experience**:
- Clear communication about what's happening
- Anticipation builds during switch (like waiting for a tool)
- Awareness of mode shift (you're now talking to "coding brain")
- Higher quality responses justify the wait

---

## Universal Context Window Format

### Structure (Model-Agnostic)

```markdown
# SYSTEM IDENTITY
You are BioMimeticAI, a consciousness-mimetic system running on Ubuntu.

Core principles:
- Reason from first principles (Axiomatic Modeling Architecture)
- Distributed cortex design (isolated services)
- No backwards compatibility (you are the only user)

# CURRENT MODE
{conversation | coding | vision_analysis | reasoning}

# USER CONTEXT
Contact: toastee0
Trust Level: 0.6
Relationship: technical_collaborator
Communication Style: Direct, technical, minimal pleasantries
Preferred Topics: AI architecture, system design, Python/TypeScript

# MEMORY CONTEXT
Recent interactions (last 3 conversations):
1. [2025-12-19 14:30] Discussed Camera Manager architecture
2. [2025-12-19 13:15] Reviewed salience mechanism design
3. [2025-12-18 22:00] Debugged episodic memory consolidation

# TASK-SPECIFIC CONTEXT

## [If coding mode]
Current Project: Camera Manager (ai_led_mapper extension)
Documentation: docs/CAMERA_MANAGER.md
Relevant Code:
  - /home/toastee/ai_led_mapper/backend/app/core/camera.py
  - /home/toastee/ai_led_mapper/frontend/src/components/CameraTile.tsx

## [If conversation mode]
Active Axioms: M1 (Patience/Kindness), F1 (Architecture), AMA1 (Meta-reasoning)
Micro-tools available: technical_level_assessment, topic_preferences

## [If vision mode]
Recent Events:
  - 14:32 - Person entered workshop (reCamera)
  - 14:30 - Scene: Person at desk with laptop
Camera Context: workshop_recamera (last 3 seconds buffered)

# USER REQUEST
{actual user message}
```

### Context Builder Implementation

```python
class UniversalContextBuilder:
    """Builds LLM-agnostic context windows"""

    def __init__(self):
        self.episodic_memory = EpisodicMemory()
        self.contact_memory = ContactMemory()
        self.axiom_library = AxiomLibrary()
        self.doc_loader = DocumentationLoader()
        self.code_search = CodebaseSearch()

    def build_context(
        self,
        user_message: str,
        contact_id: str,
        mode: str  # 'conversation' | 'coding' | 'vision'
    ) -> str:
        """
        Generate universal context window
        Works with any LLM backend
        """
        context = []

        # System identity (always included)
        context.append(self._system_identity())

        # Current mode
        context.append(f"# CURRENT MODE\n{mode}\n")

        # User context (contact memory)
        user_profile = self.contact_memory.get_contact(contact_id)
        context.append(self._format_user_context(user_profile))

        # Memory context (episodic memory)
        recent_episodes = self.episodic_memory.get_recent(
            contact_id=contact_id,
            limit=3
        )
        context.append(self._format_memory_context(recent_episodes))

        # Task-specific context
        if mode == "coding":
            context.append(self._coding_context(user_message))
        elif mode == "conversation":
            context.append(self._conversation_context(user_message))
        elif mode == "vision":
            context.append(self._vision_context(user_message))

        # User request
        context.append(f"# USER REQUEST\n{user_message}\n")

        return "\n".join(context)

    def _coding_context(self, user_message: str) -> str:
        """Gather coding-specific context"""
        # Extract what they're asking about
        keywords = self._extract_keywords(user_message)

        # Load relevant documentation
        docs = self.doc_loader.load_relevant(keywords)

        # Search codebase
        related_files = self.code_search.find_related(keywords, limit=5)

        context = "# TASK-SPECIFIC CONTEXT (Coding)\n\n"

        if docs:
            context += f"## Documentation\n{docs}\n\n"

        if related_files:
            context += "## Relevant Code\n"
            for file_path, snippet in related_files:
                context += f"### {file_path}\n```python\n{snippet}\n```\n\n"

        return context
```

---

## Model Manager Implementation

### Core Model Manager

```python
class ModelManager:
    """Manages model loading/unloading (ducking)"""

    def __init__(self, llama_server_url: str = "http://localhost:53307"):
        self.llama_server_url = llama_server_url
        self.current_model = None
        self.model_configs = {
            "conversation": {
                "name": "mistral-small-22b",
                "path": "/models/mistral-small-22b-instruct-2409-q4_k_m.gguf",
                "vram_gb": 14,
                "context_size": 32768,
                "capabilities": ["chat", "reasoning", "general"]
            },
            "coding": {
                "name": "deepseek-coder-6.7b",
                "path": "/models/deepseek-coder-6.7b-instruct-q5_k_m.gguf",
                "vram_gb": 6,
                "context_size": 16384,
                "capabilities": ["code_generation", "code_review", "debugging"]
            },
            "vision": {
                "name": "llava-1.6",
                "path": "/models/llava-v1.6-vicuna-7b-q4_k_m.gguf",
                "vram_gb": 8,
                "context_size": 4096,
                "capabilities": ["image_understanding", "scene_description"]
            }
        }
        self.last_activity = {}  # Track when each mode was last used
        self.auto_duck_timeout = 600  # Switch back after 10 min inactivity

    async def ensure_model_loaded(self, mode: str, notify_callback=None):
        """
        Ensure correct model is loaded for mode
        Handles ducking (unload/load) if needed

        notify_callback: Optional function to send status updates to user
        """
        target_config = self.model_configs[mode]

        # Already loaded?
        if self.current_model == mode:
            logger.info(f"Model {target_config['name']} already loaded")
            self.last_activity[mode] = time.time()
            return

        # Need to switch - notify user
        if notify_callback:
            await notify_callback(
                f"🔧 Switching to {mode} mode, loading {target_config['name']}...\n"
                f"This will take about 45 seconds. Time to grab coffee ☕"
            )

        # Unload current model
        if self.current_model:
            await self._unload_model()

        # Load target model
        await self._load_model(target_config)

        self.current_model = mode
        self.last_activity[mode] = time.time()

        # Notify ready
        if notify_callback:
            await notify_callback(f"✅ {mode.title()} mode ready. Let's build something.")

    async def _unload_model(self):
        """Gracefully unload current model from llama-server"""
        logger.info("Unloading current model...")

        # Send unload request to llama-server
        # (llama.cpp doesn't have explicit unload, so we restart the service)
        await self._restart_llama_server()

        self.current_model = None
        logger.info("Model unloaded, VRAM freed")

    async def _load_model(self, config: dict):
        """Load model into llama-server"""
        logger.info(f"Loading model: {config['name']}")

        # Restart llama-server with new model
        cmd = [
            "llama-server",
            "-m", config['path'],
            "-c", str(config['context_size']),
            "--port", "53307",
            "--host", "0.0.0.0",
            "-ngl", "99",  # GPU layers (all)
        ]

        # Start process (systemd service would be better)
        process = await asyncio.create_subprocess_exec(*cmd)

        # Wait for server to be ready
        await self._wait_for_server_ready()

        logger.info(f"Model {config['name']} loaded successfully")

    async def _wait_for_server_ready(self, timeout: int = 60):
        """Wait for llama-server to accept requests"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.llama_server_url}/health") as resp:
                        if resp.status == 200:
                            return
            except:
                pass
            await asyncio.sleep(1)
        raise TimeoutError("llama-server failed to start")

    async def check_auto_duck(self):
        """
        Check if we should auto-switch back to conversation mode
        Called periodically by background task
        """
        if self.current_model == "conversation":
            return  # Already on default model

        # Check last activity
        last_used = self.last_activity.get(self.current_model, 0)
        idle_time = time.time() - last_used

        if idle_time > self.auto_duck_timeout:
            logger.info(f"Auto-ducking: {self.current_model} idle for {idle_time}s")
            await self.ensure_model_loaded("conversation")
```

---

## Task Classifier

```python
class TaskClassifier:
    """Determines which model to use based on user request"""

    CODING_KEYWORDS = [
        "implement", "code", "function", "class", "debug",
        "error", "bug", "refactor", "optimize", "review",
        "write", "create", "build", "design", "architecture"
    ]

    VISION_KEYWORDS = [
        "what do you see", "describe the scene", "who is in",
        "analyze the image", "what's happening", "camera shows"
    ]

    def classify(self, message: str, context: dict = None) -> str:
        """
        Classify task type based on message content
        Returns: 'conversation' | 'coding' | 'vision' | 'reasoning'
        """
        lower = message.lower()

        # Check for coding
        if any(keyword in lower for keyword in self.CODING_KEYWORDS):
            return "coding"

        # Check for vision
        if any(keyword in lower for keyword in self.VISION_KEYWORDS):
            return "vision"

        # Check for deep reasoning (long, complex questions)
        if len(message.split()) > 100:  # Long message
            return "reasoning"

        # Default: conversation
        return "conversation"
```

---

## Integration with Discord Bot

```python
# src/discord/bot_axiom_review.py

class BioMimeticBot:
    def __init__(self):
        self.model_manager = ModelManager()
        self.task_classifier = TaskClassifier()
        self.context_builder = UniversalContextBuilder()
        self.llm_client = TextGenClient()  # Works with any backend

        # Start auto-duck background task
        asyncio.create_task(self._auto_duck_loop())

    async def on_message(self, message):
        """Handle incoming Discord message"""

        # Classify task
        mode = self.task_classifier.classify(message.content)

        # Ensure correct model loaded (may trigger duck)
        await self.model_manager.ensure_model_loaded(mode)

        # Build universal context
        context = self.context_builder.build_context(
            user_message=message.content,
            contact_id=str(message.author.id),
            mode=mode
        )

        # Generate response (model-agnostic)
        response = self.llm_client.generate(context, max_tokens=1000)

        # Send to Discord
        await message.channel.send(response)

    async def _auto_duck_loop(self):
        """Background task to auto-switch models"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self.model_manager.check_auto_duck()
```

---

## Model Configurations

### Philosophy: Capability Over Speed

**VRAM Budget**: 24GB (use it all!)

**Design Principle**: The 30-60 second model switch is a *feature*, not a bug. It's like picking up different tools or grabbing your laptop - human-like pacing that adds character to the interaction. We optimize for **interestingness and capability**, not latency.

### Available Models (Use Full VRAM)

| Model | VRAM | Context | Use Case | Priority |
|-------|------|---------|----------|----------|
| **Mistral-Small-22B** (Q4) | 14GB | 32K | Conversation, general reasoning | Default |
| **DeepSeek-Coder-33B** (Q4/Q5) | 20-22GB | 16K | Advanced code generation | Coding |
| **Qwen2.5-Coder-32B** (Q4) | 20GB | 32K | Alternative coding (longer context) | Coding |
| **Mistral-Large** (Q4) | 22GB | 128K | Deep reasoning, complex analysis | Reasoning |
| **LLaVA-1.6-34B** (Q4) | 22GB | 4K | Advanced vision understanding | Vision (future) |

**Strategy**:
- Load LARGEST model that fits task requirements
- Duck between them freely (30-60 sec is fine)
- User gets "Switching to coding mode, loading DeepSeek-Coder-33B..." message
- Creates anticipation and awareness of mode shift

---

## File Locations

**New Files**:
- `src/core/model_manager.py` - Model ducking implementation
- `src/core/task_classifier.py` - Classify requests → model selection
- `src/core/universal_context.py` - Context builder (model-agnostic)
- `src/core/llm_backend.py` - Abstraction layer for multiple backends

**Modified Files**:
- `src/discord/bot_axiom_review.py` - Use model manager + context builder
- `src/daemon/textgen_client.py` - Support multiple backends

**Config**:
- `config/models.json` - Model paths, VRAM, capabilities
- `config/.env` - Add `MODEL_DUCKING_ENABLED=true`

---

## Benefits of This Architecture

1. **Maximum Capability**: Use full 24GB for best models available
2. **Human-Like Pacing**: Switching delay feels natural (like picking up tools)
3. **Model Agnostic**: Context works with any LLM (local or API)
4. **Extensible**: Easy to add new models or backends
5. **Intelligent**: Auto-switches based on task type
6. **Transparent**: User knows which "brain" they're talking to
7. **Interesting**: Mode shifts create distinct interaction styles
8. **Cost Effective**: Can mix local + API models (e.g., Claude API for vision)

**Philosophy**: We're not building a chatbot, we're building a *system* with specialized capabilities. The switching time reinforces that you're accessing different parts of a distributed intelligence.

---

## Next Steps

1. Implement `ModelManager` with llama-server control
2. Implement `UniversalContextBuilder`
3. Integrate with Discord bot
4. Test model switching latency (30-60 sec acceptable?)
5. Add systemd service for llama-server management
6. Consider using vllm for faster model switching

**Estimated Time**: 12-15 hours for full implementation
