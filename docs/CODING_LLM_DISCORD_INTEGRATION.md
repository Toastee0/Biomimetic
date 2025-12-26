# Coding LLM Discord Integration - Design Collaboration Interface

**Purpose**: Enable the BioMimeticAI system to collaborate on its own design and implementation via Discord DM, using a coding-specialized LLM.

**Goal**: "Hey BioMimeticAI, let's discuss how to implement the Camera Manager salience engine" → AI responds with code suggestions, architectural considerations, and asks clarifying questions.

---

## Architecture Overview

```
Discord DM (toastee0)
       │
       │ Message: "Let's work on the ring buffer implementation"
       ▼
┌────────────────────────────────────────────────────────┐
│  Discord Bot (src/discord/bot_axiom_review.py)        │
│                                                        │
│  Detects: Code/design keywords ("implement", "design",│
│           "architecture", "code review")              │
│                                                        │
│  Routes to → CODING MODE                              │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  Coding LLM Router (NEW: src/core/coding_llm.py)      │
│                                                        │
│  1. Gathers context:                                  │
│     - Current task (from conversation)                │
│     - Relevant docs (CAMERA_MANAGER.md, etc.)        │
│     - Existing code (ai_led_mapper, vision_api)      │
│     - Recent commits/changes                          │
│                                                        │
│  2. Builds coding prompt:                             │
│     - System role: "Expert Python/TypeScript dev"     │
│     - Context: Documentation + code snippets          │
│     - User request: Original Discord message          │
│                                                        │
│  3. Sends to LLM backend                              │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  LLM Backend Options                                   │
│                                                        │
│  Option A: Same llama-server (port 53307)            │
│    - Already running Mistral-Small-22B                │
│    - Good at coding, might be slightly verbose        │
│                                                        │
│  Option B: Dedicated coding LLM (new port 53308)     │
│    - DeepSeek-Coder, CodeLlama, or Qwen-Coder        │
│    - Specialized for code generation                  │
│    - Requires ~14GB VRAM (conflicts with Mistral?)   │
│                                                        │
│  Option C: Claude API (external)                      │
│    - Use anthropic API for coding tasks               │
│    - Higher quality, costs money                      │
│    - No local VRAM constraint                         │
└──────────────────┬─────────────────────────────────────┘
                   │
                   │ Response: Code + explanation
                   ▼
┌────────────────────────────────────────────────────────┐
│  Response Formatter                                    │
│                                                        │
│  - Syntax highlighting (Discord markdown)             │
│  - Split long code into multiple messages             │
│  - Add file path headers                              │
│  - Include "next steps" suggestions                   │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
      Discord DM (response to toastee0)
```

---

## Implementation Plan

### Phase 1: Detection & Routing (1-2 hours)

**Modify**: `src/discord/bot_axiom_review.py`

Add trigger keywords for coding mode:
```python
CODING_KEYWORDS = [
    "implement", "code", "design", "architecture",
    "how should we", "let's work on", "help me build",
    "review this", "what's wrong with", "optimize"
]

async def detect_coding_request(message: str) -> bool:
    """Check if message is requesting coding assistance"""
    lower = message.lower()
    return any(keyword in lower for keyword in CODING_KEYWORDS)
```

Route to coding handler:
```python
@bot.event
async def on_message(message):
    # ... existing checks ...

    if await detect_coding_request(message.content):
        await handle_coding_request(message)
        return

    # ... normal conversation handling ...
```

---

### Phase 2: Context Gathering (2-3 hours)

**New file**: `src/core/coding_llm.py`

```python
class CodingLLM:
    def __init__(self):
        self.llm_client = TextGenClient()  # Reuse existing client
        self.context_builder = CodingContextBuilder()

    async def handle_request(self, user_message: str, contact_id: str):
        """Main entry point for coding requests"""

        # 1. Extract intent (what are they asking about?)
        intent = self.extract_intent(user_message)

        # 2. Gather relevant context
        context = await self.context_builder.gather(intent)

        # 3. Build prompt
        prompt = self.build_coding_prompt(user_message, context)

        # 4. Get LLM response
        response = self.llm_client.generate(prompt, max_tokens=2000)

        # 5. Format for Discord
        return self.format_response(response)
```

**Context Sources**:
```python
class CodingContextBuilder:
    def gather(self, intent: dict) -> dict:
        """Gather relevant context based on intent"""
        context = {}

        # Documentation
        if intent['topic'] == 'camera_manager':
            context['docs'] = self.read_doc('docs/CAMERA_MANAGER.md')

        # Existing code
        if intent['file_path']:
            context['current_code'] = self.read_file(intent['file_path'])

        # Recent changes
        context['recent_commits'] = self.get_recent_commits(n=5)

        # Related files (search based on keywords)
        context['related_files'] = self.search_codebase(intent['keywords'])

        return context
```

---

### Phase 3: Prompt Engineering (1 hour)

**System Prompt** for coding mode:
```python
CODING_SYSTEM_PROMPT = """You are an expert software architect and Python/TypeScript developer working on the BioMimeticAI system.

CONTEXT:
- This is a consciousness-mimetic AI system running on Ubuntu
- Multiple services gather sensory data independently
- No backwards compatibility needed (you are the only user)
- Current focus: Building Camera Manager (multi-modal perception)

YOUR ROLE:
- Suggest implementations based on existing patterns
- Point out potential issues or improvements
- Ask clarifying questions when requirements are unclear
- Provide concrete code examples
- Consider integration with existing systems

STYLE:
- Be concise but complete
- Use existing code patterns from the codebase
- Highlight trade-offs when multiple approaches exist
- Format code with proper syntax and file paths
"""
```

**Prompt Template**:
```python
def build_coding_prompt(user_message: str, context: dict) -> str:
    return f"""
{CODING_SYSTEM_PROMPT}

CURRENT TASK:
{user_message}

RELEVANT DOCUMENTATION:
{context.get('docs', 'None')}

EXISTING CODE:
{context.get('current_code', 'None')}

RELATED FILES:
{context.get('related_files', 'None')}

Please provide:
1. Your understanding of the request
2. Suggested implementation approach
3. Code examples (if applicable)
4. Questions or considerations
"""
```

---

### Phase 4: Response Formatting (1 hour)

```python
def format_response(response: str) -> list[str]:
    """Format LLM response for Discord (handle code blocks, length limits)"""
    messages = []

    # Discord message limit: 2000 chars
    # Split on code blocks first to avoid breaking syntax
    blocks = split_code_blocks(response)

    current_message = ""
    for block in blocks:
        if len(current_message) + len(block) > 1900:
            messages.append(current_message)
            current_message = block
        else:
            current_message += block

    if current_message:
        messages.append(current_message)

    return messages

def split_code_blocks(text: str) -> list[str]:
    """Split while preserving code blocks"""
    # Use regex to identify ```language blocks
    # Keep each code block intact
    # Return list of text chunks
    ...
```

**Discord Formatting**:
```python
# Python code block
```python
# /home/toastee/ai_led_mapper/backend/app/core/ring_buffer.py

class CameraRingBuffer:
    def __init__(self, max_seconds: int = 5, fps: int = 5):
        self.max_frames = max_seconds * fps
        ...
```
# TypeScript code block
```typescript
// /home/toastee/ai_led_mapper/frontend/src/components/CameraTile.tsx

interface CameraTileProps {
  camera: Camera;
  ...
}
```
```

---

### Phase 5: Conversation Threading (2 hours)

**Track coding conversations**:
```python
# Store in-memory or SQLite
coding_sessions = {
    "contact_id": {
        "session_id": "uuid",
        "topic": "ring_buffer_implementation",
        "messages": [
            {"role": "user", "content": "Let's implement ring buffers"},
            {"role": "assistant", "content": "Here's my suggestion..."},
        ],
        "context_files": ["docs/CAMERA_MANAGER.md"],
        "started_at": timestamp
    }
}
```

**Continue conversations**:
```python
@bot.command()
async def continue_coding(ctx):
    """Continue previous coding discussion"""
    session = get_active_session(ctx.author.id)
    if session:
        await ctx.send(f"Continuing: {session['topic']}")
        # Load previous context and continue
    else:
        await ctx.send("No active coding session. Start with: !code <topic>")
```

---

## LLM Backend Decision

### Model Ducking Architecture (Recommended)

**See**: `docs/MODEL_DUCKING_ARCHITECTURE.md` for complete specification

**Approach**: Dynamically switch between models based on task type:
- **Conversation**: Mistral-Small-22B (14GB VRAM)
- **Coding**: DeepSeek-Coder-6.7B (6GB VRAM)
- **Vision**: LLaVA-1.6 (8GB VRAM, future)

**Key Concept**: Only one model loaded at a time. Unload current, load needed model when task type changes. Auto-duck back to conversation mode after 10 minutes of inactivity.

**Context Builder**: Generates universal (model-agnostic) context window that works with any LLM backend.

---

## Usage Examples

### Example 1: Implementation Request
```
User: "Let's implement the ring buffer system for the Camera Manager"

Bot: "I'll help implement the ring buffer. Based on CAMERA_MANAGER.md specs:

**Suggested implementation:**

```python
# /home/toastee/ai_led_mapper/backend/app/core/ring_buffer.py

from collections import deque
import time
from typing import Optional

class CameraRingBuffer:
    def __init__(self, max_seconds: int = 5, fps: int = 5):
        self.max_frames = max_seconds * fps
        self.buffer = deque(maxlen=self.max_frames)
        self.fps = fps

    def add_frame(self, frame: bytes, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        self.buffer.append({'data': frame, 'timestamp': timestamp})

    def get_last_n_seconds(self, seconds: int) -> list:
        target_frames = min(seconds * self.fps, len(self.buffer))
        return list(self.buffer)[-target_frames:]
```

**Key decisions:**
- Using `deque(maxlen=...)` for automatic overflow handling
- Storing frames as dicts with timestamp for context
- `get_last_n_seconds()` clamps to available frames

**Questions:**
1. Should we compress frames before storing (save memory)?
2. Thread-safety needed (multiple cameras writing)?
3. Persistence to disk or memory-only?

Shall we proceed with this approach?"
```

### Example 2: Code Review
```
User: "Review this salience calculation code: [paste code]"

Bot: "**Code Review:**

**Issues Found:**
1. Missing normalization - priorities can exceed 1.0
2. No decay for stale events (events from hours ago still count)
3. schedule_score not handling timezone edge cases

**Suggested fixes:**
[code with improvements]

**Performance note:**
Current O(n) per camera - fine for <10 cameras, might need optimization for 50+

Want me to show the improved version?"
```

---

## File Locations

**New Files**:
- `src/core/coding_llm.py` - Main coding LLM handler
- `src/core/coding_context.py` - Context gathering
- `src/core/code_search.py` - Codebase search utilities

**Modified Files**:
- `src/discord/bot_axiom_review.py` - Add coding mode detection/routing

**Config**:
- `config/.env` - Add `CODING_MODE_ENABLED=true`

---

## Next Steps

1. **Implement Phase 1** (detection & routing) - Test with simple echo responses
2. **Implement Phase 2** (context gathering) - Verify correct docs/code are loaded
3. **Implement Phase 3** (prompt engineering) - Test with actual LLM
4. **Implement Phase 4** (formatting) - Ensure code blocks render properly in Discord
5. **Implement Phase 5** (threading) - Test multi-turn conversations

**Estimated Total Time**: 8-10 hours for full implementation

**Testing**: Can start with Phase 1-3 for basic "ask → respond" flow, then iterate.
