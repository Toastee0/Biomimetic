# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioMimeticAI v2.0 is a consciousness-mimetic AI system built on Axiomatic Modeling Architecture (AMA). It combines neural pattern-matching with explicit symbolic axioms to enable explainable, principled reasoning grounded in ethical and architectural first principles.

**Core Philosophy**: Reasoning from first principles rather than statistical pattern matching alone.

## Quick Start Commands

### Running the System

```bash
# Activate Python virtual environment (required for all commands)
source venv/bin/activate

# Start Discord bot (main conversation interface)
python src/discord/bot_axiom_review.py
# Or use: bash scripts/start_axiom_review_bot.sh

# Run self-training loop (continuous axiom validation)
python src/tensor_axiom/self_training_loop.py --iterations 0 --interval 300
```

### Testing & Validation

```bash
# Test all axioms against their scenarios
python src/tensor_axiom/self_training_loop.py --test-only

# Test specific axiom
python src/tensor_axiom/self_training_loop.py --axiom-id M1_kindness_over_correctness

# Run GPU tensor axiom tests
python src/tensor_axiom/test_gpu.py
```

### Database & Memory Management

```bash
# Initialize database (creates tables if missing)
python -c "from src.memory.episodic import EpisodicMemory; EpisodicMemory()"
python -c "from src.memory.contact_memory import ContactMemory; ContactMemory()"

# SQLite database location: data/biomim.db
sqlite3 data/biomim.db "SELECT COUNT(*) FROM episodes;"
sqlite3 data/biomim.db "SELECT contact_id, name, trust_level FROM contacts;"
```

### Cron Cortex Management

```bash
# Install all cortex cron jobs
bash scripts/install_cron_jobs.sh

# View installed cron jobs
crontab -l

# Monitor all cortexes in real-time
bash scripts/monitor_cortexes.sh

# Check cortex status (last run, errors)
bash scripts/cortex_status.sh

# Remove all cron jobs
bash scripts/remove_cron_jobs.sh
```

### Viewing Logs

```bash
# Real-time log monitoring
tail -f logs/axiom_evaluation.log
tail -f logs/episodic_consolidation.log
tail -f logs/contact_learning.log
tail -f logs/cron.log

# View all recent cortex activity
tail -n 100 logs/*.log
```

## System Architecture

### Distributed Cortex Design

The system mimics biological brain architecture with independent "cortexes" (brain regions) that run autonomously. Each cortex is isolated - failures don't cascade. All cortexes follow RTOS-style timeouts with hard limits.

**Real-Time Processes** (always running):
- Discord Bot: `src/discord/bot_axiom_review.py` - Main conversation interface
- LLM Server: `llama-server` on port 53307 (external dependency)

**High-Frequency Cron Jobs** (every 10-30 minutes):
- Episodic Consolidation: Calculate salience scores, mark consolidated episodes
- Axiom Spot Check: Quick validation of 3-5 random axioms
- Contact Learning: Extract patterns from conversations, create micro-tools

**Medium-Frequency Cron Jobs** (every 4-6 hours):
- Full Axiom Evaluation: Test all axioms, queue problems for review
- Memory Consolidation: Transfer episodic → semantic memory

**Low-Frequency Cron Jobs** (daily/weekly):
- Deep Learning Cycle: Full validation, health check
- Long-Term Consolidation: Archive, backup, optimize

See `docs/CORTEX_SCHEDULE.md` for complete schedules and timeouts.

### Memory System Architecture

**Episodic Memory** (`src/memory/episodic.py`):
- Stores individual conversations with context
- Salience scoring based on trust level, emotional content, axiom relevance
- Biomimetic spreading activation for contextual retrieval
- Consolidation marking for semantic transfer
- Database table: `episodes` in `data/biomim.db`

**Contact Memory** (`src/memory/contact_memory.py`):
- Structured profiles for each individual
- Trust levels (0.0-1.0, starts at 0.5)
- Communication style, preferred topics, personality traits
- Relationship type tracking
- Database table: `contacts` in `data/biomim.db`

**Semantic Memory** (`src/memory/semantic.py`):
- Generalized knowledge extracted from episodic memories
- Concept nodes with relationship edges
- Pattern abstraction from repeated experiences

**Micro-Tools System** (`data/micro_tools/`):
- Philosophy: "Expensive inference → Create cheap lookup tools"
- First time: Use LLM to understand pattern (slow)
- Creates JSON micro-tool: Fast cached lookup (fast)
- Example: User topic preferences, technical level assessments
- Linux philosophy: Many small, specialized tools

### Axiom Graph Architecture

The axiom system is a hierarchical reasoning engine with 26 axioms organized in priority layers:

**Meta-Axioms (Priority 1.0)** - Override everything:
- M1: Patience + Kindness > Being Right
- M2: Love as Joint Utility Optimization
- M3: Kindness < Cruelty (cost analysis)
- M4: Life Preservation > All Else

**Foundational Axioms (Priority 0.8)**:
- F1-F8: Architectural principles (distributed failure isolation, variable attention, etc.)
- AMA1-AMA6: Meta-reasoning principles (axiom discovery, graph structure, hybrid architecture)

**Derived Axioms (Priority 0.6)**:
- D1-D5: Composed reasoning (threat assessment, attention allocation, tool offloading)

**Domain Axioms (Priority 0.4-0.88)**:
- R1: Relationship axioms (sensory accommodation)
- E1-E2: Embodiment axioms (rover speed, speaker limits)

**Edge Relationships**:
- `implies`: A logically entails B
- `requires`: A needs B to function
- `contradicts`: A and B are incompatible
- `composes`: A and B combine to form pattern
- `specializes`: A is specific case of B
- `overrides`: A takes precedence over B

**Storage**: `data/axioms/base_axioms.json` (48KB, includes test scenarios for each axiom)

See `AXIOM_ARCHITECTURE_README.md` for comprehensive axiom documentation.

### Dynamic Prompt System

**File**: `src/core/dynamic_prompts.py`

Builds "living" system prompts that reflect the AI's current state:
- Base identity/values from `config/system_prompt.txt`
- Current memory state (episode count, contact count)
- User-specific context (name, trust, interaction history, communication style)
- Available micro-tools learned about this user
- Recent conversation history

**Result**: The AI knows what it knows. It says "I remember our last conversation about X" instead of "I don't have memory."

## Key Implementation Details

### Database Concurrency (IMPORTANT)

**Issue**: SQLite database (`data/biomim.db`) accessed concurrently by Discord bot + multiple cron cortexes.

**Current State**: Database locking can occur under heavy load (documented in recent runs).

**Required Pattern** when modifying database code:

```python
import sqlite3
import time

def get_connection():
    """Get database connection with WAL mode and timeout"""
    conn = sqlite3.connect('data/biomim.db', timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
    conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
    return conn

# Use retry logic for write operations
def retry_operation(func, max_attempts=3, backoff=1.5):
    """Retry database operations with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(backoff ** attempt)
```

Always use WAL mode and proper timeout handling in any new database access code.

### LLM Inference Client

**File**: `src/daemon/textgen_client.py`

All LLM inference goes through the `TextGenClient` class:
- Connects to `llama-server` on port 53307 (local inference)
- Streaming and non-streaming modes
- Request/response logging
- Retry logic on connection failures

**Usage**:
```python
from src.daemon.textgen_client import TextGenClient

client = TextGenClient()
response = client.generate(
    prompt="Your prompt here",
    max_tokens=500,
    temperature=0.7
)
```

### Cortex State Management

Each cortex writes status to `data/cortex_state/{cortex_name}.json`:

```json
{
  "last_run": 1733702400,
  "status": "success|error|timeout",
  "duration_seconds": 45.2,
  "items_processed": 127,
  "errors": [],
  "next_scheduled": 1733706000
}
```

The Discord bot reads these for the `!health` command (not yet implemented).

### RTOS Timeout Pattern

All cron cortexes follow this pattern:

```python
import signal
import sys

TIMEOUT_SECONDS = 60

def timeout_handler(signum, frame):
    print(f"[TIMEOUT] Process exceeded {TIMEOUT_SECONDS}s limit")
    sys.exit(124)  # Timeout exit code

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TIMEOUT_SECONDS)

try:
    # Do work here
    pass
finally:
    signal.alarm(0)  # Cancel alarm
```

Exit codes: `0` = success, `1` = error, `124` = timeout

## Discord Bot Commands

When working on bot functionality, these are the current commands:

- `!review` - Show axiom review queue
- `!inspect [number]` - Inspect specific axiom
- `!approve` - Approve current axiom
- `!reject [reason]` - Reject with reason
- `!suggest` - Get LLM improvement suggestions
- `!retest` - Re-run tests on axiom
- `!stats` - Show system statistics
- `!profile` - View your contact profile
- `!note [text]` - Add note to your profile
- `!contacts` - List all contacts

Natural language testing: Type scenarios to test axioms (e.g., "What if someone says the volume is too loud?")

Conversational AI mode: Bot acts as normal assistant when not reviewing axioms.

## GPU Acceleration Architecture (Future)

**Status**: Designed but not fully implemented

**Design Document**: `TENSOR_AXIOM_ARCHITECTURE.md` (785 lines)

**Architecture**:
- 448D axiom embeddings (semantic 256 + logic 192 + dependency 64)
- Graph Neural Network for axiom chain construction
- Differentiable execution for end-to-end learning
- Hybrid routing: Transformer (fast path) vs. Axiom (novel/risky situations)
- Expected speedup: 10-50x for complex reasoning

**Implementation Files**:
- `src/tensor_axiom/axiom_embeddings.py` - Partial implementation
- `src/tensor_axiom/axiom_attention.py` - Attention mechanism
- `src/tensor_axiom/hybrid_model.py` - Routing logic
- `src/tensor_axiom/axiom_executor.py` - Chain execution

## Directory Structure

```
BioMimeticAi/
├── config/                  # Configuration files
│   ├── .env                 # Environment variables (DISCORD_TOKEN, etc.)
│   └── system_prompt.txt    # Base system prompt
├── data/                    # Persistent data
│   ├── axioms/
│   │   ├── base_axioms.json      # 26 axioms with test scenarios
│   │   └── review_queue.json     # Axioms needing review
│   ├── biomim.db            # SQLite database (episodes, contacts)
│   ├── cortex_state/        # Cortex status JSON files
│   └── micro_tools/         # Cached user patterns (JSON)
├── docs/                    # Documentation
│   ├── CORTEX_SCHEDULE.md   # Cortex schedules and timeouts
│   └── CORTEX_INTEGRATION.md # System integration guide
├── logs/                    # Runtime logs
│   ├── axiom_evaluation.log
│   ├── episodic_consolidation.log
│   ├── contact_learning.log
│   └── cron.log
├── scripts/                 # Utility scripts
│   ├── cron/                # Cortex cron jobs
│   │   ├── axiom_spot_check.py
│   │   ├── contact_learning.py
│   │   ├── episodic_consolidation.py
│   │   └── full_axiom_evaluation.py
│   ├── install_cron_jobs.sh
│   ├── monitor_cortexes.sh
│   └── cortex_status.sh
├── src/
│   ├── core/                # Core functionality
│   │   ├── dynamic_prompts.py   # Context-aware prompt builder
│   │   └── prompts.py           # Static prompts
│   ├── daemon/              # Background services
│   │   └── textgen_client.py    # LLM inference client
│   ├── discord/             # Discord bot
│   │   └── bot_axiom_review.py  # Main bot (1,069 lines)
│   ├── memory/              # Memory systems
│   │   ├── episodic.py          # Episodic memory
│   │   ├── contact_memory.py    # Contact profiles
│   │   ├── semantic.py          # Semantic memory
│   │   └── identity.py          # AI identity/values
│   └── tensor_axiom/        # Axiom reasoning engine
│       ├── self_training_loop.py  # Self-training (598 lines)
│       ├── axiom_library.py       # Axiom management
│       ├── review_queue.py        # Review queue management
│       ├── axiom_embeddings.py    # GPU embeddings (partial)
│       └── hybrid_model.py        # Hybrid transformer-axiom (partial)
└── venv/                    # Python virtual environment
```

## Development Patterns

### Adding a New Cortex

1. Create script in `scripts/cron/new_cortex.py`
2. Follow RTOS timeout pattern (see above)
3. Write state to `data/cortex_state/new_cortex.json`
4. Log to `logs/new_cortex.log` with timestamps
5. Add to `scripts/install_cron_jobs.sh`
6. Document in `docs/CORTEX_SCHEDULE.md`

### Adding a New Axiom

1. Edit `data/axioms/base_axioms.json`
2. Add axiom with: id, name, priority, formula, description, test_scenarios
3. Add edges to existing axioms (implies, requires, etc.)
4. Run: `python src/tensor_axiom/self_training_loop.py --test-only`
5. If success < 70%, axiom goes to review queue automatically

### Adding Discord Bot Commands

1. Edit `src/discord/bot_axiom_review.py`
2. Add command using `@bot.command()` decorator
3. Update help text in bot initialization
4. Commands can use: episodic memory, contact memory, axiom library
5. Always update contact profile after interactions

### Modifying Memory Systems

1. Memory classes must handle concurrent access (see Database Concurrency)
2. Use WAL mode and retry logic for SQLite operations
3. Update database schema carefully (no migrations system yet)
4. Add indexes for frequently queried columns
5. Test with multiple concurrent processes

## Environment Variables

Located in `config/.env`:

```bash
DISCORD_TOKEN=your_discord_bot_token_here
LLM_SERVER_URL=http://localhost:53307  # llama-server
DATABASE_PATH=data/biomim.db
LOG_LEVEL=INFO
```

## Python Dependencies

The system uses a virtual environment (`venv/`) with:
- `discord.py` - Discord bot framework
- `python-dotenv` - Environment variable management
- `requests` - HTTP client for LLM server
- `sqlite3` - Database (built-in)
- `torch` - PyTorch for GPU acceleration (partial)
- `torch_geometric` - Graph neural networks (partial)

No `requirements.txt` exists yet - dependencies installed manually.

## Known Issues & Gotchas

1. **Database Locking**: Under heavy load, concurrent cortexes can experience SQLite locking. Always use WAL mode and retry logic.

2. **LLM Server Dependency**: System requires `llama-server` running on port 53307. Bot will fail if server is down. No graceful degradation yet.

3. **Cron Job Isolation**: Cron jobs must be idempotent - if a job runs twice simultaneously (shouldn't happen but could), it shouldn't corrupt state.

4. **No Migration System**: Database schema changes require manual ALTER TABLE commands. Be careful with schema modifications.

5. **Log Rotation**: Logs grow unbounded. No logrotate configuration yet. Manual cleanup required.

6. **GPU Code Incomplete**: `src/tensor_axiom/` has GPU acceleration design but partial implementation. CPU inference is production path.

7. **No Tests**: No automated test suite beyond axiom self-tests. Integration tests needed.

## Testing Philosophy

The system uses **LLM-evaluated scenario testing** instead of traditional unit tests:

- Each axiom includes test scenarios in JSON
- Self-training loop runs scenarios through LLM
- LLM evaluates if axiom was correctly applied
- Axioms with < 70% success or < 0.6 confidence are flagged
- Human review via Discord bot for problematic axioms

This is a **self-validating system** where the AI tests itself.

## Code Style Notes

- Python 3.12+ required
- No strict linting enforced
- Type hints used inconsistently
- Docstrings at module level, function level varies
- Print statements used for logging (no logging framework)
- Long files (1000+ lines) are acceptable for main components
