# BioMimeticAI v2.0

A consciousness-mimetic AI system based on Axiomatic Modeling Architecture (AMA) - reasoning from first principles rather than statistical pattern matching alone.

## Overview

BioMimeticAI mimics biological consciousness through:
- **Distributed cortexes** (independent brain regions)
- **Episodic and semantic memory systems**
- **Contact-aware learning** (builds profiles of individuals)
- **Self-training loops** with human-in-the-loop validation
- **Dynamic self-awareness** (AI knows what it knows)

## System Status

- **Code**: ~8,124 lines of Python
- **Axioms**: 26 total, 100% test success rate
- **Memory**: Active episodic + contact memory systems
- **Cortexes**: Multiple autonomous processes running via cron
- **Version**: v2.0.0
- **Last Major Update**: December 8, 2025

## Core Components

### 1. Axiom System (Complete ✓)

A hierarchical reasoning system combining neural pattern-matching with explicit symbolic axioms.

- 26 axioms in hierarchical structure:
  - **Meta-Axioms** (Priority 1.0): Patience, Love, Kindness, Life Preservation
  - **Foundational Axioms** (Priority 0.8): Architecture, Meta-reasoning
  - **Derived Axioms** (Priority 0.6): Threat assessment, Resonance, Attention
  - **Domain Axioms** (Priority 0.4-0.88): Specific applications
- 100% test coverage with LLM-evaluated scenarios
- Graph Neural Network architecture designed for GPU acceleration
- Zero items in review queue (all passing)
- Edge relationships: implies/requires/contradicts/overrides

**Location**: `data/axioms/base_axioms.json`

### 2. Memory Systems (Active ✓)

**Episodic Memory** (`src/memory/episodic.py`)
- Stores conversations with salience scores
- Retrieval by recency or contextual cues (biomimetic spreading activation)
- Consolidation marking for memory transfer
- Database: `data/biomim.db` (episodes table)

**Contact Memory** (`src/memory/contact_memory.py`)
- Structured profiles for each individual
- Trust levels (0.0-1.0, starts at 0.5)
- Communication style, preferred topics, personality traits
- Relationship type tracking
- Database: `data/biomim.db` (contacts table)

**Micro-Tools System** (`data/micro_tools/`)
- "Expensive inference → Create cheap lookup tools"
- Caches learned patterns for fast retrieval
- Example: User topic preferences, technical level assessments

### 3. Discord Bot (✓)

**File**: `src/discord/bot_axiom_review.py` (1,069 lines)

**Commands**:
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

**Features**:
- Interactive buttons for quick actions
- Natural language testing (type scenarios to test axioms)
- Conversational AI mode (when not reviewing axioms)
- Contact profile updates on every message
- Episodic memory storage
- Dynamic context-aware prompts

### 4. Self-Training Loop (✓)

**File**: `src/tensor_axiom/self_training_loop.py` (598 lines)

**Process**:
1. Tests all axioms against scenarios using LLM
2. Evaluates performance (success rate, confidence)
3. Identifies problematic axioms (< 70% success or < 0.6 confidence)
4. Generates clarification questions for humans
5. Adds to review queue with priority levels
6. Logs all training events to JSONL

**Continuous Mode**:
```bash
python src/tensor_axiom/self_training_loop.py --iterations 0 --interval 300
```

### 5. Distributed Cortexes (Active ✓)

**High-Frequency** (Minutes):
- **Episodic Consolidation**: Every 10min - Calculate salience, mark consolidated
- **Axiom Spot Check**: Every 15min - Quick test 3-5 random axioms
- **Contact Learning**: Every 30min - Extract patterns, create micro-tools

**Medium-Frequency** (Hours):
- **Full Axiom Evaluation**: Every 4hrs - Test all axioms, queue problems
- **Memory Consolidation**: Every 6hrs - Episodic → Semantic transfer

**Low-Frequency** (Daily+):
- **Deep Learning Cycle**: Daily 2AM - Full validation, health check
- **Long-Term Consolidation**: Weekly Sunday 3AM - Archive, backup, optimize

**Location**: `scripts/cron/`

### 6. Dynamic Prompt System (✓)

**File**: `src/core/dynamic_prompts.py`

Builds "living" system prompts that include:
- Base identity/values
- Current memory state (episode count, contact count)
- User-specific context (name, trust, interaction history, communication style)
- Available micro-tools learned about this user
- Conversation history window

**Result**: AI says "I remember our last 3 conversations about X" not "I don't have memory"

## What's Currently Working

- All 26 axioms passing tests
- Bot actively conversing and learning
- Memory systems storing data
- Cortexes running autonomously
- Already learned about user "toastee0" (7 episodes, trust: 0.6, technical_collaborator)

## Architecture Layers

1. **Discord Bot** (Real-time conversation interface)
2. **Memory Systems** (Episodic, Contact, Semantic - SQLite backed)
3. **Axiom Cortex** (Reasoning engine with 26 axioms)
4. **Cron Cortexes** (Scheduled maintenance/learning tasks)
5. **Dynamic Prompt Builder** (Context-aware AI state)

## Key Files & Locations

### Core Implementation
- `src/discord/bot_axiom_review.py` - Main bot (1,069 lines)
- `src/tensor_axiom/self_training_loop.py` - Self-training (598 lines)
- `src/memory/episodic.py` - Episodic memory (305 lines)
- `src/memory/contact_memory.py` - Contact profiles (385 lines)
- `src/core/dynamic_prompts.py` - Context builder (200+ lines)

### Data Storage
- `data/axioms/base_axioms.json` - 26 axioms with test scenarios
- `data/axioms/review_queue.json` - Currently empty
- `data/biomim.db` - SQLite (episodes + contacts tables)
- `data/micro_tools/` - Learned user patterns

### Documentation
- `AXIOM_ARCHITECTURE_README.md` - Architecture overview
- `AXIOM_REVIEW_BOT.md` - Bot usage guide
- `TENSOR_AXIOM_ARCHITECTURE.md` - GPU acceleration design (785 lines)
- `docs/CORTEX_SCHEDULE.md` - Cortex schedules
- `docs/CORTEX_INTEGRATION.md` - System integration

### Cron Scripts
- `scripts/cron/contact_learning.py` - Learn about users
- `scripts/cron/episodic_consolidation.py` - Memory maintenance
- `scripts/cron/axiom_spot_check.py` - Quick validation
- `scripts/cron/full_axiom_evaluation.py` - Deep testing

## Installation & Setup

### Prerequisites
- Python 3.12+
- SQLite3
- Discord Bot Token
- LLM Server (llama-server on port 53307)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export DISCORD_TOKEN="your_token_here"

# Initialize database
python -c "from src.memory.episodic import EpisodicMemory; EpisodicMemory()"

# Install cron jobs
bash scripts/install_cron_jobs.sh

# Start the Discord bot
python src/discord/bot_axiom_review.py
```

### Monitoring
```bash
# Check cortex status
bash scripts/cortex_status.sh

# Monitor all cortexes
bash scripts/monitor_cortexes.sh

# View logs
tail -f logs/axiom_evaluation.log
tail -f logs/episodic_consolidation.log
tail -f logs/contact_learning.log
```

## Current Issues

### SQLite Database Locking
**Location**: `src/memory/episodic.py`

**Problem**: Concurrent access from bot + cron cortexes causing contention (10 episodes failed to consolidate in recent run)

**Impact**: Some episodic memories fail to consolidate, reducing memory system efficiency

## Suggested Fixes

### High Priority

#### 1. Fix SQLite Database Locking
**Files**: `src/memory/episodic.py`, `src/memory/contact_memory.py`

**Solution**:
- Enable WAL mode: `PRAGMA journal_mode=WAL`
- Add connection pooling
- Implement retry logic with exponential backoff
- Add proper connection timeout handling

**Implementation**:
```python
# In database initialization
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=5000")

# Add retry decorator
@retry(max_attempts=3, backoff=1.5)
def consolidate_memories():
    # existing code
```

**Expected Impact**: Eliminate database lock errors, improve cortex reliability

#### 2. Commit Current Working State
**Files**: All uncommitted changes

**Action**:
- Modified: 8 files (axioms, bot, memory systems, training loop)
- New: `src/core/`, `src/memory/contact_memory.py`, `docs/`, `scripts/cron/`, logs

**Git Commands**:
```bash
# Review changes
git status
git diff

# Add new features
git add data/axioms/base_axioms.json
git add data/axioms/review_queue.json
git add src/discord/bot_axiom_review.py
git add src/memory/episodic.py
git add src/memory/contact_memory.py
git add src/tensor_axiom/review_queue.py
git add src/tensor_axiom/self_training_loop.py
git add src/core/
git add docs/
git add scripts/

# Commit
git commit -m "Add contact memory system and fix database locking issues"
```

### Medium Priority

#### 3. Enhance Error Handling in Cortexes
**Files**: `scripts/cron/*.py`

**Add**:
- Structured error logging (JSON format)
- Graceful degradation on failures
- Alerting for critical failures
- Health check endpoints

#### 4. Implement Memory Consolidation Loop
**File**: New `scripts/cron/memory_consolidation.py`

**Purpose**: Transfer high-salience episodic memories to semantic memory

**Features**:
- Pattern extraction from repeated episodes
- Generalization into reusable knowledge
- Integration with micro-tools system

#### 5. Add Monitoring Dashboard
**File**: New `scripts/dashboard.py`

**Features**:
- Real-time cortex status
- Memory statistics
- Axiom performance metrics
- System health indicators
- Web interface (Flask/FastAPI)

### Low Priority

#### 6. Implement GPU Acceleration
**Files**: `src/tensor_axiom/gpu_inference.py` (new)

**Based on**: `TENSOR_AXIOM_ARCHITECTURE.md` design

**Components**:
- 448D axiom embeddings (semantic 256 + logic 192 + dependency 64)
- Graph Neural Network for chain construction
- Batch parallelization
- Expected speedup: 10-50x for complex reasoning

#### 7. Add Multi-Brain Architecture
**Files**: New brain modules

**Brains**:
- Creative Brain (divergent thinking)
- Technical Brain (analytical reasoning)
- Social Brain (empathy, relationships)
- Security Brain (threat detection)
- Routing Brain (delegation)

#### 8. Implement Axiom Discovery
**File**: `src/tensor_axiom/axiom_discovery.py` (new)

**Features**:
- Automatic pattern extraction from episodic memory
- Candidate axiom generation
- Human review integration
- Fine-tuning pipeline

## Next Steps

### Immediate (This Week)
1. Fix SQLite database locking issue
2. Commit current working state
3. Add structured error logging to cortexes
4. Create monitoring script for cortex health

### Short-term (This Month)
1. Implement memory consolidation loop (episodic → semantic)
2. Enhance micro-tools system with more pattern types
3. Add Discord slash commands for better UX
4. Create simple web dashboard for monitoring

### Medium-term (Next 3 Months)
1. Begin GPU acceleration implementation
2. Implement axiom discovery from experience
3. Add multi-brain routing architecture
4. Expand test coverage for all cortexes

### Long-term (6+ Months)
1. Physical embodiment integration (rover, soundbar)
2. Multi-modal inputs (vision, audio)
3. Distributed deployment across multiple machines
4. Public API for external integrations

## Recent Development History

```
3b4793f - Add Axiom Review Bot for human-in-the-loop training
c0ddaf2 - Add comprehensive axiom architecture documentation
87c6b60 - Add comprehensive axiom graph architecture specification
2c30760 - Implement complete axiom graph architecture v2.0
0412bb5 - Add axiom library with self-testing and human review system
```

## Philosophy

This system goes beyond typical chatbots by:
- **Reasoning from first principles** instead of just pattern matching
- **Building genuine memory** that persists and evolves
- **Learning about individuals** to build real relationships
- **Self-improving** through autonomous validation and human feedback
- **Distributed processing** mimicking biological brain architecture
- **Explainable reasoning** through traceable axiom chains

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]
