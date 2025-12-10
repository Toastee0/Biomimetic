# BioMimetic AI - Cortex Integration Architecture

## Data Flow: How Cortexes Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                    CONVERSATION LAYER                        │
│                  (Discord Bot - Real-time)                   │
└────────────┬────────────────────────────────────┬───────────┘
             │                                    │
             ▼                                    ▼
     ┌───────────────┐                   ┌──────────────┐
     │  Store Episode│                   │Update Contact│
     │  to Episodic  │                   │   Profile    │
     │    Memory     │                   │  (counter++)  │
     └───────┬───────┘                   └──────┬───────┘
             │                                  │
             │                                  │
┌────────────┴──────────────────────────────────┴────────────┐
│                   MEMORY SUBSTRATES                         │
│  - data/biomim.db (episodes table)                         │
│  - data/biomim.db (contacts table)                         │
└────────────┬──────────────────────────────────┬────────────┘
             │                                  │
             │  Read every 30min                │  Read on msg
             ▼                                  ▼
    ┌────────────────┐              ┌──────────────────────┐
    │ CONTACT        │              │  DYNAMIC PROMPT      │
    │ LEARNING       │              │  BUILDER             │
    │ CORTEX         │              │                      │
    │ (Cron)         │              │  - Base identity     │
    └────────┬───────┘              │  - Memory stats      │
             │                      │  - Contact profile   │
             │ LLM Analysis         │  - Available tools   │
             │ (expensive)          │  - Learning status   │
             ▼                      └──────────┬───────────┘
    ┌────────────────┐                        │
    │  Extract:      │                        │ Inject into
    │  - Style       │                        │ conversation
    │  - Topics      │                        │
    │  - Traits      │                        ▼
    │  - Tech level  │              ┌──────────────────────┐
    └────────┬───────┘              │   LLM INFERENCE      │
             │                      │   (with full context)│
             │ Create               └──────────────────────┘
             ▼
    ┌────────────────┐
    │  MICRO-TOOLS   │
    │  (data/micro_  │
    │   tools/*.json)│
    │                │
    │  Fast lookup   │
    │  vs expensive  │
    │  re-analysis   │
    └────────────────┘
```

## Cortex Responsibilities

### 1. Discord Bot (Always Running)
**Process**: `src/discord/bot_axiom_review.py`
**Role**: Conversation interface
- Receives user messages
- Calls Dynamic Prompt Builder with user_id
- Gets context-aware system prompt
- Generates LLM response with full awareness
- Stores episode + updates contact on every message

### 2. Dynamic Prompt Builder (On-Demand)
**Module**: `src/core/dynamic_prompts.py`
**Role**: Context injection
- Loads base identity from config
- Queries episodic memory for conversation count
- Queries contact memory for user profile
- Scans micro-tools directory for learned patterns
- **Builds living prompt that reflects current state**
- AI knows what it knows!

### 3. Contact Learning Cortex (Every 30min)
**Process**: `scripts/cron/contact_learning.py`
**Role**: Pattern extraction & tool creation
- Analyzes recent conversations (expensive LLM)
- Extracts insights about each contact
- Updates contact profiles
- **Creates micro-tools** for future efficiency
- Adjusts trust levels

### 4. Episodic Consolidation (Every 10min)
**Process**: `scripts/cron/episodic_consolidation.py`
**Role**: Memory maintenance
- Calculates salience scores
- Marks episodes as consolidated
- Prevents re-processing

### 5. Axiom Cortex (Every 4hrs)
**Process**: `scripts/cron/full_axiom_evaluation.py`
**Role**: Reasoning validation
- Tests axiom library
- Identifies problems
- Generates clarification questions
- Adds to review queue

## Key Principles

### 1. Self-Awareness Through Dynamic Prompts
**Problem**: Static prompts → AI doesn't know its capabilities
**Solution**: Dynamic prompt builder queries all systems
**Result**: AI says "I have episodic memory with X conversations" not "I don't have memory"

### 2. Expensive → Cheap (Micro-tools)
**Problem**: Re-analyzing same conversations wastes inference
**Solution**: First analysis creates cached lookup tools
**Result**: Learn once, recall many times (biomimetic)

### 3. Independent Cortexes
**Problem**: Monolithic system = total failure on crash
**Solution**: Separate processes with shared data substrates
**Result**: Can restart/debug one cortex without affecting others

### 4. RTOS Timeouts
**Problem**: Runaway processes block system
**Solution**: Hard timeouts, log current state, exit
**Result**: Predictable behavior, clear failure modes

## Information Flow Example

**User sends message "What do you know about me?"**

1. **Discord Bot** receives message
2. **Contact Memory** updates interaction counter
3. **Dynamic Prompt Builder** called with user_id:
   - Queries contact profile: "Adrian, 3 interactions, technical_collaborator, trust 0.6"
   - Checks micro-tools: 2 tools available (preferred_topics, technical_level)
   - Builds prompt: "You are talking to Adrian. You've had 3 conversations. He's interested in AI, robotics."
4. **LLM** generates response with full context
5. **Episodic Memory** stores exchange
6. **Response** mentions specific known facts about user

**30 minutes later:**

7. **Contact Learning Cortex** wakes up
8. Reads last 20 episodes with this user
9. LLM analyzes: "Adrian asks detailed technical questions, prefers concise answers"
10. Updates contact profile with communication_style
11. Creates micro-tool: `{user_id}_communication_style.json`
12. Next conversation uses this cached knowledge

## Current System State

**Active Cortexes:**
- ✓ Discord Bot (running)
- ✓ Contact Learning (cron every 30min)
- ✓ Episodic Consolidation (cron every 10min)
- ✓ Axiom Evaluation (cron every 4hrs)

**Memory Systems:**
- ✓ Episodic Memory (SQLite)
- ✓ Contact Memory (SQLite)
- ✓ Micro-tools (JSON files)

**Integration:**
- ✓ Dynamic Prompt Builder queries all systems
- ✓ Bot uses context-aware prompts per user
- ✓ Learning cortex creates efficiency tools
- ✓ All cortexes log to separate files

**Next Steps:**
1. Test dynamic prompts with actual conversation
2. Verify contact learning extracts insights
3. Confirm micro-tools are created and used
4. Add !health command to show system status
