# Cortex Scheduling Architecture

## Continuous Consciousness Design

The system runs as independent cortexes (brain regions) with scheduled maintenance tasks. Each cortex is isolated - failures don't cascade. All processes are RTOS-style with hard timeouts.

## Core Principles

1. **Independence**: Each cortex can be serviced without taking down the whole mind
2. **Timeout-driven**: Processes that exceed time limits are killed, logged, and analyzed
3. **State logging**: Every process writes entry/exit logs with status
4. **Graceful degradation**: Missing/failed cortex doesn't crash others

---

## Process Categories & Schedules

### Real-Time Processes (Always Running)
**Discord Bot (Conversation Interface)**
- Service: `systemd` daemon
- Process: `src/discord/bot_axiom_review.py`
- Restart: Automatic on crash
- Logs: `logs/discord_bot.log`

**LLM Inference Server**
- Service: `llama-server` on port 53307
- Process: External (already running)
- Logs: System managed

---

### High-Frequency Cron Jobs (Minutes)

#### Episodic Memory Consolidation
**Schedule**: `*/10 * * * *` (Every 10 minutes)  
**Timeout**: 60 seconds  
**Script**: `scripts/cron/consolidate_episodes.py`

**Tasks**:
- Review unconsolidated episodes from last 10 minutes
- Calculate salience scores based on:
  - User trust level
  - Emotional content markers
  - Topic relevance to axiom system
  - Clarification question quality
- Update contact profiles with insights
- Mark episodes as consolidated

**Logs**: `logs/episodic_consolidation.log`

---

#### Axiom Spot Check
**Schedule**: `*/15 * * * *` (Every 15 minutes)  
**Timeout**: 3 minutes  
**Script**: `scripts/cron/axiom_spot_check.py`

**Tasks**:
- Randomly select 3-5 axioms
- Run quick evaluation (1-2 test scenarios each)
- Flag anomalies for deep review
- Don't generate clarification questions (just detect issues)

**Logs**: `logs/axiom_spot_check.log`

---

#### Contact Learning (PRIMARY GOAL)
**Schedule**: `*/30 * * * *` (Every 30 minutes)  
**Timeout**: 5 minutes  
**Script**: `scripts/cron/contact_learning.py`

**Tasks**:
- Analyze recent conversations with each contact
- Use LLM to extract insights (communication style, topics, traits)
- Update contact profiles with learned information
- Create micro-tools from patterns (cheap lookup vs expensive inference)
- Adjust trust levels based on interaction quality
- Add context summaries to contact notes

**Learning Principle**: 
- First time: Expensive LLM inference to understand patterns
- Creates micro-tools: Cached knowledge for fast lookup
- Future runs: Use tools instead of re-analyzing
- Linux philosophy: Many small, specialized tools

**Logs**: `logs/contact_learning.log`

---

### Medium-Frequency Cron Jobs (Hours)

#### Full Axiom Evaluation
**Schedule**: `0 */4 * * *` (Every 4 hours)  
**Timeout**: 15 minutes  
**Script**: `scripts/cron/full_axiom_evaluation.py`

**Tasks**:
- Run `SelfTrainingLoop.test_all_axioms()`
- Identify problematic axioms
- Generate clarification questions
- Add high-priority items to review queue
- Send DM notification if critical issues found

**Logs**: `logs/axiom_evaluation.log`

---

#### Memory Consolidation (Episodic → Semantic)
**Schedule**: `0 */6 * * *` (Every 6 hours)  
**Timeout**: 10 minutes  
**Script**: `scripts/cron/memory_consolidation.py`

**Tasks**:
- Extract patterns from recent episodes
- Update semantic memory concepts
- Strengthen relationship edges for frequently co-occurring concepts
- Compress old episodic entries (archive detail, keep summary)
- Update contact personality traits based on conversation patterns

**Logs**: `logs/memory_consolidation.log`

---

#### Identity Reflection
**Schedule**: `0 */12 * * *` (Every 12 hours: 00:00, 12:00)  
**Timeout**: 5 minutes  
**Script**: `scripts/cron/identity_reflection.py`

**Tasks**:
- Review recent decisions and interactions
- Check alignment with core values (from `data/identity.json`)
- Assess conversation quality metrics
- Update self-model based on feedback
- Write reflection summary to memory

**Logs**: `logs/identity_reflection.log`

---

### Low-Frequency Cron Jobs (Daily+)

#### Deep Learning Cycle
**Schedule**: `0 2 * * *` (Daily at 2:00 AM)  
**Timeout**: 30 minutes  
**Script**: `scripts/cron/deep_learning_cycle.py`

**Tasks**:
- Full axiom library validation
- Cross-reference memories for insights
- Generate meta-patterns from conversation history
- System health check (database integrity, log sizes, etc.)
- Prune low-salience memories older than 30 days
- Generate daily report

**Logs**: `logs/deep_learning.log`

---

#### Long-Term Consolidation
**Schedule**: `0 3 * * 0` (Weekly, Sunday at 3:00 AM)  
**Timeout**: 45 minutes  
**Script**: `scripts/cron/long_term_consolidation.py`

**Tasks**:
- Archive episodes older than 60 days
- Vacuum/optimize database
- Backup all state to `data/backups/`
- Rotate log files
- Generate weekly summary report

**Logs**: `logs/long_term_consolidation.log`

---

## Conversation Deliberation (Event-Driven)

**Process**: `src/background/conversation_deliberation.py`  
**Trigger**: Active conversation with idle time  
**Run Mode**: Background thread from Discord bot

**Timing**:
- Start after: 5s user idle
- Stop after: 5min user idle or new user message
- Deliberation cycles: Every 10-15s while active
- Unsolicited follow-up: Only after 45s+ idle

**Tasks per cycle**:
- Generate 2-4 alternative response branches
- Evaluate against axioms and contact profile
- Score by: relevance, truthfulness, helpfulness, relationship impact
- Keep top 2 in working memory
- Max 3-5 cycles total

**Anti-spam constraints**:
- Min 2-3s between bot messages
- Max 5 messages per minute
- 10s cooldown after 3+ rapid messages

**Logs**: `logs/deliberation.log`

---

## RTOS Timeout Handling

All cron scripts follow this pattern:

```python
#!/usr/bin/env python3
import signal
import sys
import time
from pathlib import Path

TIMEOUT_SECONDS = 60  # Adjust per task

def timeout_handler(signum, frame):
    print(f"[TIMEOUT] Process exceeded {TIMEOUT_SECONDS}s limit")
    print(f"[TIMEOUT] Current state: {get_current_state()}")
    sys.exit(124)  # Timeout exit code

def get_current_state():
    """Return string describing what we were doing when timeout hit"""
    pass

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TIMEOUT_SECONDS)

try:
    # Do work here
    pass
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)
finally:
    signal.alarm(0)  # Cancel alarm
    print(f"[COMPLETE] Finished in {time.time() - start:.2f}s")
```

Exit codes:
- `0`: Success
- `1`: Error
- `124`: Timeout

---

## Installation

```bash
# Install all cron jobs
./scripts/install_cron_jobs.sh

# View installed jobs
crontab -l

# Remove all jobs
./scripts/remove_cron_jobs.sh
```

---

## Monitoring

```bash
# Watch all cortex logs in real-time
./scripts/monitor_cortexes.sh

# Check last run status
./scripts/cortex_status.sh

# View specific cortex log
tail -f logs/axiom_evaluation.log
```

---

## State Files

Each cortex writes status to shared state:
- `data/cortex_state/episodic_consolidation.json`
- `data/cortex_state/axiom_evaluation.json`
- `data/cortex_state/memory_consolidation.json`
- `data/cortex_state/identity_reflection.json`
- `data/cortex_state/deep_learning.json`

Format:
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

The Discord bot can read these to report system health: `!health` command
