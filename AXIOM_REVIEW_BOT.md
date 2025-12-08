# Axiom Review Bot - Human-in-the-Loop Training

## Overview

The Axiom Review Bot enables human-guided self-training of the axiom cortex. It DMs you when axioms need review and allows you to approve, reject, or get improvement suggestions.

## Setup

1. **Add your Discord User ID to `.env`:**
   ```bash
   echo "DISCORD_OWNER_ID=YOUR_USER_ID_HERE" >> config/.env
   ```
   
   To find your Discord User ID:
   - Enable Developer Mode in Discord (Settings → Advanced → Developer Mode)
   - Right-click your username and select "Copy ID"

2. **Start the review bot:**
   ```bash
   ./scripts/start_axiom_review_bot.sh
   ```

3. **Start self-training loop** (in another terminal):
   ```bash
   cd /home/toastee/BioMimeticAi
   source venv/bin/activate
   python src/tensor_axiom/self_training_loop.py --iterations 0 --interval 300
   ```

## Commands (via DM)

### Review Queue
- `!review` - Show current review queue
- `!inspect [number]` - Inspect axiom from queue (default: 1)
- `!stats` - Show axiom system statistics

### Axiom Actions
- `!approve` - Approve current axiom, remove from queue
- `!reject [reason]` - Reject axiom with reason
- `!suggest` - Get LLM improvement suggestions
- `!skip` - Skip to next axiom

## Workflow

1. **Self-training loop tests axioms** using LLM inference
2. **Problematic axioms** (low success/confidence) added to review queue
3. **Bot DMs you** when high-priority items need review
4. **You inspect** axioms with `!inspect`
5. **You decide:**
   - `!approve` if axiom is correct
   - `!reject` if axiom needs rework
   - `!suggest` to get AI improvement ideas
6. **Loop continues** improving axioms based on your guidance

## Integration with Main Bot

The main PopTartee bot (`bot.py`) continues to operate normally in channels. The review bot is a separate process that:
- Uses the same token (different instance)
- Only responds to DMs from you
- Focuses solely on axiom review
- Doesn't interfere with channel conversations

## Architecture

```
┌─────────────────────┐
│  Self-Training Loop │  ← Tests axioms with LLM
│  (autonomous)       │     
└──────────┬──────────┘
           │ Adds problematic axioms
           ↓
┌─────────────────────┐
│   Review Queue      │
│  (JSONL storage)    │
└──────────┬──────────┘
           │ Bot monitors
           ↓
┌─────────────────────┐
│  Axiom Review Bot   │  ← DMs you
│  (Discord)          │
└──────────┬──────────┘
           │ Your decisions
           ↓
┌─────────────────────┐
│   Axiom Library     │  ← Updated based on feedback
│  (JSON)             │
└─────────────────────┘
```

## Logs

Training events logged to:
- `logs/self_training.jsonl` - Self-training iterations
- Review decisions tracked in `data/axioms/review_queue.json`

## Safety

- Bot only responds to DMs from configured owner (you)
- Cannot modify axioms directly, only mark for approval/rejection
- All changes require your explicit command
- Training loop tests axioms but doesn't auto-modify

## Tips

- Check `!review` periodically to see queue
- Use `!suggest` to get AI input before deciding
- Rejected axioms stay in queue for rework
- Approved axioms removed from queue but stay in library
