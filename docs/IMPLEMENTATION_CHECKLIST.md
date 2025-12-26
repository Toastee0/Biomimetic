# Model Ducking Implementation Checklist

**Version**: 1.0
**Date**: 2025-12-19
**Purpose**: Step-by-step integration guide with safety checks and rollback procedures

---

## Phase 0: Pre-Implementation Verification

### Environment Checks
- [ ] **Verify Python version**: `python --version` (need 3.12+)
- [ ] **Verify virtual environment active**: `which python` should show venv path
- [ ] **Check disk space**: `df -h /home/toastee` (need ~50GB for models)
- [ ] **Check VRAM available**: `nvidia-smi` (should show 24GB total)
- [ ] **Verify llama-server installed**: `which llama-server` or `ls /usr/local/bin/llama-server`
- [ ] **Check current bot running**: `ps aux | grep bot_axiom_review.py`
- [ ] **Backup current database**: `cp /home/toastee/BioMimeticAi/data/biomim.db /home/toastee/BioMimeticAi/data/biomim.db.backup.$(date +%Y%m%d)`

### Git Safety
- [ ] **Check git status**: `cd /home/toastee/BioMimeticAi && git status`
- [ ] **Create feature branch**: `git checkout -b feature/model-ducking`
- [ ] **Commit current state**: `git add -A && git commit -m "Pre-model-ducking snapshot"`

### Dependency Installation
- [ ] **Verify aiohttp installed**: `pip list | grep aiohttp`
- [ ] **Install if needed**: `pip install aiohttp`
- [ ] **Verify python-dotenv**: `pip list | grep python-dotenv`
- [ ] **Install if needed**: `pip install python-dotenv`

---

## Phase 1: Model Files Setup

### Download/Locate Models
- [ ] **Create models directory**: `mkdir -p /home/toastee/models`
- [ ] **Check available space**: `df -h /home/toastee/models`

#### Mistral-Small-22B (Conversation - 14GB)
- [ ] **Download** or locate existing file
- [ ] **Verify file**: `ls -lh /models/mistral-small-22b-instruct-2409-q4_k_m.gguf`
- [ ] **Test with llama-server**:
  ```bash
  llama-server -m /models/mistral-small-22b-instruct-2409-q4_k_m.gguf -c 2048 --port 53307 &
  sleep 10
  curl http://localhost:53307/health
  pkill llama-server
  ```
- [ ] **Record actual path** in notes

#### DeepSeek-Coder-33B (Coding - 22GB)
- [ ] **Download** or locate existing file
- [ ] **Verify file**: `ls -lh /models/deepseek-coder-33b-instruct-q4_k_m.gguf`
- [ ] **Test with llama-server** (same as above)
- [ ] **Record actual path** in notes

#### Update Configuration
- [ ] **Edit**: `/home/toastee/BioMimeticAi/config/models.json`
- [ ] **Update all `model_path` fields** with actual paths
- [ ] **Verify JSON syntax**: `python -m json.tool < config/models.json`

---

## Phase 2: Core Components Installation

### Task 1: LLMModelManager
- [ ] **Receive implementation** from delegated LLM
- [ ] **Save to**: `/home/toastee/BioMimeticAi/src/core/llm_model_manager.py`
- [ ] **Review code** for:
  - [ ] Proper async/await usage
  - [ ] Error handling on subprocess operations
  - [ ] Graceful shutdown logic (SIGTERM → SIGKILL)
  - [ ] Health check polling with timeout
  - [ ] Logging statements
  - [ ] Type hints present
  - [ ] Docstrings complete

#### Isolated Test (Before Integration)
- [ ] **Create test script**: `/home/toastee/BioMimeticAi/tests/test_model_manager_isolated.py`
  ```python
  import asyncio
  from src.core.llm_model_manager import LLMModelManager

  async def test_basic_load():
      manager = LLMModelManager()

      # Test conversation model
      success = await manager.ensure_model_loaded("conversation")
      assert success, "Failed to load conversation model"

      # Test model switch
      success = await manager.ensure_model_loaded("coding")
      assert success, "Failed to switch to coding model"

      # Cleanup
      await manager.shutdown()
      print("✅ Basic test passed")

  asyncio.run(test_basic_load())
  ```
- [ ] **Run test**: `python tests/test_model_manager_isolated.py`
- [ ] **Verify model switch completes**: Should take 45-120 seconds
- [ ] **Check llama-server stopped**: `ps aux | grep llama-server` (should be empty)

### Task 2: TaskClassifier
- [ ] **Receive implementation** from delegated LLM
- [ ] **Save to**: `/home/toastee/BioMimeticAi/src/core/task_classifier.py`
- [ ] **Review code** for:
  - [ ] Keyword lists loaded from config
  - [ ] Fast execution (no LLM calls)
  - [ ] Default fallback to "conversation"
  - [ ] Edge case handling

#### Isolated Test
- [ ] **Create test script**: `/home/toastee/BioMimeticAi/tests/test_classifier_isolated.py`
  ```python
  from src.core.task_classifier import TaskClassifier

  classifier = TaskClassifier()

  # Test coding classification
  assert classifier.classify("let's implement a function") == "coding"
  assert classifier.classify("debug this error") == "coding"
  assert classifier.classify("write code for X") == "coding"

  # Test conversation classification
  assert classifier.classify("hello how are you") == "conversation"
  assert classifier.classify("what's the weather") == "conversation"

  # Test reasoning classification (long message)
  long_msg = " ".join(["word"] * 150)
  assert classifier.classify(long_msg) == "reasoning"

  print("✅ Classification tests passed")
  ```
- [ ] **Run test**: `python tests/test_classifier_isolated.py`

### Task 3: UniversalContextBuilder
- [ ] **Receive implementation** from delegated LLM
- [ ] **Save to**: `/home/toastee/BioMimeticAi/src/core/universal_context.py`
- [ ] **Review code** for:
  - [ ] Integration with EpisodicMemory
  - [ ] Integration with ContactMemory
  - [ ] Markdown formatting
  - [ ] Null/empty data handling
  - [ ] Task-specific context sections

#### Isolated Test
- [ ] **Create test script**: `/home/toastee/BioMimeticAi/tests/test_context_builder_isolated.py`
  ```python
  from src.core.universal_context import UniversalContextBuilder
  from src.memory.episodic import EpisodicMemory
  from src.memory.contact_memory import ContactMemory

  builder = UniversalContextBuilder()
  episodic = EpisodicMemory()
  contacts = ContactMemory()

  # Get or create test contact
  contact = contacts.get_or_create_contact(
      user_id="test_user_123",
      username="test_user",
      display_name="Test User"
  )

  # Get recent episodes (might be empty for new test user)
  episodes = episodic.get_recent_episodes(limit=3, user_id="test_user_123")

  # Build context
  context = builder.build_context(
      user_message="let's write some code",
      contact_id="test_user_123",
      mode="coding",
      episodes=episodes
  )

  # Verify structure
  assert "# SYSTEM IDENTITY" in context
  assert "# CURRENT MODE" in context
  assert "coding" in context
  assert "# USER CONTEXT" in context
  assert "# USER REQUEST" in context

  print("✅ Context builder tests passed")
  print("\nGenerated Context Preview:")
  print(context[:500])
  ```
- [ ] **Run test**: `python tests/test_context_builder_isolated.py`

---

## Phase 3: Discord Bot Integration

**⚠️ CRITICAL: Make backups before modifying bot code**

### Pre-Integration Backup
- [ ] **Stop current bot**: `pkill -f bot_axiom_review.py`
- [ ] **Backup bot file**: `cp src/discord/bot_axiom_review.py src/discord/bot_axiom_review.py.backup`
- [ ] **Create test copy**: `cp src/discord/bot_axiom_review.py src/discord/bot_test_model_ducking.py`

### Integration Steps

#### Step 1: Add Imports (Line 46)
- [ ] **Open**: `src/discord/bot_axiom_review.py`
- [ ] **After existing imports** (~line 46), add:
  ```python
  from src.core.llm_model_manager import LLMModelManager
  from src.core.task_classifier import TaskClassifier
  from src.core.universal_context import UniversalContextBuilder
  ```
- [ ] **Save and verify syntax**: `python -m py_compile src/discord/bot_axiom_review.py`

#### Step 2: Initialize Managers (Line 55)
- [ ] **After global variables** (~line 55), add:
  ```python
  # Model Ducking Components
  model_manager = LLMModelManager()
  task_classifier = TaskClassifier()
  context_builder = UniversalContextBuilder()
  ```
- [ ] **Save and verify**: `python -m py_compile src/discord/bot_axiom_review.py`

#### Step 3: Add Shutdown Handler (End of file)
- [ ] **Before** `if __name__ == '__main__':`, add:
  ```python
  @bot.event
  async def on_close():
      """Cleanup when bot shuts down"""
      await model_manager.shutdown()
      print("[SHUTDOWN] Model manager stopped")
  ```

#### Step 4: Modify on_message Handler (Line 508+)

**Current code** (lines 507-542):
```python
# General conversation mode
print(f"[INFO] Entering conversation mode for user {message.author.name}")

async with message.channel.typing():
    # Get recent episodes
    recent_episodes = episodic.get_recent_episodes(
        limit=5,
        user_id=str(message.author.id)
    )

    # Build conversation context
    conversation_context = ""
    if recent_episodes:
        conversation_context = "Previous conversations:\n"
        for ep in recent_episodes:
            conversation_context += f"- {ep.user_message[:100]}... → {ep.bot_response[:100]}...\n"

    # Get dynamic system prompt
    full_prompt = message.content
    if conversation_context:
        full_prompt = f"{conversation_context}\nCurrent message:\n{message.content}"

    user_id = str(message.author.id)
    dynamic_system_prompt = get_conversational_prompt(user_id=user_id)

    # Generate response
    response = textgen.generate(
        full_prompt,
        system_prompt=dynamic_system_prompt,
        max_tokens=500,
        temperature=0.8
    )
```

**NEW code** (replace lines 507-542):
```python
# General conversation mode
print(f"[INFO] Entering conversation mode for user {message.author.name}")

try:
    # Step 1: Classify task type
    task_type = task_classifier.classify(message.content)
    print(f"[TASK] Classified as: {task_type}")

    # Step 2: Ensure correct model loaded
    await model_manager.ensure_model_loaded(
        task_type,
        notify_callback=lambda msg: message.channel.send(msg)
    )

    # Step 3: Build context
    async with message.channel.typing():
        # Get recent episodes
        recent_episodes = episodic.get_recent_episodes(
            limit=5,
            user_id=str(message.author.id)
        )

        # Build universal context
        context = context_builder.build_context(
            user_message=message.content,
            contact_id=str(message.author.id),
            mode=task_type,
            episodes=recent_episodes
        )

        # Get dynamic system prompt (still used)
        user_id = str(message.author.id)
        dynamic_system_prompt = get_conversational_prompt(user_id=user_id)

        # Step 4: Generate response via ModelManager
        # NOTE: This assumes model_manager.generate() exists
        # If not, we use textgen directly (model already loaded)
        response = textgen.generate(
            context,  # Use rich context instead of just message
            system_prompt=dynamic_system_prompt,
            max_tokens=500,
            temperature=0.8
        )
```

**Checklist**:
- [ ] **Find lines 507-542** in `bot_axiom_review.py`
- [ ] **Replace with new code** above
- [ ] **Save file**
- [ ] **Verify syntax**: `python -m py_compile src/discord/bot_axiom_review.py`
- [ ] **Check indentation** matches surrounding code
- [ ] **Verify all variables** still defined (task_type, context, etc.)

#### Step 5: Add Auto-Duck Background Task

- [ ] **Find**: `async def on_ready():` (around line 251-265)
- [ ] **Inside on_ready()**, add:
  ```python
  # Start auto-duck background task
  bot.loop.create_task(auto_duck_loop())
  print("[INFO] Auto-duck loop started")
  ```

- [ ] **After on_ready()**, add new function:
  ```python
  async def auto_duck_loop():
      """Background task to auto-switch models back to conversation"""
      await bot.wait_until_ready()  # Wait for bot to be ready
      while not bot.is_closed():
          try:
              await asyncio.sleep(60)  # Check every minute
              await model_manager.check_auto_duck()
          except Exception as e:
              print(f"[ERROR] Auto-duck loop error: {e}")
  ```

- [ ] **Save and verify**: `python -m py_compile src/discord/bot_axiom_review.py`

### Integration Pre-Flight Checks
- [ ] **All syntax valid**: No errors from `py_compile`
- [ ] **All imports resolve**: `python -c "from src.discord.bot_axiom_review import *"`
- [ ] **Config file exists**: `ls config/models.json`
- [ ] **Model files exist**: Verify paths in config match actual files
- [ ] **Database accessible**: `ls data/biomim.db`

---

## Phase 4: Initial Testing

### Test 1: Bot Startup
- [ ] **Set environment**: `export DISCORD_TOKEN="your_token"`
- [ ] **Start bot in foreground**: `python src/discord/bot_axiom_review.py`
- [ ] **Watch for errors** during startup
- [ ] **Check initial model load**: Should load "conversation" mode by default
- [ ] **Verify bot online** in Discord
- [ ] **Expected output**:
  ```
  [INFO] LLM inference connected
  [INFO] Model manager initialized
  [INFO] Loading default model: conversation
  [INFO] Model loaded successfully: mistral-small-22b
  [INFO] Discord bot logged in as: BioMimeticAI#1234
  [INFO] Auto-duck loop started
  ```

**If startup fails**:
- [ ] Check error message
- [ ] Verify model path in config
- [ ] Check llama-server executable location
- [ ] Review logs: `tail -50 logs/*.log`
- [ ] **ROLLBACK**: `cp src/discord/bot_axiom_review.py.backup src/discord/bot_axiom_review.py`

### Test 2: Simple Conversation
- [ ] **Send DM to bot**: "Hello, how are you?"
- [ ] **Expected**:
  - No model switch (already in conversation mode)
  - Response arrives within ~2-5 seconds
  - Normal conversation response
- [ ] **Verify in logs**: `[TASK] Classified as: conversation`

**If fails**:
- [ ] Check TaskClassifier is working
- [ ] Verify context builder doesn't crash
- [ ] Check textgen still works

### Test 3: Model Switch (Coding Request)
- [ ] **Send DM to bot**: "Let's implement a ring buffer class in Python"
- [ ] **Expected sequence**:
  1. "🔧 Switching to coding mode, loading DeepSeek-Coder-33B..."
  2. "This will take about 45 seconds. Time to grab coffee ☕"
  3. [45-120 second delay while models switch]
  4. "✅ Coding mode ready. Let's build something."
  5. [Response with code implementation]
- [ ] **Verify in logs**:
  ```
  [TASK] Classified as: coding
  [INFO] Switching from conversation to coding
  [INFO] Unloading model: mistral-small-22b
  [INFO] Loading model: deepseek-coder-33b
  [INFO] Model loaded successfully
  ```
- [ ] **Monitor VRAM**: `watch nvidia-smi` (should drop to 0GB then rise to 22GB)

**If fails**:
- [ ] Check model path for coding model
- [ ] Verify llama-server accepts model file
- [ ] Check timeout settings (might need longer for large models)
- [ ] Review subprocess errors in logs

### Test 4: Follow-Up Coding Question
- [ ] **Send DM**: "Now add error handling to that code"
- [ ] **Expected**:
  - NO model switch (already in coding mode)
  - Quick response (~2-5 seconds)
  - Code-focused answer
- [ ] **Verify in logs**: `[INFO] Model deepseek-coder-33b already loaded`

### Test 5: Auto-Duck (Return to Conversation)
- [ ] **Wait 16 minutes** without sending messages
- [ ] **Expected**:
  - After 15 min idle, bot switches back to conversation mode
  - "💭 Switching back to conversation mode..." (might not see if you're not actively looking)
- [ ] **Verify in logs**:
  ```
  [INFO] Auto-ducking: coding idle for 900s (timeout: 900s)
  [INFO] Switching from coding to conversation
  ```
- [ ] **Send simple message**: "hi" (should respond quickly, already in conversation mode)

### Test 6: Existing Commands Still Work
- [ ] **Test**: `!stats`
- [ ] **Expected**: System statistics display (existing functionality)
- [ ] **Test**: `!profile`
- [ ] **Expected**: Your contact profile
- [ ] **Verify**: No interference from model ducking

---

## Phase 5: Edge Case Testing

### Test 7: Rapid Mode Switching
- [ ] **Send**: "hello" (conversation)
- [ ] **Immediately send**: "write code for X" (coding)
- [ ] **Expected**:
  - First message might get delayed if switch in progress
  - Second message triggers switch
  - Both messages eventually answered
- [ ] **Check**: No crashes, no lost messages

### Test 8: Invalid Model Request
- [ ] **Manually edit** task_classifier to return "invalid_mode"
- [ ] **Send message**
- [ ] **Expected**: Graceful error, fallback to conversation mode
- [ ] **Revert** edit after test

### Test 9: llama-server Crash During Inference
- [ ] **Send coding request**
- [ ] **While generating**, kill llama-server: `pkill llama-server`
- [ ] **Expected**:
  - Error message to user
  - Bot doesn't crash
  - Next request triggers model reload
- [ ] **Verify**: Bot recovers gracefully

### Test 10: Model File Missing
- [ ] **Edit config**: Change coding model path to non-existent file
- [ ] **Send coding request**
- [ ] **Expected**:
  - Error during model load
  - User-friendly error message
  - Bot doesn't crash
- [ ] **Restore** correct path

---

## Phase 6: Performance & Monitoring

### Metrics to Track
- [ ] **Model switch time**: Time from classification to ready (should be 45-120s)
- [ ] **Response latency**: Time from user message to bot response
  - Conversation mode (same model): 2-5 seconds
  - Coding mode (after switch): 45-120s + inference time
  - Coding mode (already loaded): 2-5 seconds
- [ ] **VRAM usage**: `nvidia-smi` before/during/after switch
- [ ] **Memory leaks**: Monitor bot memory over 24 hours

### Create Monitoring Script
- [ ] **Create**: `scripts/monitor_model_ducking.sh`
  ```bash
  #!/bin/bash
  while true; do
      echo "=== $(date) ==="
      echo "Bot process:"
      ps aux | grep bot_axiom_review.py | grep -v grep
      echo ""
      echo "llama-server:"
      ps aux | grep llama-server | grep -v grep
      echo ""
      echo "VRAM:"
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
      echo ""
      sleep 60
  done
  ```
- [ ] **Make executable**: `chmod +x scripts/monitor_model_ducking.sh`
- [ ] **Run in background**: `./scripts/monitor_model_ducking.sh >> logs/monitoring.log &`

---

## Phase 7: Production Deployment

### Systemd Service Setup
- [ ] **Create service file**: `/etc/systemd/system/biomimetic-discord-bot.service`
  ```ini
  [Unit]
  Description=BioMimeticAI Discord Bot with Model Ducking
  After=network.target

  [Service]
  Type=simple
  User=toastee
  WorkingDirectory=/home/toastee/BioMimeticAi
  Environment="DISCORD_TOKEN=YOUR_TOKEN_HERE"
  ExecStart=/home/toastee/BioMimeticAi/venv/bin/python src/discord/bot_axiom_review.py
  Restart=always
  RestartSec=10
  StandardOutput=append:/home/toastee/BioMimeticAi/logs/bot_stdout.log
  StandardError=append:/home/toastee/BioMimeticAi/logs/bot_stderr.log

  [Install]
  WantedBy=multi-user.target
  ```
- [ ] **Reload systemd**: `sudo systemctl daemon-reload`
- [ ] **Enable service**: `sudo systemctl enable biomimetic-discord-bot`
- [ ] **Start service**: `sudo systemctl start biomimetic-discord-bot`
- [ ] **Check status**: `sudo systemctl status biomimetic-discord-bot`
- [ ] **View logs**: `journalctl -u biomimetic-discord-bot -f`

### Environment Configuration
- [ ] **Create**: `config/.env`
  ```bash
  DISCORD_TOKEN=your_actual_token_here
  LLM_SERVER_URL=http://localhost:53307
  LOG_LEVEL=INFO
  ```
- [ ] **Update bot** to load from .env:
  ```python
  from dotenv import load_dotenv
  load_dotenv('config/.env')
  DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
  ```
- [ ] **Never commit** .env to git: Add to `.gitignore`

---

## Rollback Procedures

### If Phase 3 Integration Fails
```bash
# Stop bot
pkill -f bot_axiom_review.py

# Restore backup
cp src/discord/bot_axiom_review.py.backup src/discord/bot_axiom_review.py

# Restart old version
python src/discord/bot_axiom_review.py
```

### If Model Files Don't Work
```bash
# Edit config to use only working model
nano config/models.json
# Change all modes to use same working model temporarily

# Or disable model ducking entirely:
# Comment out model switching logic, use TextGenClient directly
```

### If Database Corrupts
```bash
# Restore backup
cp data/biomim.db.backup.YYYYMMDD data/biomim.db

# Verify integrity
sqlite3 data/biomim.db "PRAGMA integrity_check;"
```

### Nuclear Option (Full Rollback)
```bash
# Stop everything
pkill -f bot_axiom_review.py
pkill llama-server

# Restore from git
cd /home/toastee/BioMimeticAi
git checkout main
git reset --hard HEAD

# Restore database
cp data/biomim.db.backup.YYYYMMDD data/biomim.db

# Restart original bot
python src/discord/bot_axiom_review.py
```

---

## Success Criteria

### Phase 1 Success
- [ ] All model files downloaded and verified
- [ ] Config file has correct paths
- [ ] Test loads successful

### Phase 2 Success
- [ ] All 3 components implemented and saved
- [ ] All isolated tests pass
- [ ] No syntax errors
- [ ] Code reviewed and approved

### Phase 3 Success
- [ ] Bot code modified without breaking existing features
- [ ] Syntax valid
- [ ] All imports resolve
- [ ] Backup created

### Phase 4 Success
- [ ] Bot starts without errors
- [ ] Simple conversation works
- [ ] Model switching works (coding request)
- [ ] Follow-up questions work (no re-switch)
- [ ] Auto-duck works (returns to conversation)
- [ ] Existing commands still work

### Phase 5 Success
- [ ] All edge cases handled gracefully
- [ ] No crashes observed
- [ ] Error messages user-friendly
- [ ] Bot recovers from failures

### Phase 6 Success
- [ ] Performance meets expectations (45-120s switch)
- [ ] No memory leaks
- [ ] Monitoring shows healthy operation

### Phase 7 Success
- [ ] Systemd service running
- [ ] Auto-starts on boot
- [ ] Logs properly captured
- [ ] Production stable

---

## Timeline Estimate

**With Tasks 1-3 Delegated to Other LLMs:**

- **Day 1 Morning**: Delegate tasks, other LLMs work on implementations
- **Day 1 Afternoon**: Phase 0-2 (environment setup + component testing) - 2-3 hours
- **Day 2 Morning**: Phase 3 (integration) - 2-3 hours
- **Day 2 Afternoon**: Phase 4-5 (testing + edge cases) - 2-3 hours
- **Day 3**: Phase 6-7 (monitoring + deployment) - 2-3 hours

**Total**: ~3 days with good parallelization

---

## Notes & Gotchas

1. **Model switching is SLOW** - 45-120 seconds. This is expected and desired.
2. **VRAM monitoring** - Use `nvidia-smi` to verify models fully unload/load
3. **Discord rate limits** - Be careful with rapid testing, Discord may throttle
4. **Database locks** - Existing issue, not caused by model ducking
5. **Log files grow** - Set up logrotate eventually
6. **First switch is slowest** - Cold start takes longer than subsequent switches

---

## Support Resources

- **Documentation**: `/home/toastee/BioMimeticAi/docs/MODEL_DUCKING_ARCHITECTURE.md`
- **Delegation tasks**: `/home/toastee/BioMimeticAi/docs/DELEGATION_TASKS_FOR_OTHER_LLMS.md`
- **Research findings**: `/home/toastee/BioMimeticAi/docs/TEAM_DELEGATION_SUMMARY.md`
- **Discord logs**: Check DMs for error messages from bot
- **System logs**: `/home/toastee/BioMimeticAi/logs/*.log`

---

**This checklist is your implementation bible. Follow it sequentially, check off each item, and you'll have a production-ready Model Ducking system.**
