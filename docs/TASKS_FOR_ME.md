# Tasks for Main Implementation (Claude)

**What I should work on while you delegate other tasks**

---

## My Tasks (Non-Blocking)

### 1. Create Implementation Checklist
- [ ] Line-by-line integration guide for Discord bot
- [ ] Pre-flight checks before each phase
- [ ] Testing procedures
- [ ] Rollback procedures if something breaks

**File**: `/home/toastee/BioMimeticAi/docs/IMPLEMENTATION_CHECKLIST.md`

---

### 2. Update System Documentation
- [ ] Add Model Ducking to README.MD (already done!)
- [ ] Update CLAUDE.md with new components (already done!)
- [ ] Create troubleshooting guide
- [ ] Create user-facing documentation

**Files**:
- `/home/toastee/BioMimeticAi/docs/TROUBLESHOOTING.md` (new)
- `/home/toastee/BioMimeticAi/docs/USER_GUIDE_MODEL_DUCKING.md` (new)

---

### 3. Create Example Configurations
- [ ] Example .env file with model paths
- [ ] Alternative models.json for different hardware configs
- [ ] Quick-start scripts

**Files**:
- `/home/toastee/BioMimeticAi/config/.env.example`
- `/home/toastee/BioMimeticAi/config/models_low_vram.json`
- `/home/toastee/BioMimeticAi/scripts/quick_start_model_ducking.sh`

---

### 4. Code Review & Quality Checks
When you get implementations back from other LLMs:
- [ ] Review for security issues
- [ ] Check error handling
- [ ] Verify logging is comprehensive
- [ ] Test integration points
- [ ] Ensure type hints present
- [ ] Verify docstrings complete

---

### 5. Integration Testing Scripts
- [ ] Create test_model_switch.py (isolated test)
- [ ] Create test_task_classifier.py
- [ ] Create test_context_builder.py
- [ ] Create integration_test.py (full flow)

**Directory**: `/home/toastee/BioMimeticAi/tests/model_ducking/`

---

## What NOT to Duplicate

Don't work on these (delegated to other LLMs):
- ❌ LLMModelManager implementation
- ❌ TaskClassifier implementation
- ❌ UniversalContextBuilder implementation
- ❌ Discord bot integration code

Work on supporting infrastructure instead!

---

## Coordination Protocol

1. **After each delegated task completes:**
   - Review the code
   - Test independently
   - Integrate into main branch
   - Update checklist

2. **Before integration:**
   - Run linter/type checker
   - Test in isolation
   - Check for naming conflicts
   - Verify imports work

3. **Documentation sync:**
   - Update docs after each component integrates
   - Keep TEAM_DELEGATION_SUMMARY.md current
   - Track issues in GitHub/notes

---

## Timeline Estimate

If you delegate Tasks 1-3 to other LLMs (parallel):
- Day 1: You get back 3 implementations
- Day 1-2: I review, test, integrate (Tasks 1-3)
- Day 2: You delegate Task 4 (integration)
- Day 2-3: I review Task 4, create tests (Task 5)
- Day 3: You delegate Task 6 (deployment)
- Day 3-4: Final testing and deployment

**Total**: 3-4 days with good parallelization vs. 1-2 weeks solo
