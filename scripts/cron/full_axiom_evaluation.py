#!/usr/bin/env python3
"""
Full Axiom Evaluation Cortex

Runs every 4 hours for comprehensive axiom testing.
Identifies problematic axioms, generates clarification questions,
and adds items to review queue.

RTOS constraints: 15 minute timeout
"""

import signal
import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tensor_axiom.self_training_loop import SelfTrainingLoop

TIMEOUT_SECONDS = 900  # 15 minutes
STATE_FILE = Path(__file__).parent.parent.parent / "data/cortex_state/axiom_evaluation.json"
LOG_FILE = Path(__file__).parent.parent.parent / "logs/axiom_evaluation.log"

start_time = time.time()
current_phase = "initialization"

def timeout_handler(signum, frame):
    elapsed = time.time() - start_time
    log(f"[TIMEOUT] Exceeded {TIMEOUT_SECONDS}s limit at phase: {current_phase}")
    save_state("timeout", elapsed, 0)
    sys.exit(124)

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

def save_state(status, duration, items_processed, problematic_count=0, errors=None):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_run": int(time.time()),
        "status": status,
        "duration_seconds": round(duration, 2),
        "axioms_tested": items_processed,
        "problematic_found": problematic_count,
        "errors": errors or [],
        "next_scheduled": int(time.time()) + 14400  # +4 hours
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def main():
    global current_phase
    
    log("="*80)
    log("[START] Full axiom evaluation cortex")
    
    errors = []
    items_processed = 0
    problematic_count = 0
    
    try:
        current_phase = "initialization"
        trainer = SelfTrainingLoop()
        
        current_phase = "testing_all"
        log("[TEST] Running full axiom library test...")
        results = trainer.test_all_axioms()
        
        if not results:
            log("[SKIP] No axioms tested")
            save_state("success", time.time() - start_time, 0)
            return
        
        items_processed = len(results)
        log(f"[COMPLETE] Tested {items_processed} axioms")
        
        current_phase = "identifying_problems"
        log("[ANALYZE] Identifying problematic axioms...")
        problematic = trainer.identify_problematic_axioms(results)
        
        problematic_count = len(problematic)
        log(f"[RESULT] Found {problematic_count} problematic axioms")
        
        if problematic_count > 0:
            log("[DETAIL] Problematic axioms:")
            for item in problematic:
                log(f"  - {item['axiom_id']}: {item['reason']}")
        
        current_phase = "complete"
        duration = time.time() - start_time
        log(f"[SUCCESS] Evaluation complete in {duration:.2f}s")
        
        save_state("success", duration, items_processed, problematic_count, errors if errors else None)
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Fatal error in phase {current_phase}: {e}"
        log(f"[FATAL] {error_msg}")
        errors.append(error_msg)
        save_state("error", duration, items_processed, problematic_count, errors)
        sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    
    try:
        main()
    finally:
        signal.alarm(0)
