#!/usr/bin/env python3
"""
Axiom Spot Check Cortex

Runs every 15 minutes for quick axiom health checks.
Selects 3-5 random axioms and runs lightweight tests.

RTOS constraints: 3 minute timeout
"""

import signal
import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tensor_axiom.axiom_library import AxiomLibrary
from src.tensor_axiom.self_training_loop import SelfTrainingLoop

TIMEOUT_SECONDS = 180  # 3 minutes
STATE_FILE = Path(__file__).parent.parent.parent / "data/cortex_state/axiom_spot_check.json"
LOG_FILE = Path(__file__).parent.parent.parent / "logs/axiom_spot_check.log"

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

def save_state(status, duration, items_processed, anomalies=None, errors=None):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_run": int(time.time()),
        "status": status,
        "duration_seconds": round(duration, 2),
        "items_processed": items_processed,
        "anomalies_found": anomalies or [],
        "errors": errors or [],
        "next_scheduled": int(time.time()) + 900  # +15 minutes
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def main():
    global current_phase
    
    log("="*80)
    log("[START] Axiom spot check cortex")
    
    errors = []
    anomalies = []
    items_processed = 0
    
    try:
        current_phase = "initialization"
        library = AxiomLibrary("data/axioms/base_axioms.json")
        library.load()
        trainer = SelfTrainingLoop()
        
        all_axioms = library.list_axioms()
        if not all_axioms:
            log("[SKIP] No axioms in library")
            save_state("success", time.time() - start_time, 0)
            return
        
        # Select 3-5 random axioms
        sample_size = min(5, max(3, len(all_axioms) // 5))
        selected = random.sample(all_axioms, sample_size)
        
        log(f"[SAMPLE] Testing {sample_size} axioms: {[a['id'] for a in selected]}")
        
        current_phase = "testing"
        for axiom in selected:
            try:
                axiom_id = axiom['id']
                test_scenarios = axiom.get('test_scenarios', [])
                
                if not test_scenarios:
                    log(f"[SKIP] {axiom_id} has no test scenarios")
                    continue
                
                # Test only first 2 scenarios for speed
                quick_scenarios = test_scenarios[:2]
                
                failures = 0
                low_confidence = 0
                
                for scenario in quick_scenarios:
                    success, confidence, reasoning = trainer.evaluate_axiom_with_llm(
                        axiom, 
                        scenario
                    )
                    
                    if not success:
                        failures += 1
                    elif confidence < 0.6:
                        low_confidence += 1
                
                # Flag if problems detected
                if failures > 0 or low_confidence >= 2:
                    anomaly = {
                        "axiom_id": axiom_id,
                        "failures": failures,
                        "low_confidence": low_confidence,
                        "timestamp": int(time.time())
                    }
                    anomalies.append(anomaly)
                    log(f"[ANOMALY] {axiom_id}: {failures} failures, {low_confidence} low confidence")
                
                items_processed += 1
                
            except Exception as e:
                error_msg = f"Failed to test {axiom.get('id')}: {e}"
                log(f"[ERROR] {error_msg}")
                errors.append(error_msg)
        
        current_phase = "complete"
        duration = time.time() - start_time
        log(f"[SUCCESS] Tested {items_processed} axioms in {duration:.2f}s")
        log(f"[RESULT] Found {len(anomalies)} anomalies")
        
        save_state("success", duration, items_processed, anomalies, errors if errors else None)
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Fatal error in phase {current_phase}: {e}"
        log(f"[FATAL] {error_msg}")
        errors.append(error_msg)
        save_state("error", duration, items_processed, anomalies, errors)
        sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    
    try:
        main()
    finally:
        signal.alarm(0)
