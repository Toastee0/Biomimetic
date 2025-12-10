#!/usr/bin/env python3
"""
Episodic Memory Consolidation Cortex

Runs every 10 minutes to process recent conversation episodes.
Calculates salience, updates contact profiles, marks as consolidated.

RTOS constraints: 60 second timeout
"""

import signal
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.memory.episodic import EpisodicMemory
from src.memory.contact_memory import ContactMemory

TIMEOUT_SECONDS = 60
STATE_FILE = Path(__file__).parent.parent.parent / "data/cortex_state/episodic_consolidation.json"
LOG_FILE = Path(__file__).parent.parent.parent / "logs/episodic_consolidation.log"

start_time = time.time()
current_phase = "initialization"

def timeout_handler(signum, frame):
    """Handle timeout - log state and exit"""
    elapsed = time.time() - start_time
    log(f"[TIMEOUT] Exceeded {TIMEOUT_SECONDS}s limit at phase: {current_phase}")
    log(f"[TIMEOUT] Elapsed: {elapsed:.2f}s")
    save_state("timeout", elapsed, 0)
    sys.exit(124)

def log(msg):
    """Write to log file with timestamp"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

def save_state(status, duration, items_processed, errors=None):
    """Save cortex state for monitoring"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_run": int(time.time()),
        "status": status,
        "duration_seconds": round(duration, 2),
        "items_processed": items_processed,
        "errors": errors or [],
        "next_scheduled": int(time.time()) + 600  # +10 minutes
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def calculate_salience(episode, contact):
    """Calculate salience score for an episode
    
    Factors:
    - User trust level (higher trust = higher salience)
    - Message length (longer = more invested)
    - Emotional markers (questions, exclamations)
    - Topic relevance (axiom-related keywords)
    """
    score = 0.5  # Base
    
    # Trust factor (0.0 - 1.0 → adds 0.0 - 0.3)
    if contact:
        trust = contact.get('trust_level', 0.5)
        score += trust * 0.3
    
    # Length factor
    msg_len = len(episode.get('user_message', ''))
    if msg_len > 200:
        score += 0.1
    elif msg_len > 500:
        score += 0.2
    
    # Emotional markers
    user_msg = episode.get('user_message', '')
    if '?' in user_msg:
        score += 0.1  # Questions are important
    if '!' in user_msg:
        score += 0.05  # Emphasis
    
    # Axiom-related keywords
    axiom_keywords = ['axiom', 'test', 'learn', 'clarif', 'truth', 'value']
    if any(kw in user_msg.lower() for kw in axiom_keywords):
        score += 0.15
    
    return min(1.0, score)  # Clamp to 1.0

def main():
    global current_phase
    
    log("="*80)
    log("[START] Episodic consolidation cortex")
    
    errors = []
    items_processed = 0
    
    try:
        current_phase = "initialization"
        episodic = EpisodicMemory()
        contacts = ContactMemory()
        
        current_phase = "fetching_episodes"
        # Get unconsolidated episodes
        episodes = episodic.get_recent_episodes(
            limit=100,
            unconsolidated_only=True
        )
        
        log(f"[FETCH] Found {len(episodes)} unconsolidated episodes")
        
        if not episodes:
            log("[SKIP] No episodes to consolidate")
            save_state("success", time.time() - start_time, 0)
            return
        
        current_phase = "processing_episodes"
        for episode in episodes:
            try:
                user_id = episode.get('user_id')
                episode_id = episode.get('episode_id')
                
                # Get contact profile
                contact = contacts.get_contact(user_id)
                
                # Calculate salience
                salience = calculate_salience(episode, contact)
                
                # Update episode salience
                episodic.update_salience(episode_id, salience)
                
                # Mark as consolidated
                episodic.mark_consolidated(episode_id)
                
                items_processed += 1
                
                if items_processed % 10 == 0:
                    log(f"[PROGRESS] Processed {items_processed}/{len(episodes)} episodes")
                
            except Exception as e:
                error_msg = f"Failed to process episode {episode.get('episode_id')}: {e}"
                log(f"[ERROR] {error_msg}")
                errors.append(error_msg)
        
        current_phase = "complete"
        duration = time.time() - start_time
        log(f"[SUCCESS] Consolidated {items_processed} episodes in {duration:.2f}s")
        
        save_state("success", duration, items_processed, errors if errors else None)
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Fatal error in phase {current_phase}: {e}"
        log(f"[FATAL] {error_msg}")
        errors.append(error_msg)
        save_state("error", duration, items_processed, errors)
        sys.exit(1)

if __name__ == "__main__":
    # Set timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    
    try:
        main()
    finally:
        signal.alarm(0)  # Cancel alarm
