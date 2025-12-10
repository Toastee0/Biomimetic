#!/usr/bin/env python3
"""
Contact Learning Cortex

Primary Goal: Learn about people through conversation analysis.
Continuously analyzes recent conversations to extract insights about contacts.

This cortex demonstrates the principle:
- Expensive inference → Create tools for future efficiency
- Learn patterns → Encode as reusable micro-tools
- Memory consolidation → Actionable knowledge

RTOS constraints: 5 minute timeout
"""

import signal
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.memory.episodic import EpisodicMemory
from src.memory.contact_memory import ContactMemory
from src.daemon.textgen_client import TextGenClient
from src.core.dynamic_prompts import get_learning_analysis_prompt

TIMEOUT_SECONDS = 300  # 5 minutes
STATE_FILE = Path(__file__).parent.parent.parent / "data/cortex_state/contact_learning.json"
LOG_FILE = Path(__file__).parent.parent.parent / "logs/contact_learning.log"

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

def save_state(status, duration, contacts_updated, insights=None, errors=None):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_run": int(time.time()),
        "status": status,
        "duration_seconds": round(duration, 2),
        "contacts_updated": contacts_updated,
        "insights_generated": len(insights) if insights else 0,
        "errors": errors or [],
        "next_scheduled": int(time.time()) + 1800  # +30 minutes
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def analyze_conversation_patterns(episodes: List[Dict], textgen: TextGenClient) -> Dict[str, any]:
    """Use LLM to extract insights from conversation history
    
    This is the EXPENSIVE inference step that we'll eventually replace
    with micro-tools as patterns become clear.
    """
    if not episodes:
        return {}
    
    # Build conversation summary for analysis
    conversation_text = ""
    for ep in episodes[-10:]:  # Last 10 exchanges
        conversation_text += f"User: {ep['user_message']}\n"
        conversation_text += f"Bot: {ep['bot_response'][:200]}\n\n"
    
    analysis_prompt = f"""Analyze this conversation history and extract key insights about the user:

{conversation_text}

Provide a JSON response with:
{{
  "communication_style": "brief description of how they communicate",
  "preferred_topics": ["topic1", "topic2", "topic3"],
  "personality_traits": ["trait1", "trait2", "trait3"],
  "relationship_type": "technical_collaborator|friend|colleague|other",
  "technical_level": "beginner|intermediate|advanced|expert",
  "interests": ["interest1", "interest2"],
  "context_summary": "2-3 sentence summary of what we've discussed",
  "specific_facts": {{
    "name_details": "any mentioned middle names, nicknames, preferences about name",
    "personal_preferences": "favorite colors, foods, activities, etc mentioned explicitly",
    "background": "any biographical details like location, work, education",
    "goals": "any stated goals or projects they're working on"
  }}
}}

Be specific and evidence-based. Extract EXACT facts mentioned, not generalizations.
For specific_facts, only include fields where explicit information was given."""
    
    system_prompt = get_learning_analysis_prompt()
    
    try:
        response = textgen.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temp for analysis
            max_tokens=500
        )
        
        if not response:
            return {}
        
        # Try to extract JSON from response
        # LLM might wrap it in markdown code blocks
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        
        insights = json.loads(json_str)
        return insights
        
    except Exception as e:
        log(f"[ERROR] LLM analysis failed: {e}")
        return {}

def create_micro_tool(insight_type: str, data: any, user_id: str) -> bool:
    """Create a micro-tool (cached knowledge) from expensive inference
    
    Instead of running LLM inference every time, we encode learned patterns
    as simple lookup tools. This is the biomimetic principle:
    - Slow learning → Fast recall
    - Expensive inference → Cheap lookup
    """
    tools_dir = Path(__file__).parent.parent.parent / "data/micro_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    
    tool_file = tools_dir / f"{user_id}_{insight_type}.json"
    
    tool_data = {
        "type": insight_type,
        "user_id": user_id,
        "data": data,
        "created_at": int(time.time()),
        "last_used": None,
        "use_count": 0,
        "confidence": 0.7  # Initial confidence
    }
    
    try:
        with open(tool_file, "w") as f:
            json.dump(tool_data, f, indent=2)
        log(f"[TOOL CREATED] {insight_type} for user {user_id}")
        return True
    except Exception as e:
        log(f"[ERROR] Failed to create tool: {e}")
        return False

def main():
    global current_phase
    
    log("="*80)
    log("[START] Contact learning cortex - GOAL: Learn about DM contacts")
    log("[PRINCIPLE] Expensive inference → Create tools for future efficiency")
    
    errors = []
    contacts_updated = 0
    all_insights = []
    
    try:
        current_phase = "initialization"
        episodic = EpisodicMemory()
        contacts = ContactMemory()
        textgen = TextGenClient()
        
        current_phase = "fetching_contacts"
        # Get all contacts we've interacted with
        contact_list = contacts.list_contacts(limit=50)
        
        if not contact_list:
            log("[SKIP] No contacts found yet")
            save_state("success", time.time() - start_time, 0)
            return
        
        log(f"[CONTACTS] Found {len(contact_list)} contacts to analyze")
        
        current_phase = "analyzing_conversations"
        for contact in contact_list:
            try:
                user_id = contact['user_id']
                username = contact['username']
                
                log(f"[ANALYZE] Learning about {username} (ID: {user_id})")
                
                # Get recent conversation history for this user
                episodes = episodic.get_recent_episodes(
                    user_id=user_id,
                    limit=20
                )
                
                if len(episodes) < 3:
                    log(f"[SKIP] Not enough conversation history for {username} ({len(episodes)} episodes)")
                    continue
                
                log(f"[INFERENCE] Running LLM analysis on {len(episodes)} episodes (expensive)")
                
                # This is the EXPENSIVE part - use LLM to extract insights
                insights = analyze_conversation_patterns(episodes, textgen)
                
                if not insights:
                    log(f"[SKIP] No insights extracted for {username}")
                    continue
                
                log(f"[INSIGHTS] Extracted: {list(insights.keys())}")
                
                # Log actual content for visibility
                for key, value in insights.items():
                    if isinstance(value, list):
                        log(f"  {key}: {value}")
                    else:
                        log(f"  {key}: {value}")
                
                all_insights.append({"user_id": user_id, "insights": insights})
                
                # Update contact profile with insights
                current_phase = f"updating_contact_{user_id}"
                
                update_data = {}
                if insights.get('communication_style'):
                    update_data['communication_style'] = insights['communication_style']
                
                if insights.get('preferred_topics'):
                    update_data['preferred_topics'] = insights['preferred_topics']
                
                if insights.get('personality_traits'):
                    update_data['personality_traits'] = insights['personality_traits']
                
                if insights.get('relationship_type'):
                    update_data['relationship_type'] = insights['relationship_type']
                
                if update_data:
                    success = contacts.update_contact(user_id, **update_data)
                    if success:
                        log(f"[UPDATE] Updated profile for {username}")
                        contacts_updated += 1
                    else:
                        log(f"[ERROR] Failed to update profile for {username}")
                
                # Add context summary as note
                if insights.get('context_summary'):
                    contacts.add_context_note(
                        user_id,
                        f"Learning cycle: {insights['context_summary']}"
                    )
                
                # Create micro-tools from learned patterns
                current_phase = f"creating_tools_{user_id}"
                
                if insights.get('preferred_topics'):
                    create_micro_tool("preferred_topics", insights['preferred_topics'], user_id)
                
                if insights.get('technical_level'):
                    create_micro_tool("technical_level", insights['technical_level'], user_id)
                
                # Store specific facts as discrete tools
                if insights.get('specific_facts'):
                    facts = insights['specific_facts']
                    create_micro_tool("specific_facts", facts, user_id)
                    log(f"[FACTS] Stored specific facts: {list(facts.keys())}")
                
                # Adjust trust based on interaction quality
                if insights.get('relationship_type') in ['technical_collaborator', 'friend']:
                    new_trust = contacts.adjust_trust(user_id, +0.1)
                    log(f"[TRUST] Increased trust for {username} to {new_trust:.2f}")
                
                log(f"[COMPLETE] Finished learning about {username}")
                
            except Exception as e:
                error_msg = f"Failed to analyze contact {contact.get('username')}: {e}"
                log(f"[ERROR] {error_msg}")
                errors.append(error_msg)
        
        current_phase = "complete"
        duration = time.time() - start_time
        log(f"[SUCCESS] Updated {contacts_updated} contacts in {duration:.2f}s")
        log(f"[INSIGHTS] Generated {len(all_insights)} insight sets")
        log(f"[PRINCIPLE] Next runs will use cached tools instead of expensive inference")
        
        save_state("success", duration, contacts_updated, all_insights, errors if errors else None)
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Fatal error in phase {current_phase}: {e}"
        log(f"[FATAL] {error_msg}")
        import traceback
        log(traceback.format_exc())
        errors.append(error_msg)
        save_state("error", duration, contacts_updated, all_insights, errors)
        sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    
    try:
        main()
    finally:
        signal.alarm(0)
