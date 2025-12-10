#!/usr/bin/env python3
"""
Memory Inspector - Non-invasive diagnostic tool for contact memory

Safely reads contact profiles and micro-tools without disrupting the running system.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.contact_memory import ContactMemory
from src.memory.episodic import EpisodicMemory

def main():
    if len(sys.argv) < 2:
        print("Usage: ./scripts/inspect_memory.py <user_id>")
        print("Example: ./scripts/inspect_memory.py 200026157335314434")
        sys.exit(1)
    
    user_id = sys.argv[1]
    
    contacts = ContactMemory()
    episodic = EpisodicMemory()
    
    print("="*80)
    print("MEMORY INSPECTION (Read-Only)")
    print("="*80)
    
    # Get contact profile
    profile = contacts.get_contact(user_id)
    
    if not profile:
        print(f"\nNo contact profile found for user_id: {user_id}")
        sys.exit(0)
    
    print(f"\n📇 CONTACT PROFILE:")
    print(f"   Username: {profile['username']}")
    print(f"   Trust Level: {profile['trust_level']:.2f}/1.0")
    print(f"   Interactions: {profile['interaction_count']}")
    print(f"   First Met: {profile['first_met']}")
    print(f"   Last Interaction: {profile['last_interaction']}")
    
    if profile.get('relationship_type'):
        print(f"   Relationship: {profile['relationship_type']}")
    
    if profile.get('communication_style'):
        print(f"   Communication Style: {profile['communication_style']}")
    
    if profile.get('preferred_topics'):
        topics = profile['preferred_topics']
        if isinstance(topics, list) and topics:
            print(f"   Preferred Topics: {', '.join(topics)}")
    
    if profile.get('personality_traits'):
        traits = profile['personality_traits']
        if isinstance(traits, list) and traits:
            print(f"   Personality Traits: {', '.join(traits)}")
    
    if profile.get('context_notes'):
        print(f"\n📝 CONTEXT NOTES:")
        notes = profile['context_notes']
        print(f"   {notes}")
    
    # Get episode count
    episodes = episodic.get_recent_episodes(user_id=user_id, limit=1000)
    print(f"\n💭 EPISODIC MEMORY:")
    print(f"   Total Episodes: {len(episodes)}")
    
    if episodes:
        print(f"   Recent conversations:")
        for ep in episodes[-3:]:
            user_msg = ep['user_message'][:60]
            bot_resp = ep['bot_response'][:60]
            print(f"     User: {user_msg}...")
            print(f"     Bot: {bot_resp}...")
            print()
    
    # Check micro-tools
    tools_dir = Path(__file__).parent.parent / "data" / "micro_tools"
    if tools_dir.exists():
        pattern = f"{user_id}_*.json"
        tool_files = list(tools_dir.glob(pattern))
        
        if tool_files:
            print(f"\n🔧 MICRO-TOOLS ({len(tool_files)} available):")
            for tool_file in tool_files:
                try:
                    with open(tool_file) as f:
                        tool = json.load(f)
                        print(f"   {tool['type']}:")
                        print(f"     Confidence: {tool['confidence']:.2f}")
                        print(f"     Use Count: {tool['use_count']}")
                        print(f"     Data: {tool['data']}")
                except:
                    pass
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
