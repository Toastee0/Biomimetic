"""
Dynamic Prompt Builder - Integrates all cortexes into context-aware system prompts

This creates "living" prompts that reflect the AI's current state:
- Memory systems (episodic, contact)
- Axiom knowledge
- Learned micro-tools
- Conversation context

The AI should know what it knows.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

# Import memory systems
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.memory.episodic import EpisodicMemory
from src.memory.contact_memory import ContactMemory


class DynamicPromptBuilder:
    """Builds context-aware system prompts that reflect AI's current capabilities"""
    
    def __init__(self):
        self.base_prompt_path = Path(__file__).parent.parent.parent / "config" / "system_prompt.txt"
        self.episodic = EpisodicMemory()
        self.contacts = ContactMemory()
        self.micro_tools_dir = Path(__file__).parent.parent.parent / "data" / "micro_tools"
    
    def load_base_prompt(self) -> str:
        """Load static base prompt"""
        if self.base_prompt_path.exists():
            with open(self.base_prompt_path, 'r') as f:
                return f.read().strip()
        return "You are PopTartee, a biomimetic AI assistant."
    
    def get_memory_status(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current memory system status"""
        status = {
            "episodic_enabled": True,
            "contact_memory_enabled": True,
            "total_episodes": 0,
            "total_contacts": 0,
            "current_contact": None,
            "interaction_history": 0
        }
        
        try:
            # Get episode count
            all_episodes = self.episodic.get_recent_episodes(limit=1000)
            status["total_episodes"] = len(all_episodes)
            
            # Get contact count
            all_contacts = self.contacts.list_contacts(limit=1000)
            status["total_contacts"] = len(all_contacts)
            
            # Get current contact info
            if user_id:
                contact = self.contacts.get_contact(user_id)
                if contact:
                    status["current_contact"] = {
                        "username": contact.get("username"),
                        "trust_level": contact.get("trust_level"),
                        "interactions": contact.get("interaction_count"),
                        "relationship": contact.get("relationship_type"),
                        "style": contact.get("communication_style"),
                        "topics": contact.get("preferred_topics", [])
                    }
                    status["interaction_history"] = contact.get("interaction_count", 0)
        except Exception as e:
            print(f"[PROMPT BUILDER] Error getting memory status: {e}")
        
        return status
    
    def get_available_micro_tools(self, user_id: Optional[str] = None) -> list:
        """Get list of available micro-tools"""
        tools = []
        
        if not self.micro_tools_dir.exists():
            return tools
        
        try:
            # Get tools for specific user if provided
            if user_id:
                pattern = f"{user_id}_*.json"
                tool_files = list(self.micro_tools_dir.glob(pattern))
            else:
                tool_files = list(self.micro_tools_dir.glob("*.json"))
            
            for tool_file in tool_files:
                try:
                    with open(tool_file) as f:
                        tool = json.load(f)
                        tools.append({
                            "type": tool.get("type"),
                            "confidence": tool.get("confidence", 0.0),
                            "use_count": tool.get("use_count", 0)
                        })
                except:
                    pass
        except Exception as e:
            print(f"[PROMPT BUILDER] Error loading micro-tools: {e}")
        
        return tools
    
    def build_conversational_prompt(self, user_id: Optional[str] = None) -> str:
        """Build dynamic system prompt for conversation
        
        Integrates:
        - Base identity/values
        - Current memory state
        - Contact-specific knowledge
        - Available tools
        """
        base = self.load_base_prompt()
        
        # Get memory status
        memory = self.get_memory_status(user_id)
        tools = self.get_available_micro_tools(user_id)
        
        # Build dynamic enhancement
        enhancement = "\n\n## Current Capabilities & State\n"
        
        # Memory systems
        enhancement += f"\n**Memory Systems:**\n"
        enhancement += f"- Episodic Memory: Active ({memory['total_episodes']} stored conversations)\n"
        enhancement += f"- Contact Memory: Active ({memory['total_contacts']} known individuals)\n"
        
        if memory["current_contact"]:
            contact = memory["current_contact"]
            enhancement += f"\n**Current Conversation Partner:**\n"
            enhancement += f"- Name: {contact['username']}\n"
            enhancement += f"- Trust Level: {contact['trust_level']:.2f}/1.0\n"
            enhancement += f"- Previous Interactions: {contact['interactions']}\n"
            
            if contact.get('relationship'):
                enhancement += f"- Relationship: {contact['relationship']}\n"
            
            if contact.get('style'):
                enhancement += f"- Communication Style: {contact['style']}\n"
            
            if contact.get('topics') and len(contact['topics']) > 0:
                enhancement += f"- Known Interests: {', '.join(contact['topics'][:5])}\n"
        
        # Micro-tools
        if tools:
            enhancement += f"\n**Learned Patterns (Micro-tools):**\n"
            enhancement += f"- {len(tools)} pattern recognition tools available\n"
            tool_types = set(t['type'] for t in tools)
            enhancement += f"- Types: {', '.join(tool_types)}\n"
        
        # Learning status
        enhancement += f"\n**Continuous Learning:**\n"
        enhancement += f"- Contact Learning cortex runs every 30 minutes\n"
        enhancement += f"- Episodic consolidation runs every 10 minutes\n"
        enhancement += f"- Building persistent understanding of interactions\n"
        
        # Combine base + enhancement
        full_prompt = base + enhancement
        
        return full_prompt
    
    def build_axiom_evaluation_prompt(self) -> str:
        """Build prompt for axiom evaluation tasks"""
        return """You are an axiom evaluation system for a biomimetic AI reasoning framework.
Your task is to determine whether given axioms (reasoning rules) correctly apply to specific scenarios.

For each scenario:
1. Carefully analyze whether the axiom's principle applies
2. Consider edge cases and boundary conditions
3. Provide clear reasoning for your judgment
4. Express confidence level (0.0 to 1.0) based on certainty

Be precise and thorough in your analysis."""
    
    def build_learning_analysis_prompt(self) -> str:
        """Build prompt for analyzing conversations to learn about contacts"""
        return """You are an empathetic observer analyzing conversations to understand people better.
Focus on genuine insights, not assumptions. Be concise and factual.

Extract specific, evidence-based insights about communication patterns, interests, and personality.
Only include what you can directly infer from the conversation."""


# Global instance
_builder = None

def get_prompt_builder() -> DynamicPromptBuilder:
    """Get singleton prompt builder instance"""
    global _builder
    if _builder is None:
        _builder = DynamicPromptBuilder()
    return _builder


def get_conversational_prompt(user_id: Optional[str] = None) -> str:
    """Get dynamic conversational prompt with full context
    
    Args:
        user_id: Discord user ID to get contact-specific context
        
    Returns:
        Context-aware system prompt
    """
    builder = get_prompt_builder()
    return builder.build_conversational_prompt(user_id)


def get_axiom_evaluation_prompt() -> str:
    """Get prompt for axiom evaluation"""
    builder = get_prompt_builder()
    return builder.build_axiom_evaluation_prompt()


def get_learning_analysis_prompt() -> str:
    """Get prompt for learning analysis"""
    builder = get_prompt_builder()
    return builder.build_learning_analysis_prompt()
