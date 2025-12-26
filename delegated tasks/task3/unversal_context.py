# /home/toastee/BioMimeticAi/src/core/universal_context.py
"""
UniversalContextBuilder - LLM-agnostic context window generator

Aggregates:
- System identity (static core principles)
- Current operating mode
- ContactMemory profile (user-specific knowledge)
- EpisodicMemory recent conversation history
- Task-specific context (coding docs/code, vision events, etc.)

Output: Clean markdown-formatted context suitable for any LLM backend.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Memory systems
from src.memory.episodic import EpisodicMemory
from src.memory.contact_memory import ContactMemory


class UniversalContextBuilder:
    """
    Builds LLM-agnostic context windows that work with any backend.

    The resulting context is structured markdown that can be prepended to the
    user message for any LLM (local GGUF, vLLM, Claude, OpenAI, etc.).
    """

    def __init__(self):
        self.episodic_memory = EpisodicMemory()
        self.contact_memory = ContactMemory()

        # Static system identity - can be extended or loaded from file if desired
        self.system_identity = (
            "You are PopTartee, a biomimetic AI assistant inspired by human cognition.\n"
            "Core principles:\n"
            "- Reason from first principles whenever possible\n"
            "- Use distributed, specialised cortexes for different tasks\n"
            "- Maintain persistent memory of interactions and individuals\n"
            "- Continuously learn and consolidate knowledge\n"
            "- Be transparent about your current mode and capabilities"
        )

    def _system_identity(self) -> str:
        """Static system identity section"""
        return "# SYSTEM IDENTITY\n" + self.system_identity + "\n"

    def _format_user_context(self, profile: Optional[Dict[str, Any]]) -> str:
        """Format contact memory profile into user context section"""
        if not profile:
            return "# USER CONTEXT\nNew or unknown contact.\n"

        lines = ["# USER CONTEXT"]
        lines.append(f"Contact: {profile.get('username', 'Unknown')}")
        if profile.get('display_name'):
            lines.append(f"Display name: {profile['display_name']}")

        lines.append(f"Trust level: {profile.get('trust_level', 0.5):.2f}/1.00")
        lines.append(f"Total interactions: {profile.get('interaction_count', 1)}")

        if profile.get('relationship_type'):
            lines.append(f"Relationship: {profile['relationship_type']}")

        if profile.get('communication_style'):
            lines.append(f"Communication style: {profile['communication_style']}")

        topics = profile.get('preferred_topics', [])
        if topics:
            if isinstance(topics, list):
                topic_str = ", ".join(topics[:5])
            else:
                topic_str = str(topics)
            lines.append(f"Known interests: {topic_str}")

        traits = profile.get('personality_traits', [])
        if traits:
            if isinstance(traits, list):
                trait_str = ", ".join(traits[:3])
            else:
                trait_str = str(traits)
            lines.append(f"Personality traits: {trait_str}")

        return "\n".join(lines) + "\n"

    def _format_episodes(self, episodes: List[Dict[str, Any]]) -> str:
        """Format recent episodic memories"""
        if not episodes:
            return "# MEMORY CONTEXT\nNo recent conversation history.\n"

        lines = ["# MEMORY CONTEXT", "Recent interactions (most recent first):"]
        for ep in episodes[:5]:  # Limit to avoid token overflow
            ts = datetime.fromtimestamp(ep['timestamp']).strftime("%Y-%m-%d %H:%M")
            user_msg = ep['user_message'][:100].replace("\n", " ")
            bot_resp = ep['bot_response'][:80].replace("\n", " ")
            lines.append(f"- [{ts}] User: {user_msg}...")
            lines.append(f"          Assistant: {bot_resp}...")

        return "\n".join(lines) + "\n"

    def _coding_context(self, user_message: str) -> str:
        """
        Placeholder for coding-specific context.
        In a full implementation this would load relevant documentation,
        search the codebase, include recent commits, etc.
        """
        # For now, provide a clear section header with a note
        return (
            "# TASK-SPECIFIC CONTEXT (Coding Mode)\n"
            "You are in specialised coding mode.\n"
            "Relevant project documentation and codebase files would be inserted here.\n"
            "Current request involves coding, implementation, debugging, or architecture.\n\n"
        )

    def _vision_context(self, user_message: str) -> str:
        """
        Placeholder for vision-specific context.
        In a full implementation this would include recent camera events,
        buffered frames, scene descriptions, etc.
        """
        return (
            "# TASK-SPECIFIC CONTEXT (Vision Mode)\n"
            "You are in vision analysis mode.\n"
            "Recent camera feed events and scene descriptions would be inserted here.\n\n"
        )

    def _conversation_context(self) -> str:
        """Default conversation mode context"""
        return (
            "# TASK-SPECIFIC CONTEXT (Conversation Mode)\n"
            "General conversation. Use contact knowledge and episodic memory naturally.\n\n"
        )

    def _reasoning_context(self) -> str:
        """Deep reasoning mode context"""
        return (
            "# TASK-SPECIFIC CONTEXT (Deep Reasoning Mode)\n"
            "Complex or long-form reasoning requested. Think step-by-step and be thorough.\n\n"
        )

    def build_context(
        self,
        user_message: str,
        contact_id: str,
        mode: str = "conversation",
        recent_episodes: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build the full universal context window.

        Args:
            user_message: The raw message from the user
            contact_id: Discord user ID (string)
            mode: One of 'conversation', 'coding', 'vision', 'reasoning'
            recent_episodes: Optional pre-fetched episodes (for performance)

        Returns:
            Complete markdown context string ready for any LLM
        """
        context_parts = []

        # 1. System identity
        context_parts.append(self._system_identity())

        # 2. Current mode
        context_parts.append(f"# CURRENT MODE\n{mode}\n")

        # 3. User / contact context
        profile = self.contact_memory.get_contact(contact_id)
        context_parts.append(self._format_user_context(profile))

        # 4. Memory context (episodic)
        if recent_episodes is None:
            recent_episodes = self.episodic_memory.get_recent_episodes(
                limit=10, user_id=contact_id
            )
        context_parts.append(self._format_episodes(recent_episodes))

        # 5. Task-specific context
        if mode == "coding":
            context_parts.append(self._coding_context(user_message))
        elif mode == "vision":
            context_parts.append(self._vision_context(user_message))
        elif mode == "reasoning":
            context_parts.append(self._reasoning_context())
        else:  # conversation
            context_parts.append(self._conversation_context())

        # 6. User request (final section)
        context_parts.append(f"# USER REQUEST\n{user_message}\n")

        return "\n".join(context_parts)


# Example usage / testing
if __name__ == "__main__":
    builder = UniversalContextBuilder()
    sample_context = builder.build_context(
        user_message="Can you help me implement a ring buffer in Python?",
        contact_id="200026157335314434",
        mode="coding"
    )
    print(sample_context)