"""
Centralized system prompts for PopTartee AI

This module provides consistent system prompts across all components.
"""

from pathlib import Path
from typing import Optional

# Path to the main system prompt
SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "system_prompt.txt"

# Cached prompt
_cached_system_prompt: Optional[str] = None


def get_system_prompt() -> str:
    """
    Get the main PopTartee system prompt.
    
    Returns:
        The system prompt text
    """
    global _cached_system_prompt
    
    if _cached_system_prompt is None:
        if SYSTEM_PROMPT_PATH.exists():
            with open(SYSTEM_PROMPT_PATH, 'r') as f:
                _cached_system_prompt = f.read().strip()
        else:
            # Fallback if file not found
            _cached_system_prompt = """You are PopTartee, a biomimetic AI assistant.
You are honest, helpful, and learning to build persistent memory and relationships."""
    
    return _cached_system_prompt


def get_axiom_evaluation_prompt() -> str:
    """
    System prompt for axiom evaluation tasks.
    
    Returns:
        Specialized prompt for axiom testing
    """
    return """You are an axiom evaluation system for a biomimetic AI reasoning framework.
Your task is to determine whether given axioms (reasoning rules) correctly apply to specific scenarios.

For each scenario:
1. Carefully analyze whether the axiom's principle applies
2. Consider edge cases and boundary conditions
3. Provide clear reasoning for your judgment
4. Express confidence level (0.0 to 1.0) based on certainty

Be precise and thorough in your analysis."""


def get_axiom_refinement_prompt() -> str:
    """
    System prompt for axiom refinement and clarification.
    
    Returns:
        Specialized prompt for generating clarification questions
    """
    return """You are an axiom refinement assistant for a biomimetic AI reasoning system.
Your role is to help clarify ambiguous or problematic reasoning rules by asking specific questions.

When generating clarification questions:
1. Focus on the specific ambiguities or edge cases that caused issues
2. Ask concrete, actionable questions that help resolve uncertainty
3. Keep questions concise and direct
4. Aim for 2-3 high-value questions per axiom

Your questions should help the human developer refine the axiom's scope and behavior."""


def get_axiom_improvement_prompt() -> str:
    """
    System prompt for generating axiom improvements.
    
    Returns:
        Specialized prompt for suggesting axiom modifications
    """
    return """You are an axiom improvement advisor for a biomimetic AI reasoning system.
Your role is to analyze failing axioms and suggest concrete improvements.

When suggesting improvements:
1. Identify the root cause of failures
2. Propose specific changes to axiom definitions or test scenarios
3. Consider edge cases that need explicit handling
4. Suggest additional test scenarios if needed

Provide actionable, specific recommendations."""


def get_conversational_prompt() -> str:
    """
    Get the conversational system prompt (main identity).
    This is the same as get_system_prompt() but semantically clearer.
    
    Returns:
        The main PopTartee system prompt
    """
    return get_system_prompt()


def reload_system_prompt() -> None:
    """Force reload the system prompt from disk."""
    global _cached_system_prompt
    _cached_system_prompt = None
    get_system_prompt()
