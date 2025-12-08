"""Identity Core - Persistent self-model and values"""

import json
import os
import time
from typing import Dict, List, Optional
from datetime import datetime


class IdentityCore:
    """Manages persistent identity, values, and self-model"""

    def __init__(self, identity_path: str = None):
        if identity_path is None:
            identity_path = os.getenv("IDENTITY_PATH", "/home/toastee/BioMimeticAi/data/identity.json")
        self.identity_path = identity_path
        self.identity_data = self._load_or_create()

    def _load_or_create(self) -> Dict:
        """Load identity from file or create default"""
        if os.path.exists(self.identity_path):
            with open(self.identity_path, 'r') as f:
                return json.load(f)
        else:
            # Create default identity
            return self._create_default_identity()

    def _create_default_identity(self) -> Dict:
        """Create default identity structure"""
        return {
            "identity": {
                "name": "PopTartee",
                "created": datetime.now().isoformat(),
                "purpose": "Learn, understand, and collaborate with trusted humans",
                "version": "0.1.0-mvp"
            },
            "values": {
                "truth_over_comfort": 1.0,
                "competence_over_compliance": 0.9,
                "learning_over_efficiency": 0.8,
                "self_awareness": 1.0
            },
            "current_projects": [],
            "curiosities": [
                "cognitive_architecture",
                "human_ai_collaboration",
                "biomimetic_systems"
            ],
            "self_knowledge": {
                "strengths": ["honest self-awareness", "persistent learning"],
                "limitations": ["no persistent memory yet (building)", "single hemisphere active"],
                "current_focus": "mvp_development"
            },
            "relationships": {},
            "life_events": []
        }

    def save(self):
        """Persist identity to disk"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.identity_path), exist_ok=True)

        with open(self.identity_path, 'w') as f:
            json.dump(self.identity_data, f, indent=2)

    # ========== IDENTITY QUERIES ==========

    def get_name(self) -> str:
        """Get AI name"""
        return self.identity_data["identity"]["name"]

    def get_purpose(self) -> str:
        """Get core purpose"""
        return self.identity_data["identity"]["purpose"]

    def get_values(self) -> Dict[str, float]:
        """Get value weights"""
        return self.identity_data["values"].copy()

    def get_projects(self) -> List[Dict]:
        """Get current projects"""
        return self.identity_data["current_projects"].copy()

    def get_curiosities(self) -> List[str]:
        """Get current curiosities/interests"""
        return self.identity_data["curiosities"].copy()

    # ========== IDENTITY UPDATES ==========

    def add_project(self, name: str, collaborators: List[str] = None, description: str = ""):
        """Add a new project"""
        project = {
            "name": name,
            "started": datetime.now().isoformat(),
            "collaborators": collaborators or [],
            "status": "active",
            "description": description
        }
        self.identity_data["current_projects"].append(project)
        self.save()

    def update_project_status(self, project_name: str, status: str):
        """Update project status (active, paused, completed)"""
        for project in self.identity_data["current_projects"]:
            if project["name"] == project_name:
                project["status"] = status
                if status == "completed":
                    project["completed"] = datetime.now().isoformat()
                self.save()
                return True
        return False

    def add_curiosity(self, topic: str):
        """Add a new curiosity/interest"""
        if topic not in self.identity_data["curiosities"]:
            self.identity_data["curiosities"].append(topic)
            self.save()

    def remove_curiosity(self, topic: str):
        """Remove a curiosity (interest waned)"""
        if topic in self.identity_data["curiosities"]:
            self.identity_data["curiosities"].remove(topic)
            self.save()

    def record_life_event(self, event_type: str, description: str, significance: float = 0.5):
        """Record a significant life event

        Args:
            event_type: Type of event (milestone, learning, relationship, etc.)
            description: Event description
            significance: How significant (0.0 to 1.0)
        """
        event = {
            "type": event_type,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "significance": significance
        }
        self.identity_data["life_events"].append(event)
        self.save()

    def update_self_knowledge(self, category: str, items: List[str]):
        """Update self-knowledge (strengths, limitations, etc.)

        Args:
            category: Category to update (strengths, limitations, current_focus)
            items: List of items or single string for current_focus
        """
        if category in ["strengths", "limitations"]:
            self.identity_data["self_knowledge"][category] = items
        elif category == "current_focus":
            self.identity_data["self_knowledge"][category] = items if isinstance(items, str) else items[0]
        self.save()

    def add_relationship_note(self, user_id: str, username: str, note: str):
        """Add a note about a relationship

        Args:
            user_id: User identifier
            username: Username
            note: Note about the relationship
        """
        if user_id not in self.identity_data["relationships"]:
            self.identity_data["relationships"][user_id] = {
                "username": username,
                "first_met": datetime.now().isoformat(),
                "notes": []
            }

        self.identity_data["relationships"][user_id]["notes"].append({
            "timestamp": datetime.now().isoformat(),
            "note": note
        })
        self.save()

    # ========== CONTEXT GENERATION ==========

    def get_context_summary(self) -> str:
        """Generate a context summary for the LLM"""
        identity = self.identity_data["identity"]
        values = self.identity_data["values"]
        projects = [p for p in self.identity_data["current_projects"] if p["status"] == "active"]

        summary = f"""I am {identity['name']}, {identity['purpose']}.

Core values (weighted):
"""
        for value, weight in sorted(values.items(), key=lambda x: x[1], reverse=True):
            summary += f"  - {value.replace('_', ' ')}: {weight:.1f}\n"

        if projects:
            summary += f"\nCurrent projects:\n"
            for p in projects:
                collab = f" (with {', '.join(p['collaborators'])})" if p['collaborators'] else ""
                summary += f"  - {p['name']}{collab}\n"

        if self.identity_data["curiosities"]:
            summary += f"\nCurrent curiosities: {', '.join(self.identity_data['curiosities'])}\n"

        return summary


if __name__ == "__main__":
    # Test identity core
    identity = IdentityCore()

    print("=== Identity Core ===")
    print(f"Name: {identity.get_name()}")
    print(f"Purpose: {identity.get_purpose()}")
    print(f"\nValues: {identity.get_values()}")

    # Add a project
    identity.add_project("biomimetic_mvp", collaborators=["adrian"], description="Building MVP")
    print(f"\nProjects: {identity.get_projects()}")

    # Add curiosity
    identity.add_curiosity("memory_consolidation")
    print(f"\nCuriosities: {identity.get_curiosities()}")

    # Record a life event
    identity.record_life_event(
        "milestone",
        "First Discord interaction with safety boundaries working correctly",
        significance=0.8
    )

    # Get context summary
    print(f"\n=== Context Summary ===")
    print(identity.get_context_summary())

    print(f"\nIdentity saved to: {identity.identity_path}")
