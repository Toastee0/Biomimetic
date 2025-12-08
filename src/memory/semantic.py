"""Semantic Memory - Long-term concept and relationship storage"""

import sqlite3
import time
import math
from typing import List, Dict, Optional, Tuple
import os


class SemanticMemory:
    """Manages semantic memory (concepts, relationships, users)"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "/home/toastee/BioMimeticAi/data/biomim.db")
        self.db_path = db_path
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    # ========== CONCEPTS ==========
    
    def store_concept(
        self,
        name: str,
        description: str = "",
        attention_weight: float = 1.0
    ) -> int:
        """Store or update a concept
        
        Args:
            name: Concept name (unique identifier)
            description: Concept description
            attention_weight: Initial attention weight
            
        Returns:
            concept_id
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = int(time.time())
        
        # Try to update existing concept
        cursor.execute("""
            UPDATE concepts
            SET description = ?, last_touched = ?, touch_count = touch_count + 1
            WHERE name = ?
        """, (description, now, name))
        
        if cursor.rowcount == 0:
            # Insert new concept
            cursor.execute("""
                INSERT INTO concepts (name, description, attention_weight, last_touched, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (name, description, attention_weight, now, now))
            concept_id = cursor.lastrowid
        else:
            # Get existing concept ID
            cursor.execute("SELECT concept_id FROM concepts WHERE name = ?", (name,))
            concept_id = cursor.fetchone()[0]
        
        conn.commit()
        conn.close()
        return concept_id
    
    def get_concept(self, name: str) -> Optional[Dict]:
        """Retrieve a concept by name
        
        Args:
            name: Concept name
            
        Returns:
            Concept dictionary or None
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM concepts WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_top_concepts(self, limit: int = 10) -> List[Dict]:
        """Get concepts with highest attention weights
        
        Args:
            limit: Maximum number of concepts to return
            
        Returns:
            List of concept dictionaries
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM concepts
            ORDER BY attention_weight DESC
            LIMIT ?
        """, (limit,))
        
        concepts = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return concepts
    
    def update_attention_weight(self, concept_id: int, new_weight: float):
        """Update attention weight for a concept
        
        Args:
            concept_id: Concept to update
            new_weight: New attention weight
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE concepts
            SET attention_weight = ?, last_touched = ?
            WHERE concept_id = ?
        """, (new_weight, int(time.time()), concept_id))
        
        conn.commit()
        conn.close()
    
    def apply_temporal_decay(self, decay_rate: float = 0.1):
        """Apply temporal decay to all concept attention weights
        
        Args:
            decay_rate: Decay rate per day (default 10%)
            
        Returns:
            Number of concepts updated
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = int(time.time())
        
        # Get all concepts
        cursor.execute("SELECT concept_id, attention_weight, last_touched FROM concepts")
        concepts = cursor.fetchall()
        
        updates = 0
        for concept_id, weight, last_touched in concepts:
            age_days = (now - last_touched) / (24 * 3600)
            decay_factor = math.exp(-age_days * decay_rate)
            new_weight = weight * decay_factor
            
            cursor.execute("""
                UPDATE concepts
                SET attention_weight = ?
                WHERE concept_id = ?
            """, (new_weight, concept_id))
            updates += 1
        
        conn.commit()
        conn.close()
        return updates
    
    # ========== RELATIONSHIPS ==========
    
    def add_relationship(
        self,
        from_concept: str,
        to_concept: str,
        relationship_type: str = "related_to",
        strength: float = 1.0
    ):
        """Add or update a relationship between concepts
        
        Args:
            from_concept: Source concept name
            to_concept: Target concept name
            relationship_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = int(time.time())
        
        # Get concept IDs (create if not exist)
        from_id = self.store_concept(from_concept)
        to_id = self.store_concept(to_concept)
        
        # Reconnect after store_concept closed the connection
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Insert or update relationship
        cursor.execute("""
            INSERT INTO relationships (from_concept, to_concept, relationship_type, strength, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(from_concept, to_concept) DO UPDATE SET
                strength = ?,
                relationship_type = ?
        """, (from_id, to_id, relationship_type, strength, now, strength, relationship_type))
        
        conn.commit()
        conn.close()
    
    def get_related_concepts(
        self,
        concept_name: str,
        min_strength: float = 0.1
    ) -> List[Dict]:
        """Get concepts related to a given concept
        
        Args:
            concept_name: Concept name
            min_strength: Minimum relationship strength
            
        Returns:
            List of related concepts with relationship info
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.*, r.strength, r.relationship_type
            FROM relationships r
            JOIN concepts c ON r.to_concept = c.concept_id
            WHERE r.from_concept = (SELECT concept_id FROM concepts WHERE name = ?)
                AND r.strength >= ?
            ORDER BY r.strength DESC
        """, (concept_name, min_strength))
        
        related = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return related
    
    # ========== USERS ==========
    
    def store_user(
        self,
        user_id: str,
        username: str,
        trust_score: float = 0.5,
        communication_style: str = "",
        notes: str = ""
    ):
        """Store or update user information
        
        Args:
            user_id: Discord user ID
            username: Discord username
            trust_score: Trust score (0.0 to 1.0, default 0.5)
            communication_style: Notes about communication style
            notes: Additional notes
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = int(time.time())
        
        # Try to update existing user
        cursor.execute("""
            UPDATE users
            SET username = ?, last_seen = ?, interaction_count = interaction_count + 1,
                communication_style = ?, notes = ?
            WHERE user_id = ?
        """, (username, now, communication_style, notes, user_id))
        
        if cursor.rowcount == 0:
            # Insert new user
            cursor.execute("""
                INSERT INTO users (user_id, username, trust_score, first_seen, last_seen, communication_style, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, username, trust_score, now, now, communication_style, notes))
        
        conn.commit()
        conn.close()
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Retrieve user information
        
        Args:
            user_id: Discord user ID
            
        Returns:
            User dictionary or None
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def update_trust_score(self, user_id: str, delta: float):
        """Adjust a user's trust score
        
        Args:
            user_id: Discord user ID
            delta: Amount to adjust trust score (can be negative)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users
            SET trust_score = MAX(0.0, MIN(1.0, trust_score + ?))
            WHERE user_id = ?
        """, (delta, user_id))
        
        conn.commit()
        conn.close()
    
    def get_trusted_users(self, min_trust: float = 0.7, limit: int = 10) -> List[Dict]:
        """Get most trusted users
        
        Args:
            min_trust: Minimum trust score
            limit: Maximum users to return
            
        Returns:
            List of user dictionaries
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM users
            WHERE trust_score >= ?
            ORDER BY trust_score DESC
            LIMIT ?
        """, (min_trust, limit))
        
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return users


if __name__ == "__main__":
    # Test semantic memory
    sem = SemanticMemory()
    
    # Test concepts
    print("=== Testing Concepts ===")
    biomim_id = sem.store_concept("biomimetic_ai", "AI that mimics biological cognition")
    memory_id = sem.store_concept("memory_systems", "Episodic and semantic memory")
    print(f"Stored concepts: {biomim_id}, {memory_id}")
    
    # Add relationship
    sem.add_relationship("biomimetic_ai", "memory_systems", "uses", 0.9)
    print("Added relationship")
    
    # Get related
    related = sem.get_related_concepts("biomimetic_ai")
    print(f"Related to biomimetic_ai: {len(related)}")
    for r in related:
        print(f"  - {r['name']} ({r['relationship_type']}, strength={r['strength']:.2f})")
    
    # Test users
    print("\n=== Testing Users ===")
    sem.store_user("200026157335314434", "toastee0", trust_score=0.9, notes="Creator, trusted collaborator")
    user = sem.get_user("200026157335314434")
    print(f"User: {user['username']} (trust={user['trust_score']:.2f}, interactions={user['interaction_count']})")
    
    # Test temporal decay
    print("\n=== Testing Temporal Decay ===")
    updates = sem.apply_temporal_decay(decay_rate=0.1)
    print(f"Applied decay to {updates} concepts")
    
    # Show top concepts
    top = sem.get_top_concepts(limit=5)
    print(f"\nTop concepts:")
    for c in top:
        print(f"  - {c['name']} (weight={c['attention_weight']:.4f})")
