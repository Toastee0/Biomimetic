"""Contact Memory - Structured memory of individual humans (address book)"""

import sqlite3
import json
import time
from typing import Dict, List, Optional, Any
from functools import wraps
import os


def retry_on_lock(max_attempts=3, backoff=1.5):
    """Decorator to retry database operations on lock errors

    Args:
        max_attempts: Maximum number of retry attempts
        backoff: Exponential backoff multiplier
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            wait_time = 0.1

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < max_attempts - 1:
                        attempt += 1
                        time.sleep(wait_time)
                        wait_time *= backoff
                    else:
                        raise

            return func(*args, **kwargs)
        return wrapper
    return decorator


class ContactMemory:
    """Manages structured profiles for individual humans the bot interacts with"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "/home/toastee/BioMimeticAi/data/biomim.db")
        self.db_path = db_path
        self._init_wal_mode()
        self._ensure_table()

    def _init_wal_mode(self):
        """Initialize WAL mode for better concurrent access"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.close()
        except Exception as e:
            print(f"[CONTACT WARNING] Could not enable WAL mode: {e}")

    def _get_connection(self):
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    
    def _ensure_table(self):
        """Ensure contacts table exists"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                display_name TEXT,
                first_met INTEGER NOT NULL,
                last_interaction INTEGER NOT NULL,
                interaction_count INTEGER DEFAULT 1,
                trust_level REAL DEFAULT 0.5,
                communication_style TEXT,
                preferred_topics TEXT,
                context_notes TEXT,
                personality_traits TEXT,
                relationship_type TEXT,
                custom_fields TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    @retry_on_lock(max_attempts=3, backoff=1.5)
    def get_or_create_contact(
        self,
        user_id: str,
        username: str,
        display_name: str = None
    ) -> Dict[str, Any]:
        """Get existing contact or create new one

        Args:
            user_id: Discord user ID
            username: Discord username
            display_name: Optional display name

        Returns:
            Contact profile dictionary
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Try to get existing contact
        cursor.execute("SELECT * FROM contacts WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        now = int(time.time())
        
        if row:
            # Update last interaction and count
            cursor.execute("""
                UPDATE contacts 
                SET last_interaction = ?,
                    interaction_count = interaction_count + 1,
                    username = ?,
                    updated_at = ?
                WHERE user_id = ?
            """, (now, username, now, user_id))
            conn.commit()
            
            # Fetch updated row
            cursor.execute("SELECT * FROM contacts WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            profile = dict(row)
        else:
            # Create new contact
            cursor.execute("""
                INSERT INTO contacts (
                    user_id, username, display_name, first_met, last_interaction,
                    interaction_count, trust_level, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, username, display_name, now, now, 1, 0.5, now, now))
            conn.commit()
            
            # Fetch new row
            cursor.execute("SELECT * FROM contacts WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            profile = dict(row)
        
        conn.close()
        
        # Parse JSON fields
        for field in ['preferred_topics', 'personality_traits', 'custom_fields']:
            if profile.get(field):
                try:
                    profile[field] = json.loads(profile[field])
                except:
                    profile[field] = []
        
        return profile
    
    @retry_on_lock(max_attempts=3, backoff=1.5)
    def update_contact(
        self,
        user_id: str,
        **updates
    ) -> bool:
        """Update contact profile fields

        Args:
            user_id: Discord user ID
            **updates: Fields to update (trust_level, communication_style, etc.)

        Returns:
            Success boolean
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Convert lists/dicts to JSON for storage
        for key in ['preferred_topics', 'personality_traits', 'custom_fields']:
            if key in updates and isinstance(updates[key], (list, dict)):
                updates[key] = json.dumps(updates[key])
        
        # Build update query
        set_parts = []
        values = []
        for key, value in updates.items():
            if key != 'user_id':  # Don't update primary key
                set_parts.append(f"{key} = ?")
                values.append(value)
        
        if not set_parts:
            conn.close()
            return False
        
        # Add updated_at timestamp
        set_parts.append("updated_at = ?")
        values.append(int(time.time()))
        
        # Add user_id for WHERE clause
        values.append(user_id)
        
        query = f"UPDATE contacts SET {', '.join(set_parts)} WHERE user_id = ?"
        
        try:
            cursor.execute(query, values)
            conn.commit()
            success = cursor.rowcount > 0
        except Exception as e:
            print(f"[CONTACT ERROR] Failed to update contact: {e}")
            success = False
        finally:
            conn.close()
        
        return success
    
    @retry_on_lock(max_attempts=3, backoff=1.5)
    def add_context_note(
        self,
        user_id: str,
        note: str
    ) -> bool:
        """Add a context note to contact (appends to existing notes)

        Args:
            user_id: Discord user ID
            note: Note to add

        Returns:
            Success boolean
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get existing notes
        cursor.execute("SELECT context_notes FROM contacts WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        existing_notes = row[0] or ""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        new_note = f"[{timestamp}] {note}"
        
        if existing_notes:
            updated_notes = f"{existing_notes}\n{new_note}"
        else:
            updated_notes = new_note
        
        cursor.execute(
            "UPDATE contacts SET context_notes = ?, updated_at = ? WHERE user_id = ?",
            (updated_notes, int(time.time()), user_id)
        )
        conn.commit()
        conn.close()
        
        return True
    
    def get_contact(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get contact profile by user_id
        
        Args:
            user_id: Discord user ID
            
        Returns:
            Contact profile dictionary or None
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM contacts WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        profile = dict(row)
        
        # Parse JSON fields
        for field in ['preferred_topics', 'personality_traits', 'custom_fields']:
            if profile.get(field):
                try:
                    profile[field] = json.loads(profile[field])
                except:
                    profile[field] = []
        
        return profile
    
    def list_contacts(
        self,
        limit: int = 50,
        min_trust: float = None,
        relationship_type: str = None
    ) -> List[Dict[str, Any]]:
        """List contacts with optional filters
        
        Args:
            limit: Maximum contacts to return
            min_trust: Minimum trust level filter
            relationship_type: Filter by relationship type
            
        Returns:
            List of contact profiles
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM contacts WHERE 1=1"
        params = []
        
        if min_trust is not None:
            query += " AND trust_level >= ?"
            params.append(min_trust)
        
        if relationship_type:
            query += " AND relationship_type = ?"
            params.append(relationship_type)
        
        query += " ORDER BY last_interaction DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        contacts = []
        for row in rows:
            profile = dict(row)
            
            # Parse JSON fields
            for field in ['preferred_topics', 'personality_traits', 'custom_fields']:
                if profile.get(field):
                    try:
                        profile[field] = json.loads(profile[field])
                    except:
                        profile[field] = []
            
            contacts.append(profile)
        
        return contacts
    
    @retry_on_lock(max_attempts=3, backoff=1.5)
    def adjust_trust(
        self,
        user_id: str,
        delta: float
    ) -> float:
        """Adjust trust level for a contact

        Args:
            user_id: Discord user ID
            delta: Amount to adjust trust (+/- 0.1 typical)

        Returns:
            New trust level
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT trust_level FROM contacts WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return 0.5
        
        current_trust = row[0]
        new_trust = max(0.0, min(1.0, current_trust + delta))  # Clamp 0-1
        
        cursor.execute(
            "UPDATE contacts SET trust_level = ?, updated_at = ? WHERE user_id = ?",
            (new_trust, int(time.time()), user_id)
        )
        conn.commit()
        conn.close()
        
        return new_trust
    
    def get_relationship_summary(self, user_id: str) -> str:
        """Generate a human-readable summary of relationship with this contact
        
        Args:
            user_id: Discord user ID
            
        Returns:
            Summary string
        """
        profile = self.get_contact(user_id)
        
        if not profile:
            return "No profile found."
        
        # Calculate relationship duration
        first_met = profile['first_met']
        last_seen = profile['last_interaction']
        duration_days = (last_seen - first_met) / 86400
        
        # Build summary
        summary = f"**{profile['username']}**"
        if profile.get('display_name'):
            summary += f" ({profile['display_name']})"
        summary += "\n"
        
        summary += f"• Known for {duration_days:.0f} days, {profile['interaction_count']} interactions\n"
        summary += f"• Trust level: {profile['trust_level']:.2f}/1.0\n"
        
        if profile.get('relationship_type'):
            summary += f"• Relationship: {profile['relationship_type']}\n"
        
        if profile.get('communication_style'):
            summary += f"• Style: {profile['communication_style']}\n"
        
        if profile.get('preferred_topics'):
            topics = profile['preferred_topics']
            if isinstance(topics, list) and topics:
                summary += f"• Interests: {', '.join(topics[:3])}\n"
        
        if profile.get('personality_traits'):
            traits = profile['personality_traits']
            if isinstance(traits, list) and traits:
                summary += f"• Traits: {', '.join(traits[:3])}\n"
        
        return summary
