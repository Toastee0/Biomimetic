"""Episodic Memory - Recent conversation storage and retrieval"""

import sqlite3
import time
from typing import List, Dict, Optional
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


class EpisodicMemory:
    """Manages episodic memory (recent conversations)"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "/home/toastee/BioMimeticAi/data/biomim.db")
        self.db_path = db_path
        self._init_wal_mode()

    def _init_wal_mode(self):
        """Initialize WAL mode for better concurrent access"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.close()
        except Exception as e:
            print(f"[EPISODIC WARNING] Could not enable WAL mode: {e}")

    def _get_connection(self):
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    
    @retry_on_lock(max_attempts=3, backoff=1.5)
    def store_episode(
        self,
        user_id: str,
        username: str,
        user_message: str,
        bot_response: str,
        hemisphere: str = "social",
        salience_score: float = 1.0
    ) -> int:
        """Store a conversation episode

        Args:
            user_id: Discord user ID
            username: Discord username
            user_message: What the user said
            bot_response: How the bot responded
            hemisphere: Which model was used ('analytical' or 'social')
            salience_score: Importance score (0.0 to 1.0)

        Returns:
            episode_id of the stored episode
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = int(time.time())

        cursor.execute("""
            INSERT INTO episodes (
                timestamp, user_id, username, user_message, bot_response,
                hemisphere, salience_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, user_id, username, user_message, bot_response, hemisphere, salience_score, now))

        episode_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return episode_id
    
    def get_recent_episodes(
        self,
        limit: int = 10,
        user_id: Optional[str] = None,
        unconsolidated_only: bool = False
    ) -> List[Dict]:
        """Retrieve recent episodes
        
        Args:
            limit: Maximum number of episodes to return
            user_id: Filter by specific user (optional)
            unconsolidated_only: Only return episodes not yet consolidated
            
        Returns:
            List of episode dictionaries
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM episodes WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if unconsolidated_only:
            query += " AND consolidated = 0"
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        episodes = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return episodes
    
    @retry_on_lock(max_attempts=3, backoff=1.5)
    def update_salience(self, episode_id: int, new_score: float) -> bool:
        """Update the salience score of an episode

        Args:
            episode_id: Episode to update
            new_score: New salience score (0.0 to 1.0)

        Returns:
            Success boolean
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE episodes
            SET salience_score = ?
            WHERE episode_id = ?
        """, (new_score, episode_id))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    @retry_on_lock(max_attempts=3, backoff=1.5)
    def mark_consolidated(self, episode_ids: List[int]) -> bool:
        """Mark episodes as consolidated into semantic memory

        Args:
            episode_ids: List of episode IDs to mark (or single int)

        Returns:
            Success boolean
        """
        # Handle single int or list
        if isinstance(episode_ids, int):
            episode_ids = [episode_ids]

        if not episode_ids:
            return False

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(episode_ids))
        cursor.execute(f"""
            UPDATE episodes
            SET consolidated = 1
            WHERE episode_id IN ({placeholders})
        """, episode_ids)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success
    
    def get_context_window(
        self,
        user_id: str,
        window_size: int = 5
    ) -> str:
        """Build a context string from recent conversations with a user
        
        Args:
            user_id: Discord user ID
            window_size: Number of recent exchanges to include
            
        Returns:
            Formatted context string
        """
        episodes = self.get_recent_episodes(limit=window_size, user_id=user_id)
        
        if not episodes:
            return ""
        
        context_parts = ["Recent conversation history:"]
        for ep in reversed(episodes):  # Oldest first
            context_parts.append(f"User: {ep['user_message']}")
            context_parts.append(f"Assistant: {ep['bot_response']}")
        
        return "\n".join(context_parts)

    def retrieve_by_cue(self, cue_message: str, user_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Retrieve episodes triggered by environmental cues (biomimetic recall)
        
        This mimics biological episodic memory: specific words/concepts in the current
        message trigger related past memories through spreading activation.
        
        Args:
            cue_message: Current message (the environmental trigger)
            user_id: Optional filter by specific user
            limit: Maximum episodes to return
            
        Returns:
            List of triggered episodes scored by relevance
        """
        # Extract cue words (simple keyword extraction)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                      "of", "with", "is", "are", "was", "were", "i", "you", "it", "this", "that"}
        
        words = cue_message.lower().split()
        cue_words = [w.strip(".,!?;:()[]\"") for w in words if len(w) > 3 and w not in stop_words]
        
        if not cue_words:
            return self.get_recent_episodes(limit=limit, user_id=user_id)
        
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search for episodes containing cue words
        keyword = cue_words[0]
        query = "SELECT * FROM episodes WHERE (LOWER(user_message) LIKE ? OR LOWER(bot_response) LIKE ?)"
        params = [f"%{keyword}%", f"%{keyword}%"]
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY salience_score DESC, timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        episodes = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return episodes if episodes else self.get_recent_episodes(limit=limit, user_id=user_id)

    def get_episode_count(self, user_id: Optional[str] = None) -> int:
        """Get total number of episodes
        
        Args:
            user_id: Optional filter by user
            
        Returns:
            Count of episodes
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE user_id = ?", (user_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM episodes")
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count


if __name__ == "__main__":
    # Test the episodic memory
    mem = EpisodicMemory()
    
    # Store a test episode
    episode_id = mem.store_episode(
        user_id="200026157335314434",
        username="toastee0",
        user_message="Hello PopTartee!",
        bot_response="Hey there! How can I help you today?",
        hemisphere="social"
    )
    
    print(f"Stored episode {episode_id}")
    
    # Retrieve recent episodes
    recent = mem.get_recent_episodes(limit=5)
    print(f"\nRecent episodes: {len(recent)}")
    for ep in recent:
        print(f"  [{ep['username']}]: {ep['user_message'][:50]}")
    
    # Build context
    context = mem.get_context_window("200026157335314434", window_size=3)
    print(f"\nContext window:\n{context}")
