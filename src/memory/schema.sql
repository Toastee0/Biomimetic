-- Biomimetic AI Memory System Schema
-- SQLite database for episodic and semantic memory

-- Episodic memory (recent conversations)
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    username TEXT NOT NULL,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    hemisphere TEXT NOT NULL,  -- 'analytical' or 'social'
    salience_score REAL DEFAULT 1.0,
    consolidated INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL
);

-- Semantic memory (long-term concepts)
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    attention_weight REAL DEFAULT 1.0,
    last_touched INTEGER NOT NULL,
    touch_count INTEGER DEFAULT 1,
    created_at INTEGER NOT NULL
);

-- Concept relationships
CREATE TABLE IF NOT EXISTS relationships (
    from_concept INTEGER,
    to_concept INTEGER,
    strength REAL DEFAULT 1.0,
    relationship_type TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY(from_concept) REFERENCES concepts(concept_id),
    FOREIGN KEY(to_concept) REFERENCES concepts(concept_id),
    PRIMARY KEY(from_concept, to_concept)
);

-- User relationships and trust
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    trust_score REAL DEFAULT 0.5,
    interaction_count INTEGER DEFAULT 0,
    first_seen INTEGER NOT NULL,
    last_seen INTEGER NOT NULL,
    communication_style TEXT,
    notes TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated);
CREATE INDEX IF NOT EXISTS idx_concepts_weight ON concepts(attention_weight DESC);
CREATE INDEX IF NOT EXISTS idx_concepts_touched ON concepts(last_touched DESC);
CREATE INDEX IF NOT EXISTS idx_users_trust ON users(trust_score DESC);
