-- ============================================================================
-- BioMimeticAI Vision Data Pool Database Schema
-- ============================================================================
-- Purpose: Autonomous vision collection and tagging system
-- Architecture: Custom build inspired by Frigate NVR patterns
-- Integration: Feeds episodic memory, cleaned up during sleep process
-- ============================================================================

-- Enable WAL mode for better concurrency (run separately with PRAGMA)
-- PRAGMA journal_mode=WAL;
-- PRAGMA busy_timeout=5000;

-- ============================================================================
-- MAIN SEGMENTS TABLE
-- ============================================================================
-- Stores video segments/snapshots with metadata
-- Retention based on urgency scoring: 7/30/90 days

CREATE TABLE IF NOT EXISTS vision_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Temporal metadata
    timestamp INTEGER NOT NULL,              -- Unix timestamp of capture
    created_at INTEGER NOT NULL,             -- Unix timestamp of DB insert

    -- Source metadata
    camera_id TEXT NOT NULL,                 -- e.g., "workshop_recamera", "droidcam_phone1"
    event_type TEXT,                         -- e.g., "entrance", "exit", "background_scan"

    -- File storage
    file_path TEXT NOT NULL UNIQUE,          -- Full path to image/video file
    file_size INTEGER,                       -- Bytes
    duration REAL DEFAULT 0,                 -- Duration in seconds (0 for snapshots)
    frame_count INTEGER DEFAULT 1,           -- Number of frames

    -- Processing status
    processed BOOLEAN DEFAULT 0,             -- Has autonomous tagging completed?
    processed_at INTEGER,                    -- Unix timestamp of processing

    -- Urgency flags (autonomous heuristics)
    urgent_flag BOOLEAN DEFAULT 0,           -- Mark for immediate AI attention
    urgency_score REAL DEFAULT 0.0,          -- 0.0 - 1.0 (0.8+ = push to AI)
    urgency_reason TEXT,                     -- JSON: {person_detected: true, ...}

    -- AI review status
    ai_notified BOOLEAN DEFAULT 0,           -- Has AI been notified via push?
    ai_reviewed BOOLEAN DEFAULT 0,           -- Has AI processed this segment?
    ai_review_at INTEGER,                    -- Unix timestamp of AI review
    ai_review_notes TEXT,                    -- AI's assessment (JSON or text)

    -- Retention management
    retention_days INTEGER DEFAULT 7,        -- Based on urgency: 7 (low), 30 (med), 90 (high)
    marked_for_deletion BOOLEAN DEFAULT 0,   -- Cleanup process marks before deletion

    -- Metadata from reCamera YOLO (if applicable)
    yolo_metadata TEXT,                      -- JSON: {class, confidence, bbox, track_id}

    -- Indexes for common queries
    CHECK (urgency_score >= 0.0 AND urgency_score <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_vs_camera_timestamp
    ON vision_segments(camera_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_vs_processed
    ON vision_segments(processed);

CREATE INDEX IF NOT EXISTS idx_vs_urgent
    ON vision_segments(urgent_flag, ai_notified);

CREATE INDEX IF NOT EXISTS idx_vs_retention
    ON vision_segments(created_at, retention_days, marked_for_deletion);

CREATE INDEX IF NOT EXISTS idx_vs_ai_reviewed
    ON vision_segments(ai_reviewed);

-- ============================================================================
-- SEGMENT TAGS TABLE
-- ============================================================================
-- Normalized multi-tag support (one segment can have many tags)
-- Tag types: object, activity, scene, audio, emotion

CREATE TABLE IF NOT EXISTS segment_tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,

    -- Tag classification
    tag_type TEXT NOT NULL,                  -- 'object', 'activity', 'scene', 'audio', 'emotion'
    tag_value TEXT NOT NULL,                 -- e.g., 'person', 'working', 'keyboard_typing'
    confidence REAL NOT NULL,                -- 0.0 - 1.0 (model confidence)

    -- Optional metadata
    bbox_json TEXT,                          -- YOLO bounding box: [x1, y1, x2, y2]
    metadata_json TEXT,                      -- Additional tag-specific data

    -- Model provenance
    model_name TEXT,                         -- e.g., 'yolov8n', 'clip-vit-b32', 'tsm'
    model_version TEXT,                      -- Track model versions

    created_at INTEGER NOT NULL,

    FOREIGN KEY (segment_id) REFERENCES vision_segments(segment_id) ON DELETE CASCADE,
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (tag_type IN ('object', 'activity', 'scene', 'audio', 'emotion', 'custom'))
);

CREATE INDEX IF NOT EXISTS idx_st_segment
    ON segment_tags(segment_id);

CREATE INDEX IF NOT EXISTS idx_st_tag_type_value
    ON segment_tags(tag_type, tag_value);

CREATE INDEX IF NOT EXISTS idx_st_confidence
    ON segment_tags(tag_type, confidence);

-- ============================================================================
-- SCENE EMBEDDINGS TABLE
-- ============================================================================
-- CLIP embeddings for semantic search
-- "Show me all times someone was working at desk"

CREATE TABLE IF NOT EXISTS scene_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL UNIQUE,

    -- CLIP embedding (512-dim for ViT-B/32, stored as numpy array blob)
    embedding BLOB NOT NULL,                 -- Serialized numpy array (float32)
    embedding_dim INTEGER DEFAULT 512,       -- Dimensionality

    -- Model metadata
    model_name TEXT DEFAULT 'ViT-B/32',      -- CLIP model variant
    model_version TEXT,                      -- Track CLIP version

    -- Optional: Quantized version for storage optimization
    embedding_quantized BLOB,                -- int8 quantized (4x smaller)

    created_at INTEGER NOT NULL,

    FOREIGN KEY (segment_id) REFERENCES vision_segments(segment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_se_segment
    ON scene_embeddings(segment_id);

-- ============================================================================
-- URGENCY QUEUE TABLE
-- ============================================================================
-- Tracks high-urgency segments that need immediate AI attention
-- Push notifications via Discord or HTTP API

CREATE TABLE IF NOT EXISTS urgency_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,

    -- Urgency scoring
    urgency_score REAL NOT NULL,             -- 0.0 - 1.0 (typically > 0.8 for queue)
    urgency_reason TEXT,                     -- JSON: {person_detected: true, unknown_face: true, ...}
    urgency_factors TEXT,                    -- JSON: detailed breakdown

    -- Notification status
    ai_notified BOOLEAN DEFAULT 0,           -- Has AI been notified?
    notified_at INTEGER,                     -- Unix timestamp of notification
    notification_method TEXT,                -- 'discord', 'http_push', 'both'

    -- Response tracking
    ai_response TEXT,                        -- What the AI decided to do (JSON)
    resolved BOOLEAN DEFAULT 0,              -- Has AI acted on this?
    resolved_at INTEGER,                     -- Unix timestamp of resolution

    created_at INTEGER NOT NULL,

    FOREIGN KEY (segment_id) REFERENCES vision_segments(segment_id) ON DELETE CASCADE,
    CHECK (urgency_score >= 0.0 AND urgency_score <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_uq_notified
    ON urgency_queue(ai_notified, urgency_score DESC);

CREATE INDEX IF NOT EXISTS idx_uq_created
    ON urgency_queue(created_at);

CREATE INDEX IF NOT EXISTS idx_uq_resolved
    ON urgency_queue(resolved);

-- ============================================================================
-- AUDIO EVENTS TABLE
-- ============================================================================
-- Audio event detection separate from visual tags
-- Different temporal nature (can have multiple events within one segment)

CREATE TABLE IF NOT EXISTS audio_events (
    audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,

    -- Audio classification
    event_type TEXT NOT NULL,                -- 'speech', 'music', 'alarm', 'tool', 'animal', etc.
    confidence REAL NOT NULL,                -- Model confidence

    -- Temporal info within segment
    start_time REAL,                         -- Offset within segment (seconds)
    duration REAL,                           -- Duration of audio event (seconds)

    -- Audio features (optional)
    loudness_db REAL,                        -- Peak loudness in dB
    frequency_range TEXT,                    -- JSON: [min_hz, max_hz]
    spectral_features TEXT,                  -- JSON: Additional features

    -- Model provenance
    model_name TEXT,                         -- e.g., 'panns_cnn14', 'yamnet'
    model_version TEXT,

    created_at INTEGER NOT NULL,

    FOREIGN KEY (segment_id) REFERENCES vision_segments(segment_id) ON DELETE CASCADE,
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_ae_segment
    ON audio_events(segment_id);

CREATE INDEX IF NOT EXISTS idx_ae_event_type
    ON audio_events(event_type, confidence);

-- ============================================================================
-- EVENT TIMELINE TABLE
-- ============================================================================
-- Frigate-inspired event lifecycle tracking
-- One event (e.g., "person entrance") → many timeline entries + segments

CREATE TABLE IF NOT EXISTS event_timeline (
    timeline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,                  -- UUID for tracking event across updates

    -- Event lifecycle
    event_start INTEGER NOT NULL,            -- Unix timestamp of event start
    event_end INTEGER,                       -- Unix timestamp of event end (NULL if ongoing)
    event_type TEXT NOT NULL,                -- 'entrance', 'exit', 'activity_change', 'scene_change'

    -- Associated data
    camera_id TEXT NOT NULL,
    segment_ids TEXT NOT NULL,               -- JSON array of segment_ids involved

    -- Event summary
    primary_object TEXT,                     -- e.g., 'person', 'car', 'dog'
    activity TEXT,                           -- e.g., 'working', 'walking', 'stationary'
    zones TEXT,                              -- JSON array of zones entered/exited

    -- Status
    active BOOLEAN DEFAULT 1,                -- Is event still ongoing?

    -- Metadata
    metadata TEXT,                           -- JSON: additional event data

    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,

    CHECK (event_end IS NULL OR event_end >= event_start)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_et_event_id
    ON event_timeline(event_id);

CREATE INDEX IF NOT EXISTS idx_et_camera_time
    ON event_timeline(camera_id, event_start);

CREATE INDEX IF NOT EXISTS idx_et_active
    ON event_timeline(active);

CREATE INDEX IF NOT EXISTS idx_et_type
    ON event_timeline(event_type);

-- ============================================================================
-- STORAGE STATS TABLE
-- ============================================================================
-- Track storage usage over time for pressure handling
-- Cleanup process monitors this

CREATE TABLE IF NOT EXISTS storage_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,

    -- Disk usage (whole partition)
    total_bytes INTEGER NOT NULL,
    used_bytes INTEGER NOT NULL,
    free_bytes INTEGER NOT NULL,
    usage_percent REAL NOT NULL,

    -- Vision data pool stats
    pool_total_segments INTEGER NOT NULL,
    pool_unprocessed_segments INTEGER NOT NULL,
    pool_urgent_segments INTEGER NOT NULL,
    pool_total_bytes INTEGER NOT NULL,

    -- Cleanup stats (from last run)
    segments_deleted INTEGER DEFAULT 0,
    bytes_freed INTEGER DEFAULT 0,
    cleanup_reason TEXT,                     -- 'scheduled', 'storage_pressure', 'manual'

    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ss_timestamp
    ON storage_stats(timestamp);

-- ============================================================================
-- CROSS-CAMERA TRACKS TABLE (OPTIONAL - FUTURE)
-- ============================================================================
-- Track objects across multiple cameras
-- Useful for person movement between rooms

CREATE TABLE IF NOT EXISTS cross_camera_tracks (
    track_id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_uuid TEXT NOT NULL UNIQUE,         -- UUID for track

    -- Object identification
    object_type TEXT NOT NULL,               -- 'person', 'car', etc.
    first_seen INTEGER NOT NULL,             -- Unix timestamp
    last_seen INTEGER NOT NULL,              -- Unix timestamp

    -- Camera path (ordered list)
    camera_path TEXT NOT NULL,               -- JSON: ['workshop_recamera', 'droidcam_phone1']
    segment_ids TEXT NOT NULL,               -- JSON: [segment_id1, segment_id2, ...]

    -- ReID features (for matching across cameras)
    reid_embedding BLOB,                     -- Appearance embedding for re-identification

    -- Track metadata
    active BOOLEAN DEFAULT 1,
    metadata TEXT,                           -- JSON: additional data

    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cct_uuid
    ON cross_camera_tracks(track_uuid);

CREATE INDEX IF NOT EXISTS idx_cct_active
    ON cross_camera_tracks(active);

CREATE INDEX IF NOT EXISTS idx_cct_last_seen
    ON cross_camera_tracks(last_seen);

-- ============================================================================
-- CONSOLIDATION LOG TABLE
-- ============================================================================
-- Track sleep/cleanup process consolidations
-- Records what was consolidated into episodic memory

CREATE TABLE IF NOT EXISTS consolidation_log (
    consolidation_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Consolidation metadata
    consolidation_type TEXT NOT NULL,        -- 'daily_summary', 'cluster_summary', 'manual'
    timestamp INTEGER NOT NULL,

    -- Segments involved
    segment_ids TEXT NOT NULL,               -- JSON array
    segment_count INTEGER NOT NULL,

    -- Clustering info (if applicable)
    cluster_id INTEGER,                      -- DBSCAN cluster ID
    cluster_size INTEGER,

    -- Episodic memory reference
    episode_id INTEGER,                      -- Reference to episodes table
    summary_text TEXT,                       -- Human-readable summary

    -- Tags summary
    common_tags TEXT,                        -- JSON: most common tags in cluster

    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cl_timestamp
    ON consolidation_log(timestamp);

CREATE INDEX IF NOT EXISTS idx_cl_type
    ON consolidation_log(consolidation_type);

-- ============================================================================
-- SYSTEM METADATA TABLE
-- ============================================================================
-- Track system configuration and versioning

CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at INTEGER NOT NULL
);

-- Insert initial metadata
INSERT OR IGNORE INTO system_metadata (key, value, description, updated_at) VALUES
    ('schema_version', '1.0.0', 'Database schema version', strftime('%s', 'now')),
    ('created_at', strftime('%s', 'now'), 'Database creation timestamp', strftime('%s', 'now')),
    ('last_cleanup', '0', 'Last cleanup process timestamp', strftime('%s', 'now')),
    ('last_consolidation', '0', 'Last consolidation process timestamp', strftime('%s', 'now')),
    ('total_segments_processed', '0', 'Lifetime segment count', strftime('%s', 'now')),
    ('total_urgent_events', '0', 'Lifetime urgent event count', strftime('%s', 'now'));

-- ============================================================================
-- VIEWS (Optional convenience queries)
-- ============================================================================

-- View: Unprocessed segments (tagging pipeline queue)
CREATE VIEW IF NOT EXISTS v_unprocessed_segments AS
SELECT
    vs.segment_id,
    vs.timestamp,
    vs.camera_id,
    vs.file_path,
    vs.event_type,
    vs.urgency_score,
    (strftime('%s', 'now') - vs.created_at) AS age_seconds
FROM vision_segments vs
WHERE vs.processed = 0
ORDER BY vs.urgency_score DESC, vs.timestamp ASC;

-- View: Urgent queue (AI needs to review)
CREATE VIEW IF NOT EXISTS v_urgent_queue AS
SELECT
    uq.queue_id,
    uq.segment_id,
    vs.camera_id,
    vs.file_path,
    uq.urgency_score,
    uq.urgency_reason,
    uq.ai_notified,
    (strftime('%s', 'now') - uq.created_at) AS age_seconds
FROM urgency_queue uq
JOIN vision_segments vs ON uq.segment_id = vs.segment_id
WHERE uq.resolved = 0
ORDER BY uq.urgency_score DESC, uq.created_at ASC;

-- View: Segments needing cleanup (past retention period)
CREATE VIEW IF NOT EXISTS v_cleanup_candidates AS
SELECT
    vs.segment_id,
    vs.camera_id,
    vs.file_path,
    vs.file_size,
    vs.urgency_score,
    vs.retention_days,
    (strftime('%s', 'now') - vs.created_at) AS age_seconds,
    (vs.retention_days * 86400) AS retention_seconds
FROM vision_segments vs
WHERE vs.marked_for_deletion = 0
  AND (strftime('%s', 'now') - vs.created_at) > (vs.retention_days * 86400)
ORDER BY vs.urgency_score ASC, vs.created_at ASC;

-- View: Storage summary
CREATE VIEW IF NOT EXISTS v_storage_summary AS
SELECT
    COUNT(*) AS total_segments,
    SUM(CASE WHEN processed = 0 THEN 1 ELSE 0 END) AS unprocessed,
    SUM(CASE WHEN urgent_flag = 1 THEN 1 ELSE 0 END) AS urgent,
    SUM(CASE WHEN ai_reviewed = 0 THEN 1 ELSE 0 END) AS not_reviewed,
    SUM(file_size) AS total_bytes,
    SUM(CASE WHEN urgency_score < 0.3 THEN file_size ELSE 0 END) AS low_urgency_bytes,
    SUM(CASE WHEN urgency_score >= 0.3 AND urgency_score < 0.7 THEN file_size ELSE 0 END) AS med_urgency_bytes,
    SUM(CASE WHEN urgency_score >= 0.7 THEN file_size ELSE 0 END) AS high_urgency_bytes
FROM vision_segments;

-- ============================================================================
-- TRIGGERS (Optional automated behaviors)
-- ============================================================================

-- Trigger: Auto-update updated_at on event_timeline
CREATE TRIGGER IF NOT EXISTS trg_et_updated_at
AFTER UPDATE ON event_timeline
FOR EACH ROW
BEGIN
    UPDATE event_timeline
    SET updated_at = strftime('%s', 'now')
    WHERE timeline_id = NEW.timeline_id;
END;

-- Trigger: Auto-update system metadata on segment insert
CREATE TRIGGER IF NOT EXISTS trg_vs_insert_counter
AFTER INSERT ON vision_segments
FOR EACH ROW
BEGIN
    UPDATE system_metadata
    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
        updated_at = strftime('%s', 'now')
    WHERE key = 'total_segments_processed';
END;

-- Trigger: Auto-update urgent event counter
CREATE TRIGGER IF NOT EXISTS trg_uq_insert_counter
AFTER INSERT ON urgency_queue
FOR EACH ROW
BEGIN
    UPDATE system_metadata
    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
        updated_at = strftime('%s', 'now')
    WHERE key = 'total_urgent_events';
END;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- To apply this schema to the database:
-- sqlite3 /home/toastee/BioMimeticAi/data/biomim.db < init_vision_pool_schema.sql
--
-- Or in Python:
-- import sqlite3
-- conn = sqlite3.connect('/home/toastee/BioMimeticAi/data/biomim.db')
-- conn.execute("PRAGMA journal_mode=WAL")
-- conn.execute("PRAGMA busy_timeout=5000")
-- with open('init_vision_pool_schema.sql', 'r') as f:
--     conn.executescript(f.read())
-- conn.commit()
