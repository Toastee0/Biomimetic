## Vision System Architecture - On-Demand Model Loading

**Hardware**: Single RTX 3090 (24GB VRAM)
**Constraint**: Cannot run large vision + text models simultaneously
**Solution**: Cron-based processing with specialized small models

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  reCamera (192.168.2.140)                   │
│  - YOLO object detection                                   │
│  - Entrance/exit tracking (3s debounce)                    │
│  - RTSP stream (rtsp://192.168.2.140:8554/stream)         │
└─────────┬──────────────────────────────────┬────────────────┘
          │                                   │
          │ POST /api/vision/event            │ RTSP Stream
          │ (with YOLO metadata)              │
          │                                   │
          ▼                                   ▼
┌─────────────────────────┐         ┌────────────────────────┐
│  Vision Event API       │         │ Background Scan        │
│  (Always Running)       │         │ (Cron: Every 1min)     │
│  - Port 8000            │         │ - Captures every 20s   │
│  - Receives events      │         │ - Catches YOLO misses  │
│  - Captures snapshots   │         │                        │
│  - Queues for processing│         │                        │
└─────────┬───────────────┘         └──────────┬─────────────┘
          │                                    │
          │ Writes to queue                   │
          ▼                                    ▼
┌──────────────────────────────────────────────────────────────┐
│           Snapshot Processing Queues (JSON)                  │
│  - snapshot_queue.json (YOLO detections)                    │
│  - background_scan_queue.json (full scene scans)            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Processed by (Cron: Every 30min)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│            Vision Processing Cortex                          │
│  - Load specialized models on-demand                         │
│  - Process queued snapshots in batch                         │
│  - Different processing for persons vs. general scans        │
└────────┬─────────────────────────────────────────────────────┘
         │
         ├─► Person Detection
         │   └─► Small Specialized Models (can coexist):
         │       ├─ Face Recognition (~200MB)  → Match contacts
         │       ├─ Age Estimation (~50MB)     → Demographics
         │       ├─ Emotion Detection (~100MB) → Mood/state
         │       └─ Gender Detection (~50MB)   → Demographics
         │
         └─► Background Scans
             └─► CLIP Model (~350MB) → "What did YOLO miss?"
                 └─ General scene understanding
                 └─ Anomaly detection

┌──────────────────────────────────────────────────────────────┐
│               Results Stored In                               │
│  - Episodic Memory (with high salience 0.85)                │
│  - Contact Memory (if face matched)                          │
│  - Queue marked as processed                                 │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. YOLO Detection Path (High Priority)

```
reCamera YOLO detects person
    ↓
3-second debounce (stable detection)
    ↓
POST /api/vision/event
    {
      "event": "entrance",
      "object": "person",
      "timestamp": 1234567890,
      "scene": ["person"],
      "yolo_detection": {
        "class": "person",
        "confidence": 0.95,
        "bbox": [x1, y1, x2, y2],
        "track_id": 123
      }
    }
    ↓
Vision API captures RTSP snapshot
    ↓
Queue snapshot with YOLO metadata
    ↓
(Later) Vision Cortex processes:
    ├─ Load face recognition model
    ├─ Crop to bbox from YOLO
    ├─ Extract face embedding
    ├─ Match against known contacts
    ├─ Estimate age, emotion, gender
    ├─ Update contact memory if matched
    └─ Store in episodic memory (salience: 0.85)
```

### 2. Background Scan Path (Safety Net)

```
Background Scan Cortex runs every minute
    ↓
Captures RTSP snapshot every 20s
    ↓
Queue for general analysis
    ↓
(Later) Vision Cortex processes:
    ├─ Load CLIP model
    ├─ Extract scene embeddings
    ├─ Compare with typical scene
    ├─ Detect anomalies
    ├─ Check for objects YOLO missed
    └─ Store in episodic memory (salience: 0.3)
```

## Model Management Strategy

### Small Models (Can Coexist with Text Model)

These models are small enough (~50-350MB) to run alongside Mistral-Small-22B:

| Model | Size | Purpose | Install |
|-------|------|---------|---------|
| Face Recognition | ~200MB | Match against contacts | `pip install insightface` |
| Age/Gender | ~50MB | Demographics | Included in InsightFace |
| Emotion | ~100MB | Mood detection | `pip install fer` |
| CLIP | ~350MB | Scene understanding | `pip install git+https://github.com/openai/CLIP.git` |

**Total**: ~700MB VRAM (leaves ~16GB for text model)

### Large Models (Require Stopping Text Model)

If needed for complex analysis:

| Model | Size | Purpose | Strategy |
|-------|------|---------|----------|
| LLaVA 7B | ~4GB | Full vision LLM | Stop text model, load LLaVA, process, restart text |
| LLaVA 13B | ~7GB | Better vision LLM | Same as above |

**Not implemented yet** - current system uses small models only.

## Cron Schedule

```bash
# Vision Event API (always running - lightweight)
# Started via: bash scripts/start_vision_api.sh

# Background Vision Scan (every minute, captures every 20s)
*/1 * * * * /home/toastee/BioMimeticAi/scripts/cron/background_vision_scan.py

# Vision Processing (every 30 minutes - processes queues)
*/30 * * * * /home/toastee/BioMimeticAi/scripts/cron/vision_processing.py
```

Install with:
```bash
bash scripts/install_vision_cron.sh  # TODO: Create this script
```

## File Structure

```
data/
├── vision/
│   ├── snapshots/                      # All captured snapshots
│   │   ├── entrance_person_1234.jpg    # YOLO detections
│   │   └── background_scan_5678.jpg    # Background scans
│   ├── snapshot_queue.json             # YOLO detections to process
│   └── background_scan_queue.json      # Background scans to process
└── cortex_state/
    ├── background_vision_scan.json     # Scan cortex status
    └── vision_processing.json          # Processing cortex status

logs/
├── background_vision_scan.log          # Scan cortex logs
└── vision_processing.log               # Processing cortex logs
```

## Queue Format

### snapshot_queue.json (YOLO Detections)

```json
[
  {
    "snapshot_path": "/path/to/entrance_person_1234.jpg",
    "event_type": "entrance",
    "detected_object": "person",
    "scene": ["person", "car"],
    "timestamp": 1234567890,
    "yolo_detection": {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 150, 300, 450],
      "track_id": 123
    },
    "queued_at": 1234567891,
    "processed": false
  }
]
```

### background_scan_queue.json (General Scans)

```json
[
  {
    "snapshot_path": "/path/to/background_scan_5678.jpg",
    "scan_type": "background",
    "timestamp": 1234567890,
    "processed": false
  }
]
```

## Processing Logic

### Person Detection Processing

```python
# In vision_processing.py
if item.get('yolo_detection') and item['detected_object'] == 'person':
    # Person-specific analysis
    analyzer = PersonAnalyzer()
    results = analyzer.analyze_person(
        image_path=snapshot_path,
        yolo_detection=item['yolo_detection']
    )

    # Results contain:
    # - face_match: Contact ID if recognized
    # - age: Estimated age
    # - emotion: Detected emotion
    # - gender: Detected gender

    # If face matched, update contact memory
    if results['face_match']:
        contact_id = results['face_match']['contact_id']
        # Update contact with visual confirmation
        # Track when they were seen
        # Update appearance notes
```

### Background Scan Processing

```python
# In vision_processing.py
if item.get('scan_type') == 'background':
    # General scene analysis
    clip_model = model_manager.load_clip()

    # Extract scene features
    features = extract_clip_features(snapshot_path)

    # Compare with normal scene
    anomaly_score = compare_with_baseline(features)

    # Check for objects YOLO might have missed
    # Store in episodic memory if anomaly detected
```

## Contact Memory Integration

When a person is recognized via face matching:

```python
# Update contact memory
contact_memory.update_contact(contact_id, {
    "last_seen_visual": timestamp,
    "visual_confirmations": count + 1,
    "recent_emotions": ["happy", "neutral"],  # Track patterns
    "appearance_notes": "Wearing blue jacket"  # Context
})

# Store in episodic memory with high salience
episodic.store_episode(
    user_id=contact_id,
    username=contact_name,
    user_message=f"Visual: {contact_name} detected at entrance",
    bot_response=json.dumps(analysis_results),
    hemisphere="sensory",
    salience_score=0.9  # Very high - recognized person
)
```

## VRAM Management

Current usage with text model:
```
Mistral-Small-22B (Q4_K_M): ~14GB VRAM
Free for vision models: ~10GB
```

Small models total: ~700MB (7% of free space)
**Safe to run alongside text model** ✓

## API Reference

### POST /api/vision/event

Receive YOLO detection event with metadata.

**Request**:
```json
{
  "event": "entrance",
  "object": "person",
  "timestamp": 1234567890,
  "scene": ["person"],
  "yolo_detection": {
    "class": "person",
    "confidence": 0.95,
    "bbox": [100, 150, 300, 450],
    "track_id": 123
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Event received and queued",
  "snapshot_captured": true,
  "queue_position": 5
}
```

## Installation

### 1. Install Dependencies

```bash
# Face recognition
pip install insightface onnxruntime-gpu

# Emotion detection
pip install fer

# CLIP
pip install git+https://github.com/openai/CLIP.git

# Image processing
pip install pillow numpy
```

### 2. Test Models

```bash
# Test model manager
python src/core/model_manager.py

# Test person analyzer
python src/core/person_analyzer.py data/vision/snapshots/test.jpg
```

### 3. Install Cron Jobs

```bash
# Add to crontab
crontab -e

# Add these lines:
*/1 * * * * /home/toastee/BioMimeticAi/scripts/cron/background_vision_scan.py >> /home/toastee/BioMimeticAi/logs/cron.log 2>&1
*/30 * * * * /home/toastee/BioMimeticAi/scripts/cron/vision_processing.py >> /home/toastee/BioMimeticAi/logs/cron.log 2>&1
```

### 4. Start Vision API

```bash
bash scripts/start_vision_api.sh
```

## Monitoring

```bash
# Watch queues
watch -n 5 'echo "Snapshot Queue:" && jq length data/vision/snapshot_queue.json && echo "Background Queue:" && jq length data/vision/background_scan_queue.json'

# Monitor processing
tail -f logs/vision_processing.log

# Check VRAM usage
python src/core/model_manager.py

# View recent snapshots
ls -lth data/vision/snapshots/ | head -20
```

## Future Enhancements

### Phase 1 (Current - Placeholders)
- [x] Queue-based architecture
- [x] Person-specific analysis framework
- [x] Background scan cortex
- [ ] Implement face recognition (InsightFace)
- [ ] Implement emotion detection (FER)
- [ ] Implement CLIP for background scans

### Phase 2 (Advanced)
- [ ] Face embedding storage in contact memory
- [ ] Gait recognition (for identification without faces)
- [ ] Clothing pattern recognition (aid identification)
- [ ] Behavior pattern analysis (normal vs. anomalous)

### Phase 3 (Large Models)
- [ ] LLaVA integration (on-demand, stop text model)
- [ ] Complex scene understanding
- [ ] Natural language scene descriptions
- [ ] Integration with axiom system for threat assessment

## Debugging

### Queue not processing?

```bash
# Check vision processing cortex status
cat data/cortex_state/vision_processing.json

# Manually run processing
python scripts/cron/vision_processing.py

# Check queue files
cat data/vision/snapshot_queue.json | jq
```

### Snapshots not capturing?

```bash
# Test RTSP connection
python src/core/rtsp_capture.py

# Check ffmpeg
which ffmpeg
ffmpeg -version
```

### Models not loading?

```bash
# Check installations
pip list | grep -E "insightface|fer|clip"

# Test model loading
python src/core/model_manager.py
```
