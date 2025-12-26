# Vision System Architecture - Complete Guide

**Hardware**: Single RTX 3090 (24GB VRAM)  
**Text Model**: Mistral-Small-22B Q4 (~14GB VRAM)  
**Vision Budget**: ~10GB VRAM available  
**Solution**: Cron-based processing with small specialized models (~700MB total)

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Data Flow](#data-flow)
4. [Components](#components)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)

---

## Overview

This vision system is designed for a single-GPU constraint where a large text model (Mistral-Small-22B) is already running. Instead of competing for VRAM with large vision models, we use:

✅ **Queue-based processing** - Capture now, analyze later  
✅ **Small specialized models** - Face recognition, emotion, CLIP (~700MB total)  
✅ **Cron scheduling** - Process in batches every 30 minutes  
✅ **Two processing paths**:
- Path A: YOLO detections (high priority person analysis)
- Path B: Background scans (safety net for missed objects)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  reCamera (192.168.2.140)                   │
│  ● YOLO object detection                                    │
│  ● Entrance/exit tracking (3s debounce)                     │
│  ● RTSP stream (rtsp://192.168.2.140:8554/stream)          │
└─────────┬──────────────────────────────────┬────────────────┘
          │                                   │
          │ POST /api/vision/event            │ RTSP Stream
          │ (with YOLO metadata)              │
          │                                   │
          ▼                                   ▼
┌─────────────────────────┐         ┌────────────────────────┐
│  Vision Event API       │         │ Background Scan        │
│  (Always Running)       │         │ (Cron: Every 1min)     │
│  ● Port 8000            │         │ ● Captures every 20s   │
│  ● Receives events      │         │ ● Uses CLIP analysis   │
│  ● Captures snapshots   │         │ ● Catches YOLO misses  │
│  ● Queues for processing│         │                        │
└─────────┬───────────────┘         └──────────┬─────────────┘
          │                                    │
          │ Writes to queue                   │
          ▼                                    ▼
┌──────────────────────────────────────────────────────────────┐
│           Snapshot Processing Queue (JSON)                   │
│  data/vision/snapshot_queue.json                            │
│  ● YOLO detections with metadata                            │
│  ● Background scans                                         │
│  ● Marked processed: false → true                           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Processed by (Cron: Every 30min)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│            Vision Processing Cortex                          │
│  ● Loads models on-demand                                    │
│  ● Batch processes queue                                     │
│  ● Unloads models after completion                           │
└────────┬─────────────────────────────────────────────────────┘
         │
         ├─► Person Detection (YOLO detections)
         │   ├─ InsightFace: Face recognition (~200MB)
         │   │  └─ 512-dim embeddings
         │   │  └─ Match against contacts (cosine similarity)
         │   ├─ FER: Emotion detection (~100MB)
         │   │  └─ 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral
         │   ├─ Age/Gender: Estimation (~50MB)
         │   │  └─ Built into InsightFace
         │   └─ Total: ~350MB
         │
         └─► General Scenes (Background scans)
             └─ CLIP ViT-B/32 (~350MB)
                └─ Scene understanding
                └─ "What did YOLO miss?"

         ↓
┌──────────────────────────────────────────────────────────────┐
│               Results Storage                                 │
│  ● Episodic Memory (salience 0.7-0.9)                       │
│  ● Contact Memory (face matches)                             │
│  ● Processed history (data/vision/processed.json)            │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Path A: YOLO Detection (High Priority)

```
1. reCamera YOLO detects person
   └─ Confidence: 0.95, BBox: [100, 150, 300, 450]

2. 3-second debounce (stable detection)

3. POST http://192.168.2.137:8000/api/vision/event
   {
     "event": "entrance",
     "object": "person",
     "timestamp": 1733850123456,
     "scene": ["person"],
     "yolo_detection": {
       "class": "person",
       "confidence": 0.95,
       "bbox": [100, 150, 300, 450],
       "track_id": 123
     }
   }

4. Vision API:
   ├─ Captures RTSP snapshot
   ├─ Saves to: data/vision/snapshots/entrance_person_1733850123456.jpg
   └─ Queues with YOLO metadata

5. Queue Entry (snapshot_queue.json):
   {
     "snapshot_path": "/home/toastee/BioMimeticAi/data/vision/snapshots/entrance_person_1733850123456.jpg",
     "event_type": "entrance",
     "detected_object": "person",
     "timestamp": 1733850123456,
     "yolo_detection": { ... },
     "processed": false
   }

6. (30 minutes later) Vision Cortex:
   ├─ Loads InsightFace, FER models
   ├─ Crops image to YOLO bbox
   ├─ Analyzes face:
   │  ├─ Embedding: [512-dim vector]
   │  ├─ Matches against contacts: "John" (confidence: 0.87)
   │  ├─ Age: ~32
   │  ├─ Gender: Male
   │  └─ Emotion: happy (0.82)
   ├─ Updates contact memory for "John"
   ├─ Stores in episodic memory (salience: 0.9)
   └─ Marks processed: true

7. Unloads models, frees VRAM
```

### Path B: Background Scan (Safety Net)

```
1. Every 20 seconds: Background scan runs

2. Captures RTSP snapshot
   └─ Saves to: data/vision/background_scans/background_scan_1733850200000.jpg

3. Analyzes with CLIP:
   Queries: ["empty room", "person in scene", "multiple people", "pet animal", ...]
   
   Results:
   ├─ "normal everyday scene": 0.65
   ├─ "empty room": 0.25
   └─ "person in scene": 0.10

4. If significant (confidence > 0.5):
   └─ Stores in episodic memory (salience: 0.4)

5. Cleanup: Deletes scans older than 24 hours
```

---

## Components

### 1. Model Manager (`src/core/model_manager.py`)
**Purpose**: Load/unload specialized models on-demand

**Features**:
- VRAM monitoring
- Load models: `load_face_recognition()`, `load_emotion_detector()`, `load_clip()`
- Unload models: `unload_model(name)`, `unload_all()`
- Analysis methods: `analyze_face()`, `detect_emotion()`, `analyze_scene_clip()`

**Models**:
| Model | Size | Purpose |
|-------|------|---------|
| InsightFace (buffalo_l) | ~200MB | Face recognition + age/gender |
| FER | ~100MB | Emotion detection (7 classes) |
| CLIP ViT-B/32 | ~350MB | Scene understanding |
| **Total** | **~700MB** | **7% of 10GB budget** |

**Test**:
```bash
python src/core/model_manager.py
```

### 2. Person Analyzer (`src/core/person_analyzer.py`)
**Purpose**: Analyze people in images, match against contacts

**Features**:
- Face matching (cosine similarity, threshold: 0.6)
- Contact registration
- Observation tracking
- Automatic contact memory updates

**Usage**:
```python
from src.core.person_analyzer import PersonAnalyzer
import numpy as np

analyzer = PersonAnalyzer()

# Analyze person
result = analyzer.analyze_person(image_np, yolo_bbox=[100, 150, 300, 450])

# Register new contact
analyzer.register_new_contact("John Doe", image_np)

# Check statistics
stats = analyzer.get_statistics()
```

### 3. Vision API (`src/core/vision_api.py`)
**Purpose**: Receive events from reCamera, queue snapshots

**Endpoints**:
- `POST /api/vision/event` - Receive entrance/exit events
- `GET /api/vision/events` - Get recent events
- `GET /api/vision/status` - System status

**Start**:
```bash
bash scripts/start_vision_api.sh
```

### 4. Vision Processing Cortex (`scripts/cron/vision_processing.py`)
**Purpose**: Process queued snapshots every 30 minutes

**Schedule**: `*/30 * * * *` (every 30 minutes)

**Process**:
1. Load queue
2. Load models (face, emotion, CLIP)
3. Process each snapshot:
   - Person: Face analysis, contact matching
   - Other: CLIP scene understanding
4. Update episodic memory
5. Mark as processed
6. Unload models

**Run manually**:
```bash
python scripts/cron/vision_processing.py
```

### 5. Background Scan (`scripts/cron/background_vision_scan.py`)
**Purpose**: Capture & analyze scene every 20 seconds (safety net)

**Schedule**: `*/1 * * * *` (every minute, internal 20s loop)

**Process**:
1. Capture snapshot every 20s
2. Analyze with CLIP
3. Detect scene changes
4. Store if significant
5. Cleanup old scans (>24h)

**Run manually**:
```bash
python scripts/cron/background_vision_scan.py
```

---

## Installation

### 1. Install Python Dependencies

```bash
cd /home/toastee/BioMimeticAi
source venv/bin/activate

# Face recognition (~200MB)
pip install insightface onnxruntime-gpu

# Emotion detection (~100MB)
pip install fer tensorflow

# CLIP (~350MB)
pip install git+https://github.com/openai/CLIP.git

# Image processing
pip install pillow numpy torch torchvision
```

### 2. Configure Cron Jobs

```bash
crontab -e
```

Add these lines:
```cron
# Background vision scan (every minute, runs 20s loop)
*/1 * * * * /home/toastee/BioMimeticAi/scripts/cron/background_vision_scan.py >> /home/toastee/BioMimeticAi/logs/cron.log 2>&1

# Vision processing (every 30 minutes)
*/30 * * * * /home/toastee/BioMimeticAi/scripts/cron/vision_processing.py >> /home/toastee/BioMimeticAi/logs/cron.log 2>&1
```

### 3. Start Vision API

```bash
bash scripts/start_vision_api.sh
```

Should see:
```
Starting Vision Event API on port 8000...
Endpoints:
  POST /api/vision/event - Receive entrance/exit events
  GET  /api/vision/events - Get recent events
  GET  /api/vision/status - Get system status
```

---

## Configuration

### Vision Models Config (`config/vision_models.json`)

```json
{
  "models": {
    "face_recognition": {
      "enabled": true,
      "model_name": "buffalo_l",
      "vram_mb": 200
    },
    "emotion": {
      "enabled": true,
      "model_name": "fer",
      "vram_mb": 100
    },
    "clip": {
      "enabled": true,
      "model_name": "ViT-B/32",
      "vram_mb": 350
    }
  },
  "total_vram_budget_mb": 700
}
```

---

## Usage

### Update reCamera Node-RED Flow

Modify your `entrance_exit_tracker.js` to send YOLO metadata:

```javascript
// In your entrance event
const event = {
    event: "entrance",
    object: "person",
    timestamp: Date.now(),
    scene: Object.keys(sceneState.objects),
    yolo_detection: {
        class: msg.payload.data.class,
        confidence: msg.payload.data.confidence,
        bbox: msg.payload.data.bbox,  // [x1, y1, x2, y2]
        track_id: msg.payload.data.track_id
    }
};

// POST to vision API
node.send({
    url: "http://192.168.2.137:8000/api/vision/event",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    payload: event
});
```

### Register a New Contact

```python
from src.core.person_analyzer import PersonAnalyzer
from PIL import Image
import numpy as np

# Load image with person's face
image = Image.open("path/to/photo.jpg")
image_np = np.array(image)

# Register
analyzer = PersonAnalyzer()
success = analyzer.register_new_contact("John Doe", image_np, metadata={
    "relationship": "friend",
    "notes": "Likes coffee"
})

if success:
    print("✓ Contact registered!")
```

### Check System Status

```bash
# Vision API status
curl http://localhost:8000/api/vision/status

# Recent events
curl http://localhost:8000/api/vision/events?limit=10

# Check cron logs
tail -f logs/cron.log

# Check vision processing log
tail -f logs/vision_processing.log
```

### Monitor VRAM Usage

```python
from src.core.model_manager import ModelManager

manager = ModelManager()

# Load models
manager.load_face_recognition()
manager.load_emotion_detector()
manager.load_clip()

# Check usage
print(f"Loaded: {manager.get_loaded_models()}")
print(f"Estimated VRAM: {manager.estimate_vram_usage()}MB")

# Unload
manager.unload_all()
```

---

## File Structure

```
BioMimeticAi/
├── src/core/
│   ├── model_manager.py         # Load/unload vision models
│   ├── person_analyzer.py       # Face matching, demographics
│   └── vision_api.py            # Flask API (always running)
│
├── scripts/
│   ├── start_vision_api.sh      # Start API service
│   └── cron/
│       ├── vision_processing.py       # Process queue (30min)
│       └── background_vision_scan.py  # Safety net (1min/20s)
│
├── data/
│   ├── contacts.json            # Known contacts with embeddings
│   └── vision/
│       ├── snapshots/           # Event snapshots
│       ├── background_scans/    # Background scans
│       ├── snapshot_queue.json  # Processing queue
│       └── processed.json       # Processing history
│
├── config/
│   └── vision_models.json       # Model configuration
│
└── docs/
    └── VISION_ARCHITECTURE.md   # This file
```

---

## VRAM Budget

```
Current Allocation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total RTX 3090:     24GB
Mistral-Small-22B:  ~14GB (Q4 quantization)
Available:          ~10GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Vision Models (loaded on-demand):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
InsightFace:        200MB  (2% of available)
FER:                100MB  (1% of available)
CLIP:               350MB  (3.5% of available)
─────────────────────────────────────────────────────
Total:              650MB  (6.5% of available)
Safety Buffer:      ~50MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Safe to run alongside Mistral-Small-22B!
```

---

## Benefits

✅ **No VRAM conflicts** - Small models coexist with text model  
✅ **Scalable** - Process in batches, not real-time  
✅ **Efficient** - Only analyze what matters (YOLO detections + 20s scans)  
✅ **Contact matching** - Face recognition integrated with memory  
✅ **Cron-based** - Fits existing cortex architecture  
✅ **Placeholder-ready** - Framework complete, models optional  

---

## Troubleshooting

### Models not loading?
```bash
# Check installations
pip list | grep -E "insightface|fer|clip"

# Test individually
python -c "import insightface; print('InsightFace OK')"
python -c "from fer import FER; print('FER OK')"
python -c "import clip; print('CLIP OK')"
```

### RTSP capture failing?
```bash
# Test RTSP stream
ffmpeg -i rtsp://192.168.2.140:8554/stream -frames:v 1 test.jpg

# Check vision API logs
tail -f logs/vision_api.log
```

### Cron jobs not running?
```bash
# Check cron logs
tail -f logs/cron.log

# Run manually
python scripts/cron/vision_processing.py
python scripts/cron/background_vision_scan.py

# Verify crontab
crontab -l
```

---

**System ready!** 🎯

The vision system will now:
1. Receive YOLO detections from reCamera
2. Queue snapshots for processing  
3. Process every 30 minutes with specialized models
4. Match faces against contacts
5. Catch YOLO misses with background scans
6. Store everything in episodic memory

Install the models when ready to activate full analysis!
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
