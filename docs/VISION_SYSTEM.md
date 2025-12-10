# Vision System Integration

Integration of reCamera with BioMimeticAI for entrance/exit detection and snapshot capture.

## Architecture

```
┌─────────────┐     RTSP Stream      ┌──────────────────┐
│  reCamera   │◄─────────────────────┤ RTSP Capture     │
│ 192.168.2.140│                      │ (ffmpeg)         │
└──────┬──────┘                       └────────┬─────────┘
       │                                       │
       │ Node-RED                              │
       │ Entrance/Exit                         │
       │ Detection                             │
       │                                       │
       ▼                                       │
┌──────────────────┐    Trigger               │
│ Vision Event API │◄──────────────────────────┘
│ Port 8000        │
└────────┬─────────┘
         │
         ├──► Snapshot Capture (on entrance)
         │    └──► RTSP stream → JPEG file
         │
         ├──► Vision Processing
         │    └──► Analyze with LLM (future)
         │    └──► Face recognition (future)
         │
         └──► Episodic Memory
              └──► Store events + snapshots
```

## Components

### 1. RTSP Capture Module
**File**: `src/core/rtsp_capture.py`

Captures single frames from RTSP stream using ffmpeg.

**Usage**:
```python
from src.core.rtsp_capture import RTSPCapture

capture = RTSPCapture(rtsp_url="rtsp://192.168.2.140:8554/stream")

# Capture snapshot
snapshot_path = capture.capture_snapshot()

# Capture as base64 (for LLM APIs)
base64_image = capture.capture_snapshot_base64()

# Test connection
if capture.test_connection():
    print("RTSP stream accessible")
```

**Features**:
- Efficient single-frame capture (no persistent connection)
- Automatic filename generation with timestamps
- Base64 encoding for LLM vision APIs
- Automatic cleanup of old snapshots
- Connection testing

**Snapshots saved to**: `data/vision/snapshots/`

### 2. Vision Event API
**File**: `src/core/vision_api.py`

Flask API that receives entrance/exit events from reCamera and triggers snapshot capture.

**Endpoints**:
- `POST /api/vision/event` - Receive entrance/exit events
- `GET /api/vision/events` - Get recent events
- `GET /api/vision/status` - Get system status

**Event Flow**:
1. reCamera Node-RED detects entrance/exit
2. Sends POST to `/api/vision/event`
3. API triggers snapshot capture (background thread)
4. Snapshot is processed and analyzed
5. Results stored in episodic memory

### 3. Vision Processor
**File**: `src/core/vision_processor.py`

Analyzes snapshots using LLM vision capabilities (placeholder - not fully implemented).

**Future Features**:
- Verify object detection accuracy
- Extract contextual details (clothing, posture, items)
- Face recognition for contact matching
- Behavioral analysis
- Intent assessment

### 4. Entrance/Exit Detection
**File**: `scripts/camera_entrance_exit_tracker.js`

Node-RED function node that runs on reCamera to detect scene changes.

**Features**:
- 3-second debounce (object must be stable)
- Tracks current scene state
- Sends entrance/exit events to core API
- Prevents false positives from flickering detections

**Deployment**: Copy to reCamera Node-RED flow

## Setup Instructions

### Prerequisites

1. **ffmpeg** (for snapshot capture):
```bash
sudo apt install ffmpeg
```

2. **reCamera Configuration**:
- IP: 192.168.2.140
- RTSP stream: rtsp://192.168.2.140:8554/stream
- SSH access configured (key: ~/.ssh/recamera_key)

3. **Core System**:
- IP: 192.168.2.137 (accessible from reCamera)
- Port 8000 available for vision API

### Installation

1. **Test RTSP Connection**:
```bash
bash scripts/test_camera_vision.sh
```

This will:
- Check ffmpeg installation
- Test RTSP stream connection
- Capture 3 test snapshots
- Test SSH access to reCamera

2. **Start Vision API**:
```bash
bash scripts/start_vision_api.sh
```

Or manually:
```bash
source venv/bin/activate
python src/core/vision_api.py
```

API will start on http://192.168.2.137:8000

3. **Deploy Node-RED Flow to reCamera**:

The entrance/exit tracker needs to be deployed to reCamera's Node-RED instance.

**Option A**: Import flow from backup
```bash
# Flow is saved in: scripts/recamera_flows_backup.json
# Import via Node-RED UI on reCamera
```

**Option B**: Manual deployment
- SSH into reCamera
- Access Node-RED (usually port 1880)
- Create function node with code from `scripts/camera_entrance_exit_tracker.js`
- Configure to POST to `http://192.168.2.137:8000/api/vision/event`

### Verification

1. **Check API Status**:
```bash
curl http://192.168.2.137:8000/api/vision/status
```

2. **Monitor Snapshots**:
```bash
watch -n 1 'ls -lh data/vision/snapshots/'
```

3. **View Recent Events**:
```bash
curl http://192.168.2.137:8000/api/vision/events?limit=10
```

4. **Trigger Test Event** (manual):
```bash
curl -X POST http://192.168.2.137:8000/api/vision/event \
  -H "Content-Type: application/json" \
  -d '{
    "event": "entrance",
    "object": "person",
    "timestamp": 1234567890000,
    "scene": ["person"]
  }'
```

This should trigger a snapshot capture.

## Configuration

### RTSP Stream URL

Update in `src/core/rtsp_capture.py` if needed:
```python
self.rtsp_url = "rtsp://192.168.2.140:8554/stream"
```

### Snapshot Directory

Default: `data/vision/snapshots/`

Change in RTSPCapture initialization:
```python
capture = RTSPCapture(snapshot_dir="/custom/path")
```

### Capture Timeout

Default: 10 seconds

Adjust per-capture:
```python
snapshot_path = capture.capture_snapshot(timeout=15)
```

### Cleanup Policy

Old snapshots are cleaned up automatically (can be configured):
```python
capture.cleanup_old_snapshots(
    max_age_hours=24,  # Delete files older than 24 hours
    keep_count=100     # Always keep 100 most recent files
)
```

## Troubleshooting

### "ffmpeg not found"
```bash
sudo apt install ffmpeg
```

### "RTSP connection failed"
1. Check reCamera is accessible:
   ```bash
   ping 192.168.2.140
   ```

2. Test RTSP stream manually:
   ```bash
   ffplay rtsp://192.168.2.140:8554/stream
   ```

3. Check reCamera RTSP server is running

### "SSH connection failed"
1. Verify SSH key is configured:
   ```bash
   ssh -i ~/.ssh/recamera_key recamera@192.168.2.140
   ```

2. Check key permissions:
   ```bash
   chmod 600 ~/.ssh/recamera_key
   ```

### "No snapshots captured"
1. Check vision API logs for errors
2. Verify RTSP stream is active
3. Test manual snapshot capture:
   ```bash
   python3 src/core/rtsp_capture.py
   ```

### "Events not arriving from reCamera"
1. Check reCamera can reach core:
   ```bash
   # From reCamera
   curl http://192.168.2.137:8000/api/vision/status
   ```

2. Verify Node-RED flow is deployed and running
3. Check Node-RED debug output

## Future Enhancements

### Vision LLM Integration
- [ ] Integrate with vision-capable LLM (LLaVA, GPT-4V, etc.)
- [ ] Verify object detection matches reality
- [ ] Extract detailed scene descriptions
- [ ] Assess behavioral patterns

### Face Recognition
- [ ] Extract face embeddings from snapshots
- [ ] Compare with known contacts
- [ ] Update contact memory with visual data
- [ ] Track when known individuals enter/exit

### Behavioral Analysis
- [ ] Body language analysis
- [ ] Gait recognition
- [ ] Intent assessment
- [ ] Anomaly detection

### Integration with Axiom System
- [ ] Trigger axiom chains on vision events
- [ ] Use axioms for threat assessment
- [ ] Generate appropriate responses
- [ ] Log axiom decisions

### Performance Optimization
- [ ] Batch snapshot processing
- [ ] GPU acceleration for vision models
- [ ] Caching of recent analyses
- [ ] Reduce RTSP capture latency

## File Structure

```
BioMimeticAi/
├── data/
│   └── vision/
│       └── snapshots/          # Captured snapshots
│           └── entrance_person_1234567890.jpg
├── docs/
│   └── VISION_SYSTEM.md        # This file
├── scripts/
│   ├── camera_entrance_exit_tracker.js  # Node-RED flow
│   ├── camera_ssh.py           # SSH helper
│   ├── test_camera_vision.sh   # Test script
│   └── start_vision_api.sh     # Start API script
└── src/
    └── core/
        ├── rtsp_capture.py     # RTSP snapshot capture
        ├── vision_api.py       # Flask API server
        └── vision_processor.py # Vision analysis (placeholder)
```

## API Reference

### POST /api/vision/event

Receive entrance/exit event from camera.

**Request**:
```json
{
  "event": "entrance",
  "object": "person",
  "timestamp": 1234567890000,
  "scene": ["person", "car"]
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Event received and stored",
  "event_id": 42
}
```

### GET /api/vision/events

Get recent vision events.

**Query Parameters**:
- `limit` (int): Max events to return (default: 50)
- `type` (string): Filter by event type ('entrance' or 'exit')

**Response**:
```json
{
  "events": [
    {
      "event": "entrance",
      "object": "person",
      "timestamp": 1234567890000,
      "received_at": 1234567891000,
      "scene": ["person"]
    }
  ],
  "total": 1
}
```

### GET /api/vision/status

Get vision system status.

**Response**:
```json
{
  "status": "online",
  "recent_event_count": 42,
  "last_event": {
    "event": "entrance",
    "object": "person",
    "timestamp": 1234567890000
  },
  "last_event_seconds_ago": 15.3
}
```
