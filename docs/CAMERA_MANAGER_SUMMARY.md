# Camera Manager - Implementation Summary

**Date**: 2025-12-19
**Status**: Design Complete, Ready for Implementation

---

## What We Built Today

Created comprehensive documentation for the **Camera Manager** - a multi-modal perception dashboard that extends `/home/toastee/ai_led_mapper` to become the sensory input system for BioMimeticAI.

### Documentation Created

1. **`CAMERA_MANAGER.md`** (700+ lines)
   - Complete architecture specification
   - API endpoints defined
   - UI mockups and component specs
   - Integration with existing BioMimeticAI systems
   - Implementation roadmap

2. **Updated `README.MD`**
   - Added Camera Manager as core component
   - Updated architecture layers
   - Documented camera types and capabilities

3. **Updated `CLAUDE.md`**
   - Added perception layer documentation
   - Noted "no backwards compatibility" principle
   - Updated directory structure
   - Added camera manager to running services

---

## Key Concepts Captured

### 1. "ADHD for AI"
The system gives LLMs the ability to switch attention between multiple sensory inputs, similar to human attention:
- Multiple cameras stream simultaneously
- Only 1-2 cameras actively processed by AI at a time
- Salience mechanism decides which deserves attention
- Ring buffers allow "looking back" when events occur

### 2. "Think Slow Like a Human"
When something interesting happens (e.g., person enters room):
- Event notification arrives *after* the event
- Ring buffer contains previous 5 seconds of frames
- AI can review what led to the event
- More human-like situational awareness

### 3. "This Server IS the AI"
BioMimeticAI is not a single program - it's the entire Ubuntu system:
- Separate services gather sensory information independently
- Each prepares data for the core AI to process
- Distributed cortex architecture mimics biological brain
- No single point of failure

---

## Architecture at a Glance

```
Camera Manager (ai_led_mapper:8001)
├── Frontend: React + TypeScript + Tailwind
│   ├── 640x480 preview tiles (up to 6 visible)
│   ├── Network scanner (nmap-based)
│   ├── Salience schedule editor
│   └── Memory browser
│
├── Backend: FastAPI + Python
│   ├── Camera drivers (RTSP, DroidCam, V4L2)
│   ├── Ring buffers (5 seconds per camera)
│   ├── Salience engine (priority calculation)
│   └── Network discovery
│
└── Integration
    ├── Vision API (port 8000) ← Events & frames
    ├── Episodic Memory ← Scene descriptions
    ├── Contact Memory ← Face recognition
    └── reCamera Events ← Entrance/exit triggers
```

---

## Camera Types Supported

1. **RTSP Cameras** (port 554, 8554)
   - reCamera E81 (with entrance/exit detection)
   - IP security cameras
   - ONVIF devices

2. **DroidCam** (port 4747)
   - Phone cameras via WiFi
   - USB mode (appears as V4L2)

3. **V4L2 Devices**
   - USB webcams
   - Virtual cameras

---

## Salience Mechanism

**Schedule-Based Priority** (0.0 - 0.5):
```json
{
  "camera": "workshop_recamera",
  "start": "09:00",
  "end": "17:00",
  "days": [0,1,2,3,4],  // Monday-Friday
  "priority": 0.5,
  "reason": "Work hours"
}
```

**Event-Based Boost** (0.0 - 0.3):
```json
{
  "source": "workshop_recamera",
  "event": "entrance",
  "boost": 0.3,
  "duration": 300,  // 5 minutes
  "reason": "Person entered"
}
```

**Total Priority** = Schedule + Event + Manual (max 1.0)

Top 1-2 cameras by priority receive AI focus.

---

## Ring Buffer System

Each enabled camera maintains:
- **Buffer size**: 5 seconds @ 5fps = 25 frames
- **Memory**: ~2-5MB per camera (JPEG compressed)
- **Purpose**: "Look back" when events trigger
- **API**: `GET /api/cameras/{id}/context?seconds=3`

Example use case:
```
14:32:00 - Person enters workshop
14:32:02 - reCamera sends entrance event
14:32:03 - Camera Manager receives event
14:32:03 - Salience engine boosts workshop camera priority
14:32:03 - AI requests last 3 seconds from ring buffer
14:32:03 - Frames sent to Vision API for analysis
14:32:05 - "Person walking to desk with laptop" → Episodic memory
```

---

## Integration Points

### 1. reCamera → Camera Manager
- reCamera sends entrance/exit events to Vision API (port 8000)
- Camera Manager subscribes to these events
- Salience engine updates camera priorities
- High-priority cameras trigger AI processing

### 2. Camera Manager → Vision API
- AI requests processing for selected camera
- Frames from ring buffer sent to Vision API
- Vision LLM analyzes scene
- Results returned with object detections, scene description

### 3. Vision API → Episodic Memory
- Scene descriptions stored as episodes
- High salience (0.7-0.9) for vision events
- Tagged with camera ID, timestamp, snapshot URL
- Retrievable via memory browser UI

### 4. Vision API → Contact Memory
- Face recognition matches known contacts
- Updates last_visual_sighting timestamp
- Stores visual confidence score
- Triggers high-salience episode for known persons

---

## Implementation Roadmap

### Phase 1: Core Camera Manager (Week 1)
- Camera abstraction classes (RTSP, DroidCam, V4L2)
- Ring buffer implementation
- Basic CRUD API endpoints
- Frontend: Camera tile component + grid layout

### Phase 2: Network Discovery (Week 1)
- nmap integration for port scanning
- RTSP/DroidCam/V4L2 verification
- Network scanner dialog UI
- Add discovered camera flow

### Phase 3: Salience System (Week 2)
- Salience engine with priority calculation
- Schedule-based + event-based priority
- Manual focus override
- Frontend: Priority visualization + schedule editor

### Phase 4: BioMimeticAI Integration (Week 2)
- Subscribe to vision API events
- Send frames to Vision API
- Store results in episodic memory
- Face recognition → contact memory
- Memory browser UI

### Phase 5: Polish & Documentation (Week 3)
- Update all docs
- User guide with screenshots
- System health monitoring
- Performance optimization
- Error handling + recovery

---

## Key Files

### New Documentation
- `/home/toastee/BioMimeticAi/docs/CAMERA_MANAGER.md` - Complete spec (700+ lines)
- `/home/toastee/BioMimeticAi/docs/CAMERA_MANAGER_SUMMARY.md` - This file

### Updated Documentation
- `/home/toastee/BioMimeticAi/README.MD` - Added Camera Manager section
- `/home/toastee/BioMimeticAi/CLAUDE.md` - Added perception layer + no-backwards-compat note

### Existing Related Files (Reference)
- `/home/toastee/BioMimeticAi/docs/VISION_SYSTEM.md` - Vision API details
- `/home/toastee/BioMimeticAi/docs/VISION_ARCHITECTURE.md` - Vision processing
- `/home/toastee/BioMimeticAi/scripts/camera_entrance_exit_tracker.js` - reCamera Node-RED
- `/home/toastee/ai_led_mapper/CAMERA_INTEGRATION.md` - Original camera notes

### Implementation Location
- `/home/toastee/ai_led_mapper` - Will be extended with Camera Manager features

---

## Design Principles

1. **No Backwards Compatibility**
   - We are the only user
   - APIs can change freely
   - No deprecation warnings needed
   - Just update and move forward

2. **Fail Isolated**
   - Each camera runs independently
   - Camera failure doesn't crash system
   - Ring buffer handles disconnections gracefully
   - Cached frames used when camera offline

3. **Human-Like Perception**
   - Multiple senses (cameras) always active
   - Only focus deeply on 1-2 at a time
   - Events trigger attention shifts
   - Memory of "what just happened" via ring buffers

4. **Event-Driven Architecture**
   - reCamera sends events (entrance/exit)
   - Salience engine updates priorities
   - AI switches focus automatically
   - Manual override always available

---

## Next Steps

1. **Read this summary** to ensure alignment with your vision
2. **Review `/home/toastee/BioMimeticAi/docs/CAMERA_MANAGER.md`** for implementation details
3. **Decide on implementation order** (follow roadmap or prioritize differently?)
4. **Begin Phase 1** when ready (camera abstraction + ring buffers)

---

## Questions Resolved Today

✅ Should this be separate service or extend ai_led_mapper? → Extend ai_led_mapper (Option B)
✅ How do multiple cameras work simultaneously? → Ring buffers, only 1-2 actively processed
✅ How does reCamera integration work? → Events via Vision API, salience boost triggers focus
✅ Network discovery method? → nmap for port scanning
✅ UI layout? → 640x480 tiles, up to 6 visible, grid layout
✅ Ring buffer purpose? → "Look back" when events happen (human-like delayed reaction)
✅ Salience mechanism? → Schedule-based + event-driven + manual override
✅ Integration with BioMimeticAI? → Vision API → Episodic Memory → Contact Memory

---

**Status**: Ready to implement. All architectural decisions documented. No need to re-explain in future conversations - just reference `/home/toastee/BioMimeticAi/docs/CAMERA_MANAGER.md`.
