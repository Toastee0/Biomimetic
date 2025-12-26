# Camera Manager - Multi-Modal Perception Dashboard

**Version**: 1.0
**Last Updated**: 2025-12-19
**Purpose**: Unified camera management system for BioMimeticAI's distributed perception architecture

---

## Philosophy

**The BioMimeticAI system IS this Ubuntu server.** Separate services gather sensory information independently and prepare it for the core AI system to process. The Camera Manager provides "ADHD-like" attention switching between multiple sensory inputs - the ability to maintain multiple video streams while focusing AI processing on only 1-2 cameras at a time.

**Key Insight**: We think slow like a human. When something interesting happens, we want to see a few seconds of *previous* video (because the notification/interest arrived after the event). Each camera maintains a ring buffer so the AI can "look back" when switching attention.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Camera Types](#camera-types)
4. [Network Discovery](#network-discovery)
5. [Ring Buffer System](#ring-buffer-system)
6. [Salience Mechanism](#salience-mechanism)
7. [UI Specification](#ui-specification)
8. [Integration Points](#integration-points)
9. [API Specification](#api-specification)
10. [Configuration Files](#configuration-files)
11. [Installation](#installation)

---

## Overview

### Current State

**ai_led_mapper** serves as the foundation:
- React + TypeScript + Tailwind frontend (Vite)
- FastAPI Python backend
- Basic camera integration (V4L2, DroidCam)
- Camera preview component

**Transformation** → **BioMimeticAI Perception Dashboard**

### Core Capabilities

1. **Multi-Camera Management**: Configure and monitor up to 6+ cameras simultaneously
2. **Live Streaming**: Each enabled camera maintains an active connection with ring buffer
3. **Selective Processing**: Only 1-2 cameras actively processed by AI at a time
4. **Event-Driven Switching**: reCamera entrance/exit events trigger attention shifts
5. **Schedule-Based Priority**: Time-based camera selection (e.g., workshop hours vs. living room hours)
6. **Memory Integration**: Scene descriptions stored in BioMimeticAI episodic memory
7. **Network Scanning**: Auto-discover cameras on local network (nmap-based)

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAMERA MANAGER DASHBOARD                         │
│                   (Extended ai_led_mapper)                          │
│                                                                     │
│  Frontend (React + TypeScript + Tailwind)                          │
│  ● 640x480 preview tiles (up to 6 visible)                         │
│  ● Camera config management                                        │
│  ● Network scanner interface                                       │
│  ● Salience schedule editor                                        │
│  ● Memory system browser                                           │
│                                                                     │
│  Backend (FastAPI + Python)                                        │
│  ● Camera connection manager                                       │
│  ● Ring buffer management (per-camera)                             │
│  ● Network discovery (nmap wrapper)                                │
│  ● Integration with BioMimeticAI vision API                        │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ├─────────────────────────────────────┐
                     │                                     │
                     ▼                                     ▼
    ┌────────────────────────────┐      ┌─────────────────────────────┐
    │   CAMERA SOURCES           │      │  BIOMIMETIC AI CORE         │
    │                            │      │                             │
    │ ● RTSP Cameras             │      │ ● Vision API (port 8000)    │
    │   - reCamera E81           │      │ ● Episodic Memory           │
    │   - IP Security Cams       │      │ ● Contact Memory            │
    │                            │      │ ● LLM Processing            │
    │ ● DroidCam (WiFi/USB)     │      │ ● Axiom System              │
    │   - Phone cameras          │      │                             │
    │                            │      └─────────────────────────────┘
    │ ● V4L2 Devices             │
    │   - USB Webcams            │
    │   - Virtual devices        │
    │                            │
    └────────────────────────────┘

                     │
                     ▼
    ┌────────────────────────────────────────────────────┐
    │         PER-CAMERA RING BUFFERS                    │
    │                                                    │
    │  Camera 1: [frame_t-5, ..., frame_t-1, frame_t]   │
    │  Camera 2: [frame_t-5, ..., frame_t-1, frame_t]   │
    │  Camera 3: [frame_t-5, ..., frame_t-1, frame_t]   │
    │  ...                                               │
    │                                                    │
    │  Buffer: 5 seconds @ 5fps = 25 frames (~2-5MB)    │
    │  AI retrieves: "Last 3 seconds" when switching     │
    └────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Camera Configuration
   ├─ User adds camera (manual or discovered)
   ├─ Config stored in data/camera_manager/cameras.json
   └─ Camera marked enabled/disabled

2. Camera Connection (when enabled)
   ├─ Backend opens stream (RTSP/DroidCam/V4L2)
   ├─ Frame capture at 5fps (conserve bandwidth)
   ├─ Frames stored in ring buffer (5 seconds = 25 frames)
   └─ Buffer continuously rotates (oldest frame discarded)

3. Salience Evaluation (every second)
   ├─ Check schedule rules (e.g., "workshop during work hours")
   ├─ Check event triggers (e.g., "person entered via reCamera")
   ├─ Calculate priority score per camera
   └─ Select highest priority camera(s) for AI processing

4. AI Processing (active focus)
   ├─ Request last N seconds from selected camera buffer
   ├─ Send to Vision API (port 8000)
   ├─ LLM analyzes scene
   ├─ Results stored in episodic memory (salience 0.7-0.9)
   └─ Continue monitoring or switch focus

5. Event-Driven Switching
   ├─ reCamera sends entrance/exit event
   ├─ Triggers immediate attention switch
   ├─ Retrieve ring buffer (captures what led to event)
   └─ Process with Vision LLM
```

---

## Camera Types

### 1. RTSP Cameras

**Protocol**: Real-Time Streaming Protocol
**Port**: 554 (default)
**Examples**: IP cameras, reCamera, ONVIF devices

**Backend Implementation**:
```python
class RTSPCamera:
    def __init__(self, url: str, fps: int = 5):
        self.url = url  # rtsp://192.168.2.140:8554/stream
        self.fps = fps
        self.process = None  # ffmpeg subprocess

    def connect(self):
        # ffmpeg -rtsp_transport tcp -i {url} -f image2pipe -vcodec mjpeg -r {fps} -
        self.process = subprocess.Popen([...])

    def get_frame(self) -> bytes:
        # Read JPEG from ffmpeg pipe
        return self.process.stdout.read(frame_size)
```

**Discovery**:
- nmap scan for port 554
- Try common RTSP paths: `/stream`, `/live`, `/h264`, `/h265`
- ONVIF discovery (optional, requires `python-onvif-zeep`)

**Configuration Example**:
```json
{
  "id": "recamera_workshop",
  "name": "Workshop reCamera E81",
  "type": "rtsp",
  "url": "rtsp://192.168.2.140:8554/stream",
  "fps": 5,
  "enabled": true,
  "events": {
    "entrance_exit": true,
    "yolo_detection": true
  }
}
```

### 2. DroidCam (Phone Cameras)

**Modes**: WiFi (HTTP MJPEG), USB (V4L2 virtual device)
**Port**: 4747 (default WiFi)
**App**: DroidCam (Android/iOS)

**Backend Implementation**:
```python
class DroidCamCamera:
    def __init__(self, ip: str, port: int = 4747):
        self.base_url = f"http://{ip}:{port}"
        self.mjpeg_url = f"{self.base_url}/mjpegfeed"
        self.frame_url = f"{self.base_url}/shot.jpg"

    async def connect(self):
        # Connect to MJPEG stream
        async with aiohttp.ClientSession() as session:
            async with session.get(self.mjpeg_url) as resp:
                # Parse MJPEG boundaries and yield frames
                ...
```

**Discovery**:
- nmap scan for port 4747
- HTTP GET to `/mjpegfeed` to verify
- mDNS/Bonjour discovery (optional)

**Configuration Example**:
```json
{
  "id": "phone_main",
  "name": "Main Phone Camera",
  "type": "droidcam",
  "ip": "192.168.2.39",
  "port": 4747,
  "fps": 5,
  "enabled": false
}
```

### 3. V4L2 Devices (USB Webcams)

**Protocol**: Video4Linux2
**Path**: `/dev/video*`
**Examples**: USB webcams, DroidCam USB mode, virtual cameras

**Backend Implementation**:
```python
import cv2

class V4L2Camera:
    def __init__(self, device: str = "/dev/video0"):
        self.device = device
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_frame(self) -> bytes:
        ret, frame = self.cap.read()
        if ret:
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
```

**Discovery**:
- Scan `/dev/video*` devices
- Test with `cv2.VideoCapture().isOpened()`

**Configuration Example**:
```json
{
  "id": "usb_webcam",
  "name": "USB Webcam",
  "type": "v4l2",
  "device": "/dev/video0",
  "fps": 5,
  "enabled": true
}
```

---

## Network Discovery

### Implementation: nmap-based scanning

**Command**:
```bash
# Scan for RTSP cameras (port 554)
nmap -p 554 --open 192.168.2.0/24 -oG -

# Scan for DroidCam (port 4747)
nmap -p 4747 --open 192.168.2.0/24 -oG -

# Scan for common camera ports (batch)
nmap -p 554,4747,8554,1935,8080 --open 192.168.2.0/24 -oG -
```

**Backend Endpoint**:
```python
@router.post("/api/cameras/scan")
async def scan_network(subnet: str = "192.168.2.0/24"):
    """
    Scan network for cameras using nmap
    Returns list of discovered devices with open camera ports
    """
    discovered = []

    # RTSP (port 554, 8554)
    rtsp_results = run_nmap(subnet, ports=[554, 8554])
    for ip, port in rtsp_results:
        # Try to verify RTSP stream
        if verify_rtsp(f"rtsp://{ip}:{port}/stream"):
            discovered.append({
                "type": "rtsp",
                "ip": ip,
                "port": port,
                "suggested_url": f"rtsp://{ip}:{port}/stream"
            })

    # DroidCam (port 4747)
    droidcam_results = run_nmap(subnet, ports=[4747])
    for ip, port in droidcam_results:
        # Verify with HTTP GET
        if verify_droidcam(ip, port):
            discovered.append({
                "type": "droidcam",
                "ip": ip,
                "port": port
            })

    # V4L2 devices (local only)
    v4l2_devices = discover_v4l2()
    discovered.extend(v4l2_devices)

    return {"discovered": discovered}
```

**Frontend UI**:
```tsx
const NetworkScanner = () => {
  const [scanning, setScanning] = useState(false);
  const [discovered, setDiscovered] = useState([]);

  const scan = async () => {
    setScanning(true);
    const response = await fetch('/api/cameras/scan', {
      method: 'POST',
      body: JSON.stringify({ subnet: '192.168.2.0/24' })
    });
    const data = await response.json();
    setDiscovered(data.discovered);
    setScanning(false);
  };

  return (
    <div>
      <button onClick={scan} disabled={scanning}>
        {scanning ? 'Scanning network...' : 'Scan for Cameras'}
      </button>
      {discovered.map(device => (
        <CameraCard device={device} onAdd={addCamera} />
      ))}
    </div>
  );
};
```

---

## Ring Buffer System

### Purpose

Cameras capture frames continuously, but AI processing is expensive and selective. Ring buffers allow:
1. **Historical Context**: When AI switches attention, it can review what happened before the trigger
2. **Memory Efficiency**: Fixed-size buffer (e.g., 5 seconds @ 5fps = 25 frames)
3. **Smooth Playback**: AI can request "last 3 seconds" as video context

### Implementation

**Per-Camera Buffer**:
```python
from collections import deque
import time

class CameraRingBuffer:
    def __init__(self, max_seconds: int = 5, fps: int = 5):
        self.max_frames = max_seconds * fps
        self.buffer = deque(maxlen=self.max_frames)
        self.fps = fps

    def add_frame(self, frame: bytes, timestamp: float = None):
        """Add frame to ring buffer (oldest frame auto-discarded)"""
        if timestamp is None:
            timestamp = time.time()
        self.buffer.append({
            'data': frame,
            'timestamp': timestamp
        })

    def get_last_n_seconds(self, seconds: int) -> list:
        """Retrieve last N seconds of frames"""
        target_frames = seconds * self.fps
        frames = list(self.buffer)[-target_frames:]
        return frames

    def get_all(self) -> list:
        """Retrieve all frames in buffer"""
        return list(self.buffer)

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
```

**Camera Manager Integration**:
```python
class CameraManager:
    def __init__(self):
        self.cameras = {}  # {camera_id: Camera instance}
        self.buffers = {}  # {camera_id: RingBuffer instance}

    def enable_camera(self, camera_id: str):
        """Start streaming and buffering"""
        camera = self.cameras[camera_id]
        buffer = CameraRingBuffer(max_seconds=5, fps=camera.fps)
        self.buffers[camera_id] = buffer

        # Start capture thread
        thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_id, camera, buffer)
        )
        thread.daemon = True
        thread.start()

    def _capture_loop(self, camera_id: str, camera, buffer):
        """Continuous capture loop"""
        while camera.enabled:
            frame = camera.get_frame()
            if frame:
                buffer.add_frame(frame)
            time.sleep(1.0 / camera.fps)

    def get_camera_context(self, camera_id: str, seconds: int = 3):
        """AI requests last N seconds from camera"""
        buffer = self.buffers.get(camera_id)
        if buffer:
            return buffer.get_last_n_seconds(seconds)
        return []
```

**API Endpoint**:
```python
@router.get("/api/cameras/{camera_id}/context")
async def get_camera_context(camera_id: str, seconds: int = 3):
    """
    Get last N seconds of frames from camera buffer
    Used by AI when switching attention to this camera
    """
    frames = camera_manager.get_camera_context(camera_id, seconds)

    # Return as base64-encoded frames for LLM processing
    return {
        "camera_id": camera_id,
        "frames": [
            {
                "data": base64.b64encode(f['data']).decode(),
                "timestamp": f['timestamp']
            }
            for f in frames
        ],
        "count": len(frames),
        "duration_seconds": seconds
    }
```

---

## Salience Mechanism

### Overview

**Goal**: Determine which camera(s) deserve AI attention at any given moment.

**Inputs**:
1. Schedule rules (time-based priority)
2. Event triggers (entrance/exit from reCamera)
3. Manual overrides (user selection)
4. Idle timeout (return to default after inactivity)

**Output**: Priority score per camera (0.0 - 1.0)

### Priority Calculation

```python
class SalienceEngine:
    def __init__(self, config_path: str):
        self.config = load_json(config_path)
        self.active_events = {}  # {camera_id: event_timestamp}

    def calculate_priority(self, camera_id: str) -> float:
        """
        Calculate priority score for camera
        Returns 0.0 - 1.0 (higher = more important)
        """
        score = 0.0
        now = time.time()

        # 1. Schedule-based priority (0.0 - 0.5)
        schedule_score = self._check_schedule(camera_id, now)
        score += schedule_score

        # 2. Event-based boost (0.0 - 0.3)
        if camera_id in self.active_events:
            event_time = self.active_events[camera_id]
            age = now - event_time
            # Decay over 300 seconds (5 minutes)
            event_score = 0.3 * max(0, 1 - (age / 300))
            score += event_score

        # 3. Manual override (0.0 - 0.2)
        if self._is_manually_selected(camera_id):
            score += 0.2

        return min(score, 1.0)

    def _check_schedule(self, camera_id: str, timestamp: float) -> float:
        """Check if current time matches any schedule rules"""
        schedule = self.config.get('schedule', [])
        current_time = datetime.fromtimestamp(timestamp).strftime('%H:%M')
        day_of_week = datetime.fromtimestamp(timestamp).weekday()

        for rule in schedule:
            if rule['camera'] != camera_id:
                continue

            # Check time range
            if rule['start'] <= current_time <= rule['end']:
                # Check day of week (if specified)
                if 'days' not in rule or day_of_week in rule['days']:
                    return rule.get('priority', 0.5)

        return 0.0

    def register_event(self, camera_id: str, event_type: str):
        """Register event from reCamera (entrance/exit)"""
        self.active_events[camera_id] = time.time()
        logger.info(f"Event registered: {event_type} on {camera_id}")

    def get_top_cameras(self, count: int = 2) -> list:
        """Get top N cameras by priority"""
        priorities = {
            cam_id: self.calculate_priority(cam_id)
            for cam_id in self.config.get('cameras', [])
        }
        sorted_cameras = sorted(
            priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_cameras[:count]
```

### Schedule Configuration

**File**: `data/camera_manager/salience_schedule.json`

```json
{
  "schedule": [
    {
      "camera": "workshop_recamera",
      "start": "09:00",
      "end": "17:00",
      "days": [0, 1, 2, 3, 4],  // Monday-Friday
      "priority": 0.5,
      "reason": "Work hours - monitor workshop activity"
    },
    {
      "camera": "living_room_rtsp",
      "start": "18:00",
      "end": "23:00",
      "days": [0, 1, 2, 3, 4, 5, 6],  // Every day
      "priority": 0.4,
      "reason": "Evening - monitor living space"
    },
    {
      "camera": "entrance_cam",
      "start": "00:00",
      "end": "23:59",
      "priority": 0.3,
      "reason": "Always monitor entrance (baseline)"
    }
  ],
  "event_triggers": [
    {
      "source": "workshop_recamera",
      "event": "entrance",
      "boost": 0.3,
      "duration": 300,  // Boost lasts 5 minutes
      "action": "activate",
      "reason": "Person entered - high interest"
    },
    {
      "source": "workshop_recamera",
      "event": "exit",
      "boost": 0.1,
      "duration": 60,
      "reason": "Person left - brief interest"
    }
  ]
}
```

### Stereo Vision Mode

For depth perception or 3D reconstruction, activate 2 cameras simultaneously:

```python
def enable_stereo_mode(left_camera_id: str, right_camera_id: str):
    """
    Activate two cameras for stereo vision
    Both cameras receive equal priority
    """
    salience_engine.set_stereo_pair(left_camera_id, right_camera_id)
    # Both cameras now return priority 1.0 when requested together
```

---

## UI Specification

### Layout: Dashboard View

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BioMimeticAI Perception Dashboard                    [Scan Network]    │
├─────────────────────────────────────────────────────────────────────────┤
│  ACTIVE CAMERAS (6 visible, scroll for more)                           │
│                                                                         │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐│
│  │ Workshop reCamera  │  │ Living Room RTSP   │  │ Phone Camera       ││
│  │ 640x480 preview    │  │ 640x480 preview    │  │ 640x480 preview    ││
│  │                    │  │                    │  │                    ││
│  │ [Live Feed]        │  │ [Live Feed]        │  │ [Offline]          ││
│  │                    │  │                    │  │                    ││
│  │ Priority: ████░ 0.8│  │ Priority: ██░░░ 0.4│  │ Priority: ░░░░░ 0.0││
│  │ AI Focus: ●        │  │ AI Focus: ○        │  │ AI Focus: ○        ││
│  │ Buffer: 25/25      │  │ Buffer: 25/25      │  │ Buffer: 0/25       ││
│  │ [Disable] [Focus]  │  │ [Disable] [Focus]  │  │ [Enable] [Config]  ││
│  └────────────────────┘  └────────────────────┘  └────────────────────┘│
│                                                                         │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐│
│  │ USB Webcam         │  │ Entrance Cam       │  │ [+ Add Camera]     ││
│  │ 640x480 preview    │  │ 640x480 preview    │  │                    ││
│  │ ...                │  │ ...                │  │ Click to add new   ││
│  └────────────────────┘  └────────────────────┘  └────────────────────┘│
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  SALIENCE & EVENTS                                                      │
│                                                                         │
│  Current Mode: [● Schedule-Based] [ Event-Driven] [ Manual]            │
│                                                                         │
│  Active Focus: Workshop reCamera (Priority: 0.8)                       │
│  Reason: Person entrance event + Work hours schedule                   │
│                                                                         │
│  Recent Events:                                                         │
│  14:32 ● Person entered workshop (reCamera trigger)                    │
│  14:30 ○ Scene assessed: Person at desk, laptop visible                │
│  14:15 ○ Schedule triggered: Workshop camera activated                 │
│                                                                         │
│  [Edit Schedule] [View Event Log]                                      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  SYSTEM STATUS                                                          │
│                                                                         │
│  Active Cameras: 4/6        Ring Buffers: 100 frames (20MB total)      │
│  AI Processing: workshop_recamera (analyzing last 3 seconds)           │
│  Memory: 142 episodes stored (vision-tagged)                           │
│  Vision API: ● Online (port 8000)                                      │
│                                                                         │
│  [Memory Browser] [System Logs] [Settings]                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Camera Tile Component

**Size**: 640x480 preview (16:9 or 4:3 aspect ratio maintained)

```tsx
interface CameraTileProps {
  camera: Camera;
  onEnable: (id: string) => void;
  onDisable: (id: string) => void;
  onFocus: (id: string) => void;
  onConfig: (id: string) => void;
}

const CameraTile: React.FC<CameraTileProps> = ({
  camera,
  onEnable,
  onDisable,
  onFocus,
  onConfig
}) => {
  const frameUrl = camera.enabled
    ? `/api/cameras/${camera.id}/frame?t=${Date.now()}`
    : '/images/camera-offline.png';

  return (
    <div className="camera-tile border rounded-lg overflow-hidden">
      {/* Preview Area */}
      <div className="relative w-[640px] h-[480px] bg-gray-900">
        <img
          src={frameUrl}
          alt={camera.name}
          className="w-full h-full object-cover"
        />

        {/* Status Overlays */}
        <div className="absolute top-2 right-2 space-y-1">
          {/* Connection Status */}
          <div className={`px-2 py-1 rounded text-xs ${
            camera.enabled ? 'bg-green-600' : 'bg-gray-600'
          }`}>
            {camera.enabled ? '● Live' : '○ Offline'}
          </div>

          {/* AI Focus Indicator */}
          {camera.ai_focus && (
            <div className="px-2 py-1 bg-blue-600 rounded text-xs animate-pulse">
              🧠 AI Focus
            </div>
          )}
        </div>

        {/* Camera Name */}
        <div className="absolute bottom-2 left-2 bg-black/60 px-3 py-1 rounded">
          {camera.name}
        </div>
      </div>

      {/* Info Bar */}
      <div className="p-3 bg-gray-800 space-y-2">
        {/* Priority Bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Priority:</span>
          <div className="flex-1 bg-gray-700 rounded h-2">
            <div
              className="bg-blue-500 h-full rounded"
              style={{ width: `${camera.priority * 100}%` }}
            />
          </div>
          <span className="text-xs text-gray-400">
            {camera.priority.toFixed(1)}
          </span>
        </div>

        {/* Ring Buffer Status */}
        {camera.enabled && (
          <div className="text-xs text-gray-400">
            Buffer: {camera.buffer_frames}/{camera.buffer_max} frames
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-2">
          {camera.enabled ? (
            <>
              <button
                onClick={() => onDisable(camera.id)}
                className="btn-secondary text-xs"
              >
                Disable
              </button>
              <button
                onClick={() => onFocus(camera.id)}
                className="btn-primary text-xs"
              >
                Focus AI
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => onEnable(camera.id)}
                className="btn-primary text-xs"
              >
                Enable
              </button>
              <button
                onClick={() => onConfig(camera.id)}
                className="btn-secondary text-xs"
              >
                Config
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
```

### Network Scanner Dialog

```tsx
const NetworkScannerDialog = () => {
  const [scanning, setScanning] = useState(false);
  const [discovered, setDiscovered] = useState([]);

  return (
    <Dialog>
      <DialogTitle>Scan Network for Cameras</DialogTitle>

      {!scanning && discovered.length === 0 && (
        <div className="text-center py-8">
          <button onClick={scan} className="btn-primary">
            Start Network Scan
          </button>
          <p className="text-sm text-gray-500 mt-2">
            Scans 192.168.2.0/24 for RTSP (554, 8554), DroidCam (4747)
          </p>
        </div>
      )}

      {scanning && (
        <div className="text-center py-8">
          <Spinner />
          <p>Scanning network... This may take 30-60 seconds</p>
        </div>
      )}

      {discovered.length > 0 && (
        <div className="space-y-3">
          <h3>Found {discovered.length} devices:</h3>
          {discovered.map(device => (
            <div key={device.ip} className="p-3 border rounded">
              <div className="flex justify-between items-center">
                <div>
                  <div className="font-semibold">
                    {device.type.toUpperCase()} Camera
                  </div>
                  <div className="text-sm text-gray-500">
                    {device.ip}:{device.port}
                  </div>
                  {device.suggested_url && (
                    <div className="text-xs text-gray-400 font-mono">
                      {device.suggested_url}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => addCamera(device)}
                  className="btn-primary"
                >
                  Add Camera
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </Dialog>
  );
};
```

### Memory Browser Integration

```tsx
const MemoryBrowser = () => {
  // Query episodic memory for vision-tagged episodes
  const [episodes, setEpisodes] = useState([]);

  useEffect(() => {
    fetch('/api/memory/episodes?tag=vision&limit=50')
      .then(r => r.json())
      .then(data => setEpisodes(data.episodes));
  }, []);

  return (
    <div className="memory-browser">
      <h2>Vision Memory</h2>

      {episodes.map(episode => (
        <div key={episode.id} className="episode-card">
          <div className="flex gap-4">
            {/* Snapshot Thumbnail */}
            <img
              src={episode.snapshot_url}
              className="w-32 h-24 object-cover rounded"
            />

            {/* Episode Details */}
            <div className="flex-1">
              <div className="text-sm text-gray-400">
                {new Date(episode.timestamp).toLocaleString()}
              </div>
              <div className="font-semibold">{episode.event}</div>
              <div className="text-sm">{episode.description}</div>
              <div className="text-xs text-gray-500">
                Camera: {episode.camera_id} | Salience: {episode.salience}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
```

---

## Integration Points

### 1. reCamera Events → Camera Manager

**Existing**: reCamera sends entrance/exit events to `http://192.168.2.137:8000/api/vision/event`

**New**: Camera Manager subscribes to these events

```python
# In Camera Manager backend
@router.post("/api/events/subscribe")
async def subscribe_to_vision_events():
    """
    Register Camera Manager as subscriber to vision events
    Can be called via webhook or polling
    """
    # Poll vision API for new events
    async def poll_events():
        while True:
            events = await fetch_vision_events()
            for event in events:
                salience_engine.register_event(
                    camera_id=event['camera_id'],
                    event_type=event['event']
                )
            await asyncio.sleep(1)

    # Start polling task
    asyncio.create_task(poll_events())
```

**Alternative**: Modify vision API to push events to Camera Manager

```python
# In BioMimeticAI vision_api.py
SUBSCRIBERS = ["http://localhost:8001/api/events/receive"]

@router.post("/api/vision/event")
async def receive_vision_event(event: VisionEvent):
    # Existing processing...

    # Notify subscribers
    for subscriber_url in SUBSCRIBERS:
        try:
            await http_client.post(subscriber_url, json=event.dict())
        except:
            logger.warning(f"Failed to notify subscriber: {subscriber_url}")
```

### 2. Camera Manager → Vision API (Frame Submission)

When AI wants to process a camera:

```python
# Camera Manager sends frames to Vision API
@router.post("/api/ai/process")
async def process_camera(camera_id: str, seconds: int = 3):
    """
    Request AI processing for specific camera
    Retrieves frames from ring buffer and sends to Vision API
    """
    # Get frames from buffer
    frames = camera_manager.get_camera_context(camera_id, seconds)

    # Send to Vision API
    response = await http_client.post(
        "http://192.168.2.137:8000/api/vision/analyze",
        json={
            "camera_id": camera_id,
            "frames": [base64.b64encode(f['data']).decode() for f in frames],
            "request_type": "scene_analysis"
        }
    )

    # Store result in episodic memory
    result = response.json()
    store_episode({
        "source": "vision",
        "camera_id": camera_id,
        "description": result['scene_description'],
        "salience": 0.8,
        "timestamp": time.time()
    })

    return result
```

### 3. Episodic Memory Storage

Vision events stored with high salience:

```python
from src.memory.episodic import EpisodicMemory

episodic = EpisodicMemory()

episodic.add_episode(
    contact_id="system",
    message={
        "type": "vision_event",
        "camera": "workshop_recamera",
        "event": "entrance",
        "object": "person",
        "scene_description": "Person entered workshop, sitting at desk with laptop",
        "snapshot_url": "/data/vision/snapshots/entrance_1234567890.jpg",
        "yolo_detections": [
            {"class": "person", "confidence": 0.95, "bbox": [100, 200, 300, 500]}
        ]
    },
    salience=0.85,  # High salience for person entrance
    metadata={
        "source": "vision",
        "camera_id": "workshop_recamera",
        "tags": ["vision", "entrance", "person"]
    }
)
```

### 4. Contact Memory (Face Recognition)

When Vision API recognizes a face:

```python
from src.memory.contact_memory import ContactMemory

contact_memory = ContactMemory()

# Vision API returns face match
face_result = {
    "contact_id": "toastee0",
    "confidence": 0.92,
    "last_seen": time.time(),
    "location": "workshop_recamera"
}

# Update contact with visual sighting
contact_memory.update_contact(
    contact_id="toastee0",
    updates={
        "last_visual_sighting": time.time(),
        "last_seen_camera": "workshop_recamera",
        "visual_confidence": 0.92
    }
)

# Add to episodic memory
episodic.add_episode(
    contact_id="toastee0",
    message={
        "type": "visual_sighting",
        "camera": "workshop_recamera",
        "confidence": 0.92,
        "description": "toastee0 seen in workshop"
    },
    salience=0.9  # High salience for known person
)
```

---

## API Specification

### Camera Management Endpoints

#### GET `/api/cameras`
List all configured cameras

**Response**:
```json
{
  "cameras": [
    {
      "id": "workshop_recamera",
      "name": "Workshop reCamera E81",
      "type": "rtsp",
      "url": "rtsp://192.168.2.140:8554/stream",
      "enabled": true,
      "connected": true,
      "fps": 5,
      "buffer_frames": 25,
      "buffer_max": 25,
      "priority": 0.8,
      "ai_focus": true
    }
  ]
}
```

#### POST `/api/cameras`
Add new camera

**Request**:
```json
{
  "name": "New Camera",
  "type": "rtsp",
  "url": "rtsp://192.168.2.150:8554/stream",
  "fps": 5,
  "enabled": false
}
```

#### PUT `/api/cameras/{camera_id}`
Update camera configuration

#### DELETE `/api/cameras/{camera_id}`
Remove camera

#### POST `/api/cameras/{camera_id}/enable`
Enable camera (start streaming and buffering)

#### POST `/api/cameras/{camera_id}/disable`
Disable camera (stop streaming)

#### GET `/api/cameras/{camera_id}/frame`
Get latest frame from camera

**Response**: JPEG image (binary)

#### GET `/api/cameras/{camera_id}/context`
Get ring buffer frames (last N seconds)

**Query Params**:
- `seconds` (int): How many seconds to retrieve (default: 3)

**Response**:
```json
{
  "camera_id": "workshop_recamera",
  "frames": [
    {
      "data": "base64_encoded_jpeg...",
      "timestamp": 1734567890.123
    }
  ],
  "count": 15,
  "duration_seconds": 3
}
```

### Salience Endpoints

#### GET `/api/salience/priorities`
Get current priority scores for all cameras

**Response**:
```json
{
  "priorities": {
    "workshop_recamera": 0.8,
    "living_room_rtsp": 0.4,
    "phone_main": 0.0
  },
  "top_cameras": [
    "workshop_recamera",
    "living_room_rtsp"
  ],
  "active_events": [
    {
      "camera_id": "workshop_recamera",
      "event": "entrance",
      "timestamp": 1734567890,
      "age_seconds": 45
    }
  ]
}
```

#### POST `/api/salience/focus`
Manually set AI focus to specific camera

**Request**:
```json
{
  "camera_id": "workshop_recamera",
  "duration": 300  // Focus for 5 minutes
}
```

#### GET `/api/salience/schedule`
Get salience schedule configuration

#### PUT `/api/salience/schedule`
Update salience schedule

### Network Discovery Endpoints

#### POST `/api/cameras/scan`
Scan network for cameras

**Request**:
```json
{
  "subnet": "192.168.2.0/24",
  "ports": [554, 4747, 8554]
}
```

**Response**:
```json
{
  "discovered": [
    {
      "type": "rtsp",
      "ip": "192.168.2.140",
      "port": 8554,
      "suggested_url": "rtsp://192.168.2.140:8554/stream",
      "verified": true
    },
    {
      "type": "droidcam",
      "ip": "192.168.2.39",
      "port": 4747,
      "verified": true
    }
  ],
  "scan_time": 45.2
}
```

#### GET `/api/cameras/discover/v4l2`
List local V4L2 devices

**Response**:
```json
{
  "devices": [
    {
      "device": "/dev/video0",
      "name": "USB Webcam",
      "capabilities": ["video_capture"]
    }
  ]
}
```

### Memory Integration Endpoints

#### GET `/api/memory/episodes`
Query episodic memory (vision-tagged)

**Query Params**:
- `tag` (string): Filter by tag (e.g., "vision")
- `camera_id` (string): Filter by camera
- `limit` (int): Max results

#### POST `/api/ai/process`
Request AI processing for camera

**Request**:
```json
{
  "camera_id": "workshop_recamera",
  "seconds": 3,
  "analysis_type": "scene_description"
}
```

**Response**:
```json
{
  "camera_id": "workshop_recamera",
  "scene_description": "Person sitting at desk with laptop, typing",
  "objects_detected": ["person", "desk", "laptop"],
  "faces_detected": [
    {
      "contact_id": "toastee0",
      "confidence": 0.92
    }
  ],
  "salience": 0.85,
  "episode_id": 142
}
```

---

## Configuration Files

### 1. Camera Configurations
**Path**: `data/camera_manager/cameras.json`

```json
{
  "cameras": [
    {
      "id": "workshop_recamera",
      "name": "Workshop reCamera E81",
      "type": "rtsp",
      "url": "rtsp://192.168.2.140:8554/stream",
      "fps": 5,
      "enabled": true,
      "buffer_seconds": 5,
      "events": {
        "entrance_exit": true,
        "yolo_detection": true
      },
      "tags": ["workshop", "entrance_detection"]
    },
    {
      "id": "living_room_rtsp",
      "name": "Living Room Camera",
      "type": "rtsp",
      "url": "rtsp://192.168.2.145:554/stream",
      "fps": 5,
      "enabled": true,
      "buffer_seconds": 5,
      "tags": ["living_room"]
    },
    {
      "id": "phone_main",
      "name": "Main Phone Camera",
      "type": "droidcam",
      "ip": "192.168.2.39",
      "port": 4747,
      "fps": 5,
      "enabled": false,
      "buffer_seconds": 5,
      "tags": ["mobile", "portable"]
    },
    {
      "id": "usb_webcam",
      "name": "USB Webcam",
      "type": "v4l2",
      "device": "/dev/video0",
      "fps": 5,
      "enabled": true,
      "buffer_seconds": 5,
      "tags": ["usb", "local"]
    }
  ]
}
```

### 2. Salience Schedule
**Path**: `data/camera_manager/salience_schedule.json`

(See [Salience Mechanism](#salience-mechanism) section for full example)

### 3. System Configuration
**Path**: `data/camera_manager/config.json`

```json
{
  "ring_buffer": {
    "max_seconds": 5,
    "fps": 5,
    "max_memory_mb": 50
  },
  "network_scan": {
    "subnet": "192.168.2.0/24",
    "ports": [554, 4747, 8554, 1935],
    "timeout": 60
  },
  "integration": {
    "vision_api_url": "http://192.168.2.137:8000",
    "biomimetic_db": "/home/toastee/BioMimeticAi/data/biomim.db"
  },
  "ui": {
    "tile_width": 640,
    "tile_height": 480,
    "max_visible_cameras": 6,
    "refresh_rate_ms": 100
  }
}
```

---

## Installation

### Prerequisites

1. **System Dependencies**:
```bash
# nmap for network scanning
sudo apt install nmap

# ffmpeg for RTSP capture
sudo apt install ffmpeg

# OpenCV for V4L2 capture
sudo apt install python3-opencv

# v4l-utils for device management
sudo apt install v4l-utils
```

2. **Python Dependencies** (backend):
```bash
cd /home/toastee/ai_led_mapper/backend
pip install -r requirements.txt
# Add new dependencies:
pip install python-nmap aiohttp
```

3. **Node Dependencies** (frontend):
```bash
cd /home/toastee/ai_led_mapper/frontend
npm install
```

### Setup Steps

1. **Clone/Update ai_led_mapper**:
```bash
cd /home/toastee/ai_led_mapper
git pull  # Or ensure latest version
```

2. **Create Camera Manager directories**:
```bash
mkdir -p /home/toastee/ai_led_mapper/data/camera_manager
mkdir -p /home/toastee/ai_led_mapper/data/ring_buffers
```

3. **Initialize configuration**:
```bash
# Copy template configs
cp /home/toastee/BioMimeticAi/docs/camera_manager_templates/* \
   /home/toastee/ai_led_mapper/data/camera_manager/
```

4. **Update systemd service** (if using):
```bash
# Edit led-mapper-backend.service to point to new Camera Manager backend
sudo systemctl edit led-mapper-backend
```

5. **Start services**:
```bash
# Backend
cd /home/toastee/ai_led_mapper/backend
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Frontend
cd /home/toastee/ai_led_mapper/frontend
npm run dev
```

6. **Verify integration**:
```bash
# Test camera API
curl http://localhost:8001/api/cameras

# Test vision API connection
curl http://192.168.2.137:8000/api/vision/status

# Test network scan
curl -X POST http://localhost:8001/api/cameras/scan
```

---

## Implementation Roadmap

### Phase 1: Core Camera Manager (Week 1)
- [ ] Backend: Camera abstraction classes (RTSP, DroidCam, V4L2)
- [ ] Backend: Ring buffer implementation
- [ ] Backend: Camera manager (enable/disable/status)
- [ ] API: Basic CRUD endpoints for cameras
- [ ] Frontend: Camera tile component (640x480 previews)
- [ ] Frontend: Grid layout with 6 visible cameras

### Phase 2: Network Discovery (Week 1)
- [ ] Backend: nmap integration for port scanning
- [ ] Backend: RTSP verification (ffprobe)
- [ ] Backend: DroidCam verification (HTTP GET)
- [ ] Backend: V4L2 device enumeration
- [ ] Frontend: Network scanner dialog
- [ ] Frontend: Add discovered camera flow

### Phase 3: Salience System (Week 2)
- [ ] Backend: Salience engine with priority calculation
- [ ] Backend: Schedule-based priority
- [ ] Backend: Event-based priority (reCamera integration)
- [ ] Backend: Manual focus override
- [ ] API: Salience endpoints
- [ ] Frontend: Priority visualization (bars)
- [ ] Frontend: Schedule editor UI
- [ ] Frontend: Event log viewer

### Phase 4: BioMimeticAI Integration (Week 2)
- [ ] Backend: Subscribe to vision API events
- [ ] Backend: Send frames to vision API for processing
- [ ] Backend: Store results in episodic memory
- [ ] Backend: Face recognition → contact memory updates
- [ ] API: Memory query endpoints
- [ ] Frontend: Memory browser component
- [ ] Frontend: Visual timeline of events

### Phase 5: Polish & Documentation (Week 3)
- [ ] Update all BioMimeticAI docs with Camera Manager integration
- [ ] Create user guide with screenshots
- [ ] Add system health monitoring
- [ ] Performance optimization (reduce bandwidth, memory)
- [ ] Error handling and recovery
- [ ] Unit tests for critical components

---

## Future Enhancements

### Advanced Vision Processing
- [ ] Object tracking across cameras (person re-identification)
- [ ] Behavior analysis (gait, posture, activity recognition)
- [ ] Anomaly detection (unusual patterns)
- [ ] Multi-camera fusion (3D pose estimation)

### Automation
- [ ] Auto-schedule learning (adapt to patterns)
- [ ] Predictive camera activation (anticipate events)
- [ ] Auto-cleanup of low-salience episodes
- [ ] Camera health monitoring (auto-restart on failure)

### UI Improvements
- [ ] Dark/light theme toggle
- [ ] Customizable tile layouts
- [ ] Keyboard shortcuts for camera switching
- [ ] Touch/gesture support for tablets
- [ ] Export camera configs as JSON

### Hardware Integration
- [ ] PTZ camera control (pan, tilt, zoom)
- [ ] Audio streaming from cameras with microphones
- [ ] Two-way audio (intercom mode)
- [ ] Motion-triggered recording

---

## Appendix: reCamera Integration Details

### reCamera Hardware
- **Model**: reCamera E81 (assuming, based on context)
- **IP**: 192.168.2.140
- **RTSP Stream**: `rtsp://192.168.2.140:8554/stream`
- **Platform**: Node-RED running on reCamera
- **Detection**: YOLO-based object detection

### Event Flow
```
1. reCamera YOLO detects objects in frame
2. Node-RED processes detections with debouncing (3 seconds)
3. Entrance/exit events sent to Vision API (192.168.2.137:8000)
4. Vision API triggers snapshot capture
5. Camera Manager receives event notification
6. Salience engine updates priority for reCamera
7. If priority high enough, AI switches focus to reCamera
8. Ring buffer frames sent to Vision API for analysis
9. Results stored in episodic memory
```

### Extending reCamera Events

To add new event types from reCamera:

**1. Update Node-RED function** on reCamera:
```javascript
// In camera_entrance_exit_tracker.js
// Add new event type
if (detected_condition) {
    node.send([{
        url: CORE_API_URL,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        payload: {
            event: "new_event_type",  // e.g., "loitering", "fall_detection"
            object: obj,
            timestamp: now,
            scene: Object.keys(sceneState.objects),
            metadata: { /* custom data */ }
        }
    }, null]);
}
```

**2. Update salience schedule**:
```json
{
  "event_triggers": [
    {
      "source": "workshop_recamera",
      "event": "fall_detection",
      "boost": 0.9,  // High priority!
      "duration": 600,
      "reason": "Potential emergency"
    }
  ]
}
```

**3. Handle in Camera Manager**:
```python
# Salience engine automatically handles new event types
# No code changes needed if using generic event boost system
```

---

## Document History

- **2025-12-19**: Initial version (comprehensive design document)
- Future updates will be tracked here

---

## Contact & Support

For questions about this system, refer to:
- `/home/toastee/BioMimeticAi/docs/VISION_SYSTEM.md` - Vision API details
- `/home/toastee/BioMimeticAi/docs/VISION_ARCHITECTURE.md` - Vision processing architecture
- `/home/toastee/ai_led_mapper/CAMERA_INTEGRATION.md` - Original camera integration notes

System is actively developed. Documentation will be updated as implementation progresses.
