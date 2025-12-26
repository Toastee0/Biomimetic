# DroidCam Integration

Integration of DroidCam as an additional camera source for the BioMimeticAI vision system.

## Overview

The vision system now supports multiple camera sources, allowing the AI to shift its attention between different viewpoints. DroidCam turns an Android phone into a wireless camera that can be used alongside the fixed reCamera.

## Use Cases

- **Flexible viewing angles**: Move the phone camera to different locations
- **Backup camera**: If reCamera fails, DroidCam provides redundancy
- **Multi-perspective monitoring**: Switch between fixed and mobile cameras
- **Development/Testing**: Test vision features without RTSP camera

## Architecture

DroidCam creates a virtual camera device (`/dev/video0`) that OpenCV can access:

```
┌─────────────────┐
│  Android Phone  │
│  DroidCam App   │
│  192.168.2.39   │
└────────┬────────┘
         │ WiFi (Port 4747)
         │
         ▼
┌────────────────────┐
│  droidcam-cli      │  (Creates virtual camera device)
│  /dev/video0       │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  DroidCamCapture   │  (OpenCV cv2.VideoCapture(0))
│  Snapshot to disk  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Camera Source Mgr  │  (Switches between RTSP/DroidCam)
│ Vision API         │
└────────────────────┘
```

## Prerequisites

### 1. Install DroidCam CLI

```bash
cd /tmp
wget -O droidcam_latest.zip https://files.dev47apps.net/linux/droidcam_2.1.3.zip
unzip droidcam_latest.zip -d droidcam
cd droidcam
sudo ./install-client
```

Verify installation:
```bash
which droidcam-cli
# Should output: /usr/bin/droidcam-cli
```

### 2. Install DroidCam App on Phone

- Download from Google Play Store: [DroidCam](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)
- Open app and note the IP address displayed (e.g., 192.168.2.39)
- Ensure phone and server are on same WiFi network

### 3. Test DroidCam Connection

```bash
# Start DroidCam manually
droidcam-cli 192.168.2.39 4747

# In another terminal, test camera access
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK!' if cap.isOpened() else 'Failed'); cap.release()"
```

## Configuration

Edit `config/camera_sources.json` to configure DroidCam:

```json
{
  "active_source": "rtsp_recamera",
  "sources": {
    "rtsp_recamera": {
      "type": "rtsp",
      "enabled": true,
      "name": "reCamera (Front Door)",
      "rtsp_url": "rtsp://192.168.2.140:8554/stream",
      "snapshot_dir": "data/vision/snapshots"
    },
    "droidcam_phone": {
      "type": "droidcam",
      "enabled": true,
      "name": "DroidCam Phone",
      "phone_ip": "192.168.2.39",
      "port": 4747,
      "snapshot_dir": "data/vision/snapshots"
    }
  }
}
```

**Configuration Options**:
- `phone_ip`: IP address shown in DroidCam app
- `port`: DroidCam port (default: 4747)
- `enabled`: Set to `false` to disable source without deleting config
- `snapshot_dir`: Where snapshots are saved (shared with RTSP by default)

## Usage

### Python API

```python
from src.core.camera_source_manager import CameraSourceManager

manager = CameraSourceManager()

# List all available sources
for name, info in manager.list_sources().items():
    print(f"{name}: {info['display_name']} ({'ACTIVE' if info['is_active'] else 'inactive'})")

# Switch to DroidCam
manager.switch_source("droidcam_phone")

# Capture snapshot from active source
snapshot_path = manager.capture_snapshot()
print(f"Saved to: {snapshot_path}")

# Capture from specific source
snapshot_path = manager.capture_snapshot(source="droidcam_phone")
```

### HTTP API

**List camera sources**:
```bash
curl http://192.168.2.137:8000/api/vision/sources
```

Response:
```json
{
  "status": "success",
  "sources": {
    "rtsp_recamera": {
      "name": "rtsp_recamera",
      "display_name": "reCamera (Front Door)",
      "type": "rtsp",
      "enabled": true,
      "is_active": true
    },
    "droidcam_phone": {
      "name": "droidcam_phone",
      "display_name": "DroidCam Phone",
      "type": "droidcam",
      "enabled": true,
      "is_active": false
    }
  },
  "active_source": "rtsp_recamera"
}
```

**Switch to DroidCam**:
```bash
curl -X POST http://192.168.2.137:8000/api/vision/sources/droidcam_phone
```

Response:
```json
{
  "status": "success",
  "message": "Switched to camera source: droidcam_phone",
  "active_source": {
    "name": "droidcam_phone",
    "display_name": "DroidCam Phone",
    "type": "droidcam",
    "is_active": true
  }
}
```

**Test connection**:
```bash
curl -X POST http://192.168.2.137:8000/api/vision/sources/droidcam_phone/test
```

Response:
```json
{
  "status": "success",
  "source": "droidcam_phone",
  "connection_ok": true
}
```

## Auto-Reconnection

The DroidCamCapture module includes automatic reconnection:

- **Polling**: If phone disconnects, it polls every 5 seconds for reconnection
- **Auto-start**: When port becomes reachable, automatically starts droidcam-cli
- **Status checking**: Verifies both process and camera accessibility

Example:
```python
from src.core.droidcam_capture import DroidCamCapture

capture = DroidCamCapture(phone_ip="192.168.2.39")

# Start with polling (if phone not immediately available)
capture.manager.restart()  # Returns immediately, polls in background

# Later captures will work once phone connects
snapshot = capture.capture_snapshot()
```

## Troubleshooting

### "Camera port not reachable"

1. Verify phone IP address in DroidCam app matches configuration
2. Ensure phone and server on same network:
   ```bash
   ping 192.168.2.39
   ```
3. Check firewall on phone (some phones block DroidCam port)

### "Camera device not found" or "Failed to open camera"

1. Check if droidcam-cli is running:
   ```bash
   pgrep -f droidcam-cli
   ```
2. Kill any stale processes:
   ```bash
   pkill -9 droidcam-cli
   ```
3. Restart via API or manager:
   ```bash
   curl -X POST http://192.168.2.137:8000/api/vision/sources/droidcam_phone/test
   ```

### "Connection failed immediately"

- DroidCam app might be closed on phone - reopen it
- Phone might be in power saving mode - disable for DroidCam
- Try restarting DroidCam app on phone

### Multiple Video Devices

If you have other cameras creating `/dev/video*` devices, OpenCV might grab wrong one:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

DroidCam creates `/dev/video0` by default. If conflicts exist, you may need to modify DroidCamCapture to use different device index.

## Testing

### Test DroidCam Capture Module

```bash
source venv/bin/activate
python src/core/droidcam_capture.py 192.168.2.39 4747
```

Expected output:
```
[DROIDCAM] Initialized - Phone: 192.168.2.39:4747
[DROIDCAM] Snapshots will be saved to: /home/user/BioMimeticAi/data/vision/snapshots
[DROIDCAM] Testing connection to 192.168.2.39:4747...
[DROIDCAM] Starting DroidCam connection...
✓ DroidCam process started: 192.168.2.39:4747
✓ Camera is accessible
✓ Connection test successful

[TEST] Capturing 3 test snapshots...
[TEST] 1/3 - Saved to /path/to/snapshot.jpg
...
```

### Test Camera Source Manager

```bash
source venv/bin/activate
python src/core/camera_source_manager.py droidcam_phone
```

This will:
1. List all available sources
2. Test active source
3. Switch to DroidCam
4. Capture test snapshot from DroidCam

## Integration with Vision Cortex

The vision cortex will automatically use the active camera source when processing snapshots. To make DroidCam the default:

1. Edit `config/camera_sources.json`:
   ```json
   {
     "active_source": "droidcam_phone",
     ...
   }
   ```

2. Restart vision API:
   ```bash
   bash scripts/start_vision_api.sh
   ```

All subsequent entrance/exit events will capture from DroidCam.

## Biomimetic Attention Shifting

The AI can now shift attention between cameras based on context:

```python
# Pseudocode for future implementation
if axiom_requires_mobility:
    camera_manager.switch_source("droidcam_phone")
    message = "Shifting attention to mobile camera for flexible perspective"
elif axiom_requires_fixed_monitoring:
    camera_manager.switch_source("rtsp_recamera")
    message = "Shifting attention to fixed door camera"

episodic.store_episode(
    user_id="system_vision",
    username="CameraSystem",
    user_message=message,
    bot_response="Attention shifted",
    hemisphere="sensory"
)
```

This mimics biological attention mechanisms where the brain shifts focus between different sensory inputs based on task requirements.

## Performance

- **RTSP (ffmpeg)**: ~2-3 seconds per snapshot (includes network stream connection)
- **DroidCam (cv2)**: ~0.5-1 second per snapshot (direct device access)

DroidCam is faster for single-frame capture since it doesn't need to negotiate RTSP stream.

## Future Enhancements

- [ ] Support multiple DroidCam devices simultaneously
- [ ] Pan/tilt control via DroidCam Pro API
- [ ] Video recording from DroidCam
- [ ] Automatic source selection based on scene analysis
- [ ] Gesture recognition to trigger camera switching
- [ ] Integration with rover control (mount phone on rover)