# Update reCamera Node-RED Flow with YOLO Metadata Support

## Quick Deploy

### Option 1: Copy the updated script to reCamera

```bash
# From your workstation
scp -i ~/.ssh/recamera_key ~/BioMimeticAi/scripts/camera_entrance_exit_tracker_with_yolo.js recamera@192.168.2.140:/tmp/
```

Then in Node-RED on the reCamera:
1. Open your existing function node that handles entrance/exit tracking
2. Replace the code with the content from `camera_entrance_exit_tracker_with_yolo.js`
3. Make sure the function node has **2 outputs** configured
4. Connect output 1 to an **http request** node with these settings:
   - Method: Use msg.method
   - URL: Use msg.url
   - Return: a parsed JSON object
5. Connect output 2 to your debug node (optional)
6. Deploy the flow

### Option 2: Manual Update in Node-RED

1. Access Node-RED: `http://192.168.2.140:1880`
2. Find your entrance/exit tracker function node
3. Copy the contents of `camera_entrance_exit_tracker_with_yolo.js`
4. Paste into the function node
5. Ensure function has **2 outputs**
6. Add/verify http request node connected to output 1
7. Deploy

## What Changed

✅ **YOLO Metadata Extraction**
- Now extracts bounding boxes, confidence scores, and track IDs from SSCMA detections
- Stores YOLO data with each object in scene state

✅ **Enhanced Event Payload**
- Entrance events now include:
  ```json
  {
    "event": "entrance",
    "object": "person",
    "timestamp": 1733850123456,
    "scene": ["person"],
    "yolo_detection": {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 150, 300, 450],
      "track_id": 0
    }
  }
  ```

✅ **Exit events preserve last known YOLO data**
- Even when object exits, vision system gets the bbox from last detection

## Verify It's Working

### 1. Check Node-RED Debug
Look for messages like:
```
✓ Entrance confirmed: person (confidence: 0.95)
✓ Exit confirmed: person (was present for 5.2s)
```

### 2. Check Vision API Status
```bash
curl http://192.168.2.137:8000/api/vision/status
```

Should show recent events.

### 3. Check Vision API Events
```bash
curl http://192.168.2.137:8000/api/vision/events | jq
```

Should show events with `yolo_detection` field.

### 4. Check Snapshot Queue
```bash
ls -lah ~/BioMimeticAi/data/vision/snapshots/
cat ~/BioMimeticAi/data/vision/snapshot_queue.json | jq
```

## Restart Node-RED on reCamera (if needed)

```bash
# SSH to reCamera
ssh -i ~/.ssh/recamera_key recamera@192.168.2.140

# Restart Node-RED service
echo "Watson64!" | sudo -S systemctl restart node-red

# Check status
sudo systemctl status node-red
```

## Troubleshooting

**No events being sent?**
- Check Node-RED debug panel for errors
- Verify http request node is configured correctly
- Check that vision API is running: `curl http://192.168.2.137:8000/api/vision/status`

**YOLO metadata missing?**
- Verify SSCMA node is sending `boxes` and `scores` in `msg.payload.data`
- Check Node-RED debug output to see detection structure

**Can't connect to vision API?**
- Verify IP address (should be 192.168.2.137 for your core system)
- Check vision API is running: `bash ~/BioMimeticAi/scripts/start_vision_api.sh`
- Verify firewall allows port 8000

## Expected Behavior

Once deployed:
1. **Detection**: reCamera YOLO detects person
2. **Debounce**: Waits 3 seconds for stable detection
3. **Entrance Event**: Sends to vision API with YOLO metadata
4. **Snapshot**: Vision API captures RTSP snapshot
5. **Queue**: Snapshot queued for processing
6. **Processing**: Vision cortex processes every 30 minutes
7. **Analysis**: Face recognition, emotion, age detection
8. **Storage**: Results in episodic memory

Move in and out of frame to test!
