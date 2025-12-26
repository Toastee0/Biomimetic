// Entrance/Exit Detection with Debouncing + YOLO Metadata
// Enhanced version that sends YOLO detection metadata to vision system

const DEBOUNCE_TIME = 3000; // 3 seconds - object must be stable before triggering event
const CORE_API_URL = "http://192.168.2.137:8000/api/vision/event"; // BioMimeticAI core endpoint

// Get current detection data
const detectionData = msg.payload?.data;
if (!detectionData) {
    return null;
}

// Extract labels and detection metadata
const labels = detectionData.labels ?? [];
const boxes = detectionData.boxes ?? [];      // YOLO bounding boxes
const scores = detectionData.scores ?? [];    // YOLO confidence scores
const target = detectionData.target ?? 0;     // Current tracked target index

if (!Array.isArray(labels)) {
    return null;
}

// Filter valid detections and build detection map with YOLO metadata
const detectionMap = {};
labels.forEach((label, idx) => {
    const str = String(label);
    if (str.length > 1) {
        const bbox = boxes[idx] || null;
        const confidence = scores[idx] || 0;
        
        detectionMap[str] = {
            class: str,
            confidence: confidence,
            bbox: bbox,  // [x1, y1, x2, y2]
            track_id: idx
        };
    }
});

const currentDetections = Object.keys(detectionMap);

// Get stored state from flow context
let sceneState = flow.get("scene_state") || {};
let pendingChanges = flow.get("pending_changes") || {};

const now = Date.now();

// Initialize scene state structure if needed
if (!sceneState.objects) {
    sceneState = {
        objects: {},        // Currently confirmed objects in scene
        lastUpdate: now
    };
}

if (!pendingChanges.enter) {
    pendingChanges = {
        enter: {},          // Objects waiting to be confirmed as entered
        exit: {}            // Objects waiting to be confirmed as exited
    };
}

// Track currently detected objects
const detectedSet = new Set(currentDetections);

// Process each currently detected object
for (const obj of currentDetections) {
    if (sceneState.objects[obj]) {
        // Object already in scene - update last seen time and YOLO data
        sceneState.objects[obj].lastSeen = now;
        sceneState.objects[obj].lastYoloData = detectionMap[obj];

        // Cancel any pending exit for this object (it's back!)
        if (pendingChanges.exit[obj]) {
            delete pendingChanges.exit[obj];
        }
    } else {
        // New object detected - add to pending entrance
        if (!pendingChanges.enter[obj]) {
            pendingChanges.enter[obj] = {
                firstSeen: now,
                count: 1,
                yoloData: detectionMap[obj]
            };
        } else {
            pendingChanges.enter[obj].count++;
            pendingChanges.enter[obj].yoloData = detectionMap[obj]; // Update with latest
        }

        // Check if object has been stable long enough to confirm entrance
        if (now - pendingChanges.enter[obj].firstSeen >= DEBOUNCE_TIME) {
            // Confirmed entrance!
            sceneState.objects[obj] = {
                enteredAt: now,
                lastSeen: now,
                lastYoloData: detectionMap[obj]
            };

            // Send entrance notification to core with YOLO metadata
            const eventPayload = {
                event: "entrance",
                object: obj,
                timestamp: now,
                scene: Object.keys(sceneState.objects),
                yolo_detection: detectionMap[obj]
            };

            // Send via HTTP request node (output 1)
            node.send([{
                url: CORE_API_URL,
                method: "POST",
                headers: { "Content-Type": "application/json" },
                payload: eventPayload
            }, null]);

            node.warn(`✓ Entrance confirmed: ${obj} (confidence: ${detectionMap[obj].confidence})`);

            // Clean up pending
            delete pendingChanges.enter[obj];
        }
    }
}

// Check for objects that have exited the scene
for (const obj in sceneState.objects) {
    if (!detectedSet.has(obj)) {
        // Object no longer detected
        if (!pendingChanges.exit[obj]) {
            pendingChanges.exit[obj] = {
                firstMissing: now,
                count: 1
            };
        } else {
            pendingChanges.exit[obj].count++;
        }

        // Check if object has been missing long enough to confirm exit
        if (now - pendingChanges.exit[obj].firstMissing >= DEBOUNCE_TIME) {
            // Confirmed exit!
            const exitTime = now;
            const duration = exitTime - sceneState.objects[obj].enteredAt;

            // Send exit notification to core (with last known YOLO data)
            const eventPayload = {
                event: "exit",
                object: obj,
                timestamp: exitTime,
                duration: duration,
                scene: Object.keys(sceneState.objects).filter(o => o !== obj),
                yolo_detection: sceneState.objects[obj].lastYoloData || null
            };

            // Send via HTTP request node (output 1)
            node.send([{
                url: CORE_API_URL,
                method: "POST",
                headers: { "Content-Type": "application/json" },
                payload: eventPayload
            }, null]);

            node.warn(`✓ Exit confirmed: ${obj} (was present for ${(duration/1000).toFixed(1)}s)`);

            // Remove from scene
            delete sceneState.objects[obj];
            delete pendingChanges.exit[obj];
        }
    } else {
        // Object is still detected - cancel any pending exit
        if (pendingChanges.exit[obj]) {
            delete pendingChanges.exit[obj];
        }
    }
}

// Save state back to flow context
sceneState.lastUpdate = now;
flow.set("scene_state", sceneState);
flow.set("pending_changes", pendingChanges);

// Debug output with YOLO info
const debugInfo = {
    currentDetections: currentDetections.map(obj => ({
        object: obj,
        confidence: detectionMap[obj].confidence,
        bbox: detectionMap[obj].bbox
    })),
    confirmedObjects: Object.keys(sceneState.objects),
    pendingEntrances: Object.keys(pendingChanges.enter),
    pendingExits: Object.keys(pendingChanges.exit),
    timestamp: now
};

// Return debug info to second output
return [null, { payload: debugInfo }];
