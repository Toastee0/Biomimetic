// Entrance/Exit Detection with Debouncing
// Replaces the simple file logging with intelligent scene tracking

const DEBOUNCE_TIME = 3000; // 3 seconds - object must be stable before triggering event
const CORE_API_URL = "http://192.168.2.137:8000/api/vision/event"; // BioMimeticAI core endpoint

// Get current labels from detection
const labels = msg.payload?.data?.labels ?? [];
if (!Array.isArray(labels)) {
    return null;
}

// Filter valid detections (length > 1)
const currentDetections = labels.filter(label => {
    const str = String(label);
    return str.length > 1;
});

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
        // Object already in scene - update last seen time
        sceneState.objects[obj].lastSeen = now;

        // Cancel any pending exit for this object (it's back!)
        if (pendingChanges.exit[obj]) {
            delete pendingChanges.exit[obj];
        }
    } else {
        // New object detected - add to pending entrance
        if (!pendingChanges.enter[obj]) {
            pendingChanges.enter[obj] = {
                firstSeen: now,
                count: 1
            };
        } else {
            pendingChanges.enter[obj].count++;
        }

        // Check if object has been stable long enough to confirm entrance
        if (now - pendingChanges.enter[obj].firstSeen >= DEBOUNCE_TIME) {
            // Confirmed entrance!
            sceneState.objects[obj] = {
                enteredAt: now,
                lastSeen: now
            };

            // Send entrance notification to core
            node.send([{
                payload: {
                    event: "entrance",
                    object: obj,
                    timestamp: now,
                    scene: Object.keys(sceneState.objects)
                }
            }, null]);

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

            // Send exit notification to core
            node.send([{
                payload: {
                    event: "exit",
                    object: obj,
                    timestamp: exitTime,
                    duration: duration,
                    scene: Object.keys(sceneState.objects).filter(o => o !== obj)
                }
            }, null]);

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

// Debug output
const debugInfo = {
    currentDetections: currentDetections,
    confirmedObjects: Object.keys(sceneState.objects),
    pendingEntrances: Object.keys(pendingChanges.enter),
    pendingExits: Object.keys(pendingChanges.exit),
    timestamp: now
};

// Return debug info to second output (for the existing debug node)
return [null, { payload: debugInfo }];
